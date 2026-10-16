// Diagnostic adapter for CUDA 13.0 CUPTI's existing PM sampling API.
// No kernels, runtime ownership, replay, cache flush, or clock control.
#include <c10/util/ApproximateClock.h>
#include <cstdio>
#include <cupti_pmsampling.h>
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static CUpti_PmSampling_Object *sampler = nullptr;
static CUpti_Profiler_Host_Object *host = nullptr;
static std::vector<std::string> names;
static std::vector<const char *> metrics;
static std::vector<uint8_t> config, data;
static std::unique_ptr<c10::ApproximateClockToUnixTimeConverter>
    clock_converter;
static std::function<c10::time_t(c10::approx_time_t)> convert_clock;
static bool clock_active = false;
extern "C" CUptiResult CUPTIAPI cuptiGetTimestamp(uint64_t *timestamp);
static void check(CUptiResult r, const char *api) {
  if (r != CUPTI_SUCCESS) {
    const char *detail = nullptr;
    cuptiGetResultString(r, &detail);
    throw std::runtime_error(std::string(api) + ": " +
                             (detail ? detail : "unknown"));
  }
}
#define CALL(api, p) check(api(&(p)), #api)
#define PARAM(type, var)                                                       \
  type var = {};                                                               \
  var.structSize = type##_STRUCT_SIZE
#define GUARDED(body)                                                          \
  try {                                                                        \
    body;                                                                      \
    return 0;                                                                  \
  } catch (const std::exception &e) {                                          \
    std::fprintf(stderr, "%s\n", e.what());                                    \
    return 1;                                                                  \
  }

// Match Kineto's lifecycle: start calibration before capture, finish after it.
// The Python convenience wrapper fixes its slope immediately at construction.
extern "C" int state_clock_begin() {
  GUARDED(if (clock_active) throw std::runtime_error("Clock already started");
          convert_clock = {};
          clock_converter =
              std::make_unique<c10::ApproximateClockToUnixTimeConverter>();
          clock_active = true;)
}
extern "C" int state_clock_finish() {
  GUARDED(if (!clock_active ||
              !clock_converter) throw std::runtime_error("Clock not started");
          convert_clock = clock_converter->makeConverter();
          clock_active = false;)
}
extern "C" uint64_t state_clock_convert(uint64_t ticks) {
  if (!convert_clock) {
    std::fprintf(stderr, "Clock conversion not initialized\n");
    return 0;
  }
  return static_cast<uint64_t>(convert_clock(ticks));
}
extern "C" int state_clock_anchor(uint64_t *values) {
  GUARDED(if (!values) throw std::invalid_argument("Null clock anchor");
          values[0] = c10::getTime(); values[3] = c10::getApproximateTime();
          CALL(cuptiGetTimestamp, values[1]);
          values[4] = c10::getApproximateTime(); values[2] = c10::getTime();)
}

extern "C" int state_pm_init(const char *list, uint64_t interval) {
  // One initialization per process. In particular, do not silently append
  // duplicate metric names or leak an earlier host object on re-entry.
  GUARDED(
      if (host || sampler ||
          !names.empty()) throw std::runtime_error("PM already initialized");
      if (!list || !*list) throw std::invalid_argument("Empty PM metric list");
      PARAM(CUpti_Profiler_Initialize_Params, init);
      CALL(cuptiProfilerInitialize, init);
      PARAM(CUpti_Device_GetChipName_Params, chip); chip.deviceIndex = 0;
      CALL(cuptiDeviceGetChipName, chip);
      PARAM(CUpti_PmSampling_GetCounterAvailability_Params, avail);
      avail.deviceIndex = 0; CALL(cuptiPmSamplingGetCounterAvailability, avail);
      std::vector<uint8_t> availability(avail.counterAvailabilityImageSize);
      avail.pCounterAvailabilityImage = availability.data();
      CALL(cuptiPmSamplingGetCounterAvailability, avail);
      PARAM(CUpti_Profiler_Host_Initialize_Params, h);
      h.profilerType = CUPTI_PROFILER_TYPE_PM_SAMPLING;
      h.pChipName = chip.pChipName;
      h.pCounterAvailabilityImage = availability.data();
      CALL(cuptiProfilerHostInitialize, h); host = h.pHostObject;
      std::string all(list); for (size_t pos = 0; pos < all.size();) {
        size_t end = all.find(',', pos);
        if (end == std::string::npos)
          end = all.size();
        names.push_back(all.substr(pos, end - pos));
        pos = end + 1;
      } for (const auto &name : names) metrics.push_back(name.c_str());
      PARAM(CUpti_Profiler_Host_ConfigAddMetrics_Params, add);
      add.pHostObject = host; add.ppMetricNames = metrics.data();
      add.numMetrics = metrics.size();
      CALL(cuptiProfilerHostConfigAddMetrics, add);
      PARAM(CUpti_Profiler_Host_GetConfigImageSize_Params, size);
      size.pHostObject = host; CALL(cuptiProfilerHostGetConfigImageSize, size);
      config.resize(size.configImageSize);
      PARAM(CUpti_Profiler_Host_GetConfigImage_Params, image);
      image.pHostObject = host; image.configImageSize = config.size();
      image.pConfigImage = config.data();
      CALL(cuptiProfilerHostGetConfigImage, image);
      PARAM(CUpti_Profiler_Host_GetNumOfPasses_Params, passes);
      passes.configImageSize = config.size();
      passes.pConfigImage = config.data();
      CALL(cuptiProfilerHostGetNumOfPasses, passes); std::fprintf(
          stderr, "PM chip=%s passes=%zu interval_ns=%llu metrics=%s\n",
          chip.pChipName, passes.numOfPasses, (unsigned long long)interval,
          list);
      if (passes.numOfPasses !=
          1) throw std::runtime_error("Refusing multipass PM configuration");
      PARAM(CUpti_PmSampling_Enable_Params, enable); enable.deviceIndex = 0;
      CALL(cuptiPmSamplingEnable, enable); sampler = enable.pPmSamplingObject;
      PARAM(CUpti_PmSampling_SetConfig_Params, set);
      set.pPmSamplingObject = sampler; set.configSize = config.size();
      set.pConfig = config.data(); set.hardwareBufferSize = 128 * 1024 * 1024;
      set.samplingInterval = interval;
      set.triggerMode = CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL;
      set.hwBufferAppendMode =
          CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_OLDEST;
      CALL(cuptiPmSamplingSetConfig, set);
      PARAM(CUpti_PmSampling_GetCounterDataSize_Params, ds);
      ds.pPmSamplingObject = sampler; ds.pMetricNames = metrics.data();
      ds.numMetrics = metrics.size(); ds.maxSamples = 200000;
      CALL(cuptiPmSamplingGetCounterDataSize, ds);
      data.resize(ds.counterDataSize);)
}

extern "C" int state_pm_start() {
  GUARDED(PARAM(CUpti_PmSampling_CounterDataImage_Initialize_Params, reset);
          reset.pPmSamplingObject = sampler;
          reset.counterDataSize = data.size(); reset.pCounterData = data.data();
          CALL(cuptiPmSamplingCounterDataImageInitialize, reset);
          PARAM(CUpti_PmSampling_Start_Params, start);
          start.pPmSamplingObject = sampler; CALL(cuptiPmSamplingStart, start);)
}

extern "C" int state_pm_stop(const char *path) {
  GUARDED(
      PARAM(CUpti_PmSampling_Stop_Params, stop);
      stop.pPmSamplingObject = sampler; CALL(cuptiPmSamplingStop, stop);
      PARAM(CUpti_PmSampling_DecodeData_Params, decode);
      decode.pPmSamplingObject = sampler;
      decode.pCounterDataImage = data.data();
      decode.counterDataImageSize = data.size();
      CUptiResult decode_result = cuptiPmSamplingDecodeData(&decode);
      // A partial image is diagnostic evidence only; never admit it as data.
      std::ofstream raw(std::string(path) + ".counterdata", std::ios::binary);
      raw.write(reinterpret_cast<const char *>(data.data()), data.size());
      raw.close(); std::ofstream diagnostic(std::string(path) + ".decode.json");
      diagnostic << "{\"result\":" << (int)decode_result
                 << ",\"stop_reason\":" << (int)decode.decodeStopReason
                 << ",\"overflow\":" << (int)decode.overflow
                 << ",\"counter_data_size\":" << data.size() << "}\n";
      diagnostic.close(); if (!raw || !diagnostic) throw std::runtime_error(
          "Counter evidence write failed");
      check(decode_result, "cuptiPmSamplingDecodeData");
      std::fprintf(stderr, "PM decode reason=%d overflow=%u\n",
                   (int)decode.decodeStopReason, decode.overflow);
      if (decode.overflow ||
          decode.decodeStopReason !=
              CUPTI_PM_SAMPLING_DECODE_STOP_REASON_END_OF_RECORDS) throw std::
          runtime_error("Incomplete/overflowed PM data");
      PARAM(CUpti_PmSampling_GetCounterDataInfo_Params, info);
      info.pCounterDataImage = data.data();
      info.counterDataImageSize = data.size();
      CALL(cuptiPmSamplingGetCounterDataInfo, info); std::ofstream csv(path);
      csv.precision(17); csv << "start_ns,end_ns";
      for (const auto &name : names) csv << ',' << name; csv << '\n';
      std::vector<double> values(metrics.size());
      for (size_t i = 0; i < info.numCompletedSamples; ++i) {
        PARAM(CUpti_PmSampling_CounterData_GetSampleInfo_Params, sample);
        sample.pPmSamplingObject = sampler;
        sample.pCounterDataImage = data.data();
        sample.counterDataImageSize = data.size();
        sample.sampleIndex = i;
        CALL(cuptiPmSamplingCounterDataGetSampleInfo, sample);
        PARAM(CUpti_Profiler_Host_EvaluateToGpuValues_Params, eval);
        eval.pHostObject = host;
        eval.pCounterDataImage = data.data();
        eval.counterDataImageSize = data.size();
        eval.rangeIndex = i;
        eval.ppMetricNames = metrics.data();
        eval.numMetrics = metrics.size();
        eval.pMetricValues = values.data();
        CALL(cuptiProfilerHostEvaluateToGpuValues, eval);
        csv << sample.startTimestamp << ',' << sample.endTimestamp;
        for (double value : values)
          csv << ',' << value;
        csv << '\n';
      } if (!csv ||
            !raw) throw std::runtime_error("Counter output write failed");
      std::fprintf(stderr, "PM completed_samples=%zu total=%zu populated=%zu\n",
                   info.numCompletedSamples, info.numTotalSamples,
                   info.numPopulatedSamples);)
}

extern "C" int state_pm_close() {
  GUARDED(
      if (sampler) {
        PARAM(CUpti_PmSampling_Disable_Params, disable);
        disable.pPmSamplingObject = sampler;
        CALL(cuptiPmSamplingDisable, disable);
        sampler = nullptr;
      } if (host) {
        PARAM(CUpti_Profiler_Host_Deinitialize_Params, deinit);
        deinit.pHostObject = host;
        CALL(cuptiProfilerHostDeinitialize, deinit);
        host = nullptr;
      })
}
