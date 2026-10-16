// CPU-only failure injection for preservation; no CUDA context or GPU work.
#include "stage5/reduction_state_sampling.cpp"
static bool api_error = false;
extern "C" CUptiResult __wrap_cuptiPmSamplingStop(CUpti_PmSampling_Stop_Params*) { return CUPTI_SUCCESS; }
extern "C" CUptiResult __wrap_cuptiPmSamplingDecodeData(CUpti_PmSampling_DecodeData_Params* p) {
  p->decodeStopReason = CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL;
  p->overflow = 1;
  return api_error ? CUPTI_ERROR_OUT_OF_MEMORY : CUPTI_SUCCESS;
}
int main(int argc,char**argv) {
  if(argc!=2)return 2;
  sampler = reinterpret_cast<CUpti_PmSampling_Object*>(1);
  data.assign(16,0x5a);
  for(int i=0;i<2;i++) {
    api_error = i==1;
    std::string path=std::string(argv[1])+std::to_string(i);
    if(state_pm_stop(path.c_str())!=1)return 3;
  }
  sampler=nullptr;
  return 0;
}
