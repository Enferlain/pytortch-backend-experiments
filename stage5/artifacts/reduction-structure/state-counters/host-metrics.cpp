#include <cstdio>
#include <cupti_profiler_host.h>
int main(){
 CUpti_Profiler_Host_Initialize_Params h={};h.structSize=CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE;h.profilerType=CUPTI_PROFILER_TYPE_PM_SAMPLING;h.pChipName="GA102";
 auto r=cuptiProfilerHostInitialize(&h);printf("host init=%d\n",r);if(r)return 1;
 for(int type=0;type<CUPTI_METRIC_TYPE__COUNT;type++){
 CUpti_Profiler_Host_GetBaseMetrics_Params p={};p.structSize=CUpti_Profiler_Host_GetBaseMetrics_Params_STRUCT_SIZE;p.pHostObject=h.pHostObject;p.metricType=(CUpti_MetricType)type;
 r=cuptiProfilerHostGetBaseMetrics(&p);printf("metric type=%d result=%d count=%zu\n",type,r,p.numMetrics);if(r)return 1;
 for(size_t i=0;i<p.numMetrics;i++)printf("%d %s\n",type,p.ppMetricNames[i]);
 }
 CUpti_Profiler_Host_Deinitialize_Params q={};q.structSize=CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE;q.pHostObject=h.pHostObject;return cuptiProfilerHostDeinitialize(&q);
}
