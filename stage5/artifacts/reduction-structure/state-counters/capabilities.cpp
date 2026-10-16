#include <cstdio>
#include <cuda.h>
#include <cupti_version.h>
#include <cupti_profiler_target.h>
#include <cupti_pmsampling.h>
static void show(const char* n, CUptiResult r) {const char* s=nullptr;cuptiGetResultString(r,&s);printf("%s=%d %s\n",n,(int)r,s?s:"");}
int main() {
 printf("cuInit=%d\n",(int)cuInit(0));
 CUcontext initialContext=nullptr;cuDevicePrimaryCtxRetain(&initialContext,0);cuCtxSetCurrent(initialContext);
 uint32_t version=0;show("cuptiGetVersion",cuptiGetVersion(&version));printf("CUPTI version=%u\n",version);
 CUpti_Profiler_Initialize_Params init={};init.structSize=CUpti_Profiler_Initialize_Params_STRUCT_SIZE;show("ProfilerInitialize",cuptiProfilerInitialize(&init));
 for(int d=0;d<2;d++) {
  char name[256]={};cuDeviceGetName(name,sizeof(name),d);printf("device=%d name=%s\n",d,name);
  for(int api=0;api<4;api++) {
   CUpti_Profiler_DeviceSupported_Params p={};p.structSize=CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE;p.cuDevice=d;p.api=(CUpti_Profiler_API)api;
   show("DeviceSupported",cuptiProfilerDeviceSupported(&p));
   printf("api=%d overall=%d architecture=%d sli=%d vGpu=%d confidential=%d cmp=%d wsl=%d\n",api,p.isSupported,p.architecture,p.sli,p.vGpu,p.confidentialCompute,p.cmp,p.wsl);
  }
 }
 CUcontext ctx=nullptr;printf("primary retain=%d\n",(int)cuDevicePrimaryCtxRetain(&ctx,0));printf("set current=%d\n",(int)cuCtxSetCurrent(ctx));
 CUpti_PmSampling_Enable_Params p={};p.structSize=CUpti_PmSampling_Enable_Params_STRUCT_SIZE;p.deviceIndex=0;
 auto r=cuptiPmSamplingEnable(&p);show("PM enable device0",r);
 if(r==CUPTI_SUCCESS){CUpti_PmSampling_Disable_Params q={};q.structSize=CUpti_PmSampling_Disable_Params_STRUCT_SIZE;q.pPmSamplingObject=p.pPmSamplingObject;show("PM disable",cuptiPmSamplingDisable(&q));}
}
