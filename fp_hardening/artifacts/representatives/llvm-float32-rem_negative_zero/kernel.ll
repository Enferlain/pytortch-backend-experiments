; ModuleID = 'kernel'
source_filename = "kernel"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cubecl_fp_rem_negative_zero_float32(ptr addrspace(1) noalias %0, ptr addrspace(1) noalias readonly %1, ptr addrspace(1) noalias %2, ptr addrspace(1) noalias readonly %3) #0 {
entry_block2v1:
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %v83 = mul i32 %v15, 256
  %v84 = add i32 %v14, %v83
  %v86 = mul i32 %v16, 256
  %v88 = mul i32 %v86, 1
  %v89 = add i32 %v84, %v88
  %v90 = zext i32 %v19 to i64
  %v91 = zext i32 %v20 to i64
  %v92 = mul i64 %v90, %v91
  %v93 = zext i32 %v21 to i64
  %v94 = mul i64 %v92, %v93
  %v95 = zext i32 %v18 to i64
  %v96 = zext i32 %v20 to i64
  %v97 = mul i64 %v95, %v96
  %v98 = add i64 %v94, %v97
  %v99 = zext i32 %v17 to i64
  %v100 = add i64 %v98, %v99
  %v102 = mul i64 %v100, 256
  %v103 = zext i32 %v89 to i64
  %v104 = add i64 %v102, %v103
  %v106 = getelementptr inbounds nuw i64, ptr addrspace(1) %3, i64 0
  %v107 = load i64, ptr addrspace(1) %v106, align 8
  %v108 = icmp ult i64 %v104, %v107
  br i1 %v108, label %then_block3v1, label %else_block4v1

then_block3v1:                                    ; preds = %entry_block2v1
  %v109 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %v104
  %v110 = load float, ptr addrspace(1) %v109, align 4
  %v112 = frem float -0.000000e+00, %v110
  %v113 = getelementptr inbounds nuw float, ptr addrspace(1) %2, i64 %v104
  store float %v112, ptr addrspace(1) %v113, align 4
  br label %if_merge_block5v1

else_block4v1:                                    ; preds = %entry_block2v1
  br label %if_merge_block5v1

if_merge_block5v1:                                ; preds = %else_block4v1, %then_block3v1
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.tid.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.laneid() #1

attributes #0 = { "nvvm.reqntid"="256,1,1" "target-cpu"="sm_86" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
