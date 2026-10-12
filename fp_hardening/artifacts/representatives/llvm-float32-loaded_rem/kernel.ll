; ModuleID = 'kernel'
source_filename = "kernel"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cubecl_fp_loaded_rem_float32(ptr addrspace(1) noalias readonly %0, ptr addrspace(1) noalias readonly %1, ptr addrspace(1) noalias %2, ptr addrspace(1) noalias readonly %3) #0 {
entry_block2v1:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %v82 = mul i32 %v14, 256
  %v83 = add i32 %v13, %v82
  %v85 = mul i32 %v15, 256
  %v87 = mul i32 %v85, 1
  %v88 = add i32 %v83, %v87
  %v89 = zext i32 %v18 to i64
  %v90 = zext i32 %v19 to i64
  %v91 = mul i64 %v89, %v90
  %v92 = zext i32 %v20 to i64
  %v93 = mul i64 %v91, %v92
  %v94 = zext i32 %v17 to i64
  %v95 = zext i32 %v19 to i64
  %v96 = mul i64 %v94, %v95
  %v97 = add i64 %v93, %v96
  %v98 = zext i32 %v16 to i64
  %v99 = add i64 %v97, %v98
  %v101 = mul i64 %v99, 256
  %v102 = zext i32 %v88 to i64
  %v103 = add i64 %v101, %v102
  %v105 = getelementptr inbounds nuw i64, ptr addrspace(1) %3, i64 0
  %v106 = load i64, ptr addrspace(1) %v105, align 8
  %v107 = icmp ult i64 %v103, %v106
  br i1 %v107, label %then_block3v1, label %else_block4v1

then_block3v1:                                    ; preds = %entry_block2v1
  %v108 = getelementptr inbounds nuw float, ptr addrspace(1) %0, i64 %v103
  %v109 = load float, ptr addrspace(1) %v108, align 4
  %v110 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %v103
  %v111 = load float, ptr addrspace(1) %v110, align 4
  %v112 = frem float %v109, %v111
  %v113 = getelementptr inbounds nuw float, ptr addrspace(1) %2, i64 %v103
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
