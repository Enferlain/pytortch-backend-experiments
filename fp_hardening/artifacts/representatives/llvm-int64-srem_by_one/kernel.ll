; ModuleID = 'kernel'
source_filename = "kernel"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cubecl_fp_srem_by_one_int64(ptr addrspace(1) noalias %0, ptr addrspace(1) noalias %1, ptr addrspace(1) noalias %2, ptr addrspace(1) noalias readonly %3) #0 {
entry_block2v1:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %v84 = mul i32 %v16, 256
  %v85 = add i32 %v15, %v84
  %v87 = mul i32 %v17, 256
  %v89 = mul i32 %v87, 1
  %v90 = add i32 %v85, %v89
  %v91 = zext i32 %v20 to i64
  %v92 = zext i32 %v21 to i64
  %v93 = mul i64 %v91, %v92
  %v94 = zext i32 %v22 to i64
  %v95 = mul i64 %v93, %v94
  %v96 = zext i32 %v19 to i64
  %v97 = zext i32 %v21 to i64
  %v98 = mul i64 %v96, %v97
  %v99 = add i64 %v95, %v98
  %v100 = zext i32 %v18 to i64
  %v101 = add i64 %v99, %v100
  %v103 = mul i64 %v101, 256
  %v104 = zext i32 %v90 to i64
  %v105 = add i64 %v103, %v104
  %v107 = getelementptr inbounds nuw i64, ptr addrspace(1) %3, i64 0
  %v108 = load i64, ptr addrspace(1) %v107, align 8
  %v109 = icmp ult i64 %v105, %v108
  br i1 %v109, label %then_block3v1, label %else_block4v1

then_block3v1:                                    ; preds = %entry_block2v1
  %v111 = getelementptr inbounds nuw i64, ptr addrspace(1) %2, i64 %v105
  store i64 0, ptr addrspace(1) %v111, align 8
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
