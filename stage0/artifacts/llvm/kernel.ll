; ModuleID = 'kernel'
source_filename = "kernel"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cubecl_sin_add(ptr addrspace(1) noalias readonly %0, ptr addrspace(1) noalias %1, ptr addrspace(1) noalias readonly %2) #0 {
entry_block2v1:
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %v81 = mul i32 %v13, 256
  %v82 = add i32 %v12, %v81
  %v84 = mul i32 %v14, 256
  %v86 = mul i32 %v84, 1
  %v87 = add i32 %v82, %v86
  %v88 = zext i32 %v17 to i64
  %v89 = zext i32 %v18 to i64
  %v90 = mul i64 %v88, %v89
  %v91 = zext i32 %v19 to i64
  %v92 = mul i64 %v90, %v91
  %v93 = zext i32 %v16 to i64
  %v94 = zext i32 %v18 to i64
  %v95 = mul i64 %v93, %v94
  %v96 = add i64 %v92, %v95
  %v97 = zext i32 %v15 to i64
  %v98 = add i64 %v96, %v97
  %v100 = mul i64 %v98, 256
  %v101 = zext i32 %v87 to i64
  %v102 = add i64 %v100, %v101
  %v104 = getelementptr inbounds nuw i64, ptr addrspace(1) %2, i64 0
  %v105 = load i64, ptr addrspace(1) %v104, align 8
  %v106 = icmp ult i64 %v102, %v105
  br i1 %v106, label %then_block3v1, label %else_block4v1

then_block3v1:                                    ; preds = %entry_block2v1
  %v107 = getelementptr inbounds nuw float, ptr addrspace(1) %0, i64 %v102
  %v108 = load float, ptr addrspace(1) %v107, align 4
  %v109 = call float @llvm.sin.f32(float %v108)
  %v111 = fadd contract float %v109, 1.000000e+00
  %v112 = getelementptr inbounds nuw float, ptr addrspace(1) %1, i64 %v102
  store float %v111, ptr addrspace(1) %v112, align 4
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sin.f32(float) #2

attributes #0 = { "nvvm.reqntid"="256,1,1" "target-cpu"="sm_86" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
