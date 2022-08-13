	.text
	.file	"test_1.c"
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movl	$0, -4(%rbp)
	callq	omp_get_num_devices@PLT
	movl	%eax, %esi
	leaq	.L.str(%rip), %rdi
	movb	$0, %al
	callq	printf@PLT
	callq	omp_get_num_devices@PLT
	movl	%eax, -12(%rbp)
	movl	$0, -16(%rbp)
.LBB0_1:                                # %for.cond
                                        # =>This Inner Loop Header: Depth=1
	movl	-16(%rbp), %eax
	cmpl	-12(%rbp), %eax
	jge	.LBB0_6
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-16(%rbp), %eax
	movl	%eax, -20(%rbp)
	movslq	-20(%rbp), %rsi
	movq	%rsp, %r9
	addq	$-64, %r9
	movq	%r9, -32(%rbp)                  # 8-byte Spill
	movq	%r9, %rsp
	movl	$1, (%r9)
	movl	$0, 4(%r9)
	movq	$0, 8(%r9)
	movq	$0, 16(%r9)
	movq	$0, 24(%r9)
	movq	$0, 32(%r9)
	movq	$0, 40(%r9)
	movq	$0, 48(%r9)
	movq	$0, 56(%r9)
	leaq	.L__unnamed_1(%rip), %rdi
	movl	$4294967295, %edx               # imm = 0xFFFFFFFF
	xorl	%ecx, %ecx
	movq	.__omp_offloading_33_7b8f0a40_main_l11.region_id@GOTPCREL(%rip), %r8
	callq	__tgt_target_kernel@PLT
	cmpl	$0, %eax
	je	.LBB0_4
# %bb.3:                                # %omp_offload.failed
                                        #   in Loop: Header=BB0_1 Depth=1
	callq	__omp_offloading_33_7b8f0a40_main_l11
.LBB0_4:                                # %omp_offload.cont
                                        #   in Loop: Header=BB0_1 Depth=1
	jmp	.LBB0_5
.LBB0_5:                                # %for.inc
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-16(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -16(%rbp)
	jmp	.LBB0_1
.LBB0_6:                                # %for.end
	xorl	%eax, %eax
	movq	%rbp, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function __omp_offloading_33_7b8f0a40_main_l11
	.type	__omp_offloading_33_7b8f0a40_main_l11,@function
__omp_offloading_33_7b8f0a40_main_l11:  # @__omp_offloading_33_7b8f0a40_main_l11
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	callq	omp_get_device_num@PLT
	movl	%eax, -4(%rbp)                  # 4-byte Spill
	callq	omp_is_initial_device$ompvariant$S2$s6$Phost
	movl	-4(%rbp), %esi                  # 4-byte Reload
	movl	%eax, %ecx
	leaq	.L.str.3(%rip), %rdx
	leaq	.L.str.2(%rip), %rax
	cmpl	$0, %ecx
	cmovneq	%rax, %rdx
	leaq	.L.str.1(%rip), %rdi
	movb	$0, %al
	callq	printf@PLT
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	__omp_offloading_33_7b8f0a40_main_l11, .Lfunc_end1-__omp_offloading_33_7b8f0a40_main_l11
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90                         # -- Begin function omp_is_initial_device$ompvariant$S2$s6$Phost
	.type	omp_is_initial_device$ompvariant$S2$s6$Phost,@function
omp_is_initial_device$ompvariant$S2$s6$Phost: # @"omp_is_initial_device$ompvariant$S2$s6$Phost"
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$1, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	omp_is_initial_device$ompvariant$S2$s6$Phost, .Lfunc_end2-omp_is_initial_device$ompvariant$S2$s6$Phost
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90                         # -- Begin function .omp_offloading.requires_reg
	.type	.omp_offloading.requires_reg,@function
.omp_offloading.requires_reg:           # @.omp_offloading.requires_reg
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movl	$1, %edi
	callq	__tgt_register_requires@PLT
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	.omp_offloading.requires_reg, .Lfunc_end3-.omp_offloading.requires_reg
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"No. of Devices: %d\n"
	.size	.L.str, 20

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"Device %d is: %s\n"
	.size	.L.str.1, 18

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"Host"
	.size	.L.str.2, 5

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"Device"
	.size	.L.str.3, 7

	.type	.__omp_offloading_33_7b8f0a40_main_l11.region_id,@object # @.__omp_offloading_33_7b8f0a40_main_l11.region_id
	.section	.rodata,"a",@progbits
	.weak	.__omp_offloading_33_7b8f0a40_main_l11.region_id
.__omp_offloading_33_7b8f0a40_main_l11.region_id:
	.byte	0                               # 0x0
	.size	.__omp_offloading_33_7b8f0a40_main_l11.region_id, 1

	.type	.L__unnamed_2,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_2:
	.asciz	";unknown;unknown;0;0;;"
	.size	.L__unnamed_2, 23

	.type	.L__unnamed_1,@object           # @1
	.section	.data.rel.ro,"aw",@progbits
	.p2align	3
.L__unnamed_1:
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	22                              # 0x16
	.quad	.L__unnamed_2
	.size	.L__unnamed_1, 24

	.type	.omp_offloading.entry_name,@object # @.omp_offloading.entry_name
	.section	.rodata.str1.16,"aMS",@progbits,1
	.p2align	4
.omp_offloading.entry_name:
	.asciz	"__omp_offloading_33_7b8f0a40_main_l11"
	.size	.omp_offloading.entry_name, 38

	.type	.omp_offloading.entry.__omp_offloading_33_7b8f0a40_main_l11,@object # @.omp_offloading.entry.__omp_offloading_33_7b8f0a40_main_l11
	.section	omp_offloading_entries,"aw",@progbits
	.weak	.omp_offloading.entry.__omp_offloading_33_7b8f0a40_main_l11
.omp_offloading.entry.__omp_offloading_33_7b8f0a40_main_l11:
	.quad	.__omp_offloading_33_7b8f0a40_main_l11.region_id
	.quad	.omp_offloading.entry_name
	.quad	0                               # 0x0
	.long	0                               # 0x0
	.long	0                               # 0x0
	.size	.omp_offloading.entry.__omp_offloading_33_7b8f0a40_main_l11, 32

	.section	.init_array.0,"aw",@init_array
	.p2align	3
	.quad	.omp_offloading.requires_reg
	.type	.Lllvm.embedded.object,@object  # @llvm.embedded.object
	.section	.llvm.offloading,"e",@llvm_offloading
	.p2align	3
.Lllvm.embedded.object:
	.asciz	"\020\377\020\255\001\000\000\000\260\020\000\000\000\000\000\000 \000\000\000\000\000\000\000(\000\000\000\000\000\000\000\005\000\001\000\000\000\000\000H\000\000\000\000\000\000\000\002\000\000\000\000\000\000\000\220\000\000\000\000\000\000\000\035\020\000\000\000\000\000\000n\000\000\000\000\000\000\000u\000\000\000\000\000\000\000i\000\000\000\000\000\000\000\211\000\000\000\000\000\000\000\000arch\000triple\000nvptx64-nvidia-cuda\000sm_70\000\000//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 7.0\n.target sm_70\n.address_size 64\n\n\t// .weak\t__omp_offloading_33_7b8f0a40_main_l11\n.extern .func  (.param .b32 func_retval0) __kmpc_target_init\n(\n\t.param .b64 __kmpc_target_init_param_0,\n\t.param .b32 __kmpc_target_init_param_1,\n\t.param .b32 __kmpc_target_init_param_2,\n\t.param .b32 __kmpc_target_init_param_3\n)\n;\n.extern .func  (.param .b32 func_retval0) __llvm_omp_vprintf\n(\n\t.param .b64 __llvm_omp_vprintf_param_0,\n\t.param .b64 __llvm_omp_vprintf_param_1,\n\t.param .b32 __llvm_omp_vprintf_param_2\n)\n;\n.extern .func  (.param .b32 func_retval0) omp_get_device_num\n()\n;\n.func  (.param .b32 func_retval0) omp_is_initial_device$ompvariant$S2$s6$Pnohost\n()\n;\n.extern .func __kmpc_target_deinit\n(\n\t.param .b64 __kmpc_target_deinit_param_0,\n\t.param .b32 __kmpc_target_deinit_param_1,\n\t.param .b32 __kmpc_target_deinit_param_2\n)\n;\n.weak .global .align 4 .u32 __omp_rtl_debug_kind;\n.weak .global .align 4 .u32 __omp_rtl_assume_teams_oversubscription;\n.weak .global .align 4 .u32 __omp_rtl_assume_threads_oversubscription;\n.weak .global .align 4 .u32 __omp_rtl_assume_no_thread_state;\n.global .align 1 .b8 __unnamed_1[23] = {59, 117, 110, 107, 110, 111, 119, 110, 59, 117, 110, 107, 110, 111, 119, 110, 59, 48, 59, 48, 59, 59, 0};\n.global .align 8 .u64 __unnamed_2[3] = {8589934592, 94489280512, generic(__unnamed_1)};\n.global .align 1 .b8 _$_str[18] = {68, 101, 118, 105, 99, 101, 32, 37, 100, 32, 105, 115, 58, 32, 37, 115, 10, 0};\n.global .align 1 .b8 _$_str1[5] = {72, 111, 115, 116, 0};\n.global .align 1 .b8 _$_str2[7] = {68, 101, 118, 105, 99, 101, 0};\n.weak .global .align 1 .u8 __omp_offloading_33_7b8f0a40_main_l11_exec_mode = 1;\n\n.weak .entry __omp_offloading_33_7b8f0a40_main_l11()\n{\n\t.local .align 8 .b8 \t__local_depot0[16];\n\t.reg .b64 \t%SP;\n\t.reg .b64 \t%SPL;\n\t.reg .pred \t%p<3>;\n\t.reg .b32 \t%r<12>;\n\t.reg .b64 \t%rd<13>;\n\n\tmov.u64 \t%SPL, __local_depot0;\n\tcvta.local.u64 \t%SP, %SPL;\n\tmov.u64 \t%rd1, __unnamed_2;\n\tcvta.global.u64 \t%rd2, %rd1;\n\tmov.u32 \t%r1, 1;\n\t{ // callseq 0, 0\n\t.reg .b32 temp_param_reg;\n\t.param .b64 param0;\n\tst.param.b64 \t[param0+0], %rd2;\n\t.param .b32 param1;\n\tst.param.b32 \t[param1+0], %r1;\n\t.param .b32 param2;\n\tst.param.b32 \t[param2+0], %r1;\n\t.param .b32 param3;\n\tst.param.b32 \t[param3+0], %r1;\n\t.param .b32 retval0;\n\tcall.uni (retval0), \n\t__kmpc_target_init, \n\t(\n\tparam0, \n\tparam1, \n\tparam2, \n\tparam3\n\t);\n\tld.param.b32 \t%r2, [retval0+0];\n\t} // callseq 0\n\tsetp.ne.s32 \t%p1, %r2, -1;\n\t@%p1 bra \t$L__BB0_2;\n\tbra.uni \t$L__BB0_1;\n$L__BB0_1:\n\t{ // callseq 1, 0\n\t.reg .b32 temp_param_reg;\n\t.param .b32 retval0;\n\tcall.uni (retval0), \n\tomp_get_device_num, \n\t(\n\t);\n\tld.param.b32 \t%r4, [retval0+0];\n\t} // callseq 1\n\t{ // callseq 2, 0\n\t.reg .b32 temp_param_reg;\n\t.param .b32 retval0;\n\tcall.uni (retval0), \n\tomp_is_initial_device$ompvariant$S2$s6$Pnohost, \n\t(\n\t);\n\tld.param.b32 \t%r6, [retval0+0];\n\t} // callseq 2\n\tsetp.ne.s32 \t%p2, %r6, 0;\n\tmov.u64 \t%rd3, _$_str2;\n\tcvta.global.u64 \t%rd4, %rd3;\n\tmov.u64 \t%rd5, _$_str1;\n\tcvta.global.u64 \t%rd6, %rd5;\n\tselp.b64 \t%rd7, %rd6, %rd4, %p2;\n\tst.u32 \t[%SP+0], %r4;\n\tst.u64 \t[%SP+8], %rd7;\n\tmov.u64 \t%rd8, _$_str;\n\tcvta.global.u64 \t%rd9, %rd8;\n\tadd.u64 \t%rd10, %SP, 0;\n\tmov.u32 \t%r8, 16;\n\t{ // callseq 3, 0\n\t.reg .b32 temp_param_reg;\n\t.param .b64 param0;\n\tst.param.b64 \t[param0+0], %rd9;\n\t.param .b64 param1;\n\tst.param.b64 \t[param1+0], %rd10;\n\t.param .b32 param2;\n\tst.param.b32 \t[param2+0], %r8;\n\t.param .b32 retval0;\n\tcall.uni (retval0), \n\t__llvm_omp_vprintf, \n\t(\n\tparam0, \n\tparam1, \n\tparam2\n\t);\n\tld.param.b32 \t%r9, [retval0+0];\n\t} // callseq 3\n\tmov.u64 \t%rd11, __unnamed_2;\n\tcvta.global.u64 \t%rd12, %rd11;\n\tmov.u32 \t%r11, 1;\n\t{ // callseq 4, 0\n\t.reg .b32 temp_param_reg;\n\t.param .b64 param0;\n\tst.param.b64 \t[param0+0], %rd12;\n\t.param .b32 param1;\n\tst.param.b32 \t[param1+0], %r11;\n\t.param .b32 param2;\n\tst.param.b32 \t[param2+0], %r11;\n\tcall.uni \n\t__kmpc_target_deinit, \n\t(\n\tparam0, \n\tparam1, \n\tparam2\n\t);\n\t} // callseq 4\n\tret;\n$L__BB0_2:\n\tret;\n\n}\n.func  (.param .b32 func_retval0) omp_is_initial_device$ompvariant$S2$s6$Pnohost()\n{\n\t.reg .b32 \t%r<2>;\n\n\tmov.u32 \t%r1, 0;\n\tst.param.b32 \t[func_retval0+0], %r1;\n\tret;\n\n}\n\000\000"
	.size	.Lllvm.embedded.object, 4272

	.ident	"clang version 15.0.0 (https://www.github.com/llvm/llvm-project 340b48b267b96ec534aec373dbf19a9464cedcfc)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym printf
	.addrsig_sym omp_get_num_devices
	.addrsig_sym __omp_offloading_33_7b8f0a40_main_l11
	.addrsig_sym omp_get_device_num
	.addrsig_sym omp_is_initial_device$ompvariant$S2$s6$Phost
	.addrsig_sym __tgt_target_kernel
	.addrsig_sym .omp_offloading.requires_reg
	.addrsig_sym __tgt_register_requires
	.addrsig_sym .__omp_offloading_33_7b8f0a40_main_l11.region_id
