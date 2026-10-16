
/tmp/stage5-memory-idle-verify.so:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 c9 2f 00 00 	mov    0x2fc9(%rip),%rax        # 3fd8 <__gmon_start__>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	call   *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <_ZdlPvm@plt-0x10>:
    1020:	ff 35 ca 2f 00 00    	push   0x2fca(%rip)        # 3ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 cc 2f 00 00    	jmp    *0x2fcc(%rip)        # 3ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001030 <_ZdlPvm@plt>:
    1030:	ff 25 ca 2f 00 00    	jmp    *0x2fca(%rip)        # 4000 <_ZdlPvm@CXXABI_1.3.9>
    1036:	68 00 00 00 00       	push   $0x0
    103b:	e9 e0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001040 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>:
    1040:	ff 25 c2 2f 00 00    	jmp    *0x2fc2(%rip)        # 4008 <_ZNSt6chrono3_V212steady_clock3nowEv@GLIBCXX_3.4.19>
    1046:	68 01 00 00 00       	push   $0x1
    104b:	e9 d0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001050 <cuLaunchHostFunc@plt>:
    1050:	ff 25 ba 2f 00 00    	jmp    *0x2fba(%rip)        # 4010 <cuLaunchHostFunc>
    1056:	68 02 00 00 00       	push   $0x2
    105b:	e9 c0 ff ff ff       	jmp    1020 <_init+0x20>

0000000000001060 <_ZnwmRKSt9nothrow_t@plt>:
    1060:	ff 25 b2 2f 00 00    	jmp    *0x2fb2(%rip)        # 4018 <_ZnwmRKSt9nothrow_t@GLIBCXX_3.4>
    1066:	68 03 00 00 00       	push   $0x3
    106b:	e9 b0 ff ff ff       	jmp    1020 <_init+0x20>

Disassembly of section .plt.got:

0000000000001070 <__cxa_finalize@plt>:
    1070:	ff 25 52 2f 00 00    	jmp    *0x2f52(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    1076:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000001080 <deregister_tm_clones>:
    1080:	48 8d 3d a1 2f 00 00 	lea    0x2fa1(%rip),%rdi        # 4028 <completed.0>
    1087:	48 8d 05 9a 2f 00 00 	lea    0x2f9a(%rip),%rax        # 4028 <completed.0>
    108e:	48 39 f8             	cmp    %rdi,%rax
    1091:	74 15                	je     10a8 <deregister_tm_clones+0x28>
    1093:	48 8b 05 36 2f 00 00 	mov    0x2f36(%rip),%rax        # 3fd0 <_ITM_deregisterTMCloneTable>
    109a:	48 85 c0             	test   %rax,%rax
    109d:	74 09                	je     10a8 <deregister_tm_clones+0x28>
    109f:	ff e0                	jmp    *%rax
    10a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    10a8:	c3                   	ret
    10a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000010b0 <register_tm_clones>:
    10b0:	48 8d 3d 71 2f 00 00 	lea    0x2f71(%rip),%rdi        # 4028 <completed.0>
    10b7:	48 8d 35 6a 2f 00 00 	lea    0x2f6a(%rip),%rsi        # 4028 <completed.0>
    10be:	48 29 fe             	sub    %rdi,%rsi
    10c1:	48 89 f0             	mov    %rsi,%rax
    10c4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    10c8:	48 c1 f8 03          	sar    $0x3,%rax
    10cc:	48 01 c6             	add    %rax,%rsi
    10cf:	48 d1 fe             	sar    $1,%rsi
    10d2:	74 14                	je     10e8 <register_tm_clones+0x38>
    10d4:	48 8b 05 05 2f 00 00 	mov    0x2f05(%rip),%rax        # 3fe0 <_ITM_registerTMCloneTable>
    10db:	48 85 c0             	test   %rax,%rax
    10de:	74 08                	je     10e8 <register_tm_clones+0x38>
    10e0:	ff e0                	jmp    *%rax
    10e2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    10e8:	c3                   	ret
    10e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000010f0 <__do_global_dtors_aux>:
    10f0:	f3 0f 1e fa          	endbr64
    10f4:	80 3d 2d 2f 00 00 00 	cmpb   $0x0,0x2f2d(%rip)        # 4028 <completed.0>
    10fb:	75 2b                	jne    1128 <__do_global_dtors_aux+0x38>
    10fd:	55                   	push   %rbp
    10fe:	48 83 3d c2 2e 00 00 	cmpq   $0x0,0x2ec2(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    1105:	00 
    1106:	48 89 e5             	mov    %rsp,%rbp
    1109:	74 0c                	je     1117 <__do_global_dtors_aux+0x27>
    110b:	48 8b 3d 0e 2f 00 00 	mov    0x2f0e(%rip),%rdi        # 4020 <__dso_handle>
    1112:	e8 59 ff ff ff       	call   1070 <__cxa_finalize@plt>
    1117:	e8 64 ff ff ff       	call   1080 <deregister_tm_clones>
    111c:	c6 05 05 2f 00 00 01 	movb   $0x1,0x2f05(%rip)        # 4028 <completed.0>
    1123:	5d                   	pop    %rbp
    1124:	c3                   	ret
    1125:	0f 1f 00             	nopl   (%rax)
    1128:	c3                   	ret
    1129:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001130 <frame_dummy>:
    1130:	f3 0f 1e fa          	endbr64
    1134:	e9 77 ff ff ff       	jmp    10b0 <register_tm_clones>
    1139:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001140 <_ZL13idle_callbackPv>:
    1140:	55                   	push   %rbp
    1141:	48 89 e5             	mov    %rsp,%rbp
    1144:	41 54                	push   %r12
    1146:	49 89 fc             	mov    %rdi,%r12
    1149:	53                   	push   %rbx
    114a:	e8 f1 fe ff ff       	call   1040 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
    114f:	48 89 c3             	mov    %rax,%rbx
    1152:	0f 1f 00             	nopl   (%rax)
    1155:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    115c:	00 00 00 00 
    1160:	e8 db fe ff ff       	call   1040 <_ZNSt6chrono3_V212steady_clock3nowEv@plt>
    1165:	48 89 c2             	mov    %rax,%rdx
    1168:	48 29 da             	sub    %rbx,%rdx
    116b:	49 3b 14 24          	cmp    (%r12),%rdx
    116f:	72 ef                	jb     1160 <_ZL13idle_callbackPv+0x20>
    1171:	66 48 0f 6e c3       	movq   %rbx,%xmm0
    1176:	66 48 0f 6e c8       	movq   %rax,%xmm1
    117b:	49 8b 44 24 08       	mov    0x8(%r12),%rax
    1180:	4c 89 e7             	mov    %r12,%rdi
    1183:	66 0f 6c c1          	punpcklqdq %xmm1,%xmm0
    1187:	be 10 00 00 00       	mov    $0x10,%esi
    118c:	0f 11 00             	movups %xmm0,(%rax)
    118f:	5b                   	pop    %rbx
    1190:	41 5c                	pop    %r12
    1192:	5d                   	pop    %rbp
    1193:	e9 98 fe ff ff       	jmp    1030 <_ZdlPvm@plt>
    1198:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    119f:	00 

00000000000011a0 <state_idle_enqueue>:
    11a0:	48 85 d2             	test   %rdx,%rdx
    11a3:	74 6b                	je     1210 <state_idle_enqueue+0x70>
    11a5:	48 81 fe a0 25 26 00 	cmp    $0x2625a0,%rsi
    11ac:	77 62                	ja     1210 <state_idle_enqueue+0x70>
    11ae:	55                   	push   %rbp
    11af:	48 89 e5             	mov    %rsp,%rbp
    11b2:	53                   	push   %rbx
    11b3:	48 89 fb             	mov    %rdi,%rbx
    11b6:	bf 10 00 00 00       	mov    $0x10,%edi
    11bb:	48 83 ec 18          	sub    $0x18,%rsp
    11bf:	48 89 75 e8          	mov    %rsi,-0x18(%rbp)
    11c3:	48 8b 35 f6 2d 00 00 	mov    0x2df6(%rip),%rsi        # 3fc0 <_ZSt7nothrow@GLIBCXX_3.4>
    11ca:	48 89 55 e0          	mov    %rdx,-0x20(%rbp)
    11ce:	e8 8d fe ff ff       	call   1060 <_ZnwmRKSt9nothrow_t@plt>
    11d3:	48 85 c0             	test   %rax,%rax
    11d6:	74 62                	je     123a <state_idle_enqueue+0x9a>
    11d8:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
    11dc:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    11e0:	48 89 df             	mov    %rbx,%rdi
    11e3:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    11e7:	48 8d 35 52 ff ff ff 	lea    -0xae(%rip),%rsi        # 1140 <_ZL13idle_callbackPv>
    11ee:	48 89 50 08          	mov    %rdx,0x8(%rax)
    11f2:	48 89 c2             	mov    %rax,%rdx
    11f5:	48 89 08             	mov    %rcx,(%rax)
    11f8:	e8 53 fe ff ff       	call   1050 <cuLaunchHostFunc@plt>
    11fd:	85 c0                	test   %eax,%eax
    11ff:	75 1f                	jne    1220 <state_idle_enqueue+0x80>
    1201:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    1205:	c9                   	leave
    1206:	c3                   	ret
    1207:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    120e:	00 00 
    1210:	b8 01 00 00 00       	mov    $0x1,%eax
    1215:	c3                   	ret
    1216:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    121d:	00 00 00 
    1220:	48 8b 7d e8          	mov    -0x18(%rbp),%rdi
    1224:	be 10 00 00 00       	mov    $0x10,%esi
    1229:	89 45 e8             	mov    %eax,-0x18(%rbp)
    122c:	e8 ff fd ff ff       	call   1030 <_ZdlPvm@plt>
    1231:	8b 45 e8             	mov    -0x18(%rbp),%eax
    1234:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    1238:	c9                   	leave
    1239:	c3                   	ret
    123a:	b8 02 00 00 00       	mov    $0x2,%eax
    123f:	eb c0                	jmp    1201 <state_idle_enqueue+0x61>

Disassembly of section .fini:

0000000000001244 <_fini>:
    1244:	f3 0f 1e fa          	endbr64
    1248:	48 83 ec 08          	sub    $0x8,%rsp
    124c:	48 83 c4 08          	add    $0x8,%rsp
    1250:	c3                   	ret
