// Copyright 2025 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// BEGIN_CONFIG
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// exception: "Subprocess call failed: xls/tools/eval_ir_main --testvector_textproto=testvector.pbtxt --use_llvm_jit sample.ir --logtostderr\n\nSubprocess stderr:\nI0507 08:50:25.217742 1769711 init.cc:762] Linux version 5.10.0-smp-1105.47.0.0 (builder@builder:/usr/src/linux) (clang version (2cbe5a33a5fda257747d75863bd9ccb8920b9249), LLD trunk) #1 [v5.10.0-1105.47.0.0] SMP @1740169297\nI0507 08:50:25.217828 1769711 init.cc:779] Enabled npx experiments: TCMALLOC_MIN_HOT_ACCESS_HINT_ABLATION,TEST_ONLY_L3_AWARE,TEST_ONLY_TCMALLOC_HUGE_CACHE_RELEASE_30S\nI0507 08:50:25.217839 1769711 init.cc:844] Process id 1769711\nI0507 08:50:25.217859 1769711 init.cc:849] Current working directory temp_directory_PSmb5M\nI0507 08:50:25.217861 1769711 init.cc:851] Current timezone is PDT (currently UTC -07:00)\nI0507 08:50:25.217869 1769711 init.cc:855] Built on May 7 2025 01:01:54 (1746604914)\nI0507 08:50:25.217871 1769711 init.cc:856]  at xls:\nI0507 08:50:25.217873 1769711 init.cc:857]  as xls/tools:eval_ir_main\nI0507 08:50:25.217875 1769711 init.cc:858]  for gcc-4.X.Y-crosstool-v18-llvm\nI0507 08:50:25.217877 1769711 init.cc:861]  from changelist 755729679 with baseline 755729679 in a mint client based on \nI0507 08:50:25.217879 1769711 init.cc:867] Build tool: release 2025.04.26-1 (mainline @751384015)\nI0507 08:50:25.217881 1769711 init.cc:868] Build target: eval_ir_main\nI0507 08:50:25.217885 1769711 init.cc:875] Command line arguments:\nI0507 08:50:25.217887 1769711 init.cc:877] argv[0]: \'/xls/tools/eval_ir_main\'\nI0507 08:50:25.217891 1769711 init.cc:877] argv[1]: \'--testvector_textproto=testvector.pbtxt\'\nI0507 08:50:25.217893 1769711 init.cc:877] argv[2]: \'--use_llvm_jit\'\nI0507 08:50:25.217896 1769711 init.cc:877] argv[3]: \'sample.ir\'\nI0507 08:50:25.217898 1769711 init.cc:877] argv[4]: \'--logtostderr\'\nI0507 08:50:25.221193 1769711 logger.cc:310] Enabling threaded logging for severity WARNING\nI0507 08:50:25.221534 1769711 mlock.cc:219] mlock()-ed 4096 bytes for BuildID, using 1 syscalls.\n*** SIGSEGV STACK OVERFLOW (see cppstackoverflow) received by PID 1769711 (TID 1769711) on cpu 40; stack trace: ***\nPC: @     0x7f2767db731a  (unknown)  (unknown)\n    @     0x558000e98585       1904  FailureSignalHandler()\n    @     0x7f276864ae80  (unknown)  (unknown)\n    @     0x7f2767d9bbe6  (unknown)  (unknown)\nI0507 08:51:06.052205 1769711 process_state.cc:304] RAW: ExecuteFailureCallbacks() safe\nI0507 08:51:06.052217 1769711 process_state.cc:1324] RAW: FailureSignalHandler(): starting unsafe phase\nI0507 08:51:06.052224 1769711 coreutil.cc:358] RAW: Attempting to connect to coredump socket @core\nI0507 08:51:06.052254 1769711 coreutil.cc:362] RAW: Failed to connect to coredump socket @core\nI0507 08:51:06.052279 1769711 coreutil.cc:269] RAW: Attempting to dump core\nI0507 08:51:06.061635 1769711 coreutil.cc:310] RAW: WriteCoreDumpWith(core) returns: 0\nW0507 08:51:06.061671 1769711 process_state.cc:1376] --- CPU registers: ---\n   r8=1  r9=7ffc0dd031e0 r10=15b2 r11=0 r12=1 r13=1 r14=1 r15=1 rdi=1 rsi=1\n  rbp=7ffc0e4edd60 rbx=1 rdx=1 rax=7ffc0e4fc6c8 rcx=1 rsp=7ffc0dd014b0\n  rip=7f2767db731a efl=10283 cgf=2b000000000033 err=6 trp=e msk=0\n  cr2=7ffc0dd014b0\nW0507 08:51:06.062197 1769711 process_state.cc:1380] **************************************************\nSTACK OVERFLOW DETECTED in operation on 0x7ffc0dd014b0 (stack ends at 0x7ffc0dd03000)\nSee cppstackoverflow\n**************************************************\n--- Stack contents: ---\n  --- Stack boundaries: [0x7ffc0dd03000, 0x7ffc0e503000) --- (8192KiB)\n  0x00007ffc0dd03000: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03020: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03040: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03060: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03080: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd030a0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd030c0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd030e0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03100: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03120: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03140: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03160: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd03180: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffc0dd031a0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\nW0507 08:51:06.062934 1769711 thread.cc:1805] --- Thread 7f2768477cc0 (name: main/1769711) stack: ---\nW0507 08:51:06.064213 1769711 thread.cc:1805]     @     0x558000e9891d  FailureSignalHandler()\nW0507 08:51:06.064321 1769711 thread.cc:1805]     @     0x7f276864ae80  __restore_rt\nW0507 08:51:06.064847 1769711 thread.cc:1805]     @     0x7f2767d9bbe6  (unknown)\nW0507 08:51:06.065238 1769711 thread.cc:1805] --- Thread 7f2768425700 (name: ExitTimeoutWatcher/1769714) stack: ---\nstack used: 10 KiB of 36 KiB\nW0507 08:51:06.065647 1769711 thread.cc:1805]     @     0x7f276864a19a  nanosleep\nW0507 08:51:06.067020 1769711 thread.cc:1805]     @     0x558000e6b94b  AbslInternalSleepFor\nW0507 08:51:06.068364 1769711 thread.cc:1805]     @     0x558000cf5ebc  (anonymous namespace)::ExitTimeoutWatcher()\nW0507 08:51:06.068437 1769711 thread.cc:1805]     @     0x7f27686417db  start_thread\nW0507 08:51:06.068615 1769711 thread.cc:1805]     @     0x7f27685b405f  clone\nW0507 08:51:06.068786 1769711 thread.cc:1805] --- Thread 7f276841a700 (name: ThreadLivenessWatcher/1769715) stack: ---\nstack used: 10 KiB of 36 KiB\nW0507 08:51:06.068794 1769711 thread.cc:1805]     @     0x7f276864a19a  nanosleep\nW0507 08:51:06.068796 1769711 thread.cc:1805]     @     0x558000e6b94b  AbslInternalSleepFor\nW0507 08:51:06.070092 1769711 thread.cc:1805]     @     0x558000cf60f9  (anonymous namespace)::ThreadLivenessWatcher()\nW0507 08:51:06.070109 1769711 thread.cc:1805]     @     0x7f27686417db  start_thread\nW0507 08:51:06.070112 1769711 thread.cc:1805]     @     0x7f27685b405f  clone\nW0507 08:51:06.070350 1769711 thread.cc:1805] --- Thread 7f276840f700 (name: MemoryReleaser/1769716) stack: ---\nstack used: 10 KiB of 1960 KiB\nW0507 08:51:06.070366 1769711 thread.cc:1805]     @     0x7f276864a19a  nanosleep\nW0507 08:51:06.070368 1769711 thread.cc:1805]     @     0x558000e6b94b  AbslInternalSleepFor\nW0507 08:51:06.071398 1769711 thread.cc:1805]     @     0x558000e89093  MallocExtension_Internal_ProcessBackgroundActions\nW0507 08:51:06.072422 1769711 thread.cc:1805]     @     0x558000cf1ce3  Thread::ThreadBody()\nW0507 08:51:06.072432 1769711 thread.cc:1805]     @     0x7f27686417db  start_thread\nW0507 08:51:06.072434 1769711 thread.cc:1805]     @     0x7f27685b405f  clone\nW0507 08:51:06.072438 1769711 thread.cc:1805] creator: 0x558000cf27c3 0x558001056f69 0x558000e954b0 0x558000e94aa4 0x558000e65503 0x557ffe7951e2 0x557ffe75da94 0x7f27684db3d4 0x557ffe75d02a\nW0507 08:51:06.072789 1769711 thread.cc:1805] --- Thread 7f2768100700 (name: Logger/1769717) stack: ---\nstack used: 10 KiB of 1960 KiB\nW0507 08:51:06.073859 1769711 thread.cc:1805]     @     0x558000e6becc  AbslInternalPerThreadSemWait\nW0507 08:51:06.075046 1769711 thread.cc:1805]     @     0x558000ef4b54  absl::CondVar::WaitCommon()\nW0507 08:51:06.076035 1769711 thread.cc:1805]     @     0x558000cf680c  threadlogger::(anonymous namespace)::LoggingThread::Run()\nW0507 08:51:06.076046 1769711 thread.cc:1805]     @     0x558000cf1ce3  Thread::ThreadBody()\nW0507 08:51:06.076054 1769711 thread.cc:1805]     @     0x7f27686417db  start_thread\nW0507 08:51:06.076056 1769711 thread.cc:1805]     @     0x7f27685b405f  clone\nW0507 08:51:06.076060 1769711 thread.cc:1805] creator: 0x558000cf27c3 0x558000cf6350 0x558000e954b0 0x558000e94aa4 0x558000e65503 0x557ffe7951e2 0x557ffe75da94 0x7f27684db3d4 0x557ffe75d02a\nW0507 08:51:06.076066 1769711 thread.cc:1805] ---- Processed 5 threads ----\nW0507 08:51:06.076088 1769711 thread.cc:1805] --- Memory map: ---\nW0507 08:51:06.076462 1769711 thread.cc:1805]   build=\nW0507 08:51:06.076470 1769711 thread.cc:1805]   557ffda00000-557ffda01000: $/eval_ir_main\nW0507 08:51:06.076474 1769711 thread.cc:1805]   557ffda01000-558001059000: $/eval_ir_main (@1000)\nW0507 08:51:06.076490 1769711 thread.cc:1805]   7f2768101000-7f2768103000: /usr/v5/lib64/libnss-2.27.so\nW0507 08:51:06.076496 1769711 thread.cc:1805]   7f2768107000-7f2768111000: /usr/v5/lib64/libnss_files-2.27.so\nW0507 08:51:06.076527 1769711 thread.cc:1805]   7f276847a000-7f276861c000: /usr/v5/lib64/libc-2.27.so\nW0507 08:51:06.076534 1769711 thread.cc:1805]   7f2768628000-7f276862b000: /usr/v5/lib64/libdl-2.27.so\nW0507 08:51:06.076539 1769711 thread.cc:1805]   7f276862d000-7f2768633000: /usr/v5/lib64/librt-2.27.so\nW0507 08:51:06.076546 1769711 thread.cc:1805]   7f2768636000-7f276864d000: /usr/v5/lib64/libpthread-2.27.so\nW0507 08:51:06.076553 1769711 thread.cc:1805]   7f2768654000-7f27687c5000: /usr/v5/lib64/libm-2.27.so\nW0507 08:51:06.076558 1769711 thread.cc:1805]   7f27687ca000-7f27687f5000: /usr/v5/lib64/ld-2.27.so\nW0507 08:51:06.076568 1769711 thread.cc:1805]   7ffc0e543000-7ffc0e545000: [vdso]\nW0507 08:51:06.076571 1769711 thread.cc:1805]   ffffffffff600000-ffffffffff601000: [vsyscall]\nI0507 08:51:06.076583 1769711 process_state.cc:308] RAW: ExecuteFailureCallbacks() unsafe\nE0507 08:51:06.076593 1769711 process_state.cc:808] RAW: Raising signal 11 with default behavior\nI0507 08:51:06.076611 1769711 process_state.cc:1438] RAW: FailureSignalHandler() exiting\n"
// issue: "https://github.com/google/xls/issues/2142"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_FUNCTION
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 128
//   proc_ticks: 0
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified; .*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
// }
// inputs {
//   function_args {
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x3_3234_00cd_bb22, bits[54]:0x3c_b873_7a43_cc8f); bits[8]:0x55"
//     args: "(bits[57]:0x1, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x15_5555_5555_5555); bits[8]:0x7f"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x55"
//     args: "(bits[57]:0x1dc_10a8_c846_00ea, bits[52]:0x8_51dd_2fee_4c65, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x40"
//     args: "(bits[57]:0x0, bits[52]:0x5_5555_5555_5555, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0x40_0000_0000_0000, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x7f"
//     args: "(bits[57]:0x4_0000, bits[52]:0x0, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x64"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x1000, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x9c"
//     args: "(bits[57]:0x2_0000_0000_0000, bits[52]:0xc_46f9_1489_0d56, bits[54]:0x15_5555_5555_5555); bits[8]:0xaa"
//     args: "(bits[57]:0x65_bdbe_43fa_76fb, bits[52]:0x6_d2b1_219d_1331, bits[54]:0x2000_0000_0000); bits[8]:0x4"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x15_5555_5555_5555); bits[8]:0x80"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0x0, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x15_5555_5555_5555); bits[8]:0xaa"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x10_0000); bits[8]:0xaa"
//     args: "(bits[57]:0x0, bits[52]:0x0, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x10"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x0"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xaa"
//     args: "(bits[57]:0x0, bits[52]:0x0, bits[54]:0x2_0000); bits[8]:0xaa"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x55"
//     args: "(bits[57]:0x0, bits[52]:0x2_f69a_cafd_7a53, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0x0, bits[52]:0x0, bits[54]:0x0); bits[8]:0xff"
//     args: "(bits[57]:0x10_0000_0000_0000, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x10"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0xf_18fc_12a4_6405); bits[8]:0x0"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x4_50bc_de68_3495, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x0"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x0, bits[54]:0x1_0000_0000_0000); bits[8]:0x10"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x6_760c_3927_c0c2, bits[54]:0x35_04bf_9f96_33ed); bits[8]:0xaa"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x2, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x100); bits[8]:0xff"
//     args: "(bits[57]:0xed_799f_c5ab_892b, bits[52]:0x10, bits[54]:0x15_5555_5555_5555); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x1000_0000_0000); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xa_f261_6084_dd8e, bits[54]:0x15_5555_5555_5555); bits[8]:0xaa"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x4"
//     args: "(bits[57]:0x176_4ea8_d140_5627, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xff"
//     args: "(bits[57]:0x2, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xaa"
//     args: "(bits[57]:0x0, bits[52]:0x800, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x0"
//     args: "(bits[57]:0xad_3479_3fed_e4c8, bits[52]:0x5_5555_5555_5555, bits[54]:0x15_5555_5555_5555); bits[8]:0x0"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x100_0000); bits[8]:0x0"
//     args: "(bits[57]:0x40, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x20_0000); bits[8]:0x0"
//     args: "(bits[57]:0x80_0000_0000, bits[52]:0x10, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0xe3"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xa3"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xff"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x5_5555_5555_5555, bits[54]:0x0); bits[8]:0xff"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x5_5555_5555_5555, bits[54]:0x0); bits[8]:0xff"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x7"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x15_5555_5555_5555); bits[8]:0x8c"
//     args: "(bits[57]:0xf8_25a0_d026_a4ad, bits[52]:0xf_68ae_e735_0a83, bits[54]:0x15_5555_5555_5555); bits[8]:0x0"
//     args: "(bits[57]:0x0, bits[52]:0x7_0512_d008_72b3, bits[54]:0x15_5555_5555_5555); bits[8]:0x0"
//     args: "(bits[57]:0x20_0000, bits[52]:0x0, bits[54]:0x100_0000); bits[8]:0xaa"
//     args: "(bits[57]:0x0, bits[52]:0x6_43e4_2f23_4135, bits[54]:0x15_5555_5555_5555); bits[8]:0x58"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0xd_4fee_e733_4698, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xd_70ce_02a8_0ae3, bits[54]:0x11_2deb_ae86_0880); bits[8]:0x7f"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x80"
//     args: "(bits[57]:0x100, bits[52]:0x2_6462_af7a_e339, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xaa"
//     args: "(bits[57]:0x4000_0000_0000, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0xff"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xeb"
//     args: "(bits[57]:0x40_0000, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x20"
//     args: "(bits[57]:0x800_0000, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x40"
//     args: "(bits[57]:0x103_7343_5972_9077, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x80"
//     args: "(bits[57]:0x0, bits[52]:0x0, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x0"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x4_fc7f_e61c_47e5, bits[54]:0x80_0000); bits[8]:0x7f"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x9a"
//     args: "(bits[57]:0x40, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x15_5555_5555_5555); bits[8]:0x7f"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x200_0000_0000, bits[54]:0x20); bits[8]:0x55"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x0, bits[54]:0x19_a97b_cd4c_f1e1); bits[8]:0x55"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x15_5555_5555_5555); bits[8]:0x40"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x10_0000, bits[54]:0x40_0000); bits[8]:0xcb"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x80_0000, bits[54]:0x1d_9298_d004_94ba); bits[8]:0x7f"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x0, bits[54]:0x3c_0cda_e174_9051); bits[8]:0x55"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x5_0552_1ccc_1715, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x76"
//     args: "(bits[57]:0x2000, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x17_a23f_8b35_65a2); bits[8]:0x2"
//     args: "(bits[57]:0x0, bits[52]:0x0, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x0"
//     args: "(bits[57]:0x8b_8f37_77ef_262f, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x17_b7e5_263b_c05e); bits[8]:0x7f"
//     args: "(bits[57]:0x8000, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x5d"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x800_0000_0000, bits[54]:0x0); bits[8]:0x7f"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x0, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0xbf"
//     args: "(bits[57]:0x7f_48e1_74f7_ae21, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x55"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2000_0000_0000); bits[8]:0xaa"
//     args: "(bits[57]:0xdf_efc1_7098_e251, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x15_5555_5555_5555); bits[8]:0xaa"
//     args: "(bits[57]:0x151_75b3_bb8e_3917, bits[52]:0x4_0000_0000_0000, bits[54]:0x20_0000_0000_0000); bits[8]:0xff"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x1000_0000, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0x0, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x0); bits[8]:0xaa"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x10_0000, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x2e_de21_c6fc_a9ec, bits[52]:0x4_8ce2_74cb_efce, bits[54]:0x3_761a_d230_1d6a); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x0, bits[54]:0x0); bits[8]:0xaa"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x800_0000); bits[8]:0x0"
//     args: "(bits[57]:0x2000_0000_0000, bits[52]:0x2000_0000, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x8"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x400_0000, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xaa"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x20_0000, bits[54]:0x0); bits[8]:0x0"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x38"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x0, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x3_f4cc_166b_ec16, bits[54]:0x0); bits[8]:0xaa"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x1_c9da_4435_5a02, bits[54]:0x0); bits[8]:0x20"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0x4"
//     args: "(bits[57]:0xb4_d793_f312_fbf9, bits[52]:0xb_9500_ec32_8ace, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x10_0000_0000_0000); bits[8]:0xaa"
//     args: "(bits[57]:0x0, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0xff"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0x0"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x0, bits[54]:0x1f_ffff_ffff_ffff); bits[8]:0xaa"
//     args: "(bits[57]:0x8_0000_0000, bits[52]:0x5_5555_5555_5555, bits[54]:0x800); bits[8]:0x16"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x15_5555_5555_5555); bits[8]:0x0"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x0, bits[54]:0x0); bits[8]:0x55"
//     args: "(bits[57]:0x1c4_2575_b5c4_6413, bits[52]:0x5_5555_5555_5555, bits[54]:0x3c_1867_a6c1_b5c4); bits[8]:0x0"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x0); bits[8]:0xb8"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x39_6f05_8af4_1dfd); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x1000, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xe7"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x8000_0000, bits[54]:0x1_0000_0000_0000); bits[8]:0x55"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x1_caef_2750_10c1, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xd8"
//     args: "(bits[57]:0x1e2_8dd5_d028_4cd1, bits[52]:0x0, bits[54]:0x100_0000_0000); bits[8]:0x8"
//     args: "(bits[57]:0x0, bits[52]:0x40_0000_0000, bits[54]:0x15_5555_5555_5555); bits[8]:0x80"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x7_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x55"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0x4_c9f1_286c_1f0f, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x7f"
//     args: "(bits[57]:0x20_07c1_cbe9_c8f2, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xe0"
//     args: "(bits[57]:0x1e1_dcd4_c4bc_9b2f, bits[52]:0xf_ffff_ffff_ffff, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0x1"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x80_0000); bits[8]:0xae"
//     args: "(bits[57]:0x1ff_ffff_ffff_ffff, bits[52]:0x2_567f_f4f2_8d9d, bits[54]:0x1c_43e2_d5ee_cbd1); bits[8]:0xff"
//     args: "(bits[57]:0xa3_4447_5390_878a, bits[52]:0x0, bits[54]:0x28_19bb_36af_6ec6); bits[8]:0xff"
//     args: "(bits[57]:0xaa_aaaa_aaaa_aaaa, bits[52]:0x100, bits[54]:0x2_0000_0000_0000); bits[8]:0x55"
//     args: "(bits[57]:0xff_ffff_ffff_ffff, bits[52]:0x5_5555_5555_5555, bits[54]:0x3f_ffff_ffff_ffff); bits[8]:0xaa"
//     args: "(bits[57]:0x155_5555_5555_5555, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x2a_aaaa_aaaa_aaaa); bits[8]:0xff"
//     args: "(bits[57]:0x1c7_0131_fef6_e7ca, bits[52]:0xa_aaaa_aaaa_aaaa, bits[54]:0x0); bits[8]:0xff"
//   }
// }
// 
// END_CONFIG
const W32_V1494 = u32:0x5d6;
type x4 = xN[bool:0x0][8];
type x15 = (bool, bool, bool, bool, bool);
fn x9(x10: x4) -> (bool, bool, bool, bool, bool) {
    {
        let x11: bool = x10 >= x10;
        let x12: bool = xor_reduce(x10);
        let x13: bool = x11[0+:bool];
        let x14: bool = x13 / bool:0x1;
        (x13, x13, x13, x13, x11)
    }
}
fn main(x0: (u57, u52, u54), x1: s8) -> (u8, u8, u4) {
    {
        let x2: u4 = (x1 as xN[bool:0x0][8])[-4:];
        let x3: u16 = decode<u16>(x2);
        let x5: x4[W32_V1494] = "Xyw9cpNkDk=R5_?^:_QOx;!nFlUCZejU1~f^R-%G?J/4/4Jb_Q><2x P5OIMAVB(`}W(~%IxBB)j\\jCeV rWu&-%c(]8[jWL0GQf:1+L4Zo7k\'i-q7k)A),R+Je`9M0csWV?:p#S~mM~bK/^6\"=|UfASl9nRs[i-Si}x]B5 9#[,uYv,tHU`eyd _O\"RPaXe1M\"l>t_t~j6Z+R(0g!5+5F]rnvz^NTZ$+*$x:xy2|Hv!bc[Jdx4!]Vn;2csq,+F\'\"ZCw%8.&vkcV{_8n3Yk8G~G\'>rW!jRA=YG%\"rmM7bQg}\'E\"qb C]u3E[C[+z)8VE`AJSA}-Um5!{D]\";fxZ/\\Bfn?!8/N6prc97|.>x;S\\~f>6%Z$l$LUvUXcFolx)2nE;[`.<z5>dtP^tSF\"=2|b~kh6+8<$?%I,,|U[m`h{p\"<d;Y>+^9.lWFxy_uc)Pg-@<Z~!4vtXm316+bUaW{qv]gI}*W*mn;=U\'D&>\"A4%?}KT0QVovn.A+\\Ti;~ZK#X!)0d2MMMu@2D?JjXNBt9Za,smRxd]YV ^Sa]HuXFH#z?h~*WRXwALZv1<@=`Eg Vv8}/dt|_Lb4?Qt/QGf}@)#P6]o@<(*t}/;b}~mU09+xyg082Z|V.JdD$dktU>}aE}Zn^`n[\"C/P]Zr9:hp\\zf)Of:m[\"Amz*p!5kjFfte>U24B?3W-o,i d}U\\qM>Goa%%@\'Y-WcKLnqZE8,:1Tt<WbFxcFJ!XW?u3-|cLvo&LODT}mS=&y*wK8?pw%?~*C|Na>,x6<42;XB**k08lOGUcTtiR_JBlH.GTAP]XS/*`Jc%+;L\'TU8Uk;}iY8:Ka}z-!NV7eJg3]rHpr:~5Iao,Q1l>W5W0@>p6G1?j[G6GQ7!Z|RC+XWJL<hOZ9E,az60d:h*1a%fG9H<e2Jz6UYE\\v\\@LlbBqg*FFBo/nnCIaNcsza[1Iko 1kN)Z|u*a+Gfts@(z=Z _DE09$XCvd}HbM3`x~:=4HF-$;sN8hDV!ZBZt,wbt>c.V\"_%$??/^y)|+<6X(|@\"=Qz4pKc]a01I\'48%4-SoBw{cez59KTK(Sm[bR[nCCp@/i}w>JM}\"}^0J`:e(Y!q]t$!lR8fPYc3qbGW#Moq9DWh5-~HbZLbj3xxCE`*vxE^?l97Irb&@l9>T\'Rad]8[OYCZ5[[ |86G(wq7`sVRa(HW[Lo(qBRH$@kW`yp1jDs@(8)\"yKF+\"=*hsE-X?Rf[dotbeF1`;\'\'S,7zhf&!V-%zDhy[p*\'>|he,nXx~<pUF :&s=\\y\\\" !#0E-BqnVq@=u7qazLTi~6JM_r/+IQId jQr6zMUVh~ ey-x90iU?N38 kZX(8%x9^\"O%yI?Xsx*gjf<=GwrC5I8Ilt #e4-Iy_zdxAxcx-]:n1eq6kaoM:gmv+X?@BHlN[r[<KabZ%bm7IX6?&cFK]TE]a\"FIJ)cVAZ2k_d_AR2`4?k*Eqg1Z&h4ZY3eiwG=?dKwbxT$@~y)o`F<\\w<fqG!9y%";
        let x6: u8 = (x1 as u8)[x2+:u8];
        let x7: u4 = x2[:];
        let x8: u4 = bit_slice_update(x2, x3, x2);
        let x16: x15[W32_V1494] = map(x5, x9);
        let x17: u8 = !x6;
        let x18: u8 = gate!(x17 as u4 > x2, x6);
        let x19: u4 = x7 % u4:0xf;
        let x20: bool = x2 <= x2;
        let x21: xN[bool:0x0][3] = x2[-5:-1];
        let x22: bool = x20 | x20;
        let x23: u8 = x17[:];
        let x24: u7 = x17[0+:xN[bool:0x0][7]];
        let x25: u52 = x0.1;
        let x26: u16 = !x3;
        let x27: u4 = x2 % u4:0x0;
        (x23, x18, x8)
    }
}
