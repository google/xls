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
// exception: "Subprocess call failed: /xls/tools/eval_proc_main --testvector_textproto=/export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_tmpdirxjdqapeu/temp_directory_94vT5a/testvector.pbtxt --ticks=128 --backend=serial_jit /export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_tmpdirxjdqapeu/temp_directory_94vT5a/sample.ir --logtostderr\n\nSubprocess stderr:\nI0423 04:52:51.246703 3389796 init_google.cc:761] Linux version 5.10.0-smp-1105.47.0.0 (builder@builder:/usr/src/linux) (clang version -trunk (2cbe5a33a5fda257747d75863bd9ccb8920b9249), LLD -trunk) #1 [v5.10.0-1105.47.0.0] SMP @1740169297\nI0423 04:52:51.246779 3389796 init_google.cc:778] Enabled /npx experiments: TCMALLOC_MIN_HOT_ACCESS_HINT_ABLATION,TEST_ONLY_L3_AWARE,TEST_ONLY_USER_VCPU\nI0423 04:52:51.246794 3389796 init_google.cc:843] Process id 3389796\nI0423 04:52:51.246812 3389796 init_google.cc:848] Current working directory /export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_tmpdirxjdqapeu/temp_directory_94vT5a\nI0423 04:52:51.246814 3389796 init_google.cc:850] Current timezone is PDT (currently UTC -07:00)\nI0423 04:52:51.246822 3389796 init_google.cc:854] Built on Apr 23 2025 00:19:01 (1745392741)\nI0423 04:52:51.246974 3389796 init_google.cc:855]  at xls-@ugbnu14.prod.google.com://cloud/buildrabbit-username/buildrabbit-client/\nI0423 04:52:51.246976 3389796 init_google.cc:856]  as //xls/tools:eval_proc_main\nI0423 04:52:51.246978 3389796 init_google.cc:857]  for gcc-4.X.Y-crosstool-v18-llvm-grtev4-k8.k8\nI0423 04:52:51.246981 3389796 init_google.cc:860]  from changelist 750479567 with baseline 750479567 in a mint client based on //depot/\nI0423 04:52:51.246982 3389796 init_google.cc:866] Build tool: , release 2025.04.14-1 (mainline @747171332)\nI0423 04:52:51.246984 3389796 init_google.cc:867] Build target: -out/k8-opt/bin/xls/tools/eval_proc_main\nI0423 04:52:51.246986 3389796 init_google.cc:874] Command line arguments:\nI0423 04:52:51.246987 3389796 init_google.cc:876] argv[0]: \'/xls/tools/eval_proc_main\'\nI0423 04:52:51.246993 3389796 init_google.cc:876] argv[1]: \'--testvector_textproto=/export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_tmpdirxjdqapeu/temp_directory_94vT5a/testvector.pbtxt\'\nI0423 04:52:51.246996 3389796 init_google.cc:876] argv[2]: \'--ticks=128\'\nI0423 04:52:51.247000 3389796 init_google.cc:876] argv[3]: \'--backend=serial_jit\'\nI0423 04:52:51.247002 3389796 init_google.cc:876] argv[4]: \'/export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_tmpdirxjdqapeu/temp_directory_94vT5a/sample.ir\'\nI0423 04:52:51.247004 3389796 init_google.cc:876] argv[5]: \'--logtostderr\'\nI0423 04:52:51.249180 3389796 logger.cc:310] Enabling threaded logging for severity WARNING\nI0423 04:52:51.250134 3389796 mlock.cc:219] mlock()-ed 4096 bytes for BuildID, using 1 syscalls.\n*** SIGSEGV STACK OVERFLOW (see /cppstackoverflow) received by PID 3389796 (TID 3389796) on cpu 157; stack trace: ***\nPC: @     0x7f81f2fef72c  (unknown)  (unknown)\n    @     0x55c1309b8045       1904  FailureSignalHandler()\n    @     0x7f81f385ce80  (unknown)  (unknown)\n    @     0x7f81f2fca5c7      52944  (unknown)\n    @     0x55c12e9cc4d7        304  xls::ProcJit::Tick()\n    @     0x55c12ea78aea        512  xls::SerialProcRuntime::TickInternal()\n    @     0x55c12ea7b3e4        208  xls::ProcRuntime::Tick()\n    @     0x55c12e89ef48       1008  xls::(anonymous namespace)::EvaluateProcs()\n    @     0x55c12e8930a1       1120  main\n    @     0x7f81f36ed3d4        192  __libc_start_main\n    @     0x55c12e89002a  (unknown)  ../sysdeps/x86_64/start.S:120 _start\nI0423 04:53:35.231672 3389796 process_state.cc:304] RAW: ExecuteFailureCallbacks() safe\nI0423 04:53:35.231679 3389796 process_state.cc:1324] RAW: FailureSignalHandler(): starting unsafe phase\nI0423 04:53:35.231684 3389796 coreutil.cc:358] RAW: Attempting to connect to coredump socket @core\nI0423 04:53:35.231719 3389796 coreutil.cc:362] RAW: Failed to connect to coredump socket @core\nI0423 04:53:35.231739 3389796 coreutil.cc:269] RAW: Attempting to dump core\nI0423 04:53:35.232303 3389796 coreutil.cc:310] RAW: WriteCoreDumpWith(/export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/test_outputs6z9yam_s/google_log_dir/core) returns: 0\nW0423 04:53:35.232331 3389796 process_state.cc:1376] --- CPU registers: ---\n   r8=7ffca9a4dd50  r9=21 r10=7ffca9a4f370 r11=1133 r12=0 r13=0 r14=0 r15=0\n  rdi=1 rsi=1 rbp=7ffcaa23d320 rbx=5bb rdx=1 rax=0 rcx=0 rsp=7ffca9a4dd50\n  rip=7f81f2fef72c efl=10297 cgf=2b000000000033 err=6 trp=e msk=0\n  cr2=7ffca9a4dd50\nW0423 04:53:35.232951 3389796 process_state.cc:1380] **************************************************\nSTACK OVERFLOW DETECTED in operation on 0x7ffca9a4dd50 (stack ends at 0x7ffca9a4e000)\nSee /cppstackoverflow\n**************************************************\n--- Stack contents: ---\n  --- Stack boundaries: [0x7ffca9a4e000, 0x7ffcaa24e000) --- (8192KiB)\n  0x00007ffca9a4e000: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e020: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e040: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e060: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e080: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e0a0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e0c0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e0e0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e100: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e120: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e140: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e160: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e180: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\n  0x00007ffca9a4e1a0: 0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000\nW0423 04:53:35.233673 3389796 thread.cc:1798] --- Thread 7f81f3689c40 (name: main/3389796) stack: ---\nW0423 04:53:35.234561 3389796 thread.cc:1798]     @     0x55c1309b83dd  FailureSignalHandler()\nW0423 04:53:35.234648 3389796 thread.cc:1798]     @     0x7f81f385ce80  __restore_rt\nW0423 04:53:35.235171 3389796 thread.cc:1798]     @     0x7f81f2fca5c7  (unknown)\nW0423 04:53:35.235177 3389796 thread.cc:1798]     @     0x55c12e9cc4d7  xls::ProcJit::Tick()\nW0423 04:53:35.235180 3389796 thread.cc:1798]     @     0x55c12ea78aea  xls::SerialProcRuntime::TickInternal()\nW0423 04:53:35.235181 3389796 thread.cc:1798]     @     0x55c12ea7b3e4  xls::ProcRuntime::Tick()\nW0423 04:53:35.235183 3389796 thread.cc:1798]     @     0x55c12e89ef48  xls::(anonymous namespace)::EvaluateProcs()\nW0423 04:53:35.235185 3389796 thread.cc:1798]     @     0x55c12e8930a1  main\nW0423 04:53:35.235187 3389796 thread.cc:1798]     @     0x7f81f36ed3d4  __libc_start_main\nW0423 04:53:35.235189 3389796 thread.cc:1798]     @     0x55c12e89002a  ../sysdeps/x86_64/start.S:120 _start\nW0423 04:53:35.235706 3389796 thread.cc:1798] --- Thread 7f81f3634700 (name: ExitTimeoutWatcher/3389816) stack: ---\nstack used: 8 KiB of 36 KiB\nW0423 04:53:35.235887 3389796 thread.cc:1798]     @     0x7f81f385c19a  nanosleep\nW0423 04:53:35.236656 3389796 thread.cc:1798]     @     0x55c13098dc6b  AbslInternalSleepFor\nW0423 04:53:35.237384 3389796 thread.cc:1798]     @     0x55c1308194dc  (anonymous namespace)::ExitTimeoutWatcher()\nW0423 04:53:35.237422 3389796 thread.cc:1798]     @     0x7f81f38537db  start_thread\nW0423 04:53:35.237572 3389796 thread.cc:1798]     @     0x7f81f37c605f  clone\nW0423 04:53:35.237754 3389796 thread.cc:1798] --- Thread 7f81f3629700 (name: ThreadLivenessWatcher/3389817) stack: ---\nstack used: 8 KiB of 36 KiB\nW0423 04:53:35.237758 3389796 thread.cc:1798]     @     0x7f81f385c19a  nanosleep\nW0423 04:53:35.237760 3389796 thread.cc:1798]     @     0x55c13098dc6b  AbslInternalSleepFor\nW0423 04:53:35.238530 3389796 thread.cc:1798]     @     0x55c130819719  (anonymous namespace)::ThreadLivenessWatcher()\nW0423 04:53:35.238533 3389796 thread.cc:1798]     @     0x7f81f38537db  start_thread\nW0423 04:53:35.238535 3389796 thread.cc:1798]     @     0x7f81f37c605f  clone\nW0423 04:53:35.238639 3389796 thread.cc:1798] --- Thread 7f81f361e700 (name: MemoryReleaser/3389818) stack: ---\nstack used: 8 KiB of 1960 KiB\nW0423 04:53:35.238643 3389796 thread.cc:1798]     @     0x7f81f385c19a  nanosleep\nW0423 04:53:35.238645 3389796 thread.cc:1798]     @     0x55c13098dc6b  AbslInternalSleepFor\nW0423 04:53:35.239523 3389796 thread.cc:1798]     @     0x55c1309aa113  MallocExtension_Internal_ProcessBackgroundActions\nW0423 04:53:35.240513 3389796 thread.cc:1798]     @     0x55c130815303  Thread::ThreadBody()\nW0423 04:53:35.240520 3389796 thread.cc:1798]     @     0x7f81f38537db  start_thread\nW0423 04:53:35.240523 3389796 thread.cc:1798]     @     0x7f81f37c605f  clone\nW0423 04:53:35.240527 3389796 thread.cc:1798] creator: 0x55c130815de3 0x55c130b636a9 0x55c1309b6170 0x55c1309b57e4 0x55c130989a23 0x55c12e8c1b22 0x55c12e891071 0x7f81f36ed3d4 0x55c12e89002a\nW0423 04:53:35.240801 3389796 thread.cc:1798] --- Thread 7f81f330f700 (name: Logger/3389820) stack: ---\nstack used: 8 KiB of 1960 KiB\nW0423 04:53:35.241586 3389796 thread.cc:1798]     @     0x55c13098e1ec  AbslInternalPerThreadSemWait\nW0423 04:53:35.242421 3389796 thread.cc:1798]     @     0x55c130a10b54  absl::CondVar::WaitCommon()\nW0423 04:53:35.243341 3389796 thread.cc:1798]     @     0x55c130819e2c  threadlogger::(anonymous namespace)::LoggingThread::Run()\nW0423 04:53:35.243351 3389796 thread.cc:1798]     @     0x55c130815303  Thread::ThreadBody()\nW0423 04:53:35.243353 3389796 thread.cc:1798]     @     0x7f81f38537db  start_thread\nW0423 04:53:35.243355 3389796 thread.cc:1798]     @     0x7f81f37c605f  clone\nW0423 04:53:35.243358 3389796 thread.cc:1798] creator: 0x55c130815de3 0x55c130819970 0x55c1309b6170 0x55c1309b57e4 0x55c130989a23 0x55c12e8c1b22 0x55c12e891071 0x7f81f36ed3d4 0x55c12e89002a\nW0423 04:53:35.243363 3389796 thread.cc:1798] ---- Processed 5 threads ----\nW0423 04:53:35.243393 3389796 thread.cc:1798] --- Memory map: ---\nW0423 04:53:35.243844 3389796 thread.cc:1798]   build=/export/hda3//local_ram_fs_dirs/8.xls..xls-.11605143829041.c8564a224093156f/logs.8.xls..xls-.11605143829041/tmp/107793437/tmp/tmpz3zogloy/bbq-31b3f689-cb25-40db-909d-dc8e9faba9e1/-out/k8-opt\nW0423 04:53:35.243854 3389796 thread.cc:1798]   55c12de00000-55c12de01000: $build/bin/xls/fuzzer/fuzz_integration_test.runfiles//xls/tools/eval_proc_main\nW0423 04:53:35.243860 3389796 thread.cc:1798]   55c12de01000-55c130b6a000: $build/bin/xls/fuzzer/fuzz_integration_test.runfiles//xls/tools/eval_proc_main (@1000)\nW0423 04:53:35.243883 3389796 thread.cc:1798]   7f81f3310000-7f81f3312000: /usr/grte/v5/lib64/libnss_borg-2.27.so\nW0423 04:53:35.243923 3389796 thread.cc:1798]   7f81f3316000-7f81f3320000: /usr/grte/v5/lib64/libnss_files-2.27.so\nW0423 04:53:35.243939 3389796 thread.cc:1798]   7f81f368c000-7f81f382e000: /usr/grte/v5/lib64/libc-2.27.so\nW0423 04:53:35.243946 3389796 thread.cc:1798]   7f81f383a000-7f81f383d000: /usr/grte/v5/lib64/libdl-2.27.so\nW0423 04:53:35.243952 3389796 thread.cc:1798]   7f81f383f000-7f81f3845000: /usr/grte/v5/lib64/librt-2.27.so\nW0423 04:53:35.243956 3389796 thread.cc:1798]   7f81f3848000-7f81f385f000: /usr/grte/v5/lib64/libpthread-2.27.so\nW0423 04:53:35.243961 3389796 thread.cc:1798]   7f81f3866000-7f81f39d7000: /usr/grte/v5/lib64/libm-2.27.so\nW0423 04:53:35.243966 3389796 thread.cc:1798]   7f81f39dc000-7f81f3a07000: /usr/grte/v5/lib64/ld-2.27.so\nW0423 04:53:35.243973 3389796 thread.cc:1798]   7ffcaa395000-7ffcaa397000: [vdso]\nW0423 04:53:35.243977 3389796 thread.cc:1798]   ffffffffff600000-ffffffffff601000: [vsyscall]\nI0423 04:53:35.243988 3389796 process_state.cc:308] RAW: ExecuteFailureCallbacks() unsafe\nE0423 04:53:35.243997 3389796 process_state.cc:808] RAW: Raising signal 11 with default behavior\nI0423 04:53:35.244008 3389796 process_state.cc:1438] RAW: FailureSignalHandler() exiting\n"
// issue: "https://github.com/google/xls/issues/2049"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: false
//   simulate: false
//   use_system_verilog: false
//   timeout_seconds: 1500
//   calls_per_sample: 0
//   proc_ticks: 128
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
//   channel_inputs {
//     inputs {
//       channel_name: "sample__x14"
//       values: "(bits[6]:0x20, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x15, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x0, bits[36]:0xe_ece0_17e4)"
//       values: "(bits[6]:0x2a, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x4, bits[36]:0xf_a8ba_a5af)"
//       values: "(bits[6]:0x4, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x3f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x15, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x2a, bits[36]:0x1000_0000)"
//       values: "(bits[6]:0x1f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x4, bits[36]:0x0)"
//       values: "(bits[6]:0x2a, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x2a, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x8, bits[36]:0x5_3c48_9ea4)"
//       values: "(bits[6]:0x0, bits[36]:0xb_a87b_640d)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x2a, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0xc, bits[36]:0x80_0000)"
//       values: "(bits[6]:0x3f, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x1, bits[36]:0x2_d7fb_bdef)"
//       values: "(bits[6]:0x2a, bits[36]:0x3_59b4_14b7)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0x8, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x10, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x1f, bits[36]:0x7_8436_35b0)"
//       values: "(bits[6]:0x1f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x1f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x2a, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x3f, bits[36]:0x5_a190_9255)"
//       values: "(bits[6]:0x15, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x15, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x15, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x2a, bits[36]:0x4_f3b1_72b8)"
//       values: "(bits[6]:0x2a, bits[36]:0x9_123d_89a9)"
//       values: "(bits[6]:0x15, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x17, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x1f, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x10, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x4, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x2a, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x3f, bits[36]:0x8)"
//       values: "(bits[6]:0x1f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x3f, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0x100_0000)"
//       values: "(bits[6]:0x15, bits[36]:0x8)"
//       values: "(bits[6]:0x20, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x2a, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x10, bits[36]:0x0)"
//       values: "(bits[6]:0x20, bits[36]:0x9_dd47_be49)"
//       values: "(bits[6]:0x1f, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x17, bits[36]:0x20_0000)"
//       values: "(bits[6]:0x10, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x8, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0x2a, bits[36]:0x20)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0x8000_0000)"
//       values: "(bits[6]:0x8, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x2a, bits[36]:0x0)"
//       values: "(bits[6]:0x2, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x3f, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x1f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x23, bits[36]:0xb_98fb_da22)"
//       values: "(bits[6]:0x2a, bits[36]:0x0)"
//       values: "(bits[6]:0x2, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x20, bits[36]:0x10_0000)"
//       values: "(bits[6]:0x0, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x15, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0xe, bits[36]:0xd_28bb_29fc)"
//       values: "(bits[6]:0x20, bits[36]:0x0)"
//       values: "(bits[6]:0x2, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0x0)"
//       values: "(bits[6]:0x1f, bits[36]:0x0)"
//       values: "(bits[6]:0x15, bits[36]:0x0)"
//       values: "(bits[6]:0x3e, bits[36]:0x1_0000_0000)"
//       values: "(bits[6]:0x3f, bits[36]:0x0)"
//       values: "(bits[6]:0x3f, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x3f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0xe, bits[36]:0x0)"
//       values: "(bits[6]:0x0, bits[36]:0xa_d171_d6a4)"
//       values: "(bits[6]:0x20, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x3f, bits[36]:0x2_0000_0000)"
//       values: "(bits[6]:0x15, bits[36]:0xf_4e61_1e6c)"
//       values: "(bits[6]:0x30, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0x4_41bf_3dee)"
//       values: "(bits[6]:0x3f, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x0, bits[36]:0x5_5555_5555)"
//       values: "(bits[6]:0x2, bits[36]:0xf_ffff_ffff)"
//       values: "(bits[6]:0x2, bits[36]:0x0)"
//       values: "(bits[6]:0x1f, bits[36]:0x80)"
//       values: "(bits[6]:0x1f, bits[36]:0x0)"
//       values: "(bits[6]:0x15, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x0, bits[36]:0xa_aaaa_aaaa)"
//       values: "(bits[6]:0x3f, bits[36]:0x2)"
//       values: "(bits[6]:0x15, bits[36]:0xf_511a_b5eb)"
//       values: "(bits[6]:0x0, bits[36]:0x7_ffff_ffff)"
//       values: "(bits[6]:0x1f, bits[36]:0x4000_0000)"
//       values: "(bits[6]:0x3f, bits[36]:0x40_0000)"
//       values: "(bits[6]:0x2a, bits[36]:0x0)"
//     }
//   }
// }
// 
// END_CONFIG
const W32_V7 = u32:0x7;
type x0 = u47;
type x26 = u8;
type x36 = (bool, bool, bool);
fn x2(x3: (s36, x0[W32_V7], s6), x4: (s36, x0[W32_V7], s6), x5: (s36, x0[W32_V7], s6), x6: (s36, x0[W32_V7], s6), x7: (s36, x0[W32_V7], s6)) -> (bool, bool) {
    {
        let x8: bool = x4 != x3;
        (x8, x8)
    }
}
fn x32(x33: x26) -> (bool, bool, bool) {
    {
        let x34: bool = xor_reduce(x33);
        let x35: bool = !x34;
        (x35, x35, x35)
    }
}
proc main {
    x14: chan<(u6, u36)> in;
    config(x14: chan<(u6, u36)> in) {
        (x14,)
    }
    init {
        (s36:-22906492246, [x0:93824992236885, x0:46912496118442, x0:93824992236885, x0:70368744177663, x0:0, x0:115847473725225, x0:32768], s6:21)
    }
    next(x1: (s36, x0[W32_V7], s6)) {
        {
            let x9: (bool, bool) = x2(x1, x1, x1, x1, x1);
            let (x10, x11): (bool, bool) = x2(x1, x1, x1, x1, x1);
            let x12: u2 = decode<u2>(x10);
            let x13: bool = x12 as bool & x11;
            let x15: (token, (u6, u36)) = recv(join(), x14);
            let x16: token = x15.0;
            let x17: (u6, u36) = x15.1;
            let x18: bool = x11 & x11;
            let x19: bool = bit_slice_update(x11, x11, x12);
            let x20: u4 = x12 ++ x13 ++ x18;
            let x21: u2 = x18 as u2 & x12;
            let x22: u11 = x11 ++ x13 ++ x10 ++ x11 ++ x11 ++ x20 ++ x12;
            let x23: bool = encode(x21);
            let x24: bool = x11 * x19;
            let x25: s63 = s63:0x2aaa_aaaa_aaaa_aaaa;
            let x27: x26[1887] = ".Qvtptw>2QG8f[,}f7=.ls?X{S:la\'*p~is&IA.oLur@sX~vO]ajS\"HP!4:r5V\\GgHekS N2r=p;~<&lt3V\'Z o\\tlN83Yw|,?bQ-VnO;ZG wr&.S~VPqk%8Y\\=n2Oz//8&]<;cmWY\'R.u\"d-Un)At2ziyj=EX?Ra\'3<|>59SWC8pcId];\'09xVG6,E<(S|=De?Vk<2+z@3T>n,*DRwHUPv:Gp$IhF0|Fsg>Yl#pYqY|/R@>9*SX%:s\\*rNS[|Gpg^n&.i-qJg\"uV*IHy7~WK{&]e2o)Cgl1i$o)Io*j6rK qA/YPv80FG^Ni?%u_erpB2=6bH+]MF(KA4ek)z3>\\#\'TbN1eCZ&Z#LreQOo<_&Gme!u:4KotB`lGHln{p:UvG&MK]v^7?ri3&-\"tWfs36@o=u d7kM/[R-;S7-}nVW W*@>N\'KWl6U/*b3Z#m+k=]I,i7Ri4SvFFa2v<#0,@H>GsFb6q167(`^nvY?@;KhuHsVUe6(kc#_&-OxwWj <kOG-}`NMOwd2#J\\xk@#/2K3,X>>bm9z5_TXS8}51$9jN4z;y%iYE-DI|V7~7tk(-e/Bxb08>k[[ B\"9\'J^.V=LRh+|ja\"|CIDh#i:bE9@2rLBM{0zaUSg/{6C&W?lmf1PMk{Or1{@}qxN!5Ue-f*P(UcQcR,~s0OeOb`KIi/s05U(Q37)uUW/T+pKj,fNc3n\"kx#MEdPo&oWo|_&.4ibiScU&@p.CNZp0v[|I\\_VVtpJo.`fZ(\"=>Q#y!9<V`mQ~vNrB7#mM>D7\"}Rn`[0Ts6Ob/A[e;s3,Ns!xhms4m}>_%R~o DFc\\>l0/8;rq?TIH3H4MnfxNy-d`=%h|`D1C}4V6LlaqLO/U|&|c[_\'BVcv%3Nnf\\g!X8uxU06<4~TUpg>\"EG`Dm6\'(.0%*!{y$:|rQ^gS0?|\\.tnC[69&{7qY6eM_N dQLOwM9S?+qDqDb#?=r6g;W0v}GBK|T*\'x0fwbW!/v[|]60w0ao=jd9#]32HVtBs@G_l\"KSiVOBrL:gXa%>(P/yL,&`@akCJ\"Uxz>N)78jj19<&!88CS1\";=Xn[p1$>~)in0#txt~p>$]S%cdH20y*~b\'SWo9!%V%uS5A|\\8]2$C9]$g]\")m+jum|Z~.iUZ:#JPFcQ7Xk&*.53?0|+4Lw_3\\WPgEWkDC\'kfUqZ;InF&!Sv\'HaRJdV&e8cT~p1m<:rMZ9!83\'oJD0h_:`IkA5felA9kmg`bJ$q#JrMN:a \"\'{.Ta@(FfF?WC?I7D;gr*Y4i&Cm`5rV=5,s%>%$] DC-*Ab XGHcLZDaIkdT:tdH.`p=R~[}WS4f!H42E7]u{e&Vb+t:u[Dxe\\q48\"~(:k7[K$f`YsZy+0}6^JEG)voPXkML<Bn~l1[@]|S>i_No PoO.0[|!;Is/^R;w1^7%8;8`,WX}4T5DX7)m5~@?[(Uqs(k;AX>rKLUU!)L$Pxh;D<HK/DNv:g]p4!lIt_ 1\"q7NLlH4b -AHOB(e+:h.sGBRy&yS+#9Yj:`\'~WP!>MA.~e Ouk23I(qbs>7bykdTJx5 DSVtL@\\&.><br2NbPeUM$ts\\G,GdEyt:SI@qX:Gn8\'\"\"Rx} ~)x3OOo|JgnNY3X_~*`e[9gfYl,[ZB\'fHqf5`BceT]_s+EV!>|NXP\'=c$iF;TC7JVpl6: $|rdxzN!whyWZ/ NflN{of<6UgGRl!n/OaO{,Sm!\"@oy e7WQi;Wddq|VdzNAbe05C_&qdh$)MX M+10{/VrQ0djL6Y|%e!/ybRV\";HcFf9Z.Y(~}fz{n;Y[/jq,}3%f9y\\rIZ|_T]xv`a]Up%k<l+\\A*@}LBOl2`h&6j_!^M#n67v%N>=?8 - smWIrYGI,(*vxkv\\k}I}Dp,1*wrfv}Gq+Bj\"Nbq:^G.";
            let x28: bool = x27 != x27;
            let x29: bool = x19 | x12 as bool;
            let x30: x26[1] = array_slice(x27, x10, x26[1]:[x27[xN[bool:0x0][32]:0], ...]);
            let x31: x26[1888] = x30 ++ x27;
            let x37: x36[1887] = map(x27, x32);
            let x38: u11 = bit_slice_update(x22, x18, x12);
            let x39: bool = or_reduce(x28);
            let x40: bool = x37 == x37;
            x1
        }
    }
}
