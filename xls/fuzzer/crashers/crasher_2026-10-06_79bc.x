// Copyright 2026 The XLS Authors
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
// exception: "Subprocess call failed: /xls/tools/eval_proc_main --testvector_textproto=testvector.pbtxt --ticks=128 --backend=serial_jit sample.ir --logtostderr\n\nSubprocess stderr:\n*** SIGSEGV (@0x55c7bdd462ce), si_code=1 received by PID 2976826 (TID 2976826) on cpu 71; stack trace: ***\nPC: @     0x55c5be9ad005  (unknown)  (anonymous namespace)::X86MCCodeEmitter::emitPrefixImpl()\n    @     0x55c5c0788787       1888  FailureSignalHandler()\n    @     0x7f7689dbfc60  1479605168  (unknown)\n    @     0x55c5be9ad005        112  (anonymous namespace)::X86MCCodeEmitter::emitPrefixImpl()\n    @     0x55c5be9ae2e0        208  (anonymous namespace)::X86MCCodeEmitter::encodeInstruction()\n    @     0x55c5c0269595        160  llvm::MCObjectStreamer::emitInstToData()\n    @     0x55c5be62f143        416  llvm::X86AsmPrinter::EmitAndCountInstruction()\n    @     0x55c5be630f5e        640  llvm::X86AsmPrinter::emitInstruction()\n    @     0x55c5be9c9c3a       4480  llvm::AsmPrinter::emitFunctionBody()\n    @     0x55c5be62aa08         48  llvm::X86AsmPrinter::runOnMachineFunction()\n    @     0x55c5bed4578d        800  llvm::MachineFunctionPass::runOnFunction()\n    @     0x55c5c019654b        432  llvm::FPPassManager::runOnFunction()\n    @     0x55c5c019c04d         48  llvm::FPPassManager::runOnModule()\n    @     0x55c5c01972ef        560  llvm::legacy::PassManagerImpl::run()\n    @     0x55c5be186d57        304  llvm::orc::SimpleCompiler::operator()()\n    @     0x55c5be1ad204        144  llvm::orc::IRCompileLayer::emit()\n    @     0x55c5be1ad7f7        160  llvm::orc::IRTransformLayer::emit()\n    @     0x55c5be1aff08        176  llvm::orc::BasicIRLayerMaterializationUnit::materialize()\n    @     0x55c5be1b7752         32  llvm::orc::InPlaceTaskDispatcher::dispatch()\n    @     0x55c5be191eb6         80  llvm::orc::ExecutionSession::dispatchOutstandingMUs()\n    @     0x55c5be195b6c        352  llvm::orc::ExecutionSession::OL_completeLookup()\n    @     0x55c5be1a4135         48  llvm::orc::InProgressFullLookupState::complete()\n    @     0x55c5be189b0f        256  llvm::orc::ExecutionSession::OL_applyQueryPhase1()\n    @     0x55c5be188ab1        160  llvm::orc::ExecutionSession::lookup()\n    @     0x55c5be19224e        224  llvm::orc::ExecutionSession::lookup()\n    @     0x55c5be192692        192  llvm::orc::ExecutionSession::lookup()\n    @     0x55c5be192da8        160  llvm::orc::ExecutionSession::lookup()\n    @     0x55c5be192e6a         64  llvm::orc::ExecutionSession::lookup()\n    @     0x55c5be1803ba        176  xls::OrcJit::LoadSymbol()\n    @     0x55c5be14516b        784  xls::JittedFunctionBase::BuildInternal()\n    @     0x55c5be1498ea        992  xls::JittedFunctionBase::Build()\n    @     0x55c5be140c2e        320  xls::ProcJit::Create()\n    @     0x55c5be13f363       2768  xls::(anonymous namespace)::CreateRuntime()\n    @     0x55c5be13ee1c       1456  xls::CreateJitSerialProcRuntime()\n    @     0x55c5bdf4b8d2        880  xls::(anonymous namespace)::EvaluateProcs()\n    @     0x55c5bdf40f42       1472  main\n    @     0x7f7689c57f12        192  __libc_start_main\n    @     0x55c5bdf3c02a  (unknown)  ../sysdeps/x86_64/start.S:120 _start\n"
// issue: "https://github.com/google/xls/issues/5109"
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
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
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve the specified pipeline length.*"
//   }
//   known_failure {
//     tool: ".*codegen_main"
//     stderr_regex: ".*Impossible to schedule proc .* as specified.*: cannot achieve full throughput.*"
//   }
//   with_valid_holdoff: false
//   codegen_ng: false
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//   }
// }
//
// END_CONFIG
const W32_V5 = u32:0x5;
type x12 = u47;
proc main {
    config() {
        ()
    }
    init {
        u35:22906492245
    }
    next(x0: u35) {
        {
            let x1: (u35,) = (x0,);
            let (x2): (u35,) = (x0,);
            let x3: u35 = x0 - x0;
            let x4: u19 = x0[-24:30];
            let x5: u35 = x3 - x2;
            let x6: u35 = -x2;
            let x7: u19 = x4 << if x2 >= u35:0xf { u35:0xf } else { x2 };
            let x8: bool = x1 != x1;
            let x9: u19 = bit_slice_update(x4, x2, x3);
            let x10: u39 = u39:0x2a_aaaa_aaaa;
            let x11: bool = x10 != x6 as u39;
            let x13: token = join();
            let x14: u35 = x3 / u35:0x2_aaaa_aaaa;
            let x15: u35 = x2 * x9 as u35;
            let x17: u19 = x3 as u19 ^ x4;
            let x18: bool = x8 & x8;
            let x19: bool = -x11;
            let x20: u35 = x1.0;
            let x21: token = join(x13);
            let x22: u35 = bit_slice_update(x2, x11, x8);
            let x23: bool = and_reduce(x3);
            let x25: u35 = {
                let x24: (u35, u35) = umulp(x15, x14);
                x24.0 + x24.1
            };
            let x26: u57 = decode<u57>(x9);
            let x27: u35 = !x0;
            let x28: u35 = -x14;
            let x29: token = join();
            x15
        }
    }
}
