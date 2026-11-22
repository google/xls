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
// exception: "Subprocess call failed: /xls/tools/eval_proc_main --testvector_textproto=testvector.pbtxt --ticks=128 --backend=serial_jit sample.ir --logtostderr\n\nSubprocess stderr:\n*** SIGSEGV (@0x55fc65750f7e), si_code=1 received by PID 595992 (TID 595992) on cpu 150; stack trace: ***\nPC: @     0x55fa663db838  (unknown)  (anonymous namespace)::X86MCCodeEmitter::emitPrefixImpl()\n    @     0x55fa681eef67       1888  FailureSignalHandler()\n    @     0x7f112e9f5c60  (unknown)  (unknown)\n    @     0x55fa663db838        112  (anonymous namespace)::X86MCCodeEmitter::emitPrefixImpl()\n    @     0x55fa663dcb40        208  (anonymous namespace)::X86MCCodeEmitter::encodeInstruction()\n    @     0x55fa67cc7f25        160  llvm::MCObjectStreamer::emitInstToData()\n    @     0x55fa660593b3        416  llvm::X86AsmPrinter::EmitAndCountInstruction()\n    @     0x55fa6605b18e        624  llvm::X86AsmPrinter::emitInstruction()\n    @     0x55fa663f78ef       4464  llvm::AsmPrinter::emitFunctionBody()\n    @     0x55fa66054c38         48  llvm::X86AsmPrinter::runOnMachineFunction()\n    @     0x55fa6677a89d        800  llvm::MachineFunctionPass::runOnFunction()\n    @     0x55fa67bef2ab        432  llvm::FPPassManager::runOnFunction()\n    @     0x55fa67bf4dbd         48  llvm::FPPassManager::runOnModule()\n    @     0x55fa67bf004f        560  llvm::legacy::PassManagerImpl::run()\n    @     0x55fa65ba9147        304  llvm::orc::SimpleCompiler::operator()()\n    @     0x55fa65bcef74        144  llvm::orc::IRCompileLayer::emit()\n    @     0x55fa65bcf567        160  llvm::orc::IRTransformLayer::emit()\n    @     0x55fa65bd1c74        176  llvm::orc::BasicIRLayerMaterializationUnit::materialize()\n    @     0x55fa65bd9502         32  llvm::orc::InPlaceTaskDispatcher::dispatch()\n    @     0x55fa65bb4776         80  llvm::orc::ExecutionSession::dispatchOutstandingMUs()\n    @     0x55fa65bb842c        352  llvm::orc::ExecutionSession::OL_completeLookup()\n    @     0x55fa65bc6295         48  llvm::orc::InProgressFullLookupState::complete()\n    @     0x55fa65babf1f        256  llvm::orc::ExecutionSession::OL_applyQueryPhase1()\n    @     0x55fa65baaec1        160  llvm::orc::ExecutionSession::lookup()\n    @     0x55fa65bb4b0e        224  llvm::orc::ExecutionSession::lookup()\n    @     0x55fa65bb4f52        192  llvm::orc::ExecutionSession::lookup()\n    @     0x55fa65bb5668        160  llvm::orc::ExecutionSession::lookup()\n    @     0x55fa65bb572a         64  llvm::orc::ExecutionSession::lookup()\n    @     0x55fa65ba2a0a        176  xls::OrcJit::LoadSymbol()\n    @     0x55fa65b66b89        784  xls::JittedFunctionBase::BuildInternal()\n    @     0x55fa65b6b27a        992  xls::JittedFunctionBase::Build()\n    @     0x55fa65b623fe        320  xls::ProcJit::Create()\n    @     0x55fa65b60ae3       2768  xls::(anonymous namespace)::CreateRuntime()\n    @     0x55fa65b6059c       1504  xls::CreateJitSerialProcRuntime()\n    @     0x55fa6596b922        880  xls::(anonymous namespace)::EvaluateProcs()\n    @     0x55fa65960f92       1472  main\n    @     0x7f112e88df12        192  __libc_start_main\n    @     0x55fa6595c02a  (unknown)  ../sysdeps/x86_64/start.S:120 _start\n"
// issue: "DO NOT SUBMIT Insert link to GitHub issue here."
// sample_options {
//   input_is_dslx: true
//   sample_type: SAMPLE_TYPE_PROC
//   ir_converter_args: "--top=main"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   ir_converter_args: "--lower_to_proc_scoped_channels=false"
//   convert_to_ir: true
//   optimize_ir: true
//   use_jit: true
//   codegen: true
//   codegen_args: "--nouse_system_verilog"
//   codegen_args: "--output_block_ir_path=sample.block.ir"
//   codegen_args: "--generator=pipeline"
//   codegen_args: "--pipeline_stages=9"
//   codegen_args: "--worst_case_throughput=2"
//   codegen_args: "--reset=rst"
//   codegen_args: "--reset_active_low=false"
//   codegen_args: "--reset_asynchronous=false"
//   codegen_args: "--reset_data_path=true"
//   simulate: true
//   simulator: "iverilog"
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
//   codegen_ng: true
//   disable_unopt_interpreter: false
//   lower_to_proc_scoped_channels: false
// }
// inputs {
//   channel_inputs {
//   }
// }
// 
// END_CONFIG
const W32_V53 = u32:0b11_0101;
type x15 = u52;
type x18 = u52;
type x35 = bool;
fn x7(x8: u52, x9: u16, x10: u52, x11: u52, x12: u52) -> (x15[4], u52, u52, u52, u52) {
    {
        let x13: u52 = -x11;
        let x14: u52 = !x12;
        let x16: x15[4] = [x11, x13, x8, x8];
        (x16, x13, x14, x13, x13)
    }
}
proc main {
    config() {
        ()
    }
    init {
        u52:274877906944
    }
    next(x0: u52) {
        {
            let x1: u52 = bit_slice_update(x0, x0, x0);
            let x2: u52 = clz(x1);
            let x3: u52 = x2 / u52:0x100_0000;
            let x4: u52 = x3[:];
            let x5: u52 = x2 >> if x1 >= u52:0x22 { u52:0x22 } else { x1 };
            let x6: u16 = x4[:16];
            let x17: (x15[4], u52, u52, u52, u52) = x7(x4, x6, x2, x4, x1);
            let x19: x18[3] = [x2, x3, x5];
            let x20: u52 = x5 / u52:0x0;
            let x21: u52 = !x1;
            let x22: token = join();
            let x23: bool = or_reduce(x21);
            let x24: x18[6] = x19 ++ x19;
            let x25: u52 = bit_slice_update(x2, x1, x4);
            let x26: u6 = encode(x2);
            let x27: u52 = x17.1;
            let x28: u53 = one_hot(x25, bool:0x1);
            let x29: x18[6] = x19 ++ x19;
            let x30: u6 = x21[0+:u6];
            let x31: (u38, u55, u24) = match x3 {
                u52:0xd_b361_4955_3554 | u52:0xa_aaaa_aaaa_aaaa => (u38:0x400, u55:0x2a_aaaa_aaaa_aaaa, u24:0x20_0000),
                _ => (u38:0x2a_aaaa_aaaa, u55:0x200, u24:0x7f_ffff),
            };
            let (x32, x33, x34) = match x3 {
                u52:0xd_b361_4955_3554 | u52:0xa_aaaa_aaaa_aaaa => (u38:0x400, u55:0x2a_aaaa_aaaa_aaaa, u24:0x20_0000),
                _ => (u38:0x2a_aaaa_aaaa, u55:0x200, u24:0x7f_ffff),
            };
            let x36: x35[W32_V53] = x28 as x35[W32_V53];
            let x37: bool = -x23;
            let x38: u52 = one_hot_sel(x23, [x0]);
            let x39: u24 = x34 & x30 as u24;
            let x40: u38 = signex(x27, x32);
            let x41: u53 = one_hot(x21, bool:0x0);
            let x42: x18[11] = array_slice(x29, x37, x18[11]:[x29[u32:0x0], ...]);
            x0
        }
    }
}
