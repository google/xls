// Copyright 2024 The XLS Authors
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

#include "xls/jit/llvm_compiler.h"

#include <cerrno>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>  // NOLINT

#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm-c/Target.h"
#include "llvm/include/llvm/Analysis/CGSCCPassManager.h"
#include "llvm/include/llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/include/llvm/IR/Argument.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/Instruction.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/PassManager.h"
#include "llvm/include/llvm/IR/Use.h"
#include "llvm/include/llvm/Passes/OptimizationLevel.h"
#include "llvm/include/llvm/Passes/PassBuilder.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "llvm/include/llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

namespace {

absl::once_flag once;
void OnceInit() {
  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();
}
}  // namespace

std::string LlvmCompiler::target_triple() const {
  return target_machine_->getTargetTriple().getTriple();
}

void LlvmCompiler::InitializeLlvm() { absl::call_once(once, OnceInit); }

absl::StatusOr<llvm::DataLayout> LlvmCompiler::CreateDataLayout() {
  LlvmCompiler::InitializeLlvm();
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                       CreateTargetMachine());
  return target_machine->createDataLayout();
}

std::unique_ptr<llvm::Module> LlvmCompiler::NewModule(std::string_view name) {
  CHECK(!module_created_) << "Only one module should be made.";
  auto module = std::make_unique<llvm::Module>(name, *GetContext());
  module->setDataLayout(data_layout_);
  module->setTargetTriple(llvm::Triple(target_triple()));
  return module;
}

absl::Status LlvmCompiler::Init() {
  XLS_ASSIGN_OR_RETURN(target_machine_, CreateTargetMachine());
  if (VLOG_IS_ON(1)) {
    std::string triple = target_machine_->getTargetTriple().normalize();
    std::string cpu = target_machine_->getTargetCPU().str();
    std::string feature_string =
        target_machine_->getTargetFeatureString().str();
    VLOG(1) << "LLVM target triple: " << triple;
    VLOG(1) << "LLVM target CPU: " << cpu;
    VLOG(1) << "LLVM target feature string: " << feature_string;
    XLS_VLOG_LINES(
        1,
        absl::StrFormat("llc invocation:\n  llc [input.ll] --filetype=obj "
                        "--relocation-model=pic -mtriple=%s -mcpu=%s -mattr=%s",
                        triple, cpu, feature_string));
  }
  data_layout_ = target_machine_->createDataLayout();
  return InitInternal();
}

std::string DumpLlvmModuleToString(const llvm::Module* module) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  module->print(ostream, nullptr);
  ostream.flush();
  return buffer;
}

namespace {

// Used for reporting a bad optimization level specification to LLVM internals.
class BadOptLevelError : public llvm::ErrorInfo<BadOptLevelError> {
 public:
  explicit BadOptLevelError(int opt_level) : opt_level_(opt_level) {}

  void log(llvm::raw_ostream& os) const override {
    os << "Invalid opt level: " << opt_level_;
  }

  std::error_code convertToErrorCode() const override {
    return std::error_code(EINVAL, std::system_category());
  }

  static char ID;

 private:
  int opt_level_;
};

char BadOptLevelError::ID;

}  // namespace

llvm::Error LlvmCompiler::PerformStandardOptimization(
    llvm::Module* bare_module) {
  // Follow the directions at llvm.org/docs/NewPassManager.html to run the
  // initial architecture independent opt passes
  llvm::CGSCCAnalysisManager cgam;
  llvm::FunctionAnalysisManager fam;
  llvm::LoopAnalysisManager lam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pass_builder;

  if (include_msan_) {
    VLOG(2) << "Building with MSAN";
    pass_builder.registerPipelineStartEPCallback(
        [](llvm::ModulePassManager& mpm, llvm::OptimizationLevel) -> void {
          mpm.addPass(
              llvm::MemorySanitizerPass(llvm::MemorySanitizerOptions()));
        });
  } else {
    VLOG(2) << "No sanitizer";
  }
  pass_builder.registerModuleAnalyses(mam);
  pass_builder.registerCGSCCAnalyses(cgam);
  pass_builder.registerFunctionAnalyses(fam);
  pass_builder.registerLoopAnalyses(lam);
  pass_builder.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel llvm_opt_level;
  switch (opt_level_) {
    case 0:
      llvm_opt_level = llvm::OptimizationLevel::O0;
      break;
    case 1:
      llvm_opt_level = llvm::OptimizationLevel::O1;
      break;
    case 2:
      llvm_opt_level = llvm::OptimizationLevel::O2;
      break;
    case 3:
      llvm_opt_level = llvm::OptimizationLevel::O3;
      break;
    default:
      return llvm::Error(std::make_unique<BadOptLevelError>(opt_level_));
  }
  llvm::ModulePassManager mpm;
  if (llvm_opt_level == llvm::OptimizationLevel::O0) {
    mpm = pass_builder.buildO0DefaultPipeline(llvm_opt_level);
  } else {
    mpm = pass_builder.buildPerModuleDefaultPipeline(llvm_opt_level);
  }
  mpm.run(*bare_module, mam);
  return llvm::Error::success();
}

// Check that every operand of every instruction is in the same function as the
// instruction itself. This is a very common error which is not checked by the
// LLVM verifier(!).
absl::Status LlvmCompiler::VerifyModule(const llvm::Module& module) {
  VLOG(4) << DumpLlvmModuleToString(&module);
  for (const llvm::Function& function : module.functions()) {
    for (const llvm::BasicBlock& basic_block : function) {
      for (const llvm::Instruction& inst : basic_block) {
        int64_t i = 0;
        for (const llvm::Use& use : inst.operands()) {
          if (const llvm::Instruction* inst_use =
                  llvm::dyn_cast<llvm::Instruction>(&use)) {
            XLS_RET_CHECK_EQ(inst_use->getFunction(), &function)
                << absl::StreamFormat(
                       "In function `%s`, operand %d of this instruction is "
                       "from different function `%s`: %s",
                       function.getName().str(), i,
                       inst_use->getFunction()->getName().str(),
                       DumpLlvmObjectToString(inst));
          } else if (const llvm::Argument* arg_use =
                         llvm::dyn_cast<llvm::Argument>(&use)) {
            XLS_RET_CHECK_EQ(arg_use->getParent(), &function)
                << absl::StreamFormat(
                       "In function `%s`, operand %d of this instruction is "
                       "from different function `%s`: %s",
                       function.getName().str(), i,
                       arg_use->getParent()->getName().str(),
                       DumpLlvmObjectToString(inst));
          }

          ++i;
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xls
