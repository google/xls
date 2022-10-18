// Copyright 2022 The XLS Authors
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

#include "xls/jit/orc_jit.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>  // NOLINT

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm-c/Target.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/Analysis/CGSCCPassManager.h"
#include "llvm/include/llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/include/llvm/IR/LegacyPassManager.h"
#include "llvm/include/llvm/IR/PassManager.h"
#include "llvm/include/llvm/Passes/OptimizationLevel.h"
#include "llvm/include/llvm/Passes/PassBuilder.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
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

std::string DumpLlvmModuleToString(const llvm::Module* module) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  module->print(ostream, nullptr);
  ostream.flush();
  return buffer;
}

// Used for reporting a bad optimization level specification to LLVM internals.
class BadOptLevelError : public llvm::ErrorInfo<BadOptLevelError> {
 public:
  BadOptLevelError(int opt_level) : opt_level_(opt_level) {}

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

OrcJit::OrcJit(int64_t opt_level, bool emit_object_code)
    : context_(std::make_unique<llvm::LLVMContext>()),
      execution_session_(
          std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")),
      opt_level_(opt_level),
      emit_object_code_(emit_object_code),
      data_layout_("") {}

OrcJit::~OrcJit() {
  if (auto err = execution_session_.endSession()) {
    execution_session_.reportError(std::move(err));
  }
}

llvm::Expected<llvm::orc::ThreadSafeModule> OrcJit::Optimizer(
    llvm::orc::ThreadSafeModule module,
    const llvm::orc::MaterializationResponsibility& responsibility) {
  llvm::Module* bare_module = module.getModuleUnlocked();

  XLS_VLOG(2) << "Unoptimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));

  llvm::CGSCCAnalysisManager cgam;
  llvm::FunctionAnalysisManager fam;
  llvm::LoopAnalysisManager lam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pass_builder;

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
      return llvm::Expected<llvm::orc::ThreadSafeModule>(
          llvm::Error(std::make_unique<BadOptLevelError>(opt_level_)));
  }
  llvm::ModulePassManager mpm;
  if (llvm_opt_level == llvm::OptimizationLevel::O0) {
    mpm = pass_builder.buildO0DefaultPipeline(llvm_opt_level);
  } else {
    mpm = pass_builder.buildPerModuleDefaultPipeline(llvm_opt_level);
  }
  mpm.run(*bare_module, mam);

  XLS_VLOG(2) << "Optimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));

  if (XLS_VLOG_IS_ON(3)) {
    // The ostream and its buffer must be declared before the
    // module_pass_manager because the destrutor of the pass manager calls flush
    // on the ostream so these must be destructed *after* the pass manager. C++
    // guarantees that the destructors are called in reverse order the obects
    // are declared.
    llvm::SmallVector<char, 0> stream_buffer;
    llvm::raw_svector_ostream ostream(stream_buffer);
    llvm::legacy::PassManager mpm;
    if (target_machine_->addPassesToEmitFile(mpm, ostream, nullptr,
                                             llvm::CGFT_AssemblyFile)) {
      XLS_VLOG(3) << "Could not create ASM generation pass!";
    }
    mpm.run(*bare_module);
    XLS_VLOG(3) << "Generated ASM:";
    XLS_VLOG_LINES(3, std::string(stream_buffer.begin(), stream_buffer.end()));
  }

  return module;
}

absl::StatusOr<std::unique_ptr<OrcJit>> OrcJit::Create(int64_t opt_level,
                                                       bool emit_object_code) {
  absl::call_once(once, OnceInit);
  std::unique_ptr<OrcJit> jit =
      absl::WrapUnique(new OrcJit(opt_level, emit_object_code));
  XLS_RETURN_IF_ERROR(jit->Init());
  return std::move(jit);
}

/* static */ absl::StatusOr<llvm::DataLayout> OrcJit::CreateDataLayout() {
  absl::call_once(once, OnceInit);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<llvm::TargetMachine> target_machine,
                       CreateTargetMachine());
  return target_machine->createDataLayout();
}

/* static */ absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
OrcJit::CreateTargetMachine() {
  auto error_or_target_builder =
      llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to detect host: ",
                     llvm::toString(error_or_target_builder.takeError())));
  }

  error_or_target_builder->setRelocationModel(llvm::Reloc::Model::PIC_);
  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }

  return std::move(error_or_target_machine.get());
}

absl::Status OrcJit::Init() {
  XLS_ASSIGN_OR_RETURN(target_machine_, CreateTargetMachine());
  data_layout_ = target_machine_->createDataLayout();

  execution_session_.runSessionLocked([this]() {
    dylib_.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix())));
  });

  auto compiler = std::make_unique<llvm::orc::SimpleCompiler>(*target_machine_);
  compile_layer_ = std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session_, object_layer_, std::move(compiler));

  if (emit_object_code_) {
    object_code_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
        execution_session_, *compile_layer_,
        [this](llvm::orc::ThreadSafeModule module,
               const llvm::orc::MaterializationResponsibility& mr) {
          llvm::SmallVector<char, 0> stream_buffer;
          llvm::raw_svector_ostream ostream(stream_buffer);
          llvm::legacy::PassManager mpm;
          if (target_machine_->addPassesToEmitFile(mpm, ostream, nullptr,
                                                   llvm::CGFT_ObjectFile)) {
            XLS_VLOG(3) << "Could not create ASM generation pass!";
          }
          mpm.run(*module.getModuleUnlocked());

          object_code_ =
              std::vector<uint8_t>(stream_buffer.begin(), stream_buffer.end());
          return module;
        });
  }

  llvm::orc::IRLayer* parent_layer =
      emit_object_code_
          ? static_cast<llvm::orc::IRLayer*>(object_code_layer_.get())
          : static_cast<llvm::orc::IRLayer*>(compile_layer_.get());
  transform_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
      execution_session_, *parent_layer,
      [this](llvm::orc::ThreadSafeModule module,
             const llvm::orc::MaterializationResponsibility& responsibility) {
        return Optimizer(std::move(module), responsibility);
      });

  return absl::OkStatus();
}

std::unique_ptr<llvm::Module> OrcJit::NewModule(std::string_view name) {
  llvm::LLVMContext* bare_context = context_.getContext();
  auto module = std::make_unique<llvm::Module>(name, *bare_context);
  module->setDataLayout(data_layout_);
  return module;
}

namespace {

// Check that every operand of every instruction is in the same function as the
// instruction itself. This is a very common error which is not checked by the
// LLVM verifier(!).
absl::Status VerifyModule(const llvm::Module& module) {
  XLS_VLOG(4) << DumpLlvmModuleToString(&module);
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

}  // namespace

absl::Status OrcJit::CompileModule(std::unique_ptr<llvm::Module>&& module) {
  XLS_RETURN_IF_ERROR(VerifyModule(*module));
  llvm::Error error = transform_layer_->add(
      dylib_, llvm::orc::ThreadSafeModule(std::move(module), context_));
  if (error) {
    return absl::UnknownError(absl::StrFormat(
        "Error compiling converted IR: %s", llvm::toString(std::move(error))));
  }
  return absl::OkStatus();
}

absl::StatusOr<llvm::JITTargetAddress> OrcJit::LoadSymbol(
    std::string_view function_name) {
  llvm::Expected<llvm::JITEvaluatedSymbol> symbol =
      execution_session_.lookup(&dylib_, function_name);
  if (!symbol) {
    return absl::InternalError(
        absl::StrFormat("Could not find start symbol \"%s\": %s", function_name,
                        llvm::toString(symbol.takeError())));
  }
  return symbol->getAddress();
}

std::string DumpLlvmModuleToString(const llvm::Module& module) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  module.print(ostream, nullptr);
  ostream.flush();
  return buffer;
}

}  // namespace xls
