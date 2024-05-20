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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Analysis/CGSCCPassManager.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/include/llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/Instruction.h"
#include "llvm/include/llvm/IR/LegacyPassManager.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/include/llvm/Passes/PassBuilder.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Transforms/Instrumentation/MemorySanitizer.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/jit/jit_emulated_tls.h"  // NOLINT: Used with MSAN
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

namespace xls {

OrcJit::OrcJit(int64_t opt_level, bool include_msan)
    : LlvmCompiler(opt_level, include_msan),
      context_(std::make_unique<llvm::LLVMContext>()),
      execution_session_(
          std::make_unique<llvm::orc::UnsupportedExecutorProcessControl>()),
      object_layer_(
          execution_session_,
          []() { return std::make_unique<llvm::SectionMemoryManager>(); }),
      dylib_(execution_session_.createBareJITDylib("main")) {}

OrcJit::~OrcJit() {
  if (auto err = execution_session_.endSession()) {
    execution_session_.reportError(std::move(err));
  }
}

llvm::Expected<llvm::orc::ThreadSafeModule> OrcJit::Optimizer(
    llvm::orc::ThreadSafeModule module,
    const llvm::orc::MaterializationResponsibility& responsibility) {
  llvm::Module* bare_module = module.getModuleUnlocked();

  VLOG(2) << "Unoptimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));
  if (jit_observer_ != nullptr &&
      jit_observer_->GetNotificationOptions().unoptimized_module) {
    jit_observer_->UnoptimizedModule(bare_module);
  }

  auto error = PerformStandardOptimization(bare_module);
  if (error) {
    return llvm::Expected<llvm::orc::ThreadSafeModule>(std::move(error));
  }

  VLOG(2) << "Optimized module IR:";
  XLS_VLOG_LINES(2, DumpLlvmModuleToString(bare_module));
  if (jit_observer_ != nullptr &&
      jit_observer_->GetNotificationOptions().optimized_module) {
    jit_observer_->OptimizedModule(bare_module);
  }

  bool observe_asm_code =
      (jit_observer_ != nullptr &&
       jit_observer_->GetNotificationOptions().assembly_code_str);
  if (VLOG_IS_ON(3) || observe_asm_code) {
    // The ostream and its buffer must be declared before the
    // module_pass_manager because the destrutor of the pass manager calls flush
    // on the ostream so these must be destructed *after* the pass manager. C++
    // guarantees that the destructors are called in reverse order the objects
    // are declared.
    llvm::SmallVector<char, 0> stream_buffer;
    llvm::raw_svector_ostream ostream(stream_buffer);
    llvm::legacy::PassManager mpm;
    if (target_machine_->addPassesToEmitFile(
            mpm, ostream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
      VLOG(3) << "Could not create ASM generation pass!";
    }
    mpm.run(*bare_module);
    VLOG(3) << "Generated ASM:";
    std::string asm_code(stream_buffer.begin(), stream_buffer.end());
    XLS_VLOG_LINES(3, asm_code);
    if (observe_asm_code) {
      jit_observer_->AssemblyCodeString(bare_module, asm_code);
    }
  }

  return module;
}

absl::StatusOr<std::unique_ptr<OrcJit>> OrcJit::Create(int64_t opt_level,
                                                       JitObserver* observer) {
  LlvmCompiler::InitializeLlvm();
#ifdef ABSL_HAVE_MEMORY_SANITIZER
  constexpr bool kHasMsan = true;
#else
  constexpr bool kHasMsan = false;
#endif
  std::unique_ptr<OrcJit> jit =
      absl::WrapUnique(new OrcJit(opt_level, kHasMsan));
  jit->SetJitObserver(observer);
  XLS_RETURN_IF_ERROR(jit->Init());
  return std::move(jit);
}

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
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

namespace {
class MsanHostEmuTls : public llvm::orc::DefinitionGenerator {
 public:
  llvm::Error tryToGenerate(llvm::orc::LookupState&, llvm::orc::LookupKind,
                            llvm::orc::JITDylib& dylib,
                            llvm::orc::JITDylibLookupFlags,
                            const llvm::orc::SymbolLookupSet& targets) final {
#ifndef ABSL_HAVE_MEMORY_SANITIZER
    // No actual msan so don't do anything.
    return llvm::Error::success();
#else
    llvm::orc::SymbolMap result;
    for (auto& kv : targets) {
      auto name = (*kv.first).str();
      if (name == "__emutls_get_address") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(&GetEmulatedMsanTLSAddr),
             llvm::JITSymbolFlags::Exported});
      }
      if (name == "__emutls_v.__msan_param_tls") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(
                 absl::bit_cast<void*>(kParamTlsEntry)),
             llvm::JITSymbolFlags::Exported});
      }
      if (name == "__emutls_v.__msan_retval_tls") {
        result[kv.first] = llvm::orc::ExecutorSymbolDef(
            {llvm::orc::ExecutorAddr::fromPtr(
                 absl::bit_cast<void*>(kRetvalTlsEntry)),
             llvm::JITSymbolFlags::Exported});
      }
    }
    if (result.empty()) {
      return llvm::Error::success();
    }
    return dylib.define(llvm::orc::absoluteSymbols(std::move(result)));
#endif
  }
};

}  // namespace

absl::Status OrcJit::InitInternal() {
  execution_session_.runSessionLocked([this]() {
    dylib_.addGenerator(std::make_unique<MsanHostEmuTls>());
    dylib_.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            data_layout_.getGlobalPrefix())));
  });

  auto compiler = std::make_unique<llvm::orc::SimpleCompiler>(*target_machine_);
  compile_layer_ = std::make_unique<llvm::orc::IRCompileLayer>(
      execution_session_, object_layer_, std::move(compiler));

  llvm::orc::IRLayer* parent_layer =
      static_cast<llvm::orc::IRLayer*>(compile_layer_.get());
  transform_layer_ = std::make_unique<llvm::orc::IRTransformLayer>(
      execution_session_, *parent_layer,
      [this](llvm::orc::ThreadSafeModule module,
             const llvm::orc::MaterializationResponsibility& responsibility) {
        return Optimizer(std::move(module), responsibility);
      });

  return absl::OkStatus();
}

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

absl::StatusOr<llvm::orc::ExecutorAddr> OrcJit::LoadSymbol(
    std::string_view function_name) {
#ifdef __APPLE__
  // On Apple systems, symbols are still prepended with an underscore.
  std::string function_name_with_underscore = absl::StrCat("_", function_name);
  function_name = function_name_with_underscore;
#endif /* __APPLE__ */
  llvm::Expected<llvm::orc::ExecutorSymbolDef> symbol =
      execution_session_.lookup(&dylib_, function_name);
  if (!symbol) {
    return absl::InternalError(
        absl::StrFormat("Could not find start symbol \"%s\": %s", function_name,
                        llvm::toString(symbol.takeError())));
  }
  return symbol->getAddress();
}

}  // namespace xls
