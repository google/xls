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

#include "xls/jit/aot_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Analysis/CGSCCPassManager.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/include/llvm/IR/Attributes.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/GlobalVariable.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/LegacyPassManager.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/Passes/PassBuilder.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/CodeGen.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "llvm/include/llvm/TargetParser/AArch64TargetParser.h"
#include "llvm/include/llvm/TargetParser/Host.h"
#include "llvm/include/llvm/TargetParser/SubtargetFeature.h"
#include "llvm/include/llvm/TargetParser/Triple.h"
#include "llvm/include/llvm/TargetParser/X86TargetParser.h"
#include "llvm/include/llvm/Transforms/Utils/Cloning.h"
#include "xls/common/status/status_macros.h"
#include "xls/jit/jit_emulated_tls.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

ABSL_FLAG(std::string, aot_target, "native",
          "CodeGen target. Supported values are \"native\", \"aarch64\", and "
          "\"x86_64\"");

namespace xls {

// static
absl::StatusOr<std::unique_ptr<AotCompiler>> AotCompiler::Create(
    bool include_msan, int64_t opt_level, JitObserver* observer) {
  LlvmCompiler::InitializeLlvm();
  auto compiler = std::unique_ptr<AotCompiler>(
      new AotCompiler(opt_level, include_msan, observer));
  XLS_RETURN_IF_ERROR(compiler->Init());
  return std::move(compiler);
}

absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
AotCompiler::CreateTargetMachine() {
  const std::string aot_target = absl::GetFlag(FLAGS_aot_target);
  if (aot_target != "native" && aot_target != "aarch64" &&
      aot_target != "x86_64") {
    return absl::InternalError(
        absl::StrCat("Unsupported value for aot_target: ", aot_target));
  }
  llvm::Triple target_triple;
  if (aot_target != "native") {
    target_triple = llvm::Triple(llvm::sys::getProcessTriple());
    target_triple.setArchName(aot_target);
  }
  auto error_or_target_builder =
      aot_target == "native"
          ? llvm::orc::JITTargetMachineBuilder::detectHost()
          : llvm::orc::JITTargetMachineBuilder(target_triple);
  if (!error_or_target_builder) {
    return absl::InternalError(
        absl::StrCat("Unable to construct machine builder: ",
                     llvm::toString(error_or_target_builder.takeError())));
  };

  error_or_target_builder->setRelocationModel(llvm::Reloc::Model::PIC_);
  // In ahead-of-time compilation we're compiling on machines we are not
  // immediately about to run on, where runtime machines may have
  // heterogeneous specifications vs the compilation machine. We assume a
  // baseline level of compatibility for all machines XLS compilations might
  // run on.
  //
  // TODO(allight): Ideally we should allow the user to select what CPU we
  // want instead of just hard-coding a haswell.
  switch (error_or_target_builder->getTargetTriple().getArch()) {
    case llvm::Triple::x86_64: {
      const std::string kBaselineCpu = "haswell";
      error_or_target_builder->setCPU(kBaselineCpu);
      // Clear out the existing features.
      error_or_target_builder->getFeatures() = llvm::SubtargetFeatures();
      // Add in features available on our "baseline" CPU.
      llvm::SmallVector<llvm::StringRef, 32> target_cpu_features;
      llvm::X86::getFeaturesForCPU(kBaselineCpu, target_cpu_features,
                                   /*NeedPlus=*/true);
      std::vector<std::string> features(target_cpu_features.begin(),
                                        target_cpu_features.end());
      error_or_target_builder->addFeatures(features);
      break;
    }
    case llvm::Triple::aarch64: {
      error_or_target_builder->getFeatures() = llvm::SubtargetFeatures();
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Compiling on unrecognized host architecture for AOT "
          "compilation: " +
          std::string{
              error_or_target_builder->getTargetTriple().getArchName()});
  }

  // NB We don't need to do anything special to force emutls because we get the
  // same base target as the orc jit which doesn't support it.
  auto error_or_target_machine = error_or_target_builder->createTargetMachine();
  if (!error_or_target_machine) {
    return absl::InternalError(
        absl::StrCat("Unable to create target machine: ",
                     llvm::toString(error_or_target_machine.takeError())));
  }

  return std::move(error_or_target_machine.get());
}

absl::Status AotCompiler::InitInternal() {
  context_ = std::make_unique<llvm::LLVMContext>();
  return absl::OkStatus();
}
namespace {

absl::Status AddWeakEmuTls(llvm::Module& module, llvm::LLVMContext* context) {
  // Add weak symbol definitions for:
  //   __emutls_v.__msan_retval_tls == kRevalTlsEntry
  //   __emutls_v.__msan_param_tls == kParamTlsEntry
  //   __emutls_get_address == call to kExportedEmulatedMsanEntrypointName
  llvm::Type* void_ptr_ty = llvm::PointerType::get(*context, 0);
  llvm::FunctionType* emutls_get_addr_type =
      llvm::FunctionType::get(void_ptr_ty, void_ptr_ty, /*isVarArg=*/false);
  llvm::Type* void_ptr_ptr_ty = llvm::PointerType::get(void_ptr_ty, 0);
  // llvm::Type* void_ptr_ptr_ty = llvm::PointerType::get(void_ptr_ty, 0);
  // Make sure its weak linkage so other emutls can override it.
  // NB module takes ownership of the pointer.
  module.insertGlobalVariable(new llvm::GlobalVariable(
      void_ptr_ty, /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      llvm::Constant::getIntegerValue(
          void_ptr_ty,
          llvm::APInt(module.getDataLayout().getPointerSize(), kParamTlsEntry)),
      "__emutls_v.__msan_param_tls"));
  module.insertGlobalVariable(new llvm::GlobalVariable(
      void_ptr_ty, /*isConstant=*/true, llvm::GlobalValue::InternalLinkage,
      llvm::Constant::getIntegerValue(
          void_ptr_ty, llvm::APInt(module.getDataLayout().getPointerSize(),
                                   kRetvalTlsEntry)),
      "__emutls_v.__msan_retval_tls"));
  llvm::FunctionCallee tls_impl = module.getOrInsertFunction(
      kExportedEmulatedMsanEntrypointName, emutls_get_addr_type);

  llvm::Function* fn = llvm::cast<llvm::Function>(
      module
          .getOrInsertFunction(
              "__emutls_get_address",
              llvm::FunctionType::get(void_ptr_ty, void_ptr_ptr_ty,
                                      /*isVarArg=*/false))
          .getCallee());
  fn->setLinkage(llvm::GlobalValue::InternalLinkage);
  fn->setAttributes(llvm::AttributeList().addFnAttribute(
      *context, llvm::StringRef("no_sanitize_memory")));
  auto basic_block =
      llvm::BasicBlock::Create(*context, "entry", fn, /*InsertBefore=*/nullptr);
  llvm::IRBuilder<> builder(basic_block);
  // NB It calls with the *pointer-to* __emutls-key so we need to dereference
  // it.
  auto call = builder.CreateCall(
      tls_impl, {builder.CreateLoad(void_ptr_ty, fn->getArg(0))});
  builder.CreateRet(call);
  return absl::OkStatus();
}

}  // namespace
absl::Status AotCompiler::CompileModule(
    std::unique_ptr<llvm::Module>&& module) {
  JitObserverRequests notification;
  if (jit_observer_ != nullptr) {
    notification = jit_observer_->GetNotificationOptions();
  }
  if (notification.unoptimized_module) {
    jit_observer_->UnoptimizedModule(module.get());
  }
  auto err = PerformStandardOptimization(module.get());
  if (err) {
    std::string mem;
    llvm::raw_string_ostream oss(mem);
    oss << err;
    return absl::InternalError(oss.str());
  }
  // To avoid the msan pass inserting stores to msan functions we only add it
  // after all the msan stuff is done.
  if (include_msan()) {
    XLS_RETURN_IF_ERROR(AddWeakEmuTls(*module, GetContext()));
  }
  if (notification.optimized_module) {
    jit_observer_->OptimizedModule(module.get());
  }
  if (notification.assembly_code_str) {
    llvm::SmallVector<char, 0> asm_stream_buffer;
    llvm::raw_svector_ostream asm_ostream(asm_stream_buffer);
    llvm::legacy::PassManager asm_mpm;
    if (target_machine_->addPassesToEmitFile(
            asm_mpm, asm_ostream, nullptr,
            llvm::CodeGenFileType::AssemblyFile)) {
      return absl::InternalError(
          "Unable to add passes for assembly code dumping");
    }
    std::unique_ptr<llvm::Module> clone = llvm::CloneModule(*module);
    asm_mpm.run(*clone);
    std::string asm_code(asm_stream_buffer.begin(), asm_stream_buffer.end());
    jit_observer_->AssemblyCodeString(module.get(), asm_code);
  }
  llvm::SmallVector<char, 0> stream_buffer;
  llvm::raw_svector_ostream ostream(stream_buffer);
  llvm::legacy::PassManager mpm;
  if (target_machine_->addPassesToEmitFile(mpm, ostream, nullptr,
                                           llvm::CodeGenFileType::ObjectFile)) {
    return absl::InternalError("Unable to add passes for object code dumping");
  }
  mpm.run(*module);
  object_code_ =
      std::vector<uint8_t>(stream_buffer.begin(), stream_buffer.end());

  return absl::OkStatus();
}

}  // namespace xls
