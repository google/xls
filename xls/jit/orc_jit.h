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

#ifndef XLS_JIT_ORC_JIT_H_
#define XLS_JIT_ORC_JIT_H_

#include <cstdint>
#include <memory>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/include/llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

namespace xls {
// A wrapper around ORC JIT which hides some of the internals of the LLVM
// interface.
class OrcJit : public LlvmCompiler {
 public:
  static constexpr int64_t kDefaultOptLevel = 2;

  ~OrcJit() override;

  absl::StatusOr<OrcJit*> AsOrcJit() override { return this; }

  // Create an LLVM orc jit. This can be used by the AOT generator to manually
  // control whether asan calls should be included. Users other than the AOT
  // compiler should use the 3-argument version above. Passing nullopt to
  // emit_msan directs the jit to use MSAN if the running binary is MSAN and
  // vice-versa.
  static absl::StatusOr<std::unique_ptr<OrcJit>> Create(
      int64_t opt_level = kDefaultOptLevel,
      bool include_observer_callbacks = false,
      JitObserver* jit_observer = nullptr);

  void SetJitObserver(JitObserver* o) { jit_observer_ = o; }

  JitObserver* jit_observer() const { return jit_observer_; }

  // Compiles the given LLVM module into the JIT's execution session.
  absl::Status CompileModule(std::unique_ptr<llvm::Module>&& module) override;

  // Returns the address of the given JIT'ed function.
  absl::StatusOr<llvm::orc::ExecutorAddr> LoadSymbol(
      std::string_view function_name);

  // Return the underlying LLVM context.
  // TODO: b/430302945 - This is not thread safe!
  llvm::LLVMContext* GetContext() override {
    return context_.withContextDo([](llvm::LLVMContext* ctxt) { return ctxt; });
  }

  absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> CreateTargetMachine()
      override;

 protected:
  absl::Status InitInternal() override;

 private:
  OrcJit(int64_t opt_level, bool include_msan, bool include_observer_callbacks);

  // Method which optimizes the given module. Used within the JIT to form an IR
  // transform layer.
  llvm::Expected<llvm::orc::ThreadSafeModule> Optimizer(
      llvm::orc::ThreadSafeModule module,
      const llvm::orc::MaterializationResponsibility& responsibility);

  llvm::orc::ThreadSafeContext context_;
  llvm::orc::ExecutionSession execution_session_;
  llvm::orc::RTDyldObjectLinkingLayer object_layer_;
  llvm::orc::JITDylib& dylib_;

  std::unique_ptr<llvm::orc::IRCompileLayer> compile_layer_;
  std::unique_ptr<llvm::orc::IRTransformLayer> transform_layer_;

  JitObserver* jit_observer_ = nullptr;
};

}  // namespace xls

#endif  // XLS_JIT_ORC_JIT_H_
