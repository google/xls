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

#ifndef XLS_JIT_LLVM_COMPILER_H_
#define XLS_JIT_LLVM_COMPILER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/Support/Error.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "llvm/include/llvm/Target/TargetMachine.h"

// LLVM is so huge that it noticeably slows down code completion. Don't have
// includes in the header to avoid this issue as much as possible.
namespace llvm {
class LLVMContext;
class Module;
}  // namespace llvm
namespace xls {

class AotCompiler;
class OrcJit;

class LlvmCompiler {
 public:
  static constexpr int64_t kDefaultOptLevel = 3;
  static void InitializeLlvm();

  virtual ~LlvmCompiler() = default;

  virtual absl::StatusOr<OrcJit*> AsOrcJit() {
    return absl::InternalError("Not an orc jit");
  }
  virtual absl::StatusOr<AotCompiler*> AsAotCompiler() {
    return absl::InternalError("Not an aot compiler");
  }

  bool IsAotCompiler() { return AsAotCompiler().ok(); }
  bool IsOrcJit() { return AsOrcJit().ok(); }

  // Gets the context pointer.
  virtual llvm::LLVMContext* GetContext() = 0;

  // Creates and returns a new LLVM module of the given name.
  //
  // May only be called once.
  //
  // TODO(allight): We should rethink the architecture of the AOT/JIT compiler
  // at some point.
  virtual std::unique_ptr<llvm::Module> NewModule(std::string_view name);

  std::string target_triple() const;

  // Compiles the given LLVM module in a manner appropriate to the compiler.
  //
  // After calling this either the entry points are available from the OrcJit or
  // the object code is available from the AotCompiler.
  //
  // This may only be called once and must be called with the module created by
  // NewModule.
  virtual absl::Status CompileModule(
      std::unique_ptr<llvm::Module>&& module) = 0;

  absl::StatusOr<llvm::DataLayout> CreateDataLayout();

  virtual absl::StatusOr<std::unique_ptr<llvm::TargetMachine>>
  CreateTargetMachine() = 0;

  int64_t opt_level() const { return opt_level_; }
  bool include_msan() const { return include_msan_; }

 protected:
  absl::Status Init();

  virtual absl::Status InitInternal() = 0;

  absl::Status VerifyModule(const llvm::Module& module);

  llvm::Error PerformStandardOptimization(llvm::Module* module);

  LlvmCompiler(int64_t opt_level, bool include_msan)
      : data_layout_(""), opt_level_(opt_level), include_msan_(include_msan) {}

  // Constructor to manually setup the compiler without Init.
  LlvmCompiler(std::unique_ptr<llvm::TargetMachine> target,
               llvm::DataLayout&& layout, int64_t opt_level, bool include_msan)
      : target_machine_(std::move(target)),
        data_layout_(layout),
        opt_level_(opt_level),
        include_msan_(include_msan) {}

  // Setup by Init
  std::unique_ptr<llvm::TargetMachine> target_machine_;
  // Setup by Init
  llvm::DataLayout data_layout_;

  int64_t opt_level_;
  // If the jitted code should include msan calls. Defaults to whatever 'this'
  // process is doing and should only be overridden for AOT generators.
  const bool include_msan_;

  bool module_created_ = false;
};

// Calls the dump method on the given LLVM object and returns the string.
template <typename T>
std::string DumpLlvmObjectToString(const T& llvm_object) {
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  llvm_object.print(ostream);
  ostream.flush();
  return buffer;
}

// Calls the dump method on the given LLVM module object and returns the string.
// DumpLlvmObjectToString cannot be used because modules are dumped slightly
// differently.
std::string DumpLlvmModuleToString(const llvm::Module* module);

}  // namespace xls

#endif  // XLS_JIT_LLVM_COMPILER_H_
