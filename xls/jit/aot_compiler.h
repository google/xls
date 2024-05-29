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

#ifndef XLS_JIT_AOT_COMPILER_H_
#define XLS_JIT_AOT_COMPILER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/Target/TargetMachine.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

namespace xls {

class AotCompiler final : public LlvmCompiler {
 public:
  static absl::StatusOr<std::unique_ptr<AotCompiler>> Create(
      bool include_msan, int64_t opt_level = LlvmCompiler::kDefaultOptLevel,
      JitObserver* observer = nullptr);

  absl::StatusOr<AotCompiler*> AsAotCompiler() override { return this; }

  // Compiles the given LLVM module into object code.
  absl::Status CompileModule(std::unique_ptr<llvm::Module>&& module) override;

  // Return the underlying LLVM context.
  llvm::LLVMContext* GetContext() override { return context_.get(); }

  absl::StatusOr<std::unique_ptr<llvm::TargetMachine>> CreateTargetMachine()
      override;

  absl::StatusOr<std::vector<uint8_t>> GetObjectCode() && {
    if (!object_code_) {
      return absl::InternalError("Object code not yet materialized");
    }
    return std::move(*object_code_);
  }

  absl::StatusOr<std::vector<uint8_t>> GetObjectCode() & {
    if (!object_code_) {
      return absl::InternalError("Object code not yet materialized");
    }
    return *object_code_;
  }

 protected:
  absl::Status InitInternal() override;

 private:
  AotCompiler(int64_t opt_level, bool include_msan, JitObserver* observer)
      : LlvmCompiler(opt_level, include_msan), jit_observer_(observer) {}

  std::unique_ptr<llvm::LLVMContext> context_ =
      std::make_unique<llvm::LLVMContext>();

  std::optional<std::vector<uint8_t>> object_code_;

  JitObserver* jit_observer_;
};

}  // namespace xls

#endif  // XLS_JIT_AOT_COMPILER_H_
