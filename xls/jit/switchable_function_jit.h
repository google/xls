// Copyright 2023 The XLS Authors
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

#ifndef XLS_JIT_SWITCHABLE_FUNCTION_JIT_H_
#define XLS_JIT_SWITCHABLE_FUNCTION_JIT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/observer.h"

namespace xls {

enum class ExecutionType {
  kDefault,
  kJit,
  kInterpreter,
};

// A wrapper for the jit structures that can be turned off at build time if
// issues arise with the LLVM jit. This class tries to match FunctionJit
// interface as much as possible.
// TODO(google/xls#1151): 2023-10-13 Implement the rest of the FunctionJit API.
class SwitchableFunctionJit {
 public:
  // Returns an object containing a host-compiled version of the specified XLS
  // function.
  static absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>> CreateJit(
      Function* xls_function, int64_t opt_level = 3,
      JitObserver* observer = nullptr);
  static absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>>
  CreateInterpreter(Function* xls_function);
  static absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>> Create(
      Function* xls_function, ExecutionType execution = ExecutionType::kDefault,
      int64_t opt_level = 3, JitObserver* observer = nullptr);

  // Executes the compiled function with the specified arguments.
  absl::StatusOr<InterpreterResult<Value>> Run(absl::Span<const Value> args);

  // As above, buth with arguments as key-value pairs.
  absl::StatusOr<InterpreterResult<Value>> Run(
      const absl::flat_hash_map<std::string, Value>& kwargs);

  // TODO(google/xls#1151): 2023-10-13 Match more of the function-jit api.

  // Returns the function that the JIT executes.
  Function* function() { return xls_function_; }

  std::optional<FunctionJit*> function_jit() {
    if (use_jit_) {
      return function_jit_.get();
    }
    return std::nullopt;
  }

 private:
  explicit SwitchableFunctionJit(Function* xls_function, bool use_jit,
                                 std::unique_ptr<FunctionJit>&& jit)
      : xls_function_(xls_function),
        use_jit_(use_jit),
        function_jit_(std::move(jit)) {}

  Function* xls_function_;
  bool use_jit_;
  std::unique_ptr<FunctionJit> function_jit_;
};
}  // namespace xls

#endif  // XLS_JIT_SWITCHABLE_FUNCTION_JIT_H_
