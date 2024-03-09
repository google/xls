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

#ifndef XLS_DSLX_RUN_COMPARATOR_H_
#define XLS_DSLX_RUN_COMPARATOR_H_

#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/test_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"

namespace xls::dslx {

// Indicates whether the RunComparator should be comparing to the JIT's IR
// execution or the IR interpreter's.
enum class CompareMode : uint8_t {
  kJit,
  kInterpreter,
};

// Helper object that is used as a post-execution hook in the interpreter,
// comparing interpreter results to results computed by the JIT to check that
// they're equivalent.
//
// Implementation note: slightly simpler to keep in object form so we can
// inspect cache state more easily than closing over it, e.g. for testing.
class RunComparator : public AbstractRunComparator {
 public:
  explicit RunComparator(CompareMode mode) : mode_(mode) {}

  absl::Status RunComparison(Package* ir_package, bool requires_implicit_token,
                             const Function* f,
                             absl::Span<InterpValue const> args,
                             const ParametricEnv* parametric_env,
                             const InterpValue& got) override;

  absl::StatusOr<InterpreterResult<xls::Value>> RunIrFunction(
      std::string_view ir_name, xls::Function* ir_function,
      absl::Span<const xls::Value> ir_args) override;

  // Returns the cached or newly-compiled jit function for ir_name.  ir_name has
  // already been mangled (see MangleDslxName) so it should be unique in the
  // program and is used as the cache key.
  //
  // Note: There is no locking in jit compilation or on the jit function cache
  // so this function is *not* thread-safe.
  absl::StatusOr<FunctionJit*> GetOrCompileJitFunction(
      std::string_view ir_name, xls::Function* ir_function);

 private:
  XLS_FRIEND_TEST(RunRoutinesTest, TestInvokedFunctionDoesJit);
  XLS_FRIEND_TEST(RunRoutinesTest, QuickcheckInvokedFunctionDoesJit);
  XLS_FRIEND_TEST(RunRoutinesTest, NoSeedStillQuickChecks);

  absl::flat_hash_map<std::string, std::unique_ptr<FunctionJit>> jit_cache_;
  CompareMode mode_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_RUN_COMPARATOR_H_
