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

#include "xls/dslx/run_routines/run_comparator.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"

namespace xls::dslx {

absl::StatusOr<FunctionJit*> RunComparator::GetOrCompileJitFunction(
    std::string_view ir_name, xls::Function* ir_function) {
  auto it = jit_cache_.find(ir_name);
  if (it != jit_cache_.end()) {
    return it->second.get();
  }
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<FunctionJit> jit,
                       FunctionJit::Create(ir_function));
  FunctionJit* result = jit.get();
  jit_cache_[ir_name] = std::move(jit);
  return result;
}

absl::Status RunComparator::RunComparison(Package* ir_package,
                                          bool requires_implicit_token,
                                          const dslx::Function* f,
                                          absl::Span<InterpValue const> args,
                                          const ParametricEnv* parametric_env,
                                          const InterpValue& got) {
  XLS_RET_CHECK(ir_package != nullptr);

  XLS_ASSIGN_OR_RETURN(
      std::string ir_name,
      MangleDslxName(f->owner()->name(), f->identifier(),
                     requires_implicit_token ? CallingConvention::kImplicitToken
                                             : CallingConvention::kTypical,
                     f->GetFreeParametricKeySet(), parametric_env));

  auto get_result = ir_package->TryGetFunction(ir_name);

  // The (converted) IR package does not include specializations of parametric
  // functions that are only called from test code, so not finding the function
  // may be benign.
  //
  // TODO(amfv): 2021-03-18 Extend IR conversion to include those functions.
  if (!get_result.has_value()) {
    XLS_LOG(WARNING) << "Could not find " << ir_name
                     << " function for JIT comparison";
    return absl::OkStatus();
  }

  xls::Function* ir_function = *get_result;

  XLS_ASSIGN_OR_RETURN(std::vector<Value> ir_args,
                       InterpValue::ConvertValuesToIr(args));

  // We need to know if the function-that-we're-doing-a-comparison-for needs an
  // implicit token.
  if (requires_implicit_token) {
    ir_args.insert(ir_args.begin(), Value::Bool(true));
    ir_args.insert(ir_args.begin(), Value::Token());
  }

  const char* mode_str = nullptr;
  Value ir_result;
  switch (mode_) {
    case CompareMode::kJit: {  // Compare to IR JIT.
      // TODO(https://github.com/google/xls/issues/506): Also compare events
      // once the DSLX interpreter supports them (and the JIT supports traces).
      XLS_ASSIGN_OR_RETURN(FunctionJit * jit,
                           GetOrCompileJitFunction(ir_name, ir_function));
      XLS_ASSIGN_OR_RETURN(ir_result, DropInterpreterEvents(jit->Run(ir_args)));
      mode_str = "JIT";
      break;
    }
    case CompareMode::kInterpreter: {  // Compare to IR interpreter.
      XLS_ASSIGN_OR_RETURN(ir_result, DropInterpreterEvents(InterpretFunction(
                                          ir_function, ir_args)));
      mode_str = "interpreter";
      break;
    }
  }

  if (requires_implicit_token) {
    // Slice off the first value.
    XLS_RET_CHECK(ir_result.element(0).IsToken());
    XLS_RET_CHECK_EQ(ir_result.size(), 2);
    Value real_ir_result = ir_result.element(1);
    ir_result = std::move(real_ir_result);
  }

  // Convert the interpreter value to an IR value so we can compare it.
  //
  // Note this conversion is lossy, but that's ok because we're just looking for
  // mismatches.
  XLS_ASSIGN_OR_RETURN(Value interp_ir_value, got.ConvertToIr());

  if (interp_ir_value != ir_result) {
    return absl::InternalError(
        absl::StrFormat("IR %s produced a different value from the DSL "
                        "interpreter for %s; IR %s: %s "
                        "DSL interpreter: %s",
                        mode_str, ir_function->name(), mode_str,
                        ir_result.ToString(), interp_ir_value.ToString()));
  }
  return absl::OkStatus();
}

absl::StatusOr<InterpreterResult<xls::Value>> RunComparator::RunIrFunction(
    std::string_view ir_name, xls::Function* ir_function,
    absl::Span<const xls::Value> ir_args) {
  XLS_ASSIGN_OR_RETURN(FunctionJit * jit,
                       GetOrCompileJitFunction(ir_name, ir_function));
  return jit->Run(ir_args);
}

}  // namespace xls::dslx
