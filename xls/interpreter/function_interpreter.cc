// Copyright 2021 The XLS Authors
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

#include "xls/interpreter/function_interpreter.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/events.h"
#include "xls/ir/keyword_args.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

// An interpreter for XLS functions.
class FunctionInterpreter final : public IrInterpreter {
 public:
  FunctionInterpreter(absl::Span<const Value> args,
                      std::optional<EvaluationObserver*> observer)
      : IrInterpreter(observer), args_(args.begin(), args.end()) {}

  absl::Status HandleParam(Param* param) final {
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         param->function_base()->GetParamIndex(param));
    if (index >= args_.size()) {
      return absl::InternalError(absl::StrFormat(
          "Parameter %s at index %d does not exist in args (of length %d)",
          param->ToString(), index, args_.size()));
    }
    return SetValueResult(param, args_[index]);
  }

 private:
  // The arguments to the Function being evaluated indexed by parameter name.
  std::vector<Value> args_;
};

}  // namespace

absl::StatusOr<InterpreterResult<Value>> InterpretFunction(
    Function* function, absl::Span<const Value> args,
    std::optional<EvaluationObserver*> observer) {
  VLOG(3) << "Interpreting function " << function->name();
  if (args.size() != function->params().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function `%s` (type: `%s`) wants %d arguments, got %d.",
        function->name(), function->GetType()->ToString(),
        function->params().size(), args.size()));
  }
  for (int64_t argno = 0; argno < args.size(); ++argno) {
    Param* param = function->param(argno);
    const Value& value = args[argno];
    Type* param_type = param->GetType();
    Type* value_type = function->package()->GetTypeForValue(value);
    if (value_type != param_type) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Got argument %s for parameter %d which is not of type %s",
          value.ToString(), argno, param_type->ToString()));
    }
  }
  FunctionInterpreter visitor(args, observer);
  XLS_RETURN_IF_ERROR(function->Accept(&visitor));
  Value result = visitor.ResolveAsValue(function->return_value());
  VLOG(2) << "Result = " << result;
  InterpreterEvents events = visitor.GetInterpreterEvents();
  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

/* static */ absl::StatusOr<InterpreterResult<Value>> InterpretFunctionKwargs(
    Function* function, const absl::flat_hash_map<std::string, Value>& args,
    std::optional<EvaluationObserver*> observer) {
  VLOG(2) << "Interpreting function " << function->name() << " with arguments:";
  XLS_ASSIGN_OR_RETURN(std::vector<Value> positional_args,
                       KeywordArgsToPositional(*function, args));
  return InterpretFunction(function, positional_args, observer);
}

}  // namespace xls
