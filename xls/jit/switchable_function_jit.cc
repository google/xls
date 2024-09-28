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

#include "xls/jit/switchable_function_jit.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/observer.h"

namespace xls {

#if EMERGENCY_FORCE_INTERPRETER == 1
#pragma clang diagnostic push
#pragma clang diagnostic warning "-W#warnings"
#warning XLS JIT is currently disabled via EMERGENCY_FORCE_INTERPRETER and by \
default won't be used.
constexpr ExecutionType kRealDefaultExecutionType = ExecutionType::kInterpreter;
#pragma clang diagnostic pop
#else
constexpr ExecutionType kRealDefaultExecutionType = ExecutionType::kJit;
#endif

absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>>
SwitchableFunctionJit::CreateJit(Function* xls_function, int64_t opt_level,
                                 JitObserver* observer) {
  XLS_ASSIGN_OR_RETURN(
      auto jit,
      FunctionJit::Create(xls_function, opt_level,
                          /*include_observer_callbacks=*/false, observer));
  return std::unique_ptr<SwitchableFunctionJit>(new SwitchableFunctionJit(
      xls_function, /*use_jit=*/true, std::move(jit)));
}

absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>>
SwitchableFunctionJit::CreateInterpreter(Function* xls_function) {
  return std::unique_ptr<SwitchableFunctionJit>(
      new SwitchableFunctionJit(xls_function, /*use_jit=*/false, nullptr));
}

absl::StatusOr<std::unique_ptr<SwitchableFunctionJit>>
SwitchableFunctionJit::Create(Function* xls_function, ExecutionType execution,
                              int64_t opt_level, JitObserver* observer) {
  if (execution == ExecutionType::kDefault) {
    execution = kRealDefaultExecutionType;
    LOG_IF(WARNING, execution == ExecutionType::kInterpreter)
        << "Interpreter is being used in place of JIT.";
  }
  switch (execution) {
    case ExecutionType::kInterpreter:
      return SwitchableFunctionJit::CreateInterpreter(xls_function);
    case ExecutionType::kJit:
      return SwitchableFunctionJit::CreateJit(xls_function, opt_level,
                                              observer);
    case ExecutionType::kDefault:
      LOG(FATAL) << "Unreachable";
  }
}

namespace {
absl::StatusOr<absl::flat_hash_map<Node*, Value>> ToValueMap(
    absl::Span<const Value> args, Function* f) {
  absl::flat_hash_map<Node*, Value> res;
  XLS_RET_CHECK_EQ(args.size(), f->params().size())
      << "Wrong number of parameters";
  int64_t i = 0;
  for (Param* p : f->params()) {
    res.emplace(p, args.at(i++));
  }
  return res;
}

absl::StatusOr<absl::flat_hash_map<Node*, Value>> ToValueMap(
    const absl::flat_hash_map<std::string, Value>& args, Function* f) {
  absl::flat_hash_map<Node*, Value> res;
  for (Param* p : f->params()) {
    XLS_RET_CHECK(args.contains(p->name()))
        << "No value for param called '" << p->name() << "' given!";
    res.emplace(p, args.at(p->name()));
  }
  return res;
}

class MemoizedIrInterpreter : public IrInterpreter {
 public:
  using IrInterpreter::IrInterpreter;
  absl::Status HandleParam(Param* param) override {
    XLS_RET_CHECK(NodeValuesMap().contains(param));
    return absl::OkStatus();
  }
};

absl::StatusOr<InterpreterResult<Value>> Interpret(
    absl::flat_hash_map<Node*, Value> values, Function* function) {
  InterpreterEvents e;
  Node* return_val = function->return_value();
  InterpreterResult<Value> res;
  MemoizedIrInterpreter interp(&values, &res.events);
  XLS_RETURN_IF_ERROR(function->Accept(&interp));
  res.value = values.at(return_val);
  return res;
}
}  // namespace

absl::StatusOr<InterpreterResult<Value>> SwitchableFunctionJit::Run(
    absl::Span<const Value> args) {
  if (use_jit_) {
    return function_jit_->Run(args);
  }
  XLS_ASSIGN_OR_RETURN(auto node_args, ToValueMap(args, function()));
  return Interpret(std::move(node_args), function());
}

absl::StatusOr<InterpreterResult<Value>> SwitchableFunctionJit::Run(
    const absl::flat_hash_map<std::string, Value>& kwargs) {
  if (use_jit_) {
    return function_jit_->Run(kwargs);
  }
  XLS_ASSIGN_OR_RETURN(auto node_args, ToValueMap(kwargs, function()));
  return Interpret(std::move(node_args), function());
}

}  // namespace xls
