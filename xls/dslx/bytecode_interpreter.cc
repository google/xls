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
#include "xls/dslx/bytecode_interpreter.h"

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

absl::Status BytecodeInterpreter::EvalAdd(const Bytecode& bytecode) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  InterpValue rhs = stack_.back();
  stack_.pop_back();

  InterpValue lhs = stack_.back();
  stack_.pop_back();

  XLS_ASSIGN_OR_RETURN(InterpValue sum, lhs.Add(rhs));
  stack_.push_back(sum);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalCall(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, bytecode.value_data());
  if (value.IsBuiltinFunction()) {
    return RunBuiltinFn(bytecode, absl::get<Builtin>(value.GetFunctionOrDie()));
  }

  return absl::UnimplementedError(
      "User-defined functions are not yet supported.");
}

absl::Status BytecodeInterpreter::EvalEq(const Bytecode& bytecode) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  InterpValue rhs = stack_.back();
  stack_.pop_back();
  InterpValue lhs = stack_.back();
  stack_.pop_back();

  stack_.push_back(InterpValue::MakeBool(lhs.Eq(rhs)));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLoad(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(int64_t slot, bytecode.integer_data());
  if (slots_->size() <= slot) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to access local data in slot %d, which out of range.",
        slot));
  }
  stack_.push_back(slots_->at(slot));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLiteral(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, bytecode.value_data());
  stack_.push_back(value);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalStore(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(int64_t slot, bytecode.integer_data());
  if (stack_.empty()) {
    return absl::InvalidArgumentError(
        "Attempted to store value from empty stack.");
  }

  if (slots_->size() <= slot) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to access local data in slot %d, which out of range.",
        slot));
  }

  slots_->at(slot) = stack_.back();
  stack_.pop_back();
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinFn(const Bytecode& bytecode,
                                               Builtin builtin) {
  switch (builtin) {
    case Builtin::kAssertEq:
      return RunBuiltinAssertEq(bytecode);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Builtin function \"%s\" not yet implemented.",
                          BuiltinToString(builtin)));
  }
}

absl::Status BytecodeInterpreter::RunBuiltinAssertEq(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AssertEq.";
  XLS_RET_CHECK_GE(stack_.size(), 2);
  InterpValue lhs = stack_[stack_.size() - 2];
  InterpValue rhs = stack_[stack_.size() - 1];

  XLS_RETURN_IF_ERROR(EvalEq(bytecode));
  if (stack_.back().IsFalse()) {
    std::string message =
        absl::StrFormat("\n  lhs: %s\n  rhs: %s\n  were not equal",
                        lhs.ToHumanString(), rhs.ToHumanString());
    return FailureErrorStatus(bytecode.source_span(), message);
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
