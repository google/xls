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

#ifndef XLS_DSLX_BYTECODE_INTERPRETER_STACK_H_
#define XLS_DSLX_BYTECODE_INTERPRETER_STACK_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// Encapsulates the bytecode interpreter stack state. This is operated on by
// builtin function implementations in addition to the interpreter itself.
class InterpreterStack {
 public:
  // Convenience helper for creating a stack with given values for testing.
  //
  // Note: stack.back() is the topmost (most recently pushed) value.
  static InterpreterStack CreateForTest(std::vector<InterpValue> stack) {
    return InterpreterStack{std::move(stack)};
  }

  InterpreterStack() = default;

  absl::StatusOr<InterpValue> Pop() {
    if (stack_.empty()) {
      return absl::InternalError("Tried to pop off an empty stack.");
    }
    InterpValue value = std::move(stack_.back());
    stack_.pop_back();
    return value;
  }

  void Push(InterpValue value) { stack_.push_back(std::move(value)); }

  const InterpValue& PeekOrDie(int64_t from_top = 0) const {
    CHECK_GE(stack_.size(), from_top + 1);
    return stack_.at(stack_.size() - from_top - 1);
  }

  // Returns a comma-delimited sequence of the interpreter values in the stack
  // as a string, with the oldest value first (left-most in the string).
  std::string ToString() const;

  bool empty() const { return stack_.empty(); }
  int64_t size() const { return stack_.size(); }

 private:
  explicit InterpreterStack(std::vector<InterpValue> stack)
      : stack_(std::move(stack)) {}

  std::vector<InterpValue> stack_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_STACK_H_
