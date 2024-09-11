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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/value_format_descriptor.h"

namespace xls::dslx {

// Encapsulates the bytecode interpreter stack state. This is operated on by
// builtin function implementations in addition to the interpreter itself.
//
// The stack holds InterpValues and optional formatting information. The
// formatting information comes from literal/numbers in the code and can is used
// provide better error messages.
class InterpreterStack {
 public:
  // Convenience helper for creating a stack with given values for testing.
  //
  // Note: stack.back() is the topmost (most recently pushed) value.
  static InterpreterStack CreateForTest(const FileTable& file_table,
                                        absl::Span<const InterpValue> stack);

  explicit InterpreterStack(const FileTable& file_table)
      : file_table_(file_table) {}

  absl::StatusOr<InterpValue> Pop() {
    XLS_ASSIGN_OR_RETURN(FormattedInterpValue value, PopFormattedValue());
    return std::move(value.value);
  }

  struct FormattedInterpValue {
    InterpValue value;
    std::optional<ValueFormatDescriptor> format_descriptor;
  };
  absl::StatusOr<FormattedInterpValue> PopFormattedValue() {
    if (stack_.empty()) {
      return absl::InternalError("Tried to pop off an empty stack.");
    }
    FormattedInterpValue value = std::move(stack_.back());
    stack_.pop_back();
    return value;
  }

  void Push(InterpValue value) {
    VLOG(3) << absl::StreamFormat("Push(%s)", value.ToString());
    stack_.push_back(FormattedInterpValue{.value = std::move(value),
                                          .format_descriptor = std::nullopt});
  }
  void PushFormattedValue(FormattedInterpValue value) {
    VLOG(3) << absl::StreamFormat(
        "PushFormattedValue(%s)",
        value.format_descriptor.has_value()
            ? value.value.ToFormattedString(*value.format_descriptor).value()
            : value.value.ToString());
    stack_.push_back(std::move(value));
  }

  const InterpValue& PeekOrDie(int64_t from_top = 0) const {
    CHECK_GE(stack_.size(), from_top + 1);
    return stack_.at(stack_.size() - from_top - 1).value;
  }

  // Returns a comma-delimited sequence of the interpreter values in the stack
  // as a string, with the oldest value first (left-most in the string).
  std::string ToString() const;

  bool empty() const { return stack_.empty(); }
  int64_t size() const { return stack_.size(); }

  const FileTable& file_table() const { return file_table_; }

 private:
  explicit InterpreterStack(const FileTable& file_table,
                            std::vector<FormattedInterpValue> stack)
      : file_table_(file_table), stack_(std::move(stack)) {}

  const FileTable& file_table_;
  std::vector<FormattedInterpValue> stack_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_BYTECODE_INTERPRETER_STACK_H_
