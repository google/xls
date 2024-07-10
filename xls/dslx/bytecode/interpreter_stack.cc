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

#include "xls/dslx/bytecode/interpreter_stack.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

/* static */ InterpreterStack InterpreterStack::CreateForTest(
    absl::Span<const InterpValue> stack) {
  std::vector<FormattedInterpValue> elements;
  elements.reserve(stack.size());
  for (const InterpValue& value : stack) {
    elements.push_back(FormattedInterpValue{.value = value,
                                            .format_descriptor = std::nullopt});
  }
  return InterpreterStack{std::move(elements)};
}

std::string InterpreterStack::ToString() const {
  return absl::StrJoin(
      stack_, ", ", [](std::string* out, const FormattedInterpValue& v) {
        if (v.format_descriptor.has_value()) {
          absl::StrAppend(
              out, v.value.ToFormattedString(*v.format_descriptor).value());
        } else {
          absl::StrAppend(out, v.value.ToString());
        }
      });
}

}  // namespace xls::dslx
