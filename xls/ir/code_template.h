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

#ifndef XLS_IR_CODE_TEMPLATE_H_
#define XLS_IR_CODE_TEMPLATE_H_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xls {

// Templating used for the Foreign Function call code generation.
// Templates contain expressions in {foo} ranges. Legitimate
// braces that should not start an expression need to be escaped with
// the same brace. So "{{foo}}" will expand to "{foo}".
//
// Depending on context, the expressions used are relevant in the language
// domain.
// In DSLX, the template will contain DSLX expressions provided by the user
// #[extern_verilog("foo {fn} (.x({a}), .y({b}), .out({return}))")]
// These  are then re-written to corresponding IR expressions when converted
// to IR.
//
// If CodeTemplate ever needed beyond foreign function calls, move in separate
// header.
class CodeTemplate {
 public:
  // Parse template, and do some syntactic checks, e.g. braces and parentheses
  // need to be balanced. On success, returns a constructed CodeTemplate.
  // Error status starts with a zero-based column number where the issue has
  // been discoverd in the template.
  static absl::StatusOr<CodeTemplate> Create(std::string_view template_text);

  // Extract the column number of the error position from the Create()-status.
  static int64_t ExtractErrorColumn(const absl::Status& s);

  // Returns the content of the expressions found in the template.
  absl::Span<const std::string> Expressions() const { return expressions_; }

  // Handling of curly-brace escaping in Substitute().
  enum class Escaped {
    kUnescape,
    kKeep,
  };

  // Given a template parameter name, return the replacment.
  using ParameterReplace = std::function<std::string(std::string_view name)>;

  // Substitute all template parameters with the value provided by the
  // "replacment_lookup" function and return the resulting string.
  // By default, the "escape_handling" is Escaped::kUnescape so that e.g.
  // '{{' will be replaced with '{' to be used as a regular template expansion.
  // If "escape_handling" is set to Escaped::kKeep all escaped characters are
  // kept escaped, so the result can be used as a valid escaped code template.
  std::string Substitute(const ParameterReplace& replacement_lookup,
                         Escaped escape_handling = Escaped::kUnescape) const;

  // Get the original template text.
  std::string ToString() const;

 private:
  CodeTemplate() = default;  // Use Create() to public construct

  // Parse template, return success. Used in the public Create()
  // method; see description there.
  absl::Status Parse(std::string_view template_text);

  // Separate arrays to be able to hand out contained expressions cheaply.
  std::vector<std::string> leading_text_;  // text up to next expression
  std::vector<std::string> expressions_;
};
}  // namespace xls

#endif  // XLS_IR_CODE_TEMPLATE_H_
