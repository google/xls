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

  // Fill template with replacement text for each expression.
  // The replacement sequence needs to contain as many elements as Expressions()
  // returns. Unescapes "{{" and "}}" sequences to "{" and "}".
  absl::StatusOr<std::string> FillTemplate(
      absl::Span<const std::string> replacements) const;

  // Recreate a template, but replace expressions.
  // Keeps {} and escaped {{ }} as-is, so that the result can be parsed
  // as template again.
  absl::StatusOr<std::string> FillEscapedTemplate(
      absl::Span<const std::string> replacements) const;

  // Get the original template text.
  std::string ToString() const;

  // There is no default constructor, but constructed CodeTemplates can
  // be assignable. We need to be explicit here to be able to use
  // this object with std::optional.
  CodeTemplate& operator=(const CodeTemplate&) = default;

 private:
  CodeTemplate() = default;  // Use Create() to public construct

  // Parse template, return success. Used in the public Create()
  // method; see description there.
  absl::Status Parse(std::string_view template_text);

  // Fill template with the choice of escaping curly braces
  // in the template as well as surrounding replacments with
  // a prefix and suffix. Underlying method to provide public
  // functionality of FillTemplate(), FillEscapedTemplate() and
  // ToString().
  absl::StatusOr<std::string> FillTemplate(
      absl::Span<const std::string> replacements, bool escape_curly,
      std::string_view expression_prefix,
      std::string_view expression_suffix) const;

  // Separate arrays to be able to hand out contained expressions cheaply.
  std::vector<std::string> leading_text_;  // text up to next expression
  std::vector<std::string> expressions_;
};
}  // namespace xls

#endif  // XLS_IR_CODE_TEMPLATE_H_
