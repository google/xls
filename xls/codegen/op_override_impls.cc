// Copyright 2022 The XLS Authors
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

#include "xls/codegen/op_override_impls.h"

#include "absl/strings/str_replace.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls::verilog {

constexpr std::array<std::string_view, 2> kGatePlaceholderAliasKeys{
    "condition", "input"};
constexpr std::array<std::string_view, 2> kGatePlaceholderAliasValues{
    "input0", "input1"};

static absl::StatusOr<std::string> GenerateFormatString(
    std::string_view fmt_string,
    const absl::flat_hash_map<std::string, std::string>& supported_placeholders,
    const absl::flat_hash_map<std::string, std::string>&
        unsupported_placeholders) {
  RE2 re(R"({(\w+)})");
  std::string placeholder;
  std::string_view piece(fmt_string);

  // Verify that all placeholder substrings are supported.
  while (RE2::FindAndConsume(&piece, re, &placeholder)) {
    if (unsupported_placeholders.contains(placeholder)) {
      // Placeholder is one of the explicitly unsupported ones. Return the
      // error specific to that placeholder.
      return absl::InvalidArgumentError(
          unsupported_placeholders.at(placeholder));
    }

    if (!supported_placeholders.contains(placeholder)) {
      // Placeholder is not a supported string. Emit an error message with a
      // sorted list of all valid placeholders.
      std::vector<std::string> all_placeholders;
      for (auto [name, value] : supported_placeholders) {
        all_placeholders.push_back(absl::StrCat("{", name, "}"));
      }
      for (auto [name, value] : unsupported_placeholders) {
        all_placeholders.push_back(absl::StrCat("{", name, "}"));
      }
      std::sort(all_placeholders.begin(), all_placeholders.end());
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid placeholder {%s} in format string. "
                          "Valid placeholders: %s",
                          placeholder, absl::StrJoin(all_placeholders, ", ")));
    }
  }

  std::string str(fmt_string);
  for (auto [name, value] : supported_placeholders) {
    absl::StrReplaceAll({{absl::StrCat("{", name, "}"), value}}, &str);
  }
  return str;
}

std::unique_ptr<OpOverride> OpOverrideAssignment::Clone() const {
  return std::make_unique<OpOverrideAssignment>(assignment_format_string_,
                                                placeholder_aliases_);
}

absl::StatusOr<NodeRepresentation> OpOverrideAssignment::Emit(
    Node* node, std::string_view name,
    absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) {
  LogicRef* ref = mb.DeclareVariable(name, node->GetType());
  absl::flat_hash_map<std::string, std::string> placeholders;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (std::holds_alternative<Expression*>(inputs[i])) {
      placeholders[absl::StrCat("input", i)] =
          std::get<Expression*>(inputs[i])->Emit(nullptr);
    }
  }
  placeholders["output"] = name;
  placeholders["width"] = absl::StrCat(node->GetType()->GetFlatBitCount());

  for (const auto& itr : placeholder_aliases_) {
    // Use `temp` because `placeholders[itr.first]` can cause a rehash that
    // invalidates the reference returned by `placeholders[itr.second]`.
    std::string temp = placeholders[itr.second];
    placeholders[itr.first] = std::move(temp);
  }

  XLS_ASSIGN_OR_RETURN(
      std::string formatted_string,
      GenerateFormatString(assignment_format_string_, placeholders, {}));

  InlineVerilogStatement* raw_statement =
      mb.assignment_section()->Add<InlineVerilogStatement>(
          node->loc(), formatted_string + ";");
  return mb.file()->Make<InlineVerilogRef>(node->loc(), ref->GetName(),
                                           raw_statement);
}

static absl::flat_hash_map<std::string, std::string> GatePlaceholders() {
  static_assert(kGatePlaceholderAliasKeys.size() ==
                kGatePlaceholderAliasValues.size());
  absl::flat_hash_map<std::string, std::string> placeholders;
  for (size_t i = 0; i < kGatePlaceholderAliasKeys.size(); ++i) {
    placeholders.insert_or_assign(kGatePlaceholderAliasKeys[i],
                                  kGatePlaceholderAliasValues[i]);
  }
  return placeholders;
}

OpOverrideGateAssignment::OpOverrideGateAssignment(std::string_view fmt_string)
    : OpOverrideAssignment(fmt_string, GatePlaceholders()) {}

std::unique_ptr<OpOverride> OpOverrideAssertion::Clone() const {
  return std::make_unique<OpOverrideAssertion>(assertion_format_string_);
}

absl::StatusOr<NodeRepresentation> OpOverrideAssertion::Emit(
    Node* node, std::string_view name,
    absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) {
  XLS_CHECK(node->Is<xls::Assert>());
  xls::Assert* asrt = node->As<xls::Assert>();
  XLS_CHECK_EQ(inputs.size(), 2);
  XLS_CHECK(std::holds_alternative<Expression*>(inputs[1]));

  Expression* condition = std::get<Expression*>(inputs[1]);

  absl::flat_hash_map<std::string, std::string> supported_placeholders;
  absl::flat_hash_map<std::string, std::string> unsupported_placeholders;
  supported_placeholders["message"] = asrt->message();
  supported_placeholders["condition"] = condition->Emit(nullptr);
  if (asrt->label().has_value()) {
    supported_placeholders["label"] = asrt->label().value();
  } else {
    unsupported_placeholders["label"] =
        "Assert format string has {label} placeholder, but assert operation "
        "has no label.";
  }
  if (mb.clock() != nullptr) {
    supported_placeholders["clk"] = mb.clock()->GetName();
  } else {
    unsupported_placeholders["clk"] =
        "Assert format string has {clk} placeholder, but block has no clock "
        "signal.";
  }
  if (mb.reset().has_value()) {
    supported_placeholders["rst"] = mb.reset()->signal->GetName();
  } else {
    unsupported_placeholders["rst"] =
        "Assert format string has {rst} placeholder, but block has no reset "
        "signal.";
  }
  XLS_ASSIGN_OR_RETURN(
      std::string assert_str,
      GenerateFormatString(assertion_format_string_, supported_placeholders,
                           unsupported_placeholders));
  return mb.AssertAlwaysComb(node->loc())
      ->statements()
      ->Add<InlineVerilogStatement>(asrt->loc(), absl::StrCat(assert_str, ";"));
}

std::unique_ptr<OpOverride> OpOverrideInstantiation::Clone() const {
  return std::make_unique<OpOverrideInstantiation>(
      instantiation_format_string_);
}

absl::StatusOr<NodeRepresentation> OpOverrideInstantiation::Emit(
    Node* node, std::string_view name,
    absl::Span<NodeRepresentation const> inputs, ModuleBuilder& mb) {
  LogicRef* ref = mb.DeclareVariable(name, node->GetType());

  absl::flat_hash_map<std::string, std::string> placeholders;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (std::holds_alternative<Expression*>(inputs[i])) {
      placeholders[absl::StrCat("input", i)] =
          std::get<Expression*>(inputs[i])->Emit(nullptr);
      placeholders[absl::StrCat("input", i, "_width")] =
          absl::StrCat(node->operand(i)->GetType()->GetFlatBitCount());
    }
  }
  placeholders["output"] = name;
  placeholders["output_width"] =
      absl::StrCat(node->GetType()->GetFlatBitCount());

  XLS_ASSIGN_OR_RETURN(
      std::string formatted_string,
      GenerateFormatString(instantiation_format_string_, placeholders, {}));

  mb.instantiation_section()->Add<InlineVerilogStatement>(node->loc(),
                                                          formatted_string);

  return ref;
}

}  // namespace xls::verilog
