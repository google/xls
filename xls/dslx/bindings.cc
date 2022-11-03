// Copyright 2020 The XLS Authors
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

#include "xls/dslx/bindings.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls::dslx {

absl::StatusOr<PositionalErrorData> GetPositionalErrorData(
    const absl::Status& status, std::optional<std::string_view> target_type) {
  auto error = [&] {
    return absl::InvalidArgumentError(
        "Provided status is not in recognized error form: " +
        status.ToString());
  };
  std::string_view s = status.message();
  std::string type_indicator;
  if (!RE2::Consume(&s, "(\\w+): ", &type_indicator)) {
    return error();
  }
  if (target_type.has_value() && type_indicator != *target_type) {
    return error();
  }
  std::vector<std::string_view> pieces =
      absl::StrSplit(s, absl::MaxSplits(' ', 1));
  if (pieces.size() < 2) {
    return absl::InvalidArgumentError(
        "Provided status does not have a standard error message");
  }
  XLS_ASSIGN_OR_RETURN(Span span, Span::FromString(pieces[0]));
  return PositionalErrorData{span, std::string(pieces[1]), type_indicator};
}

AnyNameDef BoundNodeToAnyNameDef(BoundNode bn) {
  if (std::holds_alternative<EnumDef*>(bn)) {
    return std::get<EnumDef*>(bn)->name_def();
  }
  if (std::holds_alternative<TypeDef*>(bn)) {
    return std::get<TypeDef*>(bn)->name_def();
  }
  if (std::holds_alternative<ConstantDef*>(bn)) {
    return std::get<ConstantDef*>(bn)->name_def();
  }
  if (std::holds_alternative<const NameDef*>(bn)) {
    return std::get<const NameDef*>(bn);
  }
  if (std::holds_alternative<BuiltinNameDef*>(bn)) {
    return std::get<BuiltinNameDef*>(bn);
  }
  if (std::holds_alternative<StructDef*>(bn)) {
    return std::get<StructDef*>(bn)->name_def();
  }
  if (std::holds_alternative<Import*>(bn)) {
    return std::get<Import*>(bn)->name_def();
  }
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString() << " "
                 << ToAstNode(bn)->GetNodeTypeName();
}

Span BoundNodeGetSpan(BoundNode bn) {
  if (std::holds_alternative<ConstantDef*>(bn)) {
    return std::get<ConstantDef*>(bn)->span();
  }
  if (std::holds_alternative<TypeDef*>(bn)) {
    return std::get<TypeDef*>(bn)->span();
  }
  if (std::holds_alternative<StructDef*>(bn)) {
    return std::get<StructDef*>(bn)->span();
  }
  if (std::holds_alternative<EnumDef*>(bn)) {
    return std::get<EnumDef*>(bn)->span();
  }
  if (std::holds_alternative<const NameDef*>(bn)) {
    return std::get<const NameDef*>(bn)->span();
  }
  if (std::holds_alternative<BuiltinNameDef*>(bn)) {
    Pos p("<builtin>", 0, 0);
    return Span(p, p);
  }
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
}

std::string BoundNodeGetTypeString(const BoundNode& bn) {
  // clang-format off
  if (std::holds_alternative<EnumDef*>(bn)) { return "EnumDef"; }
  if (std::holds_alternative<TypeDef*>(bn)) { return "TypeDef"; }
  if (std::holds_alternative<ConstantDef*>(bn)) { return "ConstantDef"; }
  if (std::holds_alternative<StructDef*>(bn)) { return "StructDef"; }
  if (std::holds_alternative<const NameDef*>(bn)) { return "NameDef"; }
  if (std::holds_alternative<BuiltinNameDef*>(bn)) { return "BuiltinNameDef"; }
  if (std::holds_alternative<Import*>(bn)) { return "Import"; }
  // clang-format on
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
}

Bindings Bindings::Clone() const {
  Bindings bindings(parent_);
  bindings.local_bindings_ = local_bindings_;
  bindings.fail_labels_ = fail_labels_;
  return bindings;
}

absl::StatusOr<AnyNameDef> Bindings::ResolveNameOrError(
    std::string_view name, const Span& span) const {
  XLS_ASSIGN_OR_RETURN(BoundNode bn, ResolveNodeOrError(name, span));
  return BoundNodeToAnyNameDef(bn);
}

std::optional<AnyNameDef> Bindings::ResolveNameOrNullopt(
    std::string_view name) const {
  std::optional<BoundNode> bn = ResolveNode(name);
  if (!bn) {
    return absl::nullopt;
  }
  return BoundNodeToAnyNameDef(*bn);
}

}  // namespace xls::dslx
