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
    const absl::Status& status, std::optional<absl::string_view> target_type) {
  auto error = [&] {
    return absl::InvalidArgumentError(
        "Provided status is not in recognized error form: " +
        status.ToString());
  };
  absl::string_view s = status.message();
  std::string type_indicator;
  if (!RE2::Consume(&s, "(\\w+): ", &type_indicator)) {
    return error();
  }
  if (target_type.has_value() && type_indicator != *target_type) {
    return error();
  }
  std::vector<absl::string_view> pieces =
      absl::StrSplit(s, absl::MaxSplits(' ', 1));
  if (pieces.size() < 2) {
    return absl::InvalidArgumentError(
        "Provided status does not have a standard error message");
  }
  XLS_ASSIGN_OR_RETURN(Span span, Span::FromString(pieces[0]));
  return PositionalErrorData{span, std::string(pieces[1]), type_indicator};
}

AnyNameDef BoundNodeToAnyNameDef(BoundNode bn) {
  if (absl::holds_alternative<EnumDef*>(bn)) {
    return absl::get<EnumDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<TypeDef*>(bn)) {
    return absl::get<TypeDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<ConstantDef*>(bn)) {
    return absl::get<ConstantDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<const NameDef*>(bn)) {
    return absl::get<const NameDef*>(bn);
  }
  if (absl::holds_alternative<BuiltinNameDef*>(bn)) {
    return absl::get<BuiltinNameDef*>(bn);
  }
  if (absl::holds_alternative<StructDef*>(bn)) {
    return absl::get<StructDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<Import*>(bn)) {
    return absl::get<Import*>(bn)->name_def();
  }
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString() << " "
                 << ToAstNode(bn)->GetNodeTypeName();
}

Span BoundNodeGetSpan(BoundNode bn) {
  if (absl::holds_alternative<ConstantDef*>(bn)) {
    return absl::get<ConstantDef*>(bn)->span();
  }
  if (absl::holds_alternative<TypeDef*>(bn)) {
    return absl::get<TypeDef*>(bn)->span();
  }
  if (absl::holds_alternative<StructDef*>(bn)) {
    return absl::get<StructDef*>(bn)->span();
  }
  if (absl::holds_alternative<EnumDef*>(bn)) {
    return absl::get<EnumDef*>(bn)->span();
  }
  if (absl::holds_alternative<const NameDef*>(bn)) {
    return absl::get<const NameDef*>(bn)->span();
  }
  if (absl::holds_alternative<BuiltinNameDef*>(bn)) {
    Pos p("<builtin>", 0, 0);
    return Span(p, p);
  }
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
}

std::string BoundNodeGetTypeString(const BoundNode& bn) {
  // clang-format off
  if (absl::holds_alternative<EnumDef*>(bn)) { return "EnumDef"; }
  if (absl::holds_alternative<TypeDef*>(bn)) { return "TypeDef"; }
  if (absl::holds_alternative<ConstantDef*>(bn)) { return "ConstantDef"; }
  if (absl::holds_alternative<StructDef*>(bn)) { return "StructDef"; }
  if (absl::holds_alternative<const NameDef*>(bn)) { return "NameDef"; }
  if (absl::holds_alternative<BuiltinNameDef*>(bn)) { return "BuiltinNameDef"; }
  if (absl::holds_alternative<Import*>(bn)) { return "Import"; }
  // clang-format on
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
}

absl::StatusOr<AnyNameDef> Bindings::ResolveNameOrError(
    absl::string_view name, const Span& span) const {
  XLS_ASSIGN_OR_RETURN(BoundNode bn, ResolveNodeOrError(name, span));
  return BoundNodeToAnyNameDef(bn);
}

std::optional<AnyNameDef> Bindings::ResolveNameOrNullopt(
    absl::string_view name) const {
  std::optional<BoundNode> bn = ResolveNode(name);
  if (!bn) {
    return absl::nullopt;
  }
  return BoundNodeToAnyNameDef(*bn);
}

}  // namespace xls::dslx
