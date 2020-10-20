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

#include "xls/dslx/cpp_bindings.h"

#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

absl::StatusOr<std::pair<Span, std::string>> ParseErrorGetData(
    const absl::Status& status, absl::string_view prefix) {
  absl::string_view s = status.message();
  if (absl::ConsumePrefix(&s, prefix)) {
    std::vector<absl::string_view> pieces =
        absl::StrSplit(s, absl::MaxSplits(' ', 1));
    if (pieces.size() < 2) {
      return absl::InvalidArgumentError(
          "Provided status does not have a standard error message");
    }
    XLS_ASSIGN_OR_RETURN(Span span, Span::FromString(pieces[0]));
    return std::make_pair(span, std::string(pieces[1]));
  }
  return absl::InvalidArgumentError(
      "Provided status is not in recognized error form: " + status.ToString());
}

AnyNameDef BoundNodeToAnyNameDef(BoundNode bn) {
  if (absl::holds_alternative<ConstantDef*>(bn)) {
    return absl::get<ConstantDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<TypeDef*>(bn)) {
    return absl::get<TypeDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<StructDef*>(bn)) {
    return absl::get<StructDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<EnumDef*>(bn)) {
    return absl::get<EnumDef*>(bn)->name_def();
  }
  if (absl::holds_alternative<NameDef*>(bn)) {
    return absl::get<NameDef*>(bn);
  }
  if (absl::holds_alternative<BuiltinNameDef*>(bn)) {
    return absl::get<BuiltinNameDef*>(bn);
  }
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
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
  if (absl::holds_alternative<NameDef*>(bn)) {
    return absl::get<NameDef*>(bn)->span();
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
  if (absl::holds_alternative<NameDef*>(bn)) { return "NameDef"; }
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

absl::optional<AnyNameDef> Bindings::ResolveNameOrNullopt(
    absl::string_view name) const {
  absl::optional<BoundNode> bn = ResolveNode(name);
  if (!bn) {
    return absl::nullopt;
  }
  return BoundNodeToAnyNameDef(*bn);
}

absl::StatusOr<BoundNode> ToBoundNode(AstNode* n) {
  // clang-format off
  if (auto* bn = dynamic_cast<EnumDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<TypeDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<ConstantDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<NameDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<BuiltinNameDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<StructDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<Import*>(n)) { return BoundNode(bn); }
  // clang-format on
  return absl::InvalidArgumentError("Invalid AST node for use in bindings: " +
                                    n->ToString());
}

}  // namespace xls::dslx
