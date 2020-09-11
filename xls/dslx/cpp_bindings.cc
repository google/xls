// Copyright 2020 Google LLC
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

#include "xls/common/status/status_macros.h"

namespace xls::dslx {

static AnyNameDef BoundNodeToAnyNameDef(BoundNode bn) {
  // clang-format off
  if (absl::holds_alternative<ConstantDef*>(bn)) { return absl::get<ConstantDef*>(bn)->name_def(); }  // NOLINT
  if (absl::holds_alternative<TypeDef*>(bn)) { return absl::get<TypeDef*>(bn)->name_def(); }  // NOLINT
  if (absl::holds_alternative<Struct*>(bn)) { return absl::get<Struct*>(bn)->name_def(); }  // NOLINT
  if (absl::holds_alternative<Enum*>(bn)) { return absl::get<Enum*>(bn)->name_def(); }  // NOLINT
  if (absl::holds_alternative<NameDef*>(bn)) { return absl::get<NameDef*>(bn); }  // NOLINT
  if (absl::holds_alternative<BuiltinNameDef*>(bn)) { return absl::get<BuiltinNameDef*>(bn); }  // NOLINT
  // clang-format on
  XLS_LOG(FATAL) << "Unsupported BoundNode variant: "
                 << ToAstNode(bn)->ToString();
}

xabsl::StatusOr<AnyNameDef> Bindings::ResolveNameOrError(
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

xabsl::StatusOr<BoundNode> ToBoundNode(AstNode* n) {
  // clang-format off
  if (auto* bn = dynamic_cast<Enum*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<TypeDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<ConstantDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<NameDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<BuiltinNameDef*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<Struct*>(n)) { return BoundNode(bn); }
  if (auto* bn = dynamic_cast<Import*>(n)) { return BoundNode(bn); }
  // clang-format on
  return absl::InvalidArgumentError("Invalid AST node for use in bindings: " +
                                    n->ToString());
}

}  // namespace xls::dslx
