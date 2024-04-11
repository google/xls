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

#include "xls/dslx/frontend/bindings.h"

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "re2/re2.h"

namespace xls::dslx {

std::optional<std::string_view> MaybeExtractParseNameError(
    const absl::Status& status) {
  if (status.code() != absl::StatusCode::kInvalidArgument) {
    return std::nullopt;
  }
  std::string_view name;
  if (RE2::PartialMatch(status.message(),
                        R"(Cannot find a definition for name: \"(\w+)\")",
                        &name)) {
    return name;
  }
  return std::nullopt;
}

absl::StatusOr<PositionalErrorData> GetPositionalErrorData(
    const absl::Status& status, std::optional<std::string_view> target_type) {
  auto error = [&] {
    return absl::InvalidArgumentError(
        "Provided status is not in recognized error form: " +
        status.ToString());
  };
  std::string_view s = status.message();
  std::string type_indicator;
  // Note: we permit angle braces around the filename for cases that are
  // delimiting special things like fake files or stdin; e.g.
  //
  //    <fake>:1:2
  if (!RE2::Consume(&s, "(<?\\w+>?): ", &type_indicator)) {
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
  if (std::holds_alternative<TypeAlias*>(bn)) {
    return &std::get<TypeAlias*>(bn)->name_def();
  }
  if (std::holds_alternative<ConstantDef*>(bn)) {
    return std::get<ConstantDef*>(bn)->name_def();
  }
  if (std::holds_alternative<NameDef*>(bn)) {
    return std::get<NameDef*>(bn);
  }
  if (std::holds_alternative<BuiltinNameDef*>(bn)) {
    return std::get<BuiltinNameDef*>(bn);
  }
  if (std::holds_alternative<StructDef*>(bn)) {
    return std::get<StructDef*>(bn)->name_def();
  }
  if (std::holds_alternative<Import*>(bn)) {
    return &std::get<Import*>(bn)->name_def();
  }
  LOG(FATAL) << "Unsupported BoundNode variant: " << ToAstNode(bn)->ToString()
             << " " << ToAstNode(bn)->GetNodeTypeName();
}

Span BoundNodeGetSpan(BoundNode bn) {
  return absl::visit(Visitor{
                         [](EnumDef* n) { return n->span(); },
                         [](TypeAlias* n) { return n->span(); },
                         [](ConstantDef* n) { return n->span(); },
                         [](NameDef* n) { return n->span(); },
                         [](BuiltinNameDef* n) {
                           // Builtin name defs have no real span, so we provide
                           // a fake one here.
                           Pos p("<builtin>", 0, 0);
                           return Span(p, p);
                         },
                         [](StructDef* n) { return n->span(); },
                         [](Import* n) { return n->span(); },
                     },
                     bn);
}

std::string BoundNodeGetTypeString(const BoundNode& bn) {
  // clang-format off
  if (std::holds_alternative<EnumDef*>(bn)) { return "EnumDef"; }
  if (std::holds_alternative<TypeAlias*>(bn)) { return "TypeAlias"; }
  if (std::holds_alternative<ConstantDef*>(bn)) { return "ConstantDef"; }
  if (std::holds_alternative<StructDef*>(bn)) { return "StructDef"; }
  if (std::holds_alternative<NameDef*>(bn)) { return "NameDef"; }
  if (std::holds_alternative<BuiltinNameDef*>(bn)) { return "BuiltinNameDef"; }
  if (std::holds_alternative<Import*>(bn)) { return "Import"; }
  // clang-format on
  LOG(FATAL) << "Unsupported BoundNode variant: " << ToAstNode(bn)->ToString();
}

Bindings::Bindings(Bindings* parent) : parent_(parent) {
  if (parent_ == nullptr) {
    fail_labels_.emplace();
  }
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
    return std::nullopt;
  }
  return BoundNodeToAnyNameDef(*bn);
}

absl::Status Bindings::AddFailLabel(const std::string& label,
                                    const Span& span) {
  // Traverse up to our function-scoped bindings since these labels must be
  // unique at the function scope.
  Bindings* top = this;
  while (!top->function_scoped_) {
    CHECK(top->parent_ != nullptr);
    top = top->parent_;
  }

  CHECK(top->function_scoped_);
  CHECK(top->fail_labels_.has_value());
  auto [it, inserted] = top->fail_labels_->insert(label);
  if (!inserted) {
    return ParseErrorStatus(span,
                            "A fail label must be unique within a function.");
  }

  top->fail_labels_.value().insert(label);
  return absl::OkStatus();
}

}  // namespace xls::dslx
