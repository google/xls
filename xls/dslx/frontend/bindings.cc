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
#include <vector>

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
#include "xls/dslx/status_payload.pb.h"
#include "xls/dslx/status_payload_utils.h"
#include "xls/dslx/type_system/type_info_to_proto.h"
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
    const absl::Status& status, std::optional<std::string_view> target_type,
    FileTable& file_table) {
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

  std::optional<StatusPayloadProto> payload = GetStatusPayload(status);
  if (payload.has_value() && payload->spans_size() > 0) {
    std::vector<Span> spans;
    for (const auto& span_proto : payload->spans()) {
      spans.push_back(FromProto(span_proto, file_table));
    }
    return PositionalErrorData{spans, std::string(s), type_indicator};
  }

  std::vector<std::string_view> pieces =
      absl::StrSplit(s, absl::MaxSplits(' ', 1));
  if (pieces.size() < 2) {
    return absl::InvalidArgumentError(
        "Provided status does not have a standard error message");
  }
  XLS_ASSIGN_OR_RETURN(Span span, Span::FromString(pieces[0], file_table));
  return PositionalErrorData{{span}, std::string(pieces[1]), type_indicator};
}

AnyNameDef BoundNodeToAnyNameDef(BoundNode bn) {
  return absl::visit(
      Visitor{[](NameDef* node) -> AnyNameDef { return node; },
              [](BuiltinNameDef* node) -> AnyNameDef { return node; },
              [](TypeAlias* node) -> AnyNameDef { return &node->name_def(); },
              [](Import* node) -> AnyNameDef { return &node->name_def(); },
              [](UseTreeEntry* node) -> AnyNameDef {
                return node->GetLeafNameDef().value();
              },
              [](auto* node) -> AnyNameDef { return node->name_def(); }},
      bn);
}

Span BoundNodeGetSpan(BoundNode bn, FileTable& file_table) {
  return absl::visit(Visitor{
                         [&](BuiltinNameDef* n) {
                           // Builtin name defs have no real span, so we provide
                           // a fake one here.
                           Fileno fileno = file_table.GetOrCreate("<builtin>");
                           Pos p(fileno, 0, 0);
                           return Span(p, p);
                         },
                         [](auto* n) { return n->span(); },
                     },
                     bn);
}

std::string BoundNodeGetTypeString(const BoundNode& bn) {
  return absl::visit(Visitor{[](EnumDef*) { return "EnumDef"; },
                             [](TypeAlias*) { return "TypeAlias"; },
                             [](ConstantDef*) { return "ConstantDef"; },
                             [](StructDef*) { return "StructDef"; },
                             [](ProcDef*) { return "ProcDef"; },
                             [](NameDef*) { return "NameDef"; },
                             [](BuiltinNameDef*) { return "BuiltinNameDef"; },
                             [](Import*) { return "Import"; },
                             [&](auto*) {
                               LOG(FATAL) << "Unsupported BoundNode variant: "
                                          << ToAstNode(bn)->ToString();
                               return "";
                             }},
                     bn);
}

Bindings::Bindings(Bindings* parent) : parent_(parent) {
  if (parent_ == nullptr) {
    fail_labels_.emplace();
  }
}

absl::StatusOr<AnyNameDef> Bindings::ResolveNameOrError(
    std::string_view name, const Span& span,
    const FileTable& file_table) const {
  XLS_ASSIGN_OR_RETURN(BoundNode bn,
                       ResolveNodeOrError(name, span, file_table));
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

absl::Status Bindings::AddFailLabel(const std::string& label, const Span& span,
                                    const FileTable& file_table) {
  // Traverse up to our function-scoped bindings since these labels must be
  // unique at the function scope.
  Bindings* top = this;
  while (!top->function_scoped_) {
    if (top->parent_ == nullptr) {
      return ParseErrorStatus(
          span, "Cannot use the `fail!` builtin at module top-level.",
          file_table);
    }
    top = top->parent_;
  }

  CHECK(top->function_scoped_);
  CHECK(top->fail_labels_.has_value());
  auto [it, inserted] = top->fail_labels_->insert(label);
  if (!inserted) {
    return ParseErrorStatus(
        span, "A fail label must be unique within a function.", file_table);
  }

  top->fail_labels_.value().insert(label);
  return absl::OkStatus();
}

}  // namespace xls::dslx
