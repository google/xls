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

#include "xls/dslx/frontend/proc.h"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/indent.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

bool HasMemberNamed(const ProcLikeBody& proc_body, std::string_view name) {
  for (const ProcMember* member : proc_body.members) {
    if (member->name_def()->identifier() == name) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<ProcStmt> ToProcStmt(AstNode* n) {
  if (auto* s = dynamic_cast<Function*>(n)) {
    return s;
  }
  if (auto* s = dynamic_cast<ProcMember*>(n)) {
    return s;
  }
  if (auto* s = dynamic_cast<ConstAssert*>(n)) {
    return s;
  }
  if (auto* s = dynamic_cast<ConstantDef*>(n)) {
    return s;
  }
  if (auto* s = dynamic_cast<TypeAlias*>(n)) {
    return s;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Node is not a valid ProcStmt; type: ", n->GetNodeTypeName()));
}

// -- class ProcLike

ProcLike::ProcLike(Module* owner, Span span, Span body_span, NameDef* name_def,
                   std::vector<ParametricBinding*> parametric_bindings,
                   ProcLikeBody body, bool is_public)
    : AstNode(owner),
      span_(std::move(span)),
      body_span_(std::move(body_span)),
      name_def_(name_def),
      parametric_bindings_(std::move(parametric_bindings)),
      body_(std::move(body)),
      is_public_(is_public) {
  CHECK(body_.config != nullptr);
  CHECK(body_.next != nullptr);
  CHECK(body_.init != nullptr);
}

ProcLike::~ProcLike() = default;

const XlsTuple* ProcLike::GetConfigTuple() const {
  const Function& config_fn = config();
  // Note: when the block is empty, trailing_semi is always true as an
  // invariant.
  if (config_fn.body()->trailing_semi()) {
    return nullptr;
  }
  Expr* last =
      std::get<Expr*>(config_fn.body()->statements().back()->wrapped());
  const XlsTuple* tuple = dynamic_cast<const XlsTuple*>(last);
  CHECK_NE(tuple, nullptr);
  return tuple;
}

std::vector<AstNode*> ProcLike::GetChildren(bool want_types) const {
  std::vector<AstNode*> results = {name_def()};
  for (ParametricBinding* pb : parametric_bindings_) {
    results.push_back(pb);
  }
  for (ProcMember* member : members()) {
    results.push_back(member);
  }
  for (const ProcStmt& s : body_.stmts) {
    results.push_back(ToAstNode(s));
  }
  return results;
}

std::string ProcLike::ToString() const {
  std::string pub_str = is_public() ? "pub " : "";
  std::string parametric_str;
  if (!parametric_bindings().empty()) {
    parametric_str = absl::StrFormat(
        "<%s>",
        absl::StrJoin(
            parametric_bindings(), ", ",
            [](std::string* out, ParametricBinding* parametric_binding) {
              absl::StrAppend(out, parametric_binding->ToString());
            }));
  }

  // Note: functions are emitted separately below, so we format the non-function
  // statements here.
  std::string stmts_str = absl::StrJoin(
      GetNonFunctionStmts(), "\n", [](std::string* out, const ProcStmt* stmt) {
        absl::StrAppend(out, ToAstNode(*stmt)->ToString());
      });
  if (!stmts_str.empty()) {
    absl::StrAppend(&stmts_str, "\n");
  }

  // Init functions are special, since they shouldn't be printed with
  // parentheses (since they can't take args).
  std::string init_str = Indent(
      absl::StrCat("init ", init().body()->ToString()), kRustSpacesPerIndent);

  constexpr std::string_view kTemplate = R"(%sproc %s%s {
%s%s
%s
%s
})";
  return absl::StrFormat(
      kTemplate, pub_str, name_def()->identifier(), parametric_str,
      Indent(stmts_str, kRustSpacesPerIndent),
      Indent(config().ToUndecoratedString("config"), kRustSpacesPerIndent),
      init_str,
      Indent(next().ToUndecoratedString("next"), kRustSpacesPerIndent));
}

// -- class TestProc

TestProc::~TestProc() = default;

std::string TestProc::ToString() const {
  if (expected_fail_label_.has_value()) {
    return absl::StrFormat("#[test_proc(expected_fail_label=\"%s\")]\n%s",
                           *expected_fail_label_, proc_->ToString());
  }
  return absl::StrFormat("#[test_proc]\n%s", proc_->ToString());
}

// -- class ProcMember

ProcMember::ProcMember(Module* owner, NameDef* name_def,
                       TypeAnnotation* type_annotation)
    : AstNode(owner),
      name_def_(name_def),
      type_annotation_(type_annotation),
      span_(name_def_->span().start(), type_annotation_->span().limit()) {}

ProcMember::~ProcMember() = default;

}  // namespace xls::dslx
