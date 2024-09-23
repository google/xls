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

#ifndef XLS_DSLX_FRONTEND_PROC_H_
#define XLS_DSLX_FRONTEND_PROC_H_

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// A member held in a proc, e.g. a channel declaration initialized by a
// configuration block.
//
// This is very similar to a `Param` at the moment, but we make them distinct
// types for structural clarity in the AST. Params are really "parameters to
// functions".
class ProcMember : public AstNode {
 public:
  ProcMember(Module* owner, NameDef* name_def, TypeAnnotation* type);

  ~ProcMember() override;

  AstNodeKind kind() const override { return AstNodeKind::kProcMember; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleProcMember(this);
  }

  std::string_view GetNodeTypeName() const override { return "ProcMember"; }
  std::string ToString() const override {
    return absl::StrFormat("%s: %s;", name_def_->ToString(),
                           type_annotation_->ToString());
  }

  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {name_def_, type_annotation_};
  }

  const Span& span() const { return span_; }
  NameDef* name_def() const { return name_def_; }
  TypeAnnotation* type_annotation() const { return type_annotation_; }
  const std::string& identifier() const { return name_def_->identifier(); }
  std::optional<Span> GetSpan() const override { return span_; }

 private:
  NameDef* name_def_;
  TypeAnnotation* type_annotation_;
  Span span_;
};

// TODO(leary): 2024-02-09 Extend this to allow for constant definitions, etc.
using ProcStmt = std::variant<Function*, ProcMember*, TypeAlias*, ConstAssert*,
                              ConstantDef*>;

absl::StatusOr<ProcStmt> ToProcStmt(AstNode* n);

// Constructs that we encountered during parsing that are present within a
// proc's body scope.
struct ProcLikeBody {
  // Note: this is the statements in the proc as parsed in order -- the fields
  // below are specially extracted items from these statements.
  std::vector<ProcStmt> stmts;

  Function* config;
  Function* next;
  Function* init;
  std::vector<ProcMember*> members;
};

// Note: linear time, expected only to be used only under error conditions where
// we don't mind taking time reporting.
bool HasMemberNamed(const ProcLikeBody& proc_body, std::string_view name);

class ProcLike : public AstNode {
 public:
  ProcLike(Module* owner, Span span, Span body_span, NameDef* name_def,
           std::vector<ParametricBinding*> parametric_bindings,
           ProcLikeBody body, bool is_public);

  ~ProcLike() override;

  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  const Span& span() const { return span_; }

  // Returns the span of the body block, i.e. from the opening '{' as the start
  // to the closing '}' as the limit.
  const Span& body_span() const { return body_span_; }

  std::optional<Span> GetSpan() const override { return span_; }

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  Function& config() const { return *body_.config; }
  Function& next() const { return *body_.next; }
  Function& init() const { return *body_.init; }
  absl::Span<ProcMember* const> members() const { return body_.members; }
  absl::Span<ProcStmt const> stmts() const { return body_.stmts; }
  absl::Span<ProcStmt> stmts() { return absl::MakeSpan(body_.stmts); }

  std::vector<const ProcStmt*> GetNonFunctionStmts() const {
    std::vector<const ProcStmt*> result;
    for (const ProcStmt& stmt : stmts()) {
      if (std::holds_alternative<Function*>(stmt)) {
        continue;
      }
      result.push_back(&stmt);
    }
    return result;
  }

  template <typename T>
  std::vector<const T*> GetStmtsOfType() const {
    std::vector<const T*> result;
    for (const ProcStmt& stmt : stmts()) {
      if (std::holds_alternative<T*>(stmt)) {
        result.push_back(std::get<T*>(stmt));
      }
    }
    return result;
  }

  // Note: this should be called after type checking has been performed, at
  // which point it should be validated that the last statement, if present, is
  // a tuple expression.
  absl::Nullable<const XlsTuple*> GetConfigTuple() const;

 private:
  Span span_;
  Span body_span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;

  ProcLikeBody body_;
  bool is_public_;
};

// Represents a parsed 'process' (i.e. communicating sequential process)
// specification in the DSL.
class Proc : public ProcLike {
 public:
  using ProcLike::ProcLike;

  static std::string_view GetDebugTypeName() { return "proc"; }

  AstNodeKind kind() const override { return AstNodeKind::kProc; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleProc(this);
  }
  std::string_view GetNodeTypeName() const override { return "Proc"; }
};

// Represents a construct to unit test a Proc. Analogous to TestFunction, but
// for Procs.
//
// These are specified with an annotation as follows:
// ```dslx
// #[test_proc()]
// proc test_proc { ... }
// ```
class TestProc : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "test proc"; }

  TestProc(Module* owner, Proc* proc) : AstNode(owner), proc_(proc) {}
  ~TestProc() override;

  AstNodeKind kind() const override { return AstNodeKind::kTestProc; }
  NameDef* name_def() const { return proc_->name_def(); }
  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleTestProc(this);
  }
  std::vector<AstNode*> GetChildren(bool want_types) const override {
    return {proc_};
  }
  std::string_view GetNodeTypeName() const override { return "TestProc"; }
  std::string ToString() const override;

  Proc* proc() const { return proc_; }
  std::optional<Span> GetSpan() const override { return proc_->span(); }

  const std::string& identifier() const { return proc_->identifier(); }

 private:
  Proc* proc_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_PROC_H_
