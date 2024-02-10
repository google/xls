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
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
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
    return absl::StrFormat("%s: %s", name_def_->ToString(),
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

// Represents a parsed 'process' specification in the DSL.
class Proc : public AstNode {
 public:
  static std::string_view GetDebugTypeName() { return "proc"; }

  Proc(Module* owner, Span span, NameDef* name_def,
       const std::vector<ParametricBinding*>& parametric_bindings,
       std::vector<ProcMember*> members, Function* config, Function* next,
       Function* init, bool is_public);

  ~Proc() override;

  AstNodeKind kind() const override { return AstNodeKind::kProc; }

  absl::Status Accept(AstNodeVisitor* v) const override {
    return v->HandleProc(this);
  }
  std::string_view GetNodeTypeName() const override { return "Proc"; }
  std::string ToString() const override;
  std::vector<AstNode*> GetChildren(bool want_types) const override;

  NameDef* name_def() const { return name_def_; }
  const Span& span() const { return span_; }
  std::optional<Span> GetSpan() const override { return span_; }

  const std::string& identifier() const { return name_def_->identifier(); }
  const std::vector<ParametricBinding*>& parametric_bindings() const {
    return parametric_bindings_;
  }
  bool IsParametric() const { return !parametric_bindings_.empty(); }
  bool is_public() const { return is_public_; }

  Function* config() const { return config_; }
  Function* next() const { return next_; }
  Function* init() const { return init_; }
  const std::vector<ProcMember*>& members() const { return members_; }

 private:
  Span span_;
  NameDef* name_def_;
  std::vector<ParametricBinding*> parametric_bindings_;

  Function* config_;
  Function* next_;
  Function* init_;
  std::vector<ProcMember*> members_;
  bool is_public_;
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
