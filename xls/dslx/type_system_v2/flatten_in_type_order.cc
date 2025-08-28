// Copyright 2025 The XLS Authors
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

#include "xls/dslx/type_system_v2/flatten_in_type_order.h"

#include <optional>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"

namespace xls::dslx {
namespace {

class Flattener : public AstNodeVisitorWithDefault {
 public:
  explicit Flattener(const ImportData& import_data, const AstNode* root,
                     bool include_parametric_entities)
      : import_data_(import_data),
        root_(root),
        include_parametric_entities_(include_parametric_entities) {}

  absl::Status HandleFunction(const Function* node) override {
    if (!include_parametric_entities_ && node->IsParametric()) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleProc(const Proc* node) override {
    if (!include_parametric_entities_ && node->IsParametric()) {
      return absl::OkStatus();
    }
    // Proc boundaries in the enclosing module scope are a "break point." The
    // caller needs to set up a new `TypeInfo` for the proc and then dive in. We
    // only dive in if the visitor's root is the proc subtree.
    if (node == root_) {
      // Procs have duplicate pointers to all the statements in them, so using
      // `DefaultHandler` would unnecessarily process each statement twice.
      for (ProcMember* member : node->members()) {
        XLS_RETURN_IF_ERROR(member->Accept(this));
      }
      return absl::OkStatus();
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleImpl(const Impl* node) override {
    XLS_ASSIGN_OR_RETURN(std::optional<const StructDefBase*> def,
                         GetStructOrProcDef(node->struct_ref(), import_data_));
    XLS_RET_CHECK(def.has_value());
    if (!include_parametric_entities_ && (*def)->IsParametric()) {
      return absl::OkStatus();
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    nodes_.push_back(node);
    XLS_RETURN_IF_ERROR(node->struct_ref()->Accept(this));
    for (const auto& [_, member] : node->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    // Do the equivalent `DefaultHandler`, but exclude most of the arguments.
    // We exclude the arguments because when an argument should
    // be converted depends on whether its type is determining or determined by
    // the formal argument type (it's determining it if it's based on an
    // implicit parametric). `ConvertInvocation` handles this.
    for (const ExprOrType& parametric : node->explicit_parametrics()) {
      XLS_RETURN_IF_ERROR(ToAstNode(parametric)->Accept(this));
    }

    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleLet(const Let* node) override {
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->rhs()->Accept(this));
    nodes_.push_back(node);
    for (const NameDef* name_def : node->name_def_tree()->GetNameDefs()) {
      XLS_RETURN_IF_ERROR(name_def->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleMatch(const Match* node) override {
    XLS_RETURN_IF_ERROR(node->matched()->Accept(this));
    // Prefer to visit arms that contain invocations first so that any type
    // information they produce is available when analyzing other arms whose
    // types unify with them.
    for (const MatchArm* arm : node->arms()) {
      XLS_RETURN_IF_ERROR(arm->Accept(this));
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleMatchArm(const MatchArm* node) override {
    for (const NameDefTree* name_def_tree : node->patterns()) {
      XLS_RETURN_IF_ERROR(name_def_tree->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->expr()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleRestOfTuple(const RestOfTuple* node) override {
    return absl::OkStatus();
  }

  absl::Status HandleConstantDef(const ConstantDef* node) override {
    if (node->type_annotation() != nullptr) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->value()->Accept(this));
    XLS_RETURN_IF_ERROR(node->name_def()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleUnrollFor(const UnrollFor* node) override {
    // node->body() will not be handled because unroll_for generates new
    // unrolled body statements.
    if (node->type_annotation()) {
      XLS_RETURN_IF_ERROR(node->type_annotation()->Accept(this));
    }
    XLS_RETURN_IF_ERROR(node->iterable()->Accept(this));
    XLS_RETURN_IF_ERROR(node->names()->Accept(this));
    XLS_RETURN_IF_ERROR(node->init()->Accept(this));
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleStructDef(const StructDef* node) override {
    // StructDefBase::GetChildren does not return StructMemberNodes, this is
    // blocked by https://github.com/google/xls/issues/1756.
    nodes_.push_back(node->name_def());
    for (const ParametricBinding* parametric_binding :
         node->parametric_bindings()) {
      XLS_RETURN_IF_ERROR(parametric_binding->Accept(this));
    }
    for (const StructMemberNode* member : node->members()) {
      XLS_RETURN_IF_ERROR(member->Accept(this));
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  absl::Status HandleTypeRef(const TypeRef* node) override {
    // A `TypeRef` does not expose its definition via GetChildren(), probably
    // because it is not seen as owning the definition. A definition like an
    // `EnumDef` will be processed when encountered under its lexical parent.
    // However, if the "definition" is a `ColonRef` to a type, that is not an
    // actual definition per se, and there are some tools that want to be able
    // to get a `Type` for any alleged `TypeDefinition`, so we queue it for type
    // conversion.
    if (std::holds_alternative<ColonRef*>(node->type_definition())) {
      return std::get<ColonRef*>(node->type_definition())->Accept(this);
    }
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    // Prefer conversion of invocations before nodes that may use them.
    std::vector<const AstNode*> invocations;
    std::vector<const AstNode*> non_invocations;

    const Invocation* current_invocation =
        node->kind() == AstNodeKind::kInvocation
            ? down_cast<const Invocation*>(node)
            : nullptr;
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      if (current_invocation != nullptr &&
          child == current_invocation->callee()) {
        continue;
      }
      if (child->kind() == AstNodeKind::kInvocation) {
        invocations.push_back(child);
      } else {
        non_invocations.push_back(child);
      }
    }
    for (const AstNode* child : invocations) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    for (const AstNode* child : non_invocations) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    nodes_.push_back(node);
    return absl::OkStatus();
  }

  const std::vector<const AstNode*>& nodes() const { return nodes_; }

 private:
  const ImportData& import_data_;
  const AstNode* const root_;
  const bool include_parametric_entities_;
  std::vector<const AstNode*> nodes_;
};

}  // namespace

absl::StatusOr<std::vector<const AstNode*>> FlattenInTypeOrder(
    const ImportData& import_data, const AstNode* root,
    bool include_parametric_entities) {
  Flattener flattener(import_data, root, include_parametric_entities);
  XLS_RETURN_IF_ERROR(root->Accept(&flattener));
  return flattener.nodes();
}

}  // namespace xls::dslx
