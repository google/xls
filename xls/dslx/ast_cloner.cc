// Copyright 2022 The XLS Authors
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
#include "xls/dslx/ast_cloner.h"

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast.h"

namespace xls::dslx {

// TODO(rspringer): 2022-06-06: Make sure all NameDef::definers are set
// appropriately.
// TODO(rspringer): 2022-06-06: Switch to AstNodeVisitor (without "WithDefault")
// once all nodes are supported.
class AstCloner : public AstNodeVisitorWithDefault {
 public:
  AstCloner(Module* module) : module_(module) {}

  absl::Status HandleAttr(const Attr* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    // We must've seen the NameDef here, so we don't need to visit it.
    old_to_new_[n] = module_->Make<Attr>(
        n->span(), down_cast<Expr*>(old_to_new_.at(n->lhs())),
        down_cast<NameDef*>(old_to_new_.at(n->attr())));
    return absl::OkStatus();
  }

  absl::Status HandleBuiltinNameDef(const BuiltinNameDef* n) override {
    if (!old_to_new_.contains(n)) {
      old_to_new_[n] = module_->Make<BuiltinNameDef>(n->identifier());
    }
    return absl::OkStatus();
  }

  absl::Status HandleBuiltinTypeAnnotation(
      const BuiltinTypeAnnotation* n) override {
    old_to_new_[n] =
        module_->Make<BuiltinTypeAnnotation>(n->span(), n->builtin_type());
    return absl::OkStatus();
  }

  absl::Status HandleColonRef(const ColonRef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    ColonRef::Subject new_subject;
    if (std::holds_alternative<NameRef*>(n->subject())) {
      new_subject =
          down_cast<NameRef*>(old_to_new_.at(std::get<NameRef*>(n->subject())));
    } else {
      new_subject = down_cast<ColonRef*>(
          old_to_new_.at(std::get<ColonRef*>(n->subject())));
    }

    old_to_new_[n] = module_->Make<ColonRef>(n->span(), new_subject, n->attr());
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const Function* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    NameDef* new_name_def = down_cast<NameDef*>(old_to_new_.at(n->name_def()));

    std::vector<ParametricBinding*> new_parametric_bindings;
    new_parametric_bindings.reserve(n->parametric_bindings().size());
    for (const auto* pb : n->parametric_bindings()) {
      new_parametric_bindings.push_back(
          down_cast<ParametricBinding*>(old_to_new_.at(pb)));
    }

    std::vector<Param*> new_params;
    new_params.reserve(n->params().size());
    for (const auto* param : n->params()) {
      new_params.push_back(down_cast<Param*>(old_to_new_.at(param)));
    }

    old_to_new_[n] = module_->Make<Function>(
        n->span(), new_name_def, new_parametric_bindings, new_params,
        down_cast<TypeAnnotation*>(old_to_new_.at(n->return_type())),
        down_cast<Expr*>(old_to_new_.at(n->body())), n->tag(), n->is_public());
    new_name_def->set_definer(old_to_new_.at(n));
    return absl::OkStatus();
  }

  absl::Status HandleImport(const Import* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    old_to_new_[n] = module_->Make<Import>(
        n->span(), n->subject(),
        down_cast<NameDef*>(old_to_new_.at(n->name_def())), n->alias());
    return absl::OkStatus();
  }

  absl::Status HandleLet(const Let* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    TypeAnnotation* new_type = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }

    old_to_new_[n] = module_->Make<Let>(
        n->span(), down_cast<NameDefTree*>(old_to_new_.at(n->name_def_tree())),
        new_type, down_cast<Expr*>(old_to_new_.at(n->rhs())),
        down_cast<Expr*>(old_to_new_.at(n->body())), n->is_const());
    return absl::OkStatus();
  }

  absl::Status HandleNameDef(const NameDef* n) override {
    if (!old_to_new_.contains(n)) {
      // We need to set the definer in the definer itself, not here, to avoid
      // looping (Function -> NameDef -> Function -> NameDef -> ...).
      old_to_new_[n] =
          module_->Make<NameDef>(n->span(), n->identifier(), nullptr);
    }

    return absl::OkStatus();
  }

  absl::Status HandleNameDefTree(const NameDefTree* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    if (n->is_leaf()) {
      NameDefTree::Leaf leaf;
      XLS_RETURN_IF_ERROR(std::visit(
          Visitor{
              [&](ColonRef* colon_ref) -> absl::Status {
                leaf = down_cast<ColonRef*>(old_to_new_.at(colon_ref));
                return absl::OkStatus();
              },
              [&](NameDef* name_def) -> absl::Status {
                leaf = down_cast<NameDef*>(old_to_new_.at(name_def));
                return absl::OkStatus();
              },
              [&](NameRef* name_ref) -> absl::Status {
                leaf = down_cast<NameRef*>(old_to_new_.at(name_ref));
                return absl::OkStatus();
              },
              [&](Number* number) -> absl::Status {
                leaf = down_cast<Number*>(old_to_new_.at(number));
                return absl::OkStatus();
              },
              [&](WildcardPattern* wp) -> absl::Status {
                leaf = down_cast<WildcardPattern*>(old_to_new_.at(wp));
                return absl::OkStatus();
              },
          },
          n->leaf()));
      old_to_new_[n] = module_->Make<NameDefTree>(n->span(), leaf);
      return absl::OkStatus();
    }

    NameDefTree::Nodes nodes;
    nodes.reserve(n->nodes().size());
    for (const auto& node : n->nodes()) {
      nodes.push_back(down_cast<NameDefTree*>(old_to_new_.at(node)));
    }
    old_to_new_[n] = module_->Make<NameDefTree>(n->span(), nodes);
    return absl::OkStatus();
  }

  absl::Status HandleNameRef(const NameRef* n) override {
    AnyNameDef any_name_def;
    if (std::holds_alternative<NameDef*>(n->name_def())) {
      auto* name_def = std::get<NameDef*>(n->name_def());
      XLS_RETURN_IF_ERROR(name_def->Accept(this));
      any_name_def = down_cast<NameDef*>(old_to_new_.at(name_def));
    } else {
      auto* builtin_name_def = std::get<BuiltinNameDef*>(n->name_def());
      XLS_RETURN_IF_ERROR(builtin_name_def->Accept(this));
      any_name_def =
          down_cast<BuiltinNameDef*>(old_to_new_.at(builtin_name_def));
    }

    old_to_new_[n] =
        module_->Make<NameRef>(n->span(), n->identifier(), any_name_def);
    return absl::OkStatus();
  }

  absl::Status HandleNumber(const Number* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    TypeAnnotation* new_type = nullptr;
    if (n->type_annotation() != nullptr) {
      new_type =
          down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation()));
    }
    old_to_new_[n] =
        module_->Make<Number>(n->span(), n->text(), n->number_kind(), new_type);
    return absl::OkStatus();
  }

  absl::Status HandleParam(const Param* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));
    old_to_new_[n] = module_->Make<Param>(
        down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())));
    return absl::OkStatus();
  }

  absl::Status HandleParametricBinding(const ParametricBinding* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    Expr* new_expr = nullptr;
    if (n->expr() != nullptr) {
      new_expr = down_cast<Expr*>(old_to_new_.at(n->expr()));
    }
    old_to_new_[n] = module_->Make<ParametricBinding>(
        down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        down_cast<TypeAnnotation*>(old_to_new_.at(n->type_annotation())),
        new_expr);
    return absl::OkStatus();
  }

  absl::Status HandleStructDef(const StructDef* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<ParametricBinding*> new_parametric_bindings;
    new_parametric_bindings.reserve(n->parametric_bindings().size());
    for (const auto* pb : n->parametric_bindings()) {
      new_parametric_bindings.push_back(
          down_cast<ParametricBinding*>(old_to_new_.at(pb)));
    }

    std::vector<std::pair<NameDef*, TypeAnnotation*>> new_members;
    for (const std::pair<NameDef*, TypeAnnotation*>& member : n->members()) {
      new_members.push_back(std::make_pair(
          down_cast<NameDef*>(old_to_new_.at(member.first)),
          down_cast<TypeAnnotation*>(old_to_new_.at(member.second))));
    }

    old_to_new_[n] = module_->Make<StructDef>(
        n->span(), down_cast<NameDef*>(old_to_new_.at(n->name_def())),
        new_parametric_bindings, new_members, n->is_public());
    return absl::OkStatus();
  }

  absl::Status HandleStructInstance(const StructInstance* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    // Have to explicitly visit struct def, since it's not a child.
    StructRef new_struct_ref;
    if (std::holds_alternative<StructDef*>(n->struct_def())) {
      StructDef* old_struct_def = std::get<StructDef*>(n->struct_def());
      XLS_RETURN_IF_ERROR(old_struct_def->Accept(this));
      new_struct_ref = down_cast<StructDef*>(old_to_new_.at(old_struct_def));
    } else {
      ColonRef* old_colon_ref = std::get<ColonRef*>(n->struct_def());
      XLS_RETURN_IF_ERROR(old_colon_ref->Accept(this));
      new_struct_ref = down_cast<ColonRef*>(old_to_new_.at(old_colon_ref));
    }

    absl::Span<const std::pair<std::string, Expr*>> old_members =
        n->GetUnorderedMembers();
    std::vector<std::pair<std::string, Expr*>> new_members;
    new_members.reserve(old_members.size());
    for (const auto& member : old_members) {
      new_members.push_back(std::make_pair(
          member.first, down_cast<Expr*>(old_to_new_.at(member.second))));
    }

    old_to_new_[n] =
        module_->Make<StructInstance>(n->span(), new_struct_ref, new_members);
    return absl::OkStatus();
  }

  absl::Status HandleTupleTypeAnnotation(
      const TupleTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<TypeAnnotation*> new_members;
    new_members.reserve(n->members().size());
    for (const auto* member : n->members()) {
      new_members.push_back(down_cast<TypeAnnotation*>(old_to_new_.at(member)));
    }
    old_to_new_[n] = module_->Make<TupleTypeAnnotation>(n->span(), new_members);
    return absl::OkStatus();
  }

  absl::Status HandleTypeRef(const TypeRef* n) override {
    TypeDefinition new_type_definition;

    // A TypeRef doesn't own its referenced type definition, so we have to
    // explicitly visit it.
    XLS_RETURN_IF_ERROR(std::visit(
        Visitor{[&](ColonRef* colon_ref) -> absl::Status {
                  XLS_RETURN_IF_ERROR(colon_ref->Accept(this));
                  new_type_definition =
                      down_cast<ColonRef*>(old_to_new_.at(colon_ref));
                  return absl::OkStatus();
                },
                [&](EnumDef* enum_def) -> absl::Status {
                  XLS_RETURN_IF_ERROR(enum_def->Accept(this));
                  new_type_definition =
                      down_cast<EnumDef*>(old_to_new_.at(enum_def));
                  return absl::OkStatus();
                },
                [&](StructDef* struct_def) -> absl::Status {
                  XLS_RETURN_IF_ERROR(struct_def->Accept(this));
                  new_type_definition =
                      down_cast<StructDef*>(old_to_new_.at(struct_def));
                  return absl::OkStatus();
                },
                [&](TypeDef* type_def) -> absl::Status {
                  XLS_RETURN_IF_ERROR(type_def->Accept(this));
                  new_type_definition =
                      down_cast<TypeDef*>(old_to_new_.at(type_def));
                  return absl::OkStatus();
                }},
        n->type_definition()));

    old_to_new_[n] =
        module_->Make<TypeRef>(n->span(), n->text(), new_type_definition);
    return absl::OkStatus();
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> new_parametrics;
    new_parametrics.reserve(n->parametrics().size());
    for (const auto* parametric : n->parametrics()) {
      new_parametrics.push_back(down_cast<Expr*>(old_to_new_.at(parametric)));
    }

    old_to_new_[n] = module_->Make<TypeRefTypeAnnotation>(
        n->span(), down_cast<TypeRef*>(n->type_ref()), new_parametrics);
    return absl::OkStatus();
  }

  absl::Status HandleXlsTuple(const XlsTuple* n) override {
    XLS_RETURN_IF_ERROR(VisitChildren(n));

    std::vector<Expr*> members;
    members.reserve(n->members().size());
    for (const auto* member : n->members()) {
      members.push_back(down_cast<Expr*>(old_to_new_.at(member)));
    }

    old_to_new_[n] = module_->Make<XlsTuple>(n->span(), members);
    return absl::OkStatus();
  }

  const absl::flat_hash_map<const AstNode*, AstNode*>& old_to_new() {
    return old_to_new_;
  }

 private:
  // Visits all the children of the given node, skipping those that have
  // already been processed.
  absl::Status VisitChildren(const AstNode* node) {
    for (const auto& child : node->GetChildren(/*want_types=*/true)) {
      if (!old_to_new_.contains(child)) {
        XLS_RETURN_IF_ERROR(child->Accept(this));
      }
    }
    return absl::OkStatus();
  }

  Module* module_;
  absl::flat_hash_map<const AstNode*, AstNode*> old_to_new_;
};

absl::StatusOr<AstNode*> CloneAst(AstNode* root) {
  AstCloner cloner(root->owner());
  XLS_RETURN_IF_ERROR(root->Accept(&cloner));
  return cloner.old_to_new().at(root);
}

}  // namespace xls::dslx
