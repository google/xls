// Copyright 2026 The XLS Authors
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

#include "xls/dslx/frontend/lambda_rewriter.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {
namespace {

class CollectNameRefs : public AstNodeVisitorWithDefault {
 public:
  absl::Status HandleNameRef(const NameRef* node) override {
    if (node->IsBuiltin()) {
      return DefaultHandler(node);
    }
    if (node->GetDefiner() == nullptr ||
        (node->GetDefiner()->kind() != AstNodeKind::kFunction &&
         node->GetDefiner()->kind() != AstNodeKind::kImport &&
         node->GetDefiner()->kind() != AstNodeKind::kConstantDef)) {
      XLS_RETURN_IF_ERROR(AddNameRef(node));
      if (node->GetDefiner() != nullptr && in_type_annotation_) {
        XLS_RETURN_IF_ERROR(node->GetDefiner()->Accept(this));
        XLS_RETURN_IF_ERROR(DefaultHandler(node->GetDefiner()));
      }
    }
    return DefaultHandler(node);
  }

  // ColonRefs refer to names in other namespaces (e.g., modules, struct
  // members) and are not what this visitor is trying to collect, which is
  // references to locally-defined NameDefs.
  absl::Status HandleColonRef(const ColonRef* node) override {
    return absl::OkStatus();
  }

  absl::Status HandleTypeRefTypeAnnotation(
      const TypeRefTypeAnnotation* node) override {
    AnyNameDef name_def =
        TypeDefinitionGetNameDef(node->type_ref()->type_definition());
    if (std::holds_alternative<const NameDef*>(name_def)) {
      const NameDef* nd = std::get<const NameDef*>(name_def);
      type_refs_[nd].emplace(node);
    }
    return DefaultHandler(node);
  }

  absl::Status HandleStructInstance(const StructInstance* node) override {
    XLS_RETURN_IF_ERROR(node->struct_ref()->Accept(this));
    return DefaultHandler(node);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    bool prev_in_type_annotation = in_type_annotation_;
    if (node->kind() == AstNodeKind::kTypeAnnotation) {
      in_type_annotation_ = true;
      const auto* type_annotation =
          absl::down_cast<const TypeAnnotation*>(node);
      if (type_annotation->IsAnnotation<TypeVariableTypeAnnotation>()) {
        const auto* tvta =
            type_annotation->AsAnnotation<TypeVariableTypeAnnotation>();
        XLS_RETURN_IF_ERROR(tvta->type_variable()->Accept(this));
      }
    }
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      if (child->owner() == node->owner()) {
        XLS_RETURN_IF_ERROR(child->Accept(this));
      }
    }
    in_type_annotation_ = prev_in_type_annotation;
    return absl::OkStatus();
  }

  const absl::flat_hash_set<const NameRef*>& NameRefsForDef(
      const NameDef* name_def) {
    static const absl::flat_hash_set<const NameRef*> empty_set = {};
    auto it = name_ref_info_.find(name_def);
    if (it == name_ref_info_.end()) {
      return empty_set;
    }
    return it->second.name_refs;
  }

  absl::flat_hash_set<const NameDef*> NameDefsDefinedPrior(
      const Pos start) const {
    absl::flat_hash_set<const NameDef*> result;
    for (const auto& [name_def, info] : name_ref_info_) {
      if (!info.any_used_in_type_annotation &&
          name_def->span().start() < start) {
        result.insert(name_def);
      }
    }
    return result;
  }

  absl::flat_hash_map<const NameDef*,
                      absl::flat_hash_set<const TypeRefTypeAnnotation*>>
  TypesDefinedPrior(const Pos start) const {
    absl::flat_hash_map<const NameDef*,
                        absl::flat_hash_set<const TypeRefTypeAnnotation*>>
        result;
    for (const auto& [name_def, type_refs] : type_refs_) {
      // Don't include top-level type definitions.
      std::optional<const ModuleMember*> module_member =
          name_def->owner()->FindMemberWithName(name_def->identifier());
      if (!module_member.has_value() && name_def->span().start() < start) {
        result.emplace(name_def, type_refs);
      }
    }
    return result;
  }

 private:
  struct NameRefInfo {
    bool any_used_in_type_annotation;
    absl::flat_hash_set<const NameRef*> name_refs;
  };

  absl::Status AddNameRef(const NameRef* name_ref) {
    const NameDef* nd = std::get<const NameDef*>(name_ref->name_def());
    XLS_RET_CHECK(nd != nullptr);
    if (name_ref_info_.contains(nd)) {
      NameRefInfo& info = name_ref_info_[nd];
      info.name_refs.insert(name_ref);
      if (in_type_annotation_) {
        info.any_used_in_type_annotation = true;
      }
    } else {
      NameRefInfo info = NameRefInfo{in_type_annotation_, {name_ref}};
      name_ref_info_.emplace(nd, info);
    }
    return absl::OkStatus();
  }

  absl::flat_hash_map<const NameDef*, NameRefInfo> name_ref_info_;
  absl::flat_hash_map<const NameDef*,
                      absl::flat_hash_set<const TypeRefTypeAnnotation*>>
      type_refs_;
  bool in_type_annotation_ = false;
};

class LambdaRewriter : public AstNodeVisitorWithDefault {
 public:
  explicit LambdaRewriter(const FileTable& file_table)
      : file_table_(file_table) {}

  absl::Status HandleLambda(const Lambda* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    Module* module = node->owner();
    Function* original_fn = node->function();
    Span span = node->span();
    CollectNameRefs collect_nr;
    XLS_RETURN_IF_ERROR(node->Accept(&collect_nr));

    // Parametric bindings for the struct definition.
    std::vector<ParametricBinding*> struct_parametric_bindings;
    // Parametrics in the struct type annotation.
    std::vector<ExprOrType> struct_type_parametrics;
    // Parametric values for the struct instantiation.
    std::vector<ExprOrType> struct_instance_parametrics;
    // NameDefs that have been added to the struct parametric bindings.
    absl::flat_hash_set<const NameDef*> parametric_nds;
    absl::flat_hash_map<const AstNode*, AstNode*> node_replacements;

    // If there are any parametric bindings in the containing function that are
    // referenced in the lambda, they should be added as parametric bindings to
    // the `StructDef`.
    std::optional<const Function*> containing_fn = GetContainingFunction(node);
    if (containing_fn.has_value()) {
      for (ParametricBinding* parent_binding :
           (*containing_fn)->parametric_bindings()) {
        absl::flat_hash_set<const NameRef*> name_refs =
            collect_nr.NameRefsForDef(parent_binding->name_def());
        if (name_refs.empty()) {
          continue;
        }
        XLS_RETURN_IF_ERROR(AddBindingForParentParametric(
            module, parent_binding, name_refs, struct_parametric_bindings,
            struct_type_parametrics, struct_instance_parametrics,
            parametric_nds, node_replacements));
      }
    }

    for (const auto& [original_nd, trtas] :
         collect_nr.TypesDefinedPrior(span.start())) {
      if (!parametric_nds.contains(original_nd)) {
        XLS_RETURN_IF_ERROR(ReplaceTypeRefTypeAnnotations(
            module, original_nd, trtas, struct_parametric_bindings,
            struct_type_parametrics, struct_instance_parametrics,
            parametric_nds, node_replacements));
      }
    }

    // For any NameDef that is referenced in the lambda, but defined prior to
    // the lambda, it must be captured in the struct instance, unless it was
    // already added as a parametric binding.
    std::vector<StructMemberNode*> struct_members;
    std::vector<std::pair<std::string, Expr*>> struct_instance_members;
    absl::flat_hash_set<const NameDef*> seen;
    for (const NameDef* original_name_def :
         collect_nr.NameDefsDefinedPrior(span.start())) {
      if (!parametric_nds.contains(original_name_def)) {
        AddCapture(module, original_name_def, struct_parametric_bindings,
                   struct_type_parametrics, struct_members,
                   struct_instance_members, seen);
      }
    }

    NameDef* struct_nd =
        module->Make<NameDef>(span,
                              absl::Substitute("lambda_capture_struct_at_$0",
                                               span.ToString(file_table_)),
                              /*definer=*/nullptr);
    StructDef* full_struct_def =
        module->Make<StructDef>(span, struct_nd, struct_parametric_bindings,
                                struct_members, /*is_public=*/false);
    TypeRefTypeAnnotation* struct_type_annotation =
        module->Make<TypeRefTypeAnnotation>(
            span, module->Make<TypeRef>(span, full_struct_def),
            struct_type_parametrics);
    struct_nd->set_definer(full_struct_def);

    TypeRefTypeAnnotation* struct_instance_annotation =
        module->Make<TypeRefTypeAnnotation>(
            span, module->Make<TypeRef>(span, full_struct_def),
            struct_instance_parametrics);
    StructInstance* struct_instance = module->Make<StructInstance>(
        span, struct_instance_annotation, struct_instance_members);

    Attr* instance_invocation = module->Make<Attr>(
        span, struct_instance, std::string(Lambda::kCallLambdaFn));

    // For every NameRef in the body, if it references a NameDef that has been
    // captured, replace it with a reference to the struct member.
    NameDef* self_nd = module->Make<NameDef>(
        span, KeywordToString(Keyword::kSelf), /*definer=*/nullptr);
    CloneReplacer insert_self =
        [self_nd, seen, node_replacements](
            const AstNode* node, const Module*,
            const absl::flat_hash_map<const AstNode*, AstNode*>&)
        -> std::optional<AstNode*> {
      if (node->kind() == AstNodeKind::kNameRef) {
        const NameRef* name_ref = absl::down_cast<const NameRef*>(node);
        if (name_ref->IsBuiltin()) {
          return std::nullopt;
        }
        const auto* name_def = std::get<const NameDef*>(name_ref->name_def());
        if (name_def != nullptr && seen.contains(name_def)) {
          NameRef* self_nr = node->owner()->Make<NameRef>(
              name_ref->span(), self_nd->identifier(), self_nd);
          return node->owner()->Make<Attr>(name_def->span(), self_nr,
                                           name_def->identifier(),
                                           /* in_parens= */ false);
        }
      }
      if (node_replacements.contains(node)) {
        return node_replacements.at(node);
      }
      return std::nullopt;
    };
    CloneReplacer swap_nodes =
        [node_replacements](
            const AstNode* node, const Module*,
            const absl::flat_hash_map<const AstNode*, AstNode*>&)
        -> std::optional<AstNode*> {
      if (node_replacements.contains(node)) {
        return node_replacements.at(node);
      }
      return std::nullopt;
    };
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_body,
        CloneAst(original_fn->body(),
                 ChainCloneReplacers(&PreserveTypeDefinitionsReplacer,
                                     std::move(insert_self))));
    XLS_ASSIGN_OR_RETURN(
        AstNode * cloned_return_type,
        CloneAst(original_fn->return_type(),
                 ChainCloneReplacers(&PreserveTypeDefinitionsReplacer,
                                     std::move(swap_nodes))));
    SelfTypeAnnotation* self_type = module->Make<SelfTypeAnnotation>(
        span, /*explicit_type=*/false, struct_type_annotation);
    std::vector<Param*> params = {module->Make<Param>(self_nd, self_type)};
    for (auto* param : original_fn->params()) {
      params.push_back(param);
    }
    Function* impl_fn = module->Make<Function>(
        original_fn->span(), original_fn->name_def(),
        original_fn->parametric_bindings(), params,
        absl::down_cast<TypeAnnotation*>(cloned_return_type),
        absl::down_cast<StatementBlock*>(cloned_body),
        FunctionTag::kGeneratedFromLambda,
        /*is_public=*/false, /*is_stub=*/false);
    Impl* impl = module->Make<Impl>(span, struct_type_annotation,
                                    std::vector<ImplMember>{impl_fn},
                                    /*is_public=*/false);
    impl_fn->set_impl(impl);
    full_struct_def->set_impl(impl);

    // Swap the Lambda in its parent with the attr invocation. After this step,
    // Lambdas should no longer appear in the AST.
    std::optional<ModuleMember> containing_member =
        GetContainingModuleMember(node);
    XLS_RET_CHECK(containing_member.has_value());
    auto* parent_inv = absl::down_cast<Invocation*>(node->parent());
    XLS_RET_CHECK(parent_inv != nullptr);
    if (parent_inv->callee() == node) {
      parent_inv->set_callee(instance_invocation);
    } else {
      for (int i = 0; i < parent_inv->args().size(); ++i) {
        if (parent_inv->args()[i] == node) {
          parent_inv->set_arg(i, instance_invocation);
        }
      }
    }

    XLS_RETURN_IF_ERROR(module->InsertTopBefore(ToAstNode(*containing_member),
                                                full_struct_def));
    return module->InsertTopAfter(full_struct_def, impl);
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    for (const AstNode* child : node->GetChildren(/*want_types=*/false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

 private:
  absl::Status AddBindingForParentParametric(
      Module* module, const ParametricBinding* parent_binding,
      absl::flat_hash_set<const NameRef*> name_refs,
      std::vector<ParametricBinding*>& struct_parametric_bindings,
      std::vector<ExprOrType>& struct_type_parametrics,
      std::vector<ExprOrType>& struct_instance_parametrics,
      absl::flat_hash_set<const NameDef*>& parametric_nds,
      absl::flat_hash_map<const AstNode*, AstNode*>& node_replacements) {
    NameDef* lambda_struct_nd = module->Make<NameDef>(
        parent_binding->span(), parent_binding->identifier() + "_ls",
        /*definer=*/nullptr);
    XLS_ASSIGN_OR_RETURN(AstNode * cloned_ta,
                         CloneAst(parent_binding->type_annotation()));

    AstNode* cloned_expr = nullptr;
    if (parent_binding->expr() != nullptr) {
      XLS_ASSIGN_OR_RETURN(cloned_expr, CloneAst(parent_binding->expr()));
    }
    ParametricBinding* lambda_struct_binding = module->Make<ParametricBinding>(
        lambda_struct_nd, absl::down_cast<TypeAnnotation*>(cloned_ta),
        absl::down_cast<Expr*>(cloned_expr));
    struct_parametric_bindings.push_back(lambda_struct_binding);
    NameRef* struct_type_parametric_nr =
        module->Make<NameRef>(parent_binding->span(),
                              lambda_struct_nd->identifier(), lambda_struct_nd);
    NameRef* struct_instance_parametric_nr = module->Make<NameRef>(
        parent_binding->span(), parent_binding->identifier(),
        parent_binding->name_def());
    struct_type_parametrics.push_back(struct_type_parametric_nr);
    if (parent_binding->type_annotation()
            ->IsAnnotation<GenericTypeAnnotation>()) {
      struct_instance_parametrics.push_back(
          module->Make<TypeVariableTypeAnnotation>(
              struct_instance_parametric_nr));
    } else {
      struct_instance_parametrics.push_back(struct_instance_parametric_nr);
    }
    parametric_nds.insert(parent_binding->name_def());
    for (const NameRef* original_name_ref : name_refs) {
      node_replacements.emplace(
          original_name_ref,
          module->Make<NameRef>(original_name_ref->span(),
                                lambda_struct_nd->identifier(),
                                lambda_struct_nd));
    }
    return absl::OkStatus();
  }

  absl::Status ReplaceTypeRefTypeAnnotations(
      Module* module, const NameDef* original_nd,
      absl::flat_hash_set<const TypeRefTypeAnnotation*> trtas,
      std::vector<ParametricBinding*>& struct_parametric_bindings,
      std::vector<ExprOrType>& struct_type_parametrics,
      std::vector<ExprOrType>& struct_instance_parametrics,
      absl::flat_hash_set<const NameDef*>& parametric_nds,
      absl::flat_hash_map<const AstNode*, AstNode*>& node_replacements) {
    NameDef* lambda_struct_nd = module->Make<NameDef>(
        original_nd->span(),
        absl::Substitute("$0_ls", original_nd->identifier()),
        /*definer=*/nullptr);
    ParametricBinding* lambda_struct_binding = module->Make<ParametricBinding>(
        lambda_struct_nd, module->Make<GenericTypeAnnotation>(Span::None()),
        /*expr=*/nullptr);
    struct_parametric_bindings.push_back(lambda_struct_binding);
    NameRef* struct_type_parametric_nr = module->Make<NameRef>(
        original_nd->span(), lambda_struct_nd->identifier(), lambda_struct_nd);
    struct_type_parametrics.push_back(struct_type_parametric_nr);

    XLS_ASSIGN_OR_RETURN(TypeDefinition type_def,
                         ToTypeDefinition(original_nd->definer()));
    TypeRef* lambda_type_ref =
        module->Make<TypeRef>(original_nd->span(), type_def);

    struct_instance_parametrics.push_back(module->Make<TypeRefTypeAnnotation>(
        original_nd->span(), lambda_type_ref, std::vector<ExprOrType>{}));
    parametric_nds.insert(original_nd);
    for (const TypeRefTypeAnnotation* original_type_ref : trtas) {
      node_replacements.emplace(
          original_type_ref,
          module->Make<TypeVariableTypeAnnotation>(
              module->Make<NameRef>(original_type_ref->span(),
                                    lambda_struct_nd->identifier(),
                                    lambda_struct_nd),
              /*internal=*/true));
    }
    return absl::OkStatus();
  }

  void AddCapture(
      Module* module, const NameDef* original_name_def,
      std::vector<ParametricBinding*>& struct_parametric_bindings,
      std::vector<ExprOrType>& struct_type_parametrics,
      std::vector<StructMemberNode*>& struct_members,
      std::vector<std::pair<std::string, Expr*>>& struct_instance_members,
      absl::flat_hash_set<const NameDef*>& seen) {
    // Create parametric binding with generic type to use for the context
    // variable type.
    GenericTypeAnnotation* gta =
        module->Make<GenericTypeAnnotation>(original_name_def->span());
    NameDef* generic_name_def =
        module->Make<NameDef>(original_name_def->span(),
                              absl::Substitute("parametric_type_for_$0",
                                               original_name_def->identifier()),
                              /*definer=*/gta);
    NameRef* generic_name_ref =
        module->Make<NameRef>(original_name_def->span(),
                              generic_name_def->identifier(), generic_name_def);
    struct_type_parametrics.push_back(generic_name_ref);
    struct_parametric_bindings.push_back(module->Make<ParametricBinding>(
        generic_name_def, gta, /*expr=*/nullptr));

    NameDef* struct_member_nd = module->Make<NameDef>(
        original_name_def->span(), original_name_def->identifier(),
        /*definer=*/nullptr);
    TypeVariableTypeAnnotation* tvta =
        module->Make<TypeVariableTypeAnnotation>(generic_name_ref,
                                                 /*internal=*/true);
    StructMemberNode* struct_member = module->Make<StructMemberNode>(
        Span::None(), struct_member_nd, Span::None(), tvta);
    struct_members.push_back(struct_member);

    // Make a name ref that points to the original name def. Add as a member
    // to a new struct instance.
    NameRef* struct_instance_nr = module->Make<NameRef>(
        original_name_def->span(), original_name_def->identifier(),
        original_name_def);
    struct_instance_members.push_back(
        std::make_pair(original_name_def->identifier(), struct_instance_nr));
    seen.insert(original_name_def);
  }

  const FileTable& file_table_;
};

}  // namespace

absl::Status RewriteLambdas(Module& module, const FileTable& file_table) {
  LambdaRewriter visitor(file_table);
  return module.Accept(&visitor);
}

}  // namespace xls::dslx
