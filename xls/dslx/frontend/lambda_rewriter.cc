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

#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/import_utils.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

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

  absl::flat_hash_set<const NameDef*> ConstLetsDefinedPrior(
      const Pos start) const {
    return NameDefsDefinedPriorInternal(start, is_const);
  }

  absl::flat_hash_set<const NameDef*> NameDefsDefinedPrior(
      const Pos start) const {
    return NameDefsDefinedPriorInternal(
        start, [](const NameDef* nd) { return !is_const(nd); });
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

  static bool is_const(const NameDef* name_def) {
    if (name_def->definer() == nullptr ||
        name_def->definer()->kind() != AstNodeKind::kLet) {
      return false;
    }
    const auto* let = absl::down_cast<const Let*>(name_def->definer());
    return let->is_const();
  }

  absl::flat_hash_set<const NameDef*> NameDefsDefinedPriorInternal(
      const Pos start,
      std::function<bool(const NameDef*)> name_def_filter) const {
    absl::flat_hash_set<const NameDef*> result;
    for (const auto& [name_def, info] : name_ref_info_) {
      if (!info.any_used_in_type_annotation &&
          name_def->span().start() < start) {
        if (name_def_filter(name_def)) {
          result.insert(name_def);
        }
      }
    }
    return result;
  }

  absl::flat_hash_map<const NameDef*, NameRefInfo> name_ref_info_;
  absl::flat_hash_map<const NameDef*,
                      absl::flat_hash_set<const TypeRefTypeAnnotation*>>
      type_refs_;
  bool in_type_annotation_ = false;
};

// A helper class to manage the bindings for a lambda struct definition, its
// type ref, and the struct instance.
class LambdaStructBindings {
 public:
  void AddBinding(ParametricBinding* struct_definition_parametric,
                  ExprOrType struct_type_parametric,
                  ExprOrType struct_instance_parametric) {
    if (ToAstNode(struct_instance_parametric) == nullptr) {
      return AddBinding(struct_definition_parametric, struct_type_parametric);
    }
    fully_defined_bindings_.push_back({struct_definition_parametric,
                                       struct_type_parametric,
                                       struct_instance_parametric});
  }
  void AddBinding(ParametricBinding* struct_definition_parametric,
                  ExprOrType struct_type_parametric) {
    bindings_without_instance_.push_back(
        {struct_definition_parametric, struct_type_parametric, std::nullopt});
  }

  std::vector<ParametricBinding*> StructDefBindings() const {
    std::vector<ParametricBinding*> bindings;
    bindings.reserve(TotalSize());
    for (const auto& binding : OrderedBindings()) {
      bindings.push_back(binding.struct_definition_parametric);
    }
    return bindings;
  }

  std::vector<ExprOrType> TypeRefBindings() const {
    std::vector<ExprOrType> bindings;
    bindings.reserve(TotalSize());
    for (const auto& binding : OrderedBindings()) {
      bindings.push_back(binding.struct_type_parametric);
    }
    return bindings;
  }

  std::vector<ExprOrType> StructInstanceBindings() const {
    std::vector<ExprOrType> bindings;
    bindings.reserve(fully_defined_bindings_.size());
    for (const auto& binding : fully_defined_bindings_) {
      bindings.push_back(*binding.struct_instance_parametric);
    }
    return bindings;
  }

 private:
  struct StructParametricBindingSet {
    // The parametric binding for the struct definition.
    ParametricBinding* struct_definition_parametric;

    // The corresponding expression in the struct type annotation.
    ExprOrType struct_type_parametric;

    // The expr or type used in the struct instance.
    std::optional<ExprOrType> struct_instance_parametric;
  };

  int TotalSize() const {
    return fully_defined_bindings_.size() + bindings_without_instance_.size();
  }

  std::vector<StructParametricBindingSet> OrderedBindings() const {
    std::vector<StructParametricBindingSet> results;
    results.reserve(fully_defined_bindings_.size() +
                    bindings_without_instance_.size());
    absl::c_copy(fully_defined_bindings_, std::back_inserter(results));
    absl::c_copy(bindings_without_instance_, std::back_inserter(results));

    return results;
  }

  std::vector<StructParametricBindingSet> fully_defined_bindings_;
  std::vector<StructParametricBindingSet> bindings_without_instance_;
};

class LambdaRewriter : public AstNodeRecursiveVisitor {
 public:
  explicit LambdaRewriter(const ImportData& import_data)
      : import_data_(import_data) {}

  absl::Status HandleLambda(const Lambda* node) override {
    XLS_RETURN_IF_ERROR(DefaultHandler(node));
    Module* module = node->owner();
    Function* original_fn = node->function();
    Span span = node->span();
    CollectNameRefs collect_nr;
    XLS_RETURN_IF_ERROR(node->Accept(&collect_nr));

    // Parametric bindings for the lambda struct, impl, and struct instance.
    LambdaStructBindings bindings;
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
            module, parent_binding, name_refs, &bindings, parametric_nds,
            node_replacements));
      }
    }

    for (const auto& [original_nd, trtas] :
         collect_nr.TypesDefinedPrior(span.start())) {
      if (!parametric_nds.contains(original_nd)) {
        XLS_RETURN_IF_ERROR(
            ReplaceTypeRefTypeAnnotations(module, original_nd, trtas, &bindings,
                                          parametric_nds, node_replacements));
      }
    }

    for (const NameDef* original_name_def :
         collect_nr.ConstLetsDefinedPrior(span.start())) {
      absl::flat_hash_set<const NameRef*> name_refs =
          collect_nr.NameRefsForDef(original_name_def);
      if (name_refs.empty()) {
        continue;
      }
      if (!parametric_nds.contains(original_name_def)) {
        AddConstantCapture(module, original_name_def, &bindings, name_refs,
                           node_replacements);
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
        AddCapture(module, original_name_def, &bindings, struct_members,
                   struct_instance_members, seen);
      }
    }

    NameDef* struct_nd = module->Make<NameDef>(
        span,
        absl::Substitute("lambda_capture_struct_at_$0",
                         span.ToString(import_data_.file_table())),
        /*definer=*/nullptr);
    StructDef* full_struct_def =
        module->Make<StructDef>(span, struct_nd, bindings.StructDefBindings(),
                                struct_members, /*is_public=*/false);
    TypeRefTypeAnnotation* struct_type_annotation =
        module->Make<TypeRefTypeAnnotation>(
            span, module->Make<TypeRef>(span, full_struct_def),
            bindings.TypeRefBindings());
    struct_nd->set_definer(full_struct_def);

    TypeRefTypeAnnotation* struct_instance_annotation =
        module->Make<TypeRefTypeAnnotation>(
            span, module->Make<TypeRef>(span, full_struct_def),
            bindings.StructInstanceBindings());
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
    auto* parent_inv = dynamic_cast<Invocation*>(node->parent());
    if (parent_inv == nullptr) {
      return TypeInferenceErrorStatus(
          node->span(), nullptr,
          "Lambdas are currently only supported as arguments to invocations.",
          import_data_.file_table());
    }
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

 private:
  absl::Status AddBindingForParentParametric(
      Module* module, const ParametricBinding* parent_binding,
      absl::flat_hash_set<const NameRef*> name_refs,
      LambdaStructBindings* bindings,
      absl::flat_hash_set<const NameDef*>& parametric_nds,
      absl::flat_hash_map<const AstNode*, AstNode*>& node_replacements) {
    NameDef* lambda_struct_nd = module->Make<NameDef>(
        parent_binding->span(), parent_binding->identifier() + "_ls",
        parent_binding->name_def()->definer());
    XLS_ASSIGN_OR_RETURN(AstNode * cloned_ta,
                         CloneAst(parent_binding->type_annotation()));

    std::optional<ExprOrType> cloned_default_expr_or_type;
    if (parent_binding->default_expr_or_type().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          AstNode * cloned_node,
          CloneAst(ToAstNode(*parent_binding->default_expr_or_type())));
      cloned_default_expr_or_type = ToExprOrType(cloned_node);
    }
    ParametricBinding* lambda_struct_binding = module->Make<ParametricBinding>(
        lambda_struct_nd, absl::down_cast<TypeAnnotation*>(cloned_ta),
        cloned_default_expr_or_type);
    NameRef* struct_type_parametric_nr =
        module->Make<NameRef>(parent_binding->span(),
                              lambda_struct_nd->identifier(), lambda_struct_nd);
    NameRef* struct_instance_parametric_nr = module->Make<NameRef>(
        parent_binding->span(), parent_binding->identifier(),
        parent_binding->name_def());
    ExprOrType instance_parametric = struct_instance_parametric_nr;
    if (parent_binding->type_annotation()
            ->IsAnnotation<GenericTypeAnnotation>()) {
      instance_parametric = module->Make<TypeVariableTypeAnnotation>(
          struct_instance_parametric_nr);
    }
    bindings->AddBinding(lambda_struct_binding, struct_type_parametric_nr,
                         instance_parametric);
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
      LambdaStructBindings* bindings,
      absl::flat_hash_set<const NameDef*>& parametric_nds,
      absl::flat_hash_map<const AstNode*, AstNode*>& node_replacements) {
    NameDef* lambda_struct_nd = module->Make<NameDef>(
        original_nd->span(),
        absl::Substitute("$0_ls", original_nd->identifier()),
        original_nd->definer());
    ParametricBinding* lambda_struct_binding = module->Make<ParametricBinding>(
        lambda_struct_nd, module->Make<GenericTypeAnnotation>(Span::None()),
        /*default_expr_or_type=*/std::nullopt);
    NameRef* struct_type_parametric_nr = module->Make<NameRef>(
        original_nd->span(), lambda_struct_nd->identifier(), lambda_struct_nd);

    XLS_ASSIGN_OR_RETURN(TypeDefinition type_def,
                         ToTypeDefinition(original_nd->definer()));
    TypeRef* instance_type_ref =
        module->Make<TypeRef>(original_nd->span(), type_def);
    bindings->AddBinding(
        lambda_struct_binding, struct_type_parametric_nr,
        module->Make<TypeRefTypeAnnotation>(
            original_nd->span(), instance_type_ref, std::vector<ExprOrType>{}));

    parametric_nds.insert(original_nd);

    TypeRef* lambda_type_ref = nullptr;
    for (const TypeRefTypeAnnotation* original_type_ref : trtas) {
      XLS_ASSIGN_OR_RETURN(std::optional<StructOrProcRef> struct_or_proc_ref,
                           GetStructOrProcRef(original_type_ref, import_data_));
      if (lambda_type_ref == nullptr) {
        lambda_type_ref = module->Make<TypeRef>(original_nd->span(), type_def);
      }
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

  void AddConstantCapture(
      Module* module, const NameDef* original_nd,
      LambdaStructBindings* bindings,
      absl::flat_hash_set<const NameRef*> name_refs,
      absl::flat_hash_map<const AstNode*, AstNode*>& node_replacements) {
    // Generic type parametric for the constant type definition.
    GenericTypeAnnotation* gta =
        module->Make<GenericTypeAnnotation>(original_nd->span());
    NameDef* generic_name_def = module->Make<NameDef>(
        original_nd->span(),
        absl::Substitute("parametric_type_for_$0", original_nd->identifier()),
        /*definer=*/gta);

    const Let* original_let =
        absl::down_cast<const Let*>(original_nd->definer());
    TypeAnnotation* instance_annotation = original_let->type_annotation();

    // Binding for type of constant.
    bindings->AddBinding(
        module->Make<ParametricBinding>(generic_name_def, gta,
                                        /*default_expr_or_type=*/std::nullopt),
        module->Make<NameRef>(original_nd->span(),
                              generic_name_def->identifier(), generic_name_def),
        instance_annotation);

    // Binding for value of constant.
    NameDef* value_name_def = module->Make<NameDef>(
        original_nd->span(),
        absl::Substitute("$0_lm", original_nd->identifier()),
        /*definer=*/nullptr);
    TypeVariableTypeAnnotation* tvta = module->Make<TypeVariableTypeAnnotation>(
        module->Make<NameRef>(original_nd->span(),
                              generic_name_def->identifier(), generic_name_def),
        /*internal=*/true);
    bindings->AddBinding(
        module->Make<ParametricBinding>(value_name_def, tvta,
                                        /*default_expr_or_type=*/std::nullopt),
        module->Make<NameRef>(original_nd->span(), value_name_def->identifier(),
                              value_name_def),
        module->Make<NameRef>(original_nd->span(), original_nd->identifier(),
                              original_nd));

    for (const NameRef* name_ref : name_refs) {
      node_replacements.emplace(
          name_ref,
          module->Make<NameRef>(name_ref->span(), value_name_def->identifier(),
                                value_name_def));
    }
  }

  void AddCapture(
      Module* module, const NameDef* original_name_def,
      LambdaStructBindings* bindings,
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
    bindings->AddBinding(
        module->Make<ParametricBinding>(generic_name_def, gta,
                                        /*default_expr_or_type=*/std::nullopt),
        generic_name_ref);

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

  const ImportData& import_data_;
};

}  // namespace

absl::Status RewriteLambdas(Module& module, const ImportData& import_data) {
  LambdaRewriter visitor(import_data);
  return module.Accept(&visitor);
}

}  // namespace xls::dslx
