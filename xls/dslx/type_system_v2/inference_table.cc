// Copyright 2024 The XLS Authors
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

#include "xls/dslx/type_system_v2/inference_table.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {

std::vector<std::pair<uint8_t, std::string>>* TypeInferenceFlag::flag_names_ =
    new std::vector<std::pair<uint8_t, std::string>>();

const TypeInferenceFlag TypeInferenceFlag::kNone(0, "none");
const TypeInferenceFlag TypeInferenceFlag::kMinSize(1, "min-size");
const TypeInferenceFlag TypeInferenceFlag::kStandardType(1 << 1,
                                                         "standard-type");
const TypeInferenceFlag TypeInferenceFlag::kHasPrefix(1 << 2, "has-prefix");
const TypeInferenceFlag TypeInferenceFlag::kSliceContainerSize(
    1 << 3, "slice-container-size");
const TypeInferenceFlag TypeInferenceFlag::kBitsLikeType(1 << 4,
                                                         "bits-like-type");
const TypeInferenceFlag TypeInferenceFlag::kFormalMemberType(
    1 << 5, "formal-member-type");

namespace {

// Converts an `InferenceVariableKind` to string for tracing purposes.
std::string_view InferenceVariableKindToString(InferenceVariableKind kind) {
  switch (kind) {
    case InferenceVariableKind::kType:
      return "type";
    case InferenceVariableKind::kInteger:
      return "int";
    case InferenceVariableKind::kBool:
      return "bool";
  }
}

// Converts a `TypeAnnotation` for a parametric to an `InferenceVariableKind`.
absl::StatusOr<InferenceVariableKind> TypeAnnotationToInferenceVariableKind(
    const TypeAnnotation* annotation) {
  if (annotation->IsAnnotation<BuiltinTypeAnnotation>()) {
    const auto* builtin = annotation->AsAnnotation<BuiltinTypeAnnotation>();
    switch (builtin->builtin_type()) {
      case BuiltinType::kBool:
        return InferenceVariableKind::kBool;
      case BuiltinType::kChannelIn:
      case BuiltinType::kChannelOut:
      case BuiltinType::kToken:
        break;
      default:
        return InferenceVariableKind::kInteger;
    }
  }
  if (annotation->IsAnnotation<GenericTypeAnnotation>()) {
    return InferenceVariableKind::kType;
  }
  if (GetSignednessAndBitCount(annotation).ok() ||
      annotation->IsAnnotation<TypeRefTypeAnnotation>()) {
    // Note that a type ref encountered here must currently must be an enum or
    // other integral-type alias. This is not practical to verify here, but if a
    // struct is used then it will fail in the parser.
    return InferenceVariableKind::kInteger;
  }
  return absl::InvalidArgumentError(
      absl::Substitute("Inference variables of type $0 are not supported.",
                       annotation->ToString()));
}

std::string ValueOrExprToString(
    std::variant<int64_t, const Expr*> expr_or_value) {
  return std::holds_alternative<int64_t>(expr_or_value)
             ? absl::StrCat(std::get<int64_t>(expr_or_value))
             : std::get<const Expr*>(expr_or_value)->ToString();
}

// Represents the immutable metadata for a variable in an `InferenceTable`.
class InferenceVariable {
 public:
  InferenceVariable(const AstNode* definer, const NameRef* name_ref,
                    InferenceVariableKind kind)
      : definer_(definer), name_ref_(name_ref), kind_(kind) {}

  const AstNode* definer() const { return definer_; }

  std::string_view name() const { return name_ref_->identifier(); }

  InferenceVariableKind kind() const { return kind_; }

  // Returns the `NameRef` dealt out for this variable at creation time. This
  // is used to avoid spurious creations of additional refs by table functions
  // that look up variables. However, any `NameRef` to the variable's `NameDef`
  // is equally usable.
  const NameRef* name_ref() const { return name_ref_; }

  template <typename H>
  friend H AbslHashValue(H h, const InferenceVariable& v) {
    return H::combine(std::move(h), v.definer_, v.name(), v.kind_);
  }

  std::string ToString() const {
    return absl::Substitute("InferenceVariable(name=$0, kind=$1, definer=$2)",
                            name(), InferenceVariableKindToString(kind_),
                            definer_->ToString());
  }

 private:
  const AstNode* const definer_;
  const NameRef* const name_ref_;
  const InferenceVariableKind kind_;
};

// The mutable data for a node in an `InferenceTable`.
struct NodeData {
  size_t order_added;
  std::optional<const TypeAnnotation*> type_annotation;
  std::optional<const InferenceVariable*> type_variable;
  std::optional<StartAndWidthExprs> slice_start_and_width_exprs;
};

// The mutable data for a `ParametricContext` in an `InferenceTable`.
struct MutableParametricContextData {
  // An arbitrarily-selected context that has the same `ParametricEnv` as this
  // one. This is only set in cases where "duplicate" contexts are identified by
  // the table.
  std::optional<const ParametricContext*> canonical_context;

  absl::flat_hash_map<const InferenceVariable*, ParametricContextScopedExpr>
      parametric_values;
  absl::flat_hash_map<const InferenceVariable*,
                      std::vector<const TypeAnnotation*>>
      type_annotations_per_type_variable;
  absl::flat_hash_map<const Invocation*, const Function*> callees;
};

// An auxiliary data structure for holding cached unification data associated
// with an `InferenceTable`.
class UnificationCache {
 public:
  void SetUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable,
      const absl::flat_hash_set<const NameRef*>&
          transitive_variable_dependencies,
      const TypeAnnotation* unified_type) {
    const NameDef* variable_name_def =
        std::get<const NameDef*>(variable->name_def());
    if (!parametric_context.has_value()) {
      cache_[variable_name_def].cached_type = unified_type;
    } else {
      cache_[variable_name_def]
          .cached_type_per_parametric_context[*parametric_context] =
          unified_type;
    }
    for (const NameRef* dep : transitive_variable_dependencies) {
      const NameDef* dep_def = std::get<const NameDef*>(dep->name_def());
      transitive_consumers_[dep_def].insert(variable_name_def);
    }
  }

  std::optional<const TypeAnnotation*> GetUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) {
    const auto it = cache_.find(std::get<const NameDef*>(variable->name_def()));
    if (it == cache_.end()) {
      return std::nullopt;
    }
    if (parametric_context.has_value()) {
      const auto type_for_context =
          it->second.cached_type_per_parametric_context.find(
              *parametric_context);
      if (type_for_context !=
          it->second.cached_type_per_parametric_context.end()) {
        return type_for_context->second;
      }
    }
    return it->second.cached_type;
  }

  void InvalidateVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) {
    VLOG(6) << "Invalidating unification cache for variable due to a direct "
               "change: "
            << variable->ToString();
    const auto* def = std::get<const NameDef*>(variable->name_def());
    const auto consumers = transitive_consumers_.find(def);
    if (consumers != transitive_consumers_.end()) {
      for (const NameDef* consumer : consumers->second) {
        VLOG(5) << "Invalidating unification cache for variable: "
                << consumer->ToString()
                << " due to dependency on changed variable: "
                << variable->ToString();
        if (parametric_context.has_value()) {
          const auto consumer_it = cache_.find(consumer);
          if (consumer_it != cache_.end()) {
            consumer_it->second.cached_type_per_parametric_context.erase(
                *parametric_context);
          }
        } else {
          // Note that if the invalidation is not scoped to a context, it could
          // affect any context.
          cache_.erase(consumer);
        }
      }
    }

    if (parametric_context.has_value()) {
      const auto it = cache_.find(def);
      if (it != cache_.end()) {
        it->second.cached_type_per_parametric_context.erase(
            *parametric_context);
      }
    } else {
      cache_.erase(def);
      transitive_consumers_.erase(def);
    }
  }

 private:
  struct VariableState {
    std::optional<const TypeAnnotation*> cached_type;
    absl::flat_hash_map<const ParametricContext*, const TypeAnnotation*>
        cached_type_per_parametric_context;
  };

  absl::flat_hash_map<const NameDef*, VariableState> cache_;
  absl::flat_hash_map<const NameDef*, absl::flat_hash_set<const NameDef*>>
      transitive_consumers_;
};

class InferenceTableImpl : public InferenceTable {
 public:
  absl::StatusOr<const NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer, std::string_view name,
      std::optional<const TypeAnnotation*> declaration_annotation) override {
    VLOG(6) << "DefineInternalVariable of kind " << (int)kind << " with name "
            << name << " and definer: " << definer->ToString();
    XLS_RET_CHECK(definer->GetSpan().has_value());
    Span span = *definer->GetSpan();
    const NameDef* name_def =
        definer->owner()->Make<NameDef>(span, std::string(name), definer);
    const NameRef* name_ref =
        definer->owner()->Make<NameRef>(span, std::string(name), name_def);
    AddVariable(name_def,
                std::make_unique<InferenceVariable>(definer, name_ref, kind));
    if (declaration_annotation.has_value()) {
      CHECK_NE(*declaration_annotation, nullptr);
      XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable,
                           GetVariable(name_ref));
      declaration_type_annotations_.emplace(variable, *declaration_annotation);
    }
    return name_ref;
  }

  absl::StatusOr<const NameRef*> DefineParametricVariable(
      const ParametricBinding& binding) override {
    VLOG(6) << "DefineParametricVariable of type "
            << binding.type_annotation()->ToString() << " with name "
            << binding.name_def()->ToString();
    XLS_ASSIGN_OR_RETURN(
        InferenceVariableKind kind,
        TypeAnnotationToInferenceVariableKind(binding.type_annotation()));
    const NameDef* name_def = binding.name_def();
    const NameRef* name_ref = binding.owner()->Make<NameRef>(
        name_def->span(), name_def->identifier(), name_def);
    AddVariable(name_def,
                std::make_unique<InferenceVariable>(name_def, name_ref, kind));
    XLS_RETURN_IF_ERROR(SetTypeAnnotation(name_def, binding.type_annotation()));
    return name_ref;
  }

  absl::StatusOr<ParametricContext*> AddParametricInvocation(
      const Invocation& node, const Function& callee,
      std::optional<const Function*> caller,
      std::optional<const ParametricContext*> parent_context,
      std::optional<const TypeAnnotation*> self_type,
      TypeInfo* invocation_type_info) override {
    VLOG(5) << "Add parametric invocation: " << node.ToString()
            << " from parent context: "
            << ::xls::dslx::ToString(parent_context);
    auto context = std::make_unique<ParametricContext>(
        parametric_contexts_.size(), &node,
        ParametricInvocationDetails{&callee, caller}, invocation_type_info,
        parent_context, self_type);
    const std::vector<ParametricBinding*>& bindings =
        callee.parametric_bindings();
    const std::vector<ExprOrType>& explicit_parametrics =
        node.explicit_parametrics();
    XLS_RET_CHECK(explicit_parametrics.size() <= bindings.size());
    MutableParametricContextData mutable_data;
    for (int i = 0; i < bindings.size(); i++) {
      const ParametricBinding* binding = bindings[i];
      const InferenceVariable* variable =
          variables_.at(binding->name_def()).get();
      if (i < explicit_parametrics.size()) {
        const ExprOrType value = explicit_parametrics[i];
        if (std::holds_alternative<Expr*>(value)) {
          mutable_data.parametric_values.emplace(
              variable, ParametricContextScopedExpr(parent_context,
                                                    binding->type_annotation(),
                                                    std::get<Expr*>(value)));
        }
      } else if (binding->expr() != nullptr) {
        mutable_data.parametric_values.emplace(
            variable,
            ParametricContextScopedExpr(
                context.get(), binding->type_annotation(), binding->expr()));
      }
    }
    ParametricContext* result = context.get();
    parametric_contexts_.push_back(std::move(context));
    mutable_parametric_context_data_.emplace(result, std::move(mutable_data));
    return result;
  }

  bool MapToCanonicalInvocationTypeInfo(ParametricContext* parametric_context,
                                        ParametricEnv env) override {
    CHECK(parametric_context->is_invocation());

    // `ParametricEnv` doesn't currently capture generic types, so for the time
    // being, we can't canonicalize invocations that use them.
    for (const ParametricBinding* binding :
         parametric_context->parametric_bindings()) {
      if (binding->type_annotation()->IsAnnotation<GenericTypeAnnotation>()) {
        return false;
      }
    }

    parametric_context->SetInvocationEnv(env);
    const ParametricInvocationDetails& details =
        std::get<ParametricInvocationDetails>(parametric_context->details());
    const auto it = canonical_parametric_context_.find({details.callee, env});
    if (it == canonical_parametric_context_.end()) {
      canonical_parametric_context_.emplace_hint(
          it, std::make_pair(details.callee, env), parametric_context);
      return false;
    }

    mutable_parametric_context_data_.at(parametric_context).canonical_context =
        it->second;
    parametric_context->SetTypeInfo(it->second->type_info());
    // The owner of the canonical context should have called
    // `SetParametricFreeFunctionType` on it before now.
    CHECK(std::get<ParametricInvocationDetails>(it->second->details())
              .parametric_free_function_type != nullptr);
    parametric_context->SetParametricFreeFunctionType(
        std::get<ParametricInvocationDetails>(it->second->details())
            .parametric_free_function_type);
    return true;
  }

  std::vector<const ParametricContext*> GetParametricInvocations()
      const override {
    std::vector<const ParametricContext*> result;
    for (const auto& context : parametric_contexts_) {
      if (context->is_invocation()) {
        result.push_back(context.get());
      }
    }
    return result;
  }

  absl::StatusOr<StructContextResult> GetOrCreateParametricStructContext(
      const StructDefBase* struct_def, const AstNode* node,
      ParametricEnv parametric_env, const TypeAnnotation* self_type,
      absl::FunctionRef<absl::StatusOr<TypeInfo*>()> type_info_factory)
      override {
    auto& contexts = parametric_struct_contexts_[struct_def];
    const auto it = contexts.find(parametric_env);
    if (it != contexts.end()) {
      return StructContextResult{.context = it->second, .created_new = false};
    }
    XLS_ASSIGN_OR_RETURN(TypeInfo * type_info, type_info_factory());
    auto context = std::make_unique<ParametricContext>(
        parametric_contexts_.size(), node,
        ParametricStructDetails{struct_def, parametric_env}, type_info,
        /*parent_context=*/std::nullopt, self_type);
    const ParametricContext* result = context.get();
    parametric_contexts_.push_back(std::move(context));
    contexts.emplace_hint(it, parametric_env, result);
    mutable_parametric_context_data_.emplace(result,
                                             MutableParametricContextData{});
    converted_parametric_envs_.emplace(result, parametric_env);
    return StructContextResult{.context = result, .created_new = true};
  }

  std::optional<ParametricContextScopedExpr> GetParametricValue(
      const NameDef& binding_name_def,
      const ParametricContext& context) const override {
    const InferenceVariable* variable = variables_.at(&binding_name_def).get();
    const absl::flat_hash_map<const InferenceVariable*,
                              ParametricContextScopedExpr>& values =
        mutable_parametric_context_data_.at(&context).parametric_values;
    const auto it = values.find(variable);
    return it == values.end() ? std::nullopt : std::make_optional(it->second);
  }

  absl::Status SetTypeAnnotation(const AstNode* node,
                                 const TypeAnnotation* annotation) override {
    return MutateAndCheckNodeData(
        node, [=](NodeData& data) { data.type_annotation = annotation; });
  }

  absl::Status AddTypeAnnotationToVariableForParametricContext(
      std::optional<const ParametricContext*> context, const NameRef* ref,
      const TypeAnnotation* annotation) override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable, GetVariable(ref));
    AddTypeAnnotationForParametricContextInternal(context, variable,
                                                  annotation);
    return absl::OkStatus();
  }

  absl::Status AddTypeAnnotationToVariableForParametricContext(
      std::optional<const ParametricContext*> context,
      const ParametricBinding* binding,
      const TypeAnnotation* annotation) override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable,
                         GetVariable(binding->name_def()));
    AddTypeAnnotationForParametricContextInternal(context, variable,
                                                  annotation);
    return absl::OkStatus();
  }

  absl::Status RemoveTypeAnnotationsFromTypeVariable(
      const NameRef* ref,
      absl::FunctionRef<bool(const TypeAnnotation*)> remove_predicate)
      override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable, GetVariable(ref));
    const auto it = type_annotations_per_type_variable_.find(variable);
    if (it != type_annotations_per_type_variable_.end()) {
      auto& annotations = it->second;
      size_t count_before = annotations.size();
      annotations.erase(std::remove_if(annotations.begin(), annotations.end(),
                                       [&](const TypeAnnotation* annotation) {
                                         return remove_predicate(annotation);
                                       }),
                        annotations.end());
      if (annotations.size() != count_before) {
        cache_.InvalidateVariable(/*parametric_context=*/std::nullopt,
                                  variable->name_ref());
      }
    }
    return absl::OkStatus();
  }

  void SetAnnotationFlag(const TypeAnnotation* annotation,
                         TypeInferenceFlag flag) override {
    annotation_flags_[annotation].SetFlag(flag);
  }

  TypeInferenceFlag GetAnnotationFlag(
      const TypeAnnotation* annotation) const override {
    const auto it = annotation_flags_.find(annotation);
    return it != annotation_flags_.end() ? it->second
                                         : TypeInferenceFlag::kNone;
  }

  absl::Status SetTypeVariable(const AstNode* node,
                               const NameRef* type) override {
    XLS_ASSIGN_OR_RETURN(InferenceVariable * variable, GetVariable(type));
    VLOG(6) << "SetTypeVariable node " << node->ToString() << "; type "
            << type->ToString() << " variable " << variable->ToString();
    if (variable->kind() != InferenceVariableKind::kType) {
      return absl::InvalidArgumentError(
          absl::Substitute("Setting the type of $0 to non-type variable: $1",
                           node->ToString(), variable->ToString()));
    }
    return MutateAndCheckNodeData(
        node, [=](NodeData& data) { data.type_variable = variable; });
  }

  std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const override {
    const auto it = node_data_.find(node);
    if (it == node_data_.end()) {
      return std::nullopt;
    }
    return it->second.type_annotation;
  }

  std::optional<const NameRef*> GetTypeVariable(
      const AstNode* node) const override {
    const auto it = node_data_.find(node);
    if (it == node_data_.end()) {
      return std::nullopt;
    }
    const std::optional<const InferenceVariable*>& variable =
        it->second.type_variable;
    return variable.has_value() ? std::make_optional((*variable)->name_ref())
                                : std::nullopt;
  }

  absl::StatusOr<std::optional<const TypeAnnotation*>>
  GetDeclarationTypeAnnotation(const NameRef* ref) const override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable, GetVariable(ref));
    const auto it = declaration_type_annotations_.find(variable);
    if (it == declaration_type_annotations_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  absl::StatusOr<std::vector<const TypeAnnotation*>>
  GetTypeAnnotationsForTypeVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* ref) const override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable, GetVariable(ref));
    const auto it = type_annotations_per_type_variable_.find(variable);
    std::vector<const TypeAnnotation*> result =
        (it == type_annotations_per_type_variable_.end())
            ? std::vector<const TypeAnnotation*>()
            : it->second;
    if (parametric_context.has_value()) {
      const auto& invocation_specific_annotations =
          mutable_parametric_context_data_.at(*parametric_context)
              .type_annotations_per_type_variable;
      const auto invocation_specific_it =
          invocation_specific_annotations.find(variable);
      if (invocation_specific_it != invocation_specific_annotations.end()) {
        absl::c_copy(invocation_specific_it->second,
                     std::back_inserter(result));
      }
    }
    return result;
  }

  absl::StatusOr<std::vector<const Invocation*>>
  GetInvocationsFeedingTypeVariable(const NameRef* ref) const override {
    XLS_ASSIGN_OR_RETURN(const InferenceVariable* variable, GetVariable(ref));
    const auto it = invocations_feeding_type_variable_.find(variable);
    if (it == invocations_feeding_type_variable_.end()) {
      return std::vector<const Invocation*>{};
    }
    return it->second;
  }

  void SetColonRefTarget(const ColonRef* colon_ref,
                         const AstNode* target) override {
    colon_ref_targets_[colon_ref] = target;
  }

  std::optional<const AstNode*> GetColonRefTarget(
      const ColonRef* colon_ref) const override {
    const auto it = colon_ref_targets_.find(colon_ref);
    return it == colon_ref_targets_.end() ? std::nullopt
                                          : std::make_optional(it->second);
  }

  void SetCalleeInCallerContext(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context,
      const Function* callee) override {
    if (caller_context.has_value()) {
      mutable_parametric_context_data_.at(*caller_context).callees[invocation] =
          callee;
    } else {
      callees_with_no_caller_context_[invocation] = callee;
    }
  }

  std::optional<const Function*> GetCalleeInCallerContext(
      const Invocation* invocation,
      std::optional<const ParametricContext*> caller_context) const override {
    const absl::flat_hash_map<const Invocation*, const Function*>& map =
        caller_context.has_value()
            ? mutable_parametric_context_data_.at(*caller_context).callees
            : callees_with_no_caller_context_;
    const auto it = map.find(invocation);
    return it == map.end() ? std::nullopt : std::make_optional(it->second);
  }

  absl::StatusOr<AstNode*> Clone(
      const AstNode* input, CloneReplacer replacer,
      std::optional<Module*> target_module) override {
    absl::flat_hash_map<const AstNode*, AstNode*> all_pairs;
    XLS_ASSIGN_OR_RETURN(all_pairs,
                         CloneAstAndGetAllPairs(input, target_module,

                                                std::move(replacer)));
    // VerifyClone won't pass here.
    // XLS_RETURN_IF_ERROR(
    //     VerifyClone(input, all_pairs.at(input),
    //     *input->owner()->file_table()));
    for (const auto& [old_node, new_node] : all_pairs) {
      if (old_node != new_node) {
        const auto it = node_data_.find(old_node);
        if (it != node_data_.end()) {
          NodeData copy = it->second;
          node_data_[new_node] = std::move(copy);
        }
        if (new_node->kind() == AstNodeKind::kTypeAnnotation) {
          const auto flag_it = annotation_flags_.find(
              down_cast<const TypeAnnotation*>(old_node));
          if (flag_it != annotation_flags_.end()) {
            const TypeInferenceFlag flag = flag_it->second;
            annotation_flags_.emplace(
                down_cast<const TypeAnnotation*>(new_node), flag);
          }
        }
        if (old_node->kind() == AstNodeKind::kColonRef) {
          const auto* old_node_as_colon_ref =
              down_cast<const ColonRef*>(old_node);
          std::optional<const AstNode*> target =
              GetColonRefTarget(old_node_as_colon_ref);
          if (target.has_value()) {
            SetColonRefTarget(down_cast<const ColonRef*>(new_node), *target);
          }
        }
      }
    }
    return all_pairs.at(input);
  }

  std::string ToString() const override {
    std::string result;
    absl::flat_hash_map<const AstNode*, std::vector<const ParametricContext*>>
        contexts_per_node;
    auto annotation_to_string = [this](std::string_view indent,
                                       const TypeAnnotation* annotation) {
      return absl::Substitute(
          std::string(indent) + "Annotation: $0; flag: $1\n",
          annotation->ToString(), GetAnnotationFlag(annotation).ToString());
    };

    for (const auto& context : parametric_contexts_) {
      contexts_per_node[context->node()].push_back(context.get());
    }
    for (const auto& [node, data] : node_data_) {
      absl::StrAppendFormat(&result, "Node: %s\n", node->ToString());
      absl::StrAppendFormat(&result, "  Kind: %s\n",
                            AstNodeKindToString(node->kind()));
      absl::StrAppendFormat(&result, "  Address: 0x%x\n", (uint64_t)node);
      absl::StrAppendFormat(&result, "  Module: %s\n", node->owner()->name());
      if (data.type_variable.has_value()) {
        absl::StrAppendFormat(&result, "  Variable: %s\n",
                              (*data.type_variable)->name());
      }
      if (data.type_annotation.has_value()) {
        absl::StrAppend(&result,
                        annotation_to_string("  ", *data.type_annotation));
      }
      if (data.slice_start_and_width_exprs.has_value()) {
        absl::StrAppendFormat(
            &result, "  Start: %s\n",
            ValueOrExprToString(data.slice_start_and_width_exprs->start));
        absl::StrAppendFormat(
            &result, "  Width: %s\n",
            ValueOrExprToString(data.slice_start_and_width_exprs->width));
      }
      const std::vector<const ParametricContext*>& contexts =
          contexts_per_node[node];
      if (!contexts.empty()) {
        absl::StrAppendFormat(&result, "  Parametric contexts:\n");
        for (int i = 0; i < contexts.size(); i++) {
          absl::StrAppendFormat(&result, "    %d %s\n", i,
                                ::xls::dslx::ToString(contexts[i]));
        }
      }
    }
    for (const auto& [name_def, variable] : variables_) {
      absl::StrAppendFormat(&result, "Variable: %s\n", variable->name());
      absl::StrAppendFormat(&result, "  NameDef address: 0x%x\n",
                            (uint64_t)(name_def));
      absl::StrAppendFormat(&result, "  Module: %s\n",
                            variable->definer()->owner()->name());
      const auto annotations =
          type_annotations_per_type_variable_.find(variable.get());
      if (annotations != type_annotations_per_type_variable_.end() &&
          !annotations->second.empty()) {
        absl::StrAppendFormat(&result, "  Annotations:\n");
        for (int i = 0; i < annotations->second.size(); i++) {
          absl::StrAppend(&result,
                          annotation_to_string("    ", annotations->second[i]));
        }
      }
      for (const auto& context : parametric_contexts_) {
        const MutableParametricContextData& data =
            mutable_parametric_context_data_.at(context.get());
        const auto context_annotations =
            data.type_annotations_per_type_variable.find(variable.get());
        if (context_annotations !=
                data.type_annotations_per_type_variable.end() &&
            !context_annotations->second.empty()) {
          absl::StrAppendFormat(&result, "  Annotations for context %s\n",
                                ::xls::dslx::ToString(context.get()));
          for (int i = 0; i < context_annotations->second.size(); i++) {
            absl::StrAppendFormat(&result, "    %d %s\n", i,
                                  context_annotations->second[i]->ToString());
          }
        }
      }
    }
    if (!parametric_contexts_.empty()) {
      absl::StrAppendFormat(&result, "Parametric contexts:\n");
      for (const auto& context : parametric_contexts_) {
        absl::StrAppendFormat(&result, "  %s\n",
                              ::xls::dslx::ToString(context.get()));
      }
    }
    return result;
  }

  absl::Status SetSliceStartAndWidthExprs(
      const AstNode* node, StartAndWidthExprs start_and_width) override {
    return MutateAndCheckNodeData(node, [&](NodeData& data) {
      data.slice_start_and_width_exprs = start_and_width;
    });
  }

  std::optional<StartAndWidthExprs> GetSliceStartAndWidthExprs(
      const AstNode* node) override {
    const auto it = node_data_.find(node);
    return it == node_data_.end() ? std::nullopt
                                  : it->second.slice_start_and_width_exprs;
  }

  void SetCachedUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable,
      const absl::flat_hash_set<const NameRef*>&
          transitive_variable_dependencies,
      const TypeAnnotation* unified_type) override {
    cache_.SetUnifiedTypeForVariable(GetCanonicalContext(parametric_context),
                                     variable, transitive_variable_dependencies,
                                     unified_type);
  }

  std::optional<const TypeAnnotation*> GetCachedUnifiedTypeForVariable(
      std::optional<const ParametricContext*> parametric_context,
      const NameRef* variable) override {
    return cache_.GetUnifiedTypeForVariable(
        GetCanonicalContext(parametric_context), variable);
  }

  void SetParametricEnv(const ParametricContext* parametric_context,
                        ParametricEnv env) override {
    converted_parametric_envs_[parametric_context] = std::move(env);
  }

  void SetParametricValueExprs(
      const ParametricContext* parametric_context,
      absl::flat_hash_map<const NameDef*, ExprOrType> value_exprs) override {
    parametric_value_exprs_[parametric_context] = std::move(value_exprs);
  }

  ParametricEnv GetParametricEnv(std::optional<const ParametricContext*>
                                     parametric_context) const override {
    if (parametric_context.has_value()) {
      const auto it = converted_parametric_envs_.find(*parametric_context);
      if (it != converted_parametric_envs_.end()) {
        return it->second;
      }
    }
    return ParametricEnv{};
  }

  absl::StatusOr<absl::flat_hash_map<const NameDef*, ExprOrType>>
  GetParametricValueExprs(
      const ParametricContext* parametric_context) override {
    const auto it = parametric_value_exprs_.find(parametric_context);
    if (it == parametric_value_exprs_.end()) {
      return absl::NotFoundError(absl::StrCat("No value exprs for context: ",
                                              parametric_context->ToString()));
    }
    return it->second;
  }

 private:
  std::optional<const ParametricContext*> GetCanonicalContext(
      std::optional<const ParametricContext*> context) {
    if (!context.has_value()) {
      return std::nullopt;
    }
    const auto it = mutable_parametric_context_data_.find(*context);
    if (it == mutable_parametric_context_data_.end()) {
      return std::nullopt;
    }
    return it->second.canonical_context.has_value()
               ? it->second.canonical_context
               : context;
  }

  void AddVariable(const NameDef* name_def,
                   std::unique_ptr<InferenceVariable> variable) {
    variables_.emplace(name_def, std::move(variable));
  }

  absl::StatusOr<InferenceVariable*> GetVariable(const NameRef* ref) const {
    return GetVariable(ref->name_def());
  }

  absl::StatusOr<InferenceVariable*> GetVariable(AnyNameDef name_def) const {
    if (std::holds_alternative<const NameDef*>(name_def)) {
      const auto it = variables_.find(std::get<const NameDef*>(name_def));
      if (it != variables_.end()) {
        return it->second.get();
      }
    }
    return absl::NotFoundError(
        absl::Substitute("No inference variable for NameRef: $0",
                         ToAstNode(name_def)->ToString()));
  }

  // Runs the given `mutator` on the stored `NodeData` for `node`, creating the
  // `NodeData` if it does not exist already. Then refines what is known about
  // the type variable associated with `node`, if any, and errors if there is
  // conflicting information.
  absl::Status MutateAndCheckNodeData(
      const AstNode* node, absl::AnyInvocable<void(NodeData&)> mutator) {
    const auto [it, inserted] = node_data_.emplace(node, NodeData{});
    NodeData& node_data = it->second;
    if (inserted) {
      node_data.order_added = node_data_.size();
    }
    std::optional<const InferenceVariable*> old_variable =
        node_data.type_variable;
    mutator(node_data);
    // Refine and check the associated type variable.
    if (node_data.type_variable.has_value() &&
        node_data.type_annotation.has_value()) {
      // If a node has `X` as its variable and `TVTA(X)` as its annotation, the
      // annotation is not useful and could lead to infinite looping elsewhere.
      if ((*node_data.type_annotation)
              ->IsAnnotation<TypeVariableTypeAnnotation>()) {
        const auto* tvta = (*node_data.type_annotation)
                               ->AsAnnotation<TypeVariableTypeAnnotation>();
        absl::StatusOr<InferenceVariable*> variable =
            GetVariable(tvta->type_variable());
        if (variable.ok() && *variable == *node_data.type_variable) {
          node_data.type_annotation = std::nullopt;
        }
      }
      if (node_data.type_annotation.has_value()) {
        type_annotations_per_type_variable_[*node_data.type_variable].push_back(
            *node_data.type_annotation);
      }
    }
    if (old_variable.has_value()) {
      cache_.InvalidateVariable(/*parametric_context=*/std::nullopt,
                                (*old_variable)->name_ref());
    }
    if (node_data.type_variable.has_value()) {
      if (node->kind() == AstNodeKind::kInvocation) {
        invocations_feeding_type_variable_[*node_data.type_variable].push_back(
            down_cast<const Invocation*>(node));
      }
      cache_.InvalidateVariable(/*parametric_context=*/std::nullopt,
                                (*node_data.type_variable)->name_ref());
    }
    return absl::OkStatus();
  }

  void AddTypeAnnotationForParametricContextInternal(
      std::optional<const ParametricContext*> context,
      const InferenceVariable* variable, const TypeAnnotation* annotation) {
    CHECK(variable->kind() == InferenceVariableKind::kType);
    if (context.has_value()) {
      mutable_parametric_context_data_.at(*context)
          .type_annotations_per_type_variable[variable]
          .push_back(annotation);
    } else {
      type_annotations_per_type_variable_[variable].push_back(annotation);
    }
    cache_.InvalidateVariable(GetCanonicalContext(context),
                              variable->name_ref());
  }

  // The variables of all kinds that have been defined by the user or
  // internally.
  absl::flat_hash_map<const NameDef*, std::unique_ptr<InferenceVariable>>
      variables_;
  // The type annotations that have been associated with each inference
  // variable of type-kind.
  absl::flat_hash_map<const InferenceVariable*,
                      std::vector<const TypeAnnotation*>>
      type_annotations_per_type_variable_;
  // The type annotations that were declared by the user.
  absl::flat_hash_map<const InferenceVariable*, const TypeAnnotation*>
      declaration_type_annotations_;
  // The `AstNode` objects that have associated data.
  absl::flat_hash_map<const AstNode*, NodeData> node_data_;
  // Parametric contexts and the corresponding information about parametric
  // variables.
  std::vector<std::unique_ptr<ParametricContext>> parametric_contexts_;
  absl::flat_hash_map<const ParametricContext*, MutableParametricContextData>
      mutable_parametric_context_data_;
  absl::flat_hash_map<std::pair<const Function*, ParametricEnv>,
                      const ParametricContext*>
      canonical_parametric_context_;
  absl::flat_hash_map<const TypeAnnotation*, TypeInferenceFlag>
      annotation_flags_;
  absl::flat_hash_map<const ColonRef*, const AstNode*> colon_ref_targets_;
  absl::flat_hash_map<
      const StructDefBase*,
      absl::flat_hash_map<ParametricEnv, const ParametricContext*>>
      parametric_struct_contexts_;
  absl::flat_hash_map<const ParametricContext*, ParametricEnv>
      converted_parametric_envs_;
  absl::flat_hash_map<const ParametricContext*,
                      absl::flat_hash_map<const NameDef*, ExprOrType>>
      parametric_value_exprs_;
  absl::flat_hash_map<const InferenceVariable*, std::vector<const Invocation*>>
      invocations_feeding_type_variable_;
  absl::flat_hash_map<const Invocation*, const Function*>
      callees_with_no_caller_context_;
  UnificationCache cache_;
};

}  // namespace

std::string TypeInferenceFlag::ToString() const {
  if (flags_ == 0) {
    return "none";
  }
  std::vector<std::string> names;
  for (const auto& [flags, name] : *flag_names_) {
    if (flags != 0 && (flags_ & flags) != 0) {
      names.push_back(name);
    }
  }
  return absl::StrJoin(names, ",");
}

InferenceTable::~InferenceTable() = default;

std::unique_ptr<InferenceTable> InferenceTable::Create() {
  return std::make_unique<InferenceTableImpl>();
}

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span,
    const InterpValue& value, const TypeAnnotation* type_annotation) {
  VLOG(5) << "Creating type-checked number: " << value.ToString()
          << " of type: " << type_annotation->ToString();
  Number* number = module.Make<Number>(span, value.ToString(/*humanize=*/true),
                                       NumberKind::kOther, nullptr);
  XLS_RETURN_IF_ERROR(table.SetTypeAnnotation(number, type_annotation));
  return number;
}

absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation) {
  return MakeTypeCheckedNumber(module, table, span, InterpValue::MakeS64(value),
                               type_annotation);
}

bool IsColonRefWithTypeTarget(const InferenceTable& table, const Expr* expr) {
  if (expr->kind() != AstNodeKind::kColonRef) {
    return false;
  }
  std::optional<const AstNode*> colon_ref_target =
      table.GetColonRefTarget(down_cast<const ColonRef*>(expr));
  return colon_ref_target.has_value() &&
         ((*colon_ref_target)->kind() == AstNodeKind::kTypeAlias ||
          (*colon_ref_target)->kind() == AstNodeKind::kEnumDef ||
          (*colon_ref_target)->kind() == AstNodeKind::kTypeAnnotation);
}

CloneReplacer NameRefMapper(
    InferenceTable& table,
    const absl::flat_hash_map<const NameDef*, ExprOrType>& map,
    std::optional<Module*> target_module) {
  return [table = &table, map = &map, target_module](
             const AstNode* node, Module* new_module,
             const absl::flat_hash_map<const AstNode*, AstNode*>&)
             -> absl::StatusOr<std::optional<AstNode*>> {
    if (node->kind() == AstNodeKind::kNameRef) {
      const auto* ref = down_cast<const NameRef*>(node);
      if (std::holds_alternative<const NameDef*>(ref->name_def())) {
        const auto it = map->find(std::get<const NameDef*>(ref->name_def()));
        if (it != map->end()) {
          Module* module_for_clone =
              target_module ? *target_module : new_module;
          return table->Clone(ToAstNode(it->second), &NoopCloneReplacer,
                              module_for_clone);
        }
      }
    }
    return std::nullopt;
  };
}

}  // namespace xls::dslx
