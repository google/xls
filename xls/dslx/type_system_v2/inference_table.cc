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
#include "absl/strings/substitute.h"
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
  const auto* builtin = dynamic_cast<const BuiltinTypeAnnotation*>(annotation);
  if (builtin) {
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
  const auto* array = dynamic_cast<const ArrayTypeAnnotation*>(annotation);
  if (array) {
    const auto* builtin_element_type =
        dynamic_cast<BuiltinTypeAnnotation*>(array->element_type());
    if (builtin_element_type != nullptr &&
        builtin_element_type->GetBitCount() == 0) {
      return InferenceVariableKind::kInteger;
    }
  }
  if (dynamic_cast<const GenericTypeAnnotation*>(annotation)) {
    return InferenceVariableKind::kType;
  }
  return absl::InvalidArgumentError(
      absl::Substitute("Inference variables of type $0 are not supported.",
                       annotation->ToString()));
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
  absl::flat_hash_map<const InferenceVariable*, ParametricContextScopedExpr>
      parametric_values;
  absl::flat_hash_map<const InferenceVariable*,
                      std::vector<const TypeAnnotation*>>
      type_annotations_per_type_variable;
};

class InferenceTableImpl : public InferenceTable {
 public:
  explicit InferenceTableImpl(Module& module) : module_(module) {}

  absl::StatusOr<const NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer, std::string_view name,
      std::optional<const TypeAnnotation*> declaration_annotation) override {
    VLOG(6) << "DefineInternalVariable of kind " << (int)kind << " with name "
            << name << " and definer: " << definer->ToString();
    CHECK(definer->GetSpan().has_value());
    Span span = *definer->GetSpan();
    const NameDef* name_def =
        module_.Make<NameDef>(span, std::string(name), definer);
    const NameRef* name_ref =
        module_.Make<NameRef>(span, std::string(name), name_def);
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
    const NameRef* name_ref = module_.Make<NameRef>(
        name_def->span(), name_def->identifier(), name_def);
    AddVariable(name_def,
                std::make_unique<InferenceVariable>(name_def, name_ref, kind));
    XLS_RETURN_IF_ERROR(SetTypeAnnotation(name_def, binding.type_annotation()));
    return name_ref;
  }

  absl::StatusOr<const ParametricContext*> AddParametricInvocation(
      const Invocation& node, const Function& callee,
      std::optional<const Function*> caller,
      std::optional<const ParametricContext*> parent_context,
      std::optional<const TypeAnnotation*> self_type,
      TypeInfo* invocation_type_info) override {
    VLOG(5) << "Add parametric invocation: " << node.ToString()
            << " from parent context: "
            << ::xls::dslx::ToString(parent_context);
    // If we call a function in a different module, we need to add the
    // parametric bindings here.
    if (module_.name() != callee.owner()->name()) {
      for (const ParametricBinding* binding : callee.parametric_bindings()) {
        if (!variables_.contains(binding->name_def())) {
          XLS_ASSIGN_OR_RETURN(const NameRef* name_ref,
                               DefineParametricVariable(*binding));
          XLS_RETURN_IF_ERROR(
              SetTypeAnnotation(name_ref, binding->type_annotation()));
        }
      }
    }
    auto context = std::make_unique<ParametricContext>(
        parametric_contexts_.size(), &node,
        ParametricInvocationDetails{&callee, caller}, invocation_type_info,
        parent_context, self_type);
    const std::vector<ParametricBinding*>& bindings =
        callee.parametric_bindings();
    const std::vector<ExprOrType>& explicit_parametrics =
        node.explicit_parametrics();
    CHECK(explicit_parametrics.size() <= bindings.size());
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
    const ParametricContext* result = context.get();
    parametric_contexts_.push_back(std::move(context));
    mutable_parametric_context_data_.emplace(result, std::move(mutable_data));
    return result;
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

  absl::StatusOr<const ParametricContext*> GetOrCreateParametricStructContext(
      const StructDefBase* struct_def, const AstNode* node,
      ParametricEnv parametric_env, const TypeAnnotation* self_type,
      absl::FunctionRef<absl::StatusOr<TypeInfo*>()> type_info_factory)
      override {
    auto& contexts = parametric_struct_contexts_[struct_def];
    const auto it = contexts.find(parametric_env);
    if (it != contexts.end()) {
      return it->second;
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
    return result;
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
      annotations.erase(std::remove_if(annotations.begin(), annotations.end(),
                                       [&](const TypeAnnotation* annotation) {
                                         return remove_predicate(annotation);
                                       }),
                        annotations.end());
    }
    return absl::OkStatus();
  }

  void MarkAsAutoLiteral(const TypeAnnotation* annotation) override {
    auto_literal_annotations_.insert(annotation);
  }

  bool IsAutoLiteral(const TypeAnnotation* annotation) const override {
    return auto_literal_annotations_.contains(annotation);
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
      if (mutable_parametric_context_data_.contains(*parametric_context)) {
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
    }
    return result;
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

  absl::StatusOr<AstNode*> Clone(const AstNode* input,
                                 CloneReplacer replacer) override {
    absl::flat_hash_map<const AstNode*, AstNode*> all_pairs;
    XLS_ASSIGN_OR_RETURN(
        all_pairs,
        CloneAstAndGetAllPairs(
            input, ChainCloneReplacers(&PreserveTypeDefinitionsReplacer,
                                       std::move(replacer))));
    for (const auto& [old_node, new_node] : all_pairs) {
      if (old_node != new_node) {
        const auto it = node_data_.find(old_node);
        if (it != node_data_.end()) {
          NodeData copy = it->second;
          node_data_.emplace(new_node, std::move(copy));
        }
        if (const auto* new_node_as_annotation =
                dynamic_cast<const TypeAnnotation*>(new_node);
            new_node_as_annotation != nullptr &&
            auto_literal_annotations_.contains(
                dynamic_cast<const TypeAnnotation*>(old_node))) {
          auto_literal_annotations_.insert(new_node_as_annotation);
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
          std::string(indent) + "Annotation: $0; Auto literal: $1\n",
          annotation->ToString(),
          auto_literal_annotations_.contains(annotation));
    };

    for (const auto& context : parametric_contexts_) {
      contexts_per_node[context->node()].push_back(context.get());
    }
    for (const auto& [node, data] : node_data_) {
      absl::StrAppendFormat(&result, "Node: %s\n", node->ToString());
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
            data.slice_start_and_width_exprs->start->ToString());
        absl::StrAppendFormat(
            &result, "  Width: %s\n",
            data.slice_start_and_width_exprs->width->ToString());
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

 private:
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
    mutator(node_data);
    // Refine and check the associated type variable.
    if (node_data.type_variable.has_value() &&
        node_data.type_annotation.has_value()) {
      type_annotations_per_type_variable_[*node_data.type_variable].push_back(
          *node_data.type_annotation);
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
  }

  Module& module_;
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
  absl::flat_hash_set<const TypeAnnotation*> auto_literal_annotations_;
  absl::flat_hash_map<const ColonRef*, const AstNode*> colon_ref_targets_;
  absl::flat_hash_map<
      const StructDefBase*,
      absl::flat_hash_map<ParametricEnv, const ParametricContext*>>
      parametric_struct_contexts_;
};

}  // namespace

InferenceTable::~InferenceTable() = default;

std::unique_ptr<InferenceTable> InferenceTable::Create(Module& module) {
  return std::make_unique<InferenceTableImpl>(module);
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

// Variant that takes a raw `int64_t` value for the number.
absl::StatusOr<Number*> MakeTypeCheckedNumber(
    Module& module, InferenceTable& table, const Span& span, int64_t value,
    const TypeAnnotation* type_annotation) {
  return MakeTypeCheckedNumber(module, table, span, InterpValue::MakeS64(value),
                               type_annotation);
}

}  // namespace xls::dslx
