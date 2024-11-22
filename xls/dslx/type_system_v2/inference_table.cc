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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

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
  return absl::InvalidArgumentError(
      absl::Substitute("Inference variables of type $0 are not supported.",
                       annotation->ToString()));
}

// Represents the immutable metadata for a variable in an `InferenceTable`.
class InferenceVariable {
 public:
  InferenceVariable(const AstNode* definer, std::string_view name,
                    InferenceVariableKind kind, bool parametric)
      : definer_(definer), name_(name), kind_(kind), parametric_(parametric) {}

  const AstNode* definer() const { return definer_; }

  std::string_view name() const { return name_; }

  InferenceVariableKind kind() const { return kind_; }

  bool parametric() const { return parametric_; }

  template <typename H>
  friend H AbslHashValue(H h, const InferenceVariable& v) {
    return H::combine(std::move(h), v.definer_, v.name_, v.kind_);
  }

  std::string ToString() const {
    return absl::Substitute("InferenceVariable(name=$0, kind=$1, definer=$2)",
                            name_, InferenceVariableKindToString(kind_),
                            definer_->ToString());
  }

 private:
  const AstNode* const definer_;
  const std::string name_;
  const InferenceVariableKind kind_;
  const bool parametric_;
};

// The mutable data for a node in an `InferenceTable`.
struct NodeData {
  size_t order_added;
  std::optional<const TypeAnnotation*> type_annotation;
  std::optional<const InferenceVariable*> type_variable;
};

// The mutable data for a type variable in an `InferenceTable`.
struct TypeConstraints {
  std::optional<bool> is_signed;
  std::optional<int64_t> min_width;

  // The explicit type annotation from which `is_signed` was determined, for
  // tracing and error purposes.
  std::optional<const TypeAnnotation*> signedness_definer;
};

// A visitor that finds the inference variables referred to by a given `AstNode`
// and its descendants.
class InferenceVariableDiscoveryVisitor : public AstNodeVisitorWithDefault {
 public:
  explicit InferenceVariableDiscoveryVisitor(
      absl::AnyInvocable<
          absl::StatusOr<const InferenceVariable*>(const NameRef*)>
          variable_resolver)
      : variable_resolver_(std::move(variable_resolver)) {}

  absl::Status HandleNameRef(const NameRef* ref) override {
    absl::StatusOr<const InferenceVariable*> variable = variable_resolver_(ref);
    if (variable.ok()) {
      discovered_variables_.insert(*variable);
    }
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* n) override {
    for (const AstNode* child : n->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  const absl::flat_hash_set<const InferenceVariable*>& GetDiscoveredVariables()
      const {
    return discovered_variables_;
  }

 private:
  absl::AnyInvocable<absl::StatusOr<const InferenceVariable*>(const NameRef*)>
      variable_resolver_;
  absl::flat_hash_set<const InferenceVariable*> discovered_variables_;
};

class InferenceTableImpl : public InferenceTable {
 public:
  InferenceTableImpl(Module& module, const FileTable& file_table)
      : module_(module), file_table_(file_table) {}

  absl::StatusOr<NameRef*> DefineInternalVariable(
      InferenceVariableKind kind, AstNode* definer,
      std::string_view name) override {
    CHECK(definer->GetSpan().has_value());
    Span span = *definer->GetSpan();
    NameDef* name_def = module_.Make<NameDef>(span, std::string(name), definer);
    AddVariable(name_def, std::make_unique<InferenceVariable>(
                              definer, name, kind, /*parametric=*/false));
    return module_.Make<NameRef>(span, std::string(name), name_def);
  }

  absl::StatusOr<NameRef*> DefineParametricVariable(
      const ParametricBinding& binding) override {
    XLS_ASSIGN_OR_RETURN(
        InferenceVariableKind kind,
        TypeAnnotationToInferenceVariableKind(binding.type_annotation()));
    const NameDef* name_def = binding.name_def();
    AddVariable(name_def, std::make_unique<InferenceVariable>(
                              name_def, name_def->identifier(), kind,
                              /*parametric=*/true));
    return module_.Make<NameRef>(name_def->span(), name_def->identifier(),
                                 name_def);
  }

  absl::StatusOr<const ParametricInvocation*> AddParametricInvocation(
      const Invocation& node, const Function& callee, const Function& caller,
      std::optional<const ParametricInvocation*> caller_invocation) override {
    if (caller_invocation.has_value()) {
      CHECK(&(*caller_invocation)->callee() == &caller);
    }
    auto invocation = std::make_unique<ParametricInvocation>(
        parametric_invocations_.size(), node, callee, caller,
        caller_invocation);
    const std::vector<ParametricBinding*>& bindings =
        callee.parametric_bindings();
    const std::vector<ExprOrType>& explicit_parametrics =
        node.explicit_parametrics();
    if (explicit_parametrics.size() > bindings.size()) {
      return ArgCountMismatchErrorStatus(
          node.span(),
          absl::Substitute(
              "Too many parametric values supplied; limit: $0 given: $1",
              callee.parametric_bindings().size(), explicit_parametrics.size()),
          file_table_);
    }
    absl::flat_hash_map<const InferenceVariable*, InvocationScopedExpr> values;
    for (int i = 0; i < bindings.size(); i++) {
      const ParametricBinding* binding = bindings[i];
      const InferenceVariable* variable =
          variables_.at(binding->name_def()).get();
      if (i < explicit_parametrics.size()) {
        const ExprOrType value = explicit_parametrics[i];
        if (!std::holds_alternative<Expr*>(value)) {
          return absl::InvalidArgumentError(absl::Substitute(
              "Type inference version 2 is a work in progress and doesn't yet "
              "support types as parametric values: $0",
              node.ToString()));
        }
        values.emplace(
            variable,
            InvocationScopedExpr(caller_invocation, binding->type_annotation(),
                                 std::get<Expr*>(value)));
      } else if (binding->expr() == nullptr) {
        return absl::UnimplementedError(absl::StrCat(
            "Type inference version 2 is a work in progress and doesn't yet "
            "support inferring parametrics from function arguments: ",
            invocation->ToString()));
      } else {
        values.emplace(
            variable,
            InvocationScopedExpr(invocation.get(), binding->type_annotation(),
                                 binding->expr()));
      }
    }
    const ParametricInvocation* result = invocation.get();
    parametric_invocations_.push_back(std::move(invocation));
    parametric_values_by_invocation_.emplace(result, std::move(values));
    return result;
  }

  std::vector<const ParametricInvocation*> GetParametricInvocations()
      const override {
    std::vector<const ParametricInvocation*> result;
    result.reserve(parametric_invocations_.size());
    for (const auto& invocation : parametric_invocations_) {
      result.push_back(invocation.get());
    }
    return result;
  }

  InvocationScopedExpr GetParametricValue(
      const NameDef& binding_name_def,
      const ParametricInvocation& invocation) const override {
    const InferenceVariable* variable = variables_.at(&binding_name_def).get();
    return parametric_values_by_invocation_.at(&invocation).at(variable);
  }

  absl::Status SetTypeAnnotation(const AstNode* node,
                                 const TypeAnnotation* annotation) override {
    return MutateAndCheckNodeData(
        node, [=](NodeData& data) { data.type_annotation = annotation; });
  }

  absl::Status SetTypeVariable(const AstNode* node,
                               const NameRef* type) override {
    XLS_ASSIGN_OR_RETURN(InferenceVariable * variable, GetVariable(type));
    if (variable->kind() != InferenceVariableKind::kType) {
      return absl::InvalidArgumentError(
          absl::Substitute("Setting the type of $0 to non-type variable: $1",
                           node->ToString(), variable->ToString()));
    }
    return MutateAndCheckNodeData(
        node, [=](NodeData& data) { data.type_variable = variable; });
  }

  std::vector<const AstNode*> GetStaticNodes() const override {
    return ConvertNodeSetToOrderedVector(static_nodes_);
  }

  // Returns all the nodes that have information in the table and whose types
  // have distinct concretizations in the context of the given `invocation`. The
  // nodes are returned in the order added to the table.
  std::vector<const AstNode*> GetNodesWithInvocationSpecificTypes(
      const ParametricInvocation* invocation) const override {
    absl::flat_hash_set<const AstNode*> dependents;
    for (const ParametricBinding* binding :
         invocation->callee().parametric_bindings()) {
      const InferenceVariable* variable =
          variables_.at(binding->name_def()).get();
      const auto it = variable_dependents_.find(variable);
      if (it != variable_dependents_.end()) {
        absl::c_copy(it->second, std::inserter(dependents, dependents.end()));
      }
    }
    return ConvertNodeSetToOrderedVector(dependents);
  }

  std::optional<const TypeAnnotation*> GetTypeAnnotation(
      const AstNode* node) const override {
    const auto it = node_data_.find(node);
    if (it == node_data_.end()) {
      return std::nullopt;
    }
    return it->second.type_annotation;
  }

 private:
  void AddVariable(const NameDef* name_def,
                   std::unique_ptr<InferenceVariable> variable) {
    if (variable->kind() == InferenceVariableKind::kType) {
      type_constraints_.emplace(variable.get(),
                                std::make_unique<TypeConstraints>());
    }
    variables_.emplace(name_def, std::move(variable));
  }

  absl::StatusOr<InferenceVariable*> GetVariable(const NameRef* ref) {
    if (std::holds_alternative<const NameDef*>(ref->name_def())) {
      const auto it =
          variables_.find(std::get<const NameDef*>(ref->name_def()));
      if (it != variables_.end()) {
        return it->second.get();
      }
    }
    return absl::NotFoundError(absl::Substitute(
        "No inference variable for NameRef: $0", ref->ToString()));
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
      static_nodes_.insert(node);
    }
    bool had_type_annotation_before = node_data.type_annotation.has_value();
    mutator(node_data);
    // Refine and check the associated type variable.
    if (node_data.type_variable.has_value() &&
        node_data.type_annotation.has_value()) {
      XLS_RETURN_IF_ERROR(RefineAndCheckTypeVariable(
          *node_data.type_variable, *node_data.type_annotation));
    }
    // Update the dependencies of the node.
    if (node_data.type_variable.has_value()) {
      AddDependency(node, *node_data.type_variable);
    }
    if (!had_type_annotation_before && node_data.type_annotation.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          absl::flat_hash_set<const InferenceVariable*> referenced_variables,
          GetReferencedVariables(*node_data.type_annotation));
      for (const InferenceVariable* variable : referenced_variables) {
        AddDependency(node, variable);
      }
    }
    return absl::OkStatus();
  }

  // Refines what is known about the given `variable` (which is assumed to be a
  // type-kind variable) based on the given `annotation` that it must satisfy,
  // and errors if there is a conflict with existing information.
  absl::Status RefineAndCheckTypeVariable(const InferenceVariable* variable,
                                          const TypeAnnotation* annotation) {
    const auto* builtin_annotation =
        dynamic_cast<const BuiltinTypeAnnotation*>(annotation);
    if (builtin_annotation == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Type inference version 2 does not yet support refining "
                       "and updating a variable with type annotation: ",
                       annotation->ToString()));
    }
    TypeConstraints& constraints = *type_constraints_[variable];
    if (!constraints.min_width.has_value() ||
        builtin_annotation->GetBitCount() > *constraints.min_width) {
      constraints.min_width = builtin_annotation->GetBitCount();
    }
    XLS_ASSIGN_OR_RETURN(const bool annotation_is_signed,
                         builtin_annotation->GetSignedness());
    if (constraints.is_signed.has_value() &&
        annotation_is_signed != *constraints.is_signed) {
      return SignednessMismatchErrorStatus(
          annotation, *constraints.signedness_definer, file_table_);
    } else if (!constraints.is_signed.has_value()) {
      constraints.is_signed = annotation_is_signed;
      constraints.signedness_definer = annotation;
    }
    return absl::OkStatus();
  }

  void AddDependency(const AstNode* node, const InferenceVariable* variable) {
    variable_dependents_[variable].insert(node);
    if (variable->parametric()) {
      // Nodes are considered static by default, and they lose their static-ness
      // when they are determined to depend on a parametric.
      static_nodes_.erase(node);
    }
  }

  std::vector<const AstNode*> ConvertNodeSetToOrderedVector(
      const absl::flat_hash_set<const AstNode*>& nodes) const {
    std::vector<const AstNode*> result;
    absl::c_copy(nodes, std::back_inserter(result));
    absl::c_sort(result, [this](const AstNode* x, const AstNode* y) {
      return node_data_.at(x).order_added < node_data_.at(y).order_added;
    });
    return result;
  }

  absl::StatusOr<absl::flat_hash_set<const InferenceVariable*>>
  GetReferencedVariables(const AstNode* node) {
    InferenceVariableDiscoveryVisitor visitor(
        [this](const NameRef* ref) { return GetVariable(ref); });
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    return visitor.GetDiscoveredVariables();
  }

  Module& module_;
  const FileTable& file_table_;
  // The variables of all kinds that have been defined by the user or
  // internally.
  absl::flat_hash_map<const NameDef*, std::unique_ptr<InferenceVariable>>
      variables_;
  // The constraints that have been determined for `variables_` that are
  // of `kType` kind.
  absl::flat_hash_map<const InferenceVariable*,
                      std::unique_ptr<TypeConstraints>>
      type_constraints_;
  // The `AstNode` objects that have associated data.
  absl::flat_hash_map<const AstNode*, NodeData> node_data_;
  // Which `AstNode` objects depend on which variables.
  absl::flat_hash_map<const InferenceVariable*,
                      absl::flat_hash_set<const AstNode*>>
      variable_dependents_;
  // These nodes don't depend on any parametric variables, so their associated
  // types have a single concretization.
  absl::flat_hash_set<const AstNode*> static_nodes_;
  // Parametric invocations and the corresponding information about parametric
  // variables.
  std::vector<std::unique_ptr<ParametricInvocation>> parametric_invocations_;
  absl::flat_hash_map<
      const ParametricInvocation*,
      absl::flat_hash_map<const InferenceVariable*, InvocationScopedExpr>>
      parametric_values_by_invocation_;
};

}  // namespace

InferenceTable::~InferenceTable() = default;

std::unique_ptr<InferenceTable> InferenceTable::Create(
    Module& module, const FileTable& file_table) {
  return std::make_unique<InferenceTableImpl>(module, file_table);
}

}  // namespace xls::dslx
