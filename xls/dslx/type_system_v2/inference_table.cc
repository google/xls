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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

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

// Represents the immutable metadata for a variable in an `InferenceTable`.
class InferenceVariable {
 public:
  InferenceVariable(const AstNode* definer, std::string_view name,
                    InferenceVariableKind kind)
      : definer_(definer), name_(name), kind_(kind) {}

  const AstNode* definer() const { return definer_; }

  std::string_view name() const { return name_; }

  InferenceVariableKind kind() const { return kind_; }

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
};

// The mutable data for a node in an `InferenceTable`.
struct NodeData {
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
    AddVariable(name_def,
                std::make_unique<InferenceVariable>(definer, name, kind));
    return module_.Make<NameRef>(span, std::string(name), name_def);
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

  const std::vector<const AstNode*>& GetNodes() const override {
    return nodes_in_order_added_;
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
    if (inserted) {
      nodes_in_order_added_.push_back(node);
    }
    NodeData& node_data = it->second;
    mutator(node_data);
    if (node_data.type_variable.has_value() &&
        node_data.type_annotation.has_value()) {
      return RefineAndCheckTypeVariable(*node_data.type_variable,
                                        *node_data.type_annotation);
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
  std::vector<const AstNode*> nodes_in_order_added_;
  absl::flat_hash_map<const AstNode*, NodeData> node_data_;
};

}  // namespace

InferenceTable::~InferenceTable() = default;

std::unique_ptr<InferenceTable> InferenceTable::Create(
    Module& module, const FileTable& file_table) {
  return std::make_unique<InferenceTableImpl>(module, file_table);
}

}  // namespace xls::dslx
