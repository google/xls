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

#include "xls/dslx/type_system_v2/expand_variables.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/type_system_v2/inference_table.h"

namespace xls::dslx {

namespace {

// A utility that flattens type annotation trees, with expansion of encountered
// type variables, instead of unification of those variables. This is in
// contrast to `ResolveVariableTypeAnnotations`, which converts encountered
// variables to their unifications. The flattening + expansion behavior of this
// visitor is useful for dependency analysis before we are ready to perform
// unification.
class VariableExpander : public AstNodeVisitorWithDefault {
 public:
  VariableExpander(const InferenceTable& table,
                   std::optional<const ParametricContext*> parametric_context)
      : table_(table), parametric_context_(parametric_context) {}

  absl::Status HandleTypeVariableTypeAnnotation(
      const TypeVariableTypeAnnotation* node) override {
    if (!visited_.insert(node).second) {
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(
        std::vector<const TypeAnnotation*> annotations_for_variable,
        table_.GetTypeAnnotationsForTypeVariable(parametric_context_,
                                                 node->type_variable()));
    for (const TypeAnnotation* annotation : annotations_for_variable) {
      XLS_RETURN_IF_ERROR(annotation->Accept(this));
    }
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode* node) override {
    if (!visited_.insert(node).second) {
      return absl::OkStatus();
    }
    if (std::optional<const NameRef*> variable = table_.GetTypeVariable(node);
        variable.has_value()) {
      XLS_ASSIGN_OR_RETURN(std::vector<const TypeAnnotation*> annotations,
                           table_.GetTypeAnnotationsForTypeVariable(
                               parametric_context_, *variable));
      for (const TypeAnnotation* annotation : annotations) {
        XLS_RETURN_IF_ERROR(annotation->Accept(this));
      }
    }
    if (const auto* annotation = dynamic_cast<const TypeAnnotation*>(node)) {
      annotations_.push_back(annotation);
    }
    for (const AstNode* child : node->GetChildren(/*want_types=*/true)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  const std::vector<const TypeAnnotation*>& annotations() const {
    return annotations_;
  }

 private:
  const InferenceTable& table_;
  const std::optional<const ParametricContext*> parametric_context_;
  std::vector<const TypeAnnotation*> annotations_;
  // Prevent multiple traversal in case of cycles. Cycles can be valid in
  // internally-generated `FunctionTypeAnnotation` data structures that contain
  // `ParamTypeAnnotation` or `ReturnTypeAnnotation` objects in their graph.
  absl::flat_hash_set<const AstNode*> visited_;
};

}  // namespace

std::vector<const TypeAnnotation*> ExpandVariables(
    const TypeAnnotation* annotation, const InferenceTable& table,
    std::optional<const ParametricContext*> parametric_context) {
  VariableExpander expander(table, parametric_context);
  CHECK_OK(annotation->Accept(&expander));
  return expander.annotations();
}

}  // namespace xls::dslx
