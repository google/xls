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

#ifndef XLS_PASSES_VISIBILITY_EXPR_BUILDER_H_
#define XLS_PASSES_VISIBILITY_EXPR_BUILDER_H_

#include <cstdint>
#include <optional>
#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/area_accumulated_analysis.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/expression_builder.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

class VisibilityBuilder : public ExpressionBuilder {
 public:
  static constexpr int64_t kMaxCasesToCheckImplyNoPrevCase = 100;

  VisibilityBuilder(int64_t prior_existing_id, const BddQueryEngine* bdd_engine,
                    const NodeForwardDependencyAnalysis& nda,
                    BitProvenanceAnalysis& bpa)
      : ExpressionBuilder(nda.bound_function(), bdd_engine),
        prior_existing_id_(prior_existing_id),
        nda_(nda),
        bpa_(bpa),
        visibility_expr_cache_{} {}

 private:
  // The largest node id that existed in the function before visibility
  // expressions began being added; used to ignore nodes added during
  // visibility expression construction.
  int64_t prior_existing_id_;
  // Dependency analysis is used to prevent creating visibility expressions
  // that include the source as a term in the expression.
  const NodeForwardDependencyAnalysis& nda_;
  BitProvenanceAnalysis& bpa_;
  absl::flat_hash_map<std::tuple<Node*, Node*, FunctionBase*>, Node*>
      visibility_expr_cache_;

 public:
  // BuildVisibilityIRExpr adds IR to the function representing a visibility
  // expression for a node such that the node being visible implies the
  // expression is true. The subset of visibility constraining edges to be used
  // is provided by the caller. These edges would likely be obtained from the
  // visibility API GetEdgesForMutuallyExclusiveVisibilityExpr.
  //
  // Consider this simple IR example:
  //
  //   inputs: (x: u32 , y: u32 , z: u32 , op1: u4 , op2: u4)
  //   sel1 = select(op1, {x, y, x}, y)
  //   lt1  = lt(op2, 5)
  //   and1 = and(x, sign_ext(lt1, 32))
  //   sel2 = select(op1, {y, z, y}, and1)
  //   ret tuple(sel1, sel2)
  //
  // x is used if (op1 == 0 || op1 == 2) || (op1 == 3 || op1 == 4)
  //
  // A couple of possible expressions for x are:
  //   1. or(or(eq(op1, 0), eq(op1, 2)), and(lt1, uge(op1, 3)))
  //      exact visibility expression
  //      incurs the cost of 2 'or', 2 'eq', 1 'and', and 1 'uge'
  //   2. or(or(eq(op1, 0), eq(op1, 2)), lt1)
  //      assumes visible as long as op1 == 0 || op1 == 2 || op1 < 5
  //      incurs the cost of 2 'or', 2 'eq'
  //   3. or(or(eq(op1, 0), eq(op1, 2)), uge(op1, 3))
  //      assumes visible as long as op1 == 0 || op1 == 2 || op1 >= 3
  //      incurs the cost of 2 'or', 2 'eq', and 1 'uge'
  //
  // In the above example, suppose the returned expression must be mutually
  // exclusive with the visibility of 'z'. Edges corresponding to expression (2)
  // would NOT work because 'lt1' does not imply 'z' is not visible. However,
  // edges corresponding to expression (3) would work because 'uge(op1, 3)'
  // implies 'z' is not visible.
  absl::StatusOr<Node*> BuildVisibilityIRExpr(
      FunctionBase* func, Node* node,
      const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
          conditional_edges);

 private:
  absl::StatusOr<Node*> MakeParamIfTmpFunc(Node* node, FunctionBase* func) {
    return func == TmpFunc() ? TmpFuncNodeOrParam(node) : node;
  }

  absl::StatusOr<Node*> GetSelectorIfIndependent(Node* node, Node* select,
                                                 Node* source,
                                                 FunctionBase* func);
  bool DoesCaseImplyNoPrevCase(PrioritySelect* select, int64_t case_index);

  absl::StatusOr<Node*> GetVisibilityExprForPrioritySelect(
      Node* node, PrioritySelect* select, Node* source, FunctionBase* func);
  absl::StatusOr<Node*> GetVisibilityExprForSelect(Node* node, Select* select,
                                                   Node* source,
                                                   FunctionBase* func);
  absl::StatusOr<Node*> GetVisibilityExprForAnd(Node* node, NaryOp* and_node,
                                                Node* source,
                                                FunctionBase* func);
  absl::StatusOr<Node*> GetVisibilityExprForOr(Node* node, NaryOp* or_node,
                                               Node* source,
                                               FunctionBase* func);
  absl::StatusOr<Node*> GetVisibilityExprForPredicate(
      std::optional<Node*> predicate, Node* source, FunctionBase* func);

  absl::StatusOr<Node*> BuildVisibilityExprHelper(Node* node, Node* user,
                                                  Node* source,
                                                  FunctionBase* func);
  // Builds predicate for node `u` being used by `v` on `func`.
  absl::StatusOr<Node*> BuildVisibilityExpr(Node* node, Node* user,
                                            Node* source, FunctionBase* func);
  absl::StatusOr<Node*> BuildNodeAndUserVisibleExpr(FunctionBase* func,
                                                    Node* user_uses_node,
                                                    Node* user_is_used,
                                                    Literal* always_visible);

  absl::StatusOr<Node*> BuildVisibilityIRExprFromEdges(
      FunctionBase* func, Node* node, Node* source,
      const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
          conditional_edges,
      absl::flat_hash_map<Node*, Node*>& node_to_visibility_ir_cache,
      Literal* always_visible);

  absl::StatusOr<Node*> GetNonRepeatedSourceOf(Node* operand,
                                               FunctionBase* func);
};

class VisibilityEstimator : public VisibilityBuilder {
 public:
  VisibilityEstimator(int64_t prior_existing_id,
                      const BddQueryEngine* bdd_engine,
                      const NodeForwardDependencyAnalysis& nda,
                      BitProvenanceAnalysis& bpa,
                      const AreaEstimator* area_estimator,
                      const DelayEstimator* delay_estimator)
      : VisibilityBuilder(prior_existing_id, bdd_engine, nda, bpa),
        area_analysis_(area_estimator),
        delay_analysis_(delay_estimator) {
    CHECK_OK(area_analysis_.Attach(TmpFunc()).status());
    CHECK_OK(delay_analysis_.Attach(TmpFunc()).status());
  }

 private:
  AreaAccumulatedAnalysis area_analysis_;
  CriticalPathDelayAnalysis delay_analysis_;

 public:
  struct AreaDelay {
    double area;
    int64_t delay;
  };

  absl::StatusOr<AreaDelay> GetAreaAndDelayOfVisibilityExpr(
      Node* node,
      const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
          conditional_edges);
};

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_EXPR_BUILDER_H_
