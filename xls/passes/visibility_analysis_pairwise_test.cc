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

#include "xls/passes/visibility_analysis_pairwise.h"

#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;

using OperandNode = OperandVisibilityAnalysis::OperandNode;

class PairwiseVisibilityAnalysisTest : public IrTestBase {
 protected:
  struct VisibilityAnalyses {
    NodeForwardDependencyAnalysis nda;
    LazyPostDominatorAnalysis post_dom;
    std::unique_ptr<BddQueryEngine> bdd_engine;
    std::optional<OperandVisibilityAnalysis> operand_vis;
    std::unique_ptr<VisibilityAnalysis> general_vis;
    std::unique_ptr<PairwiseVisibilityAnalysis> pairwise_vis;

    BinaryDecisionDiagram& bdd() const { return bdd_engine->bdd(); }
  };

  absl::StatusOr<std::unique_ptr<VisibilityAnalyses>> CreateVisibilityAnalyses(
      FunctionBase* f) {
    auto analyses = std::make_unique<VisibilityAnalyses>();
    XLS_RETURN_IF_ERROR(analyses->nda.Attach(f).status());
    XLS_RETURN_IF_ERROR(analyses->post_dom.Attach(f).status());
    analyses->bdd_engine = BddQueryEngine::MakeDefault();
    XLS_RETURN_IF_ERROR(analyses->bdd_engine->Populate(f).status());
    XLS_ASSIGN_OR_RETURN(analyses->operand_vis,
                         OperandVisibilityAnalysis::Create(
                             &analyses->nda, analyses->bdd_engine.get()));
    XLS_ASSIGN_OR_RETURN(analyses->general_vis,
                         VisibilityAnalysis::Create(&*analyses->operand_vis,
                                                    analyses->bdd_engine.get(),
                                                    &analyses->post_dom));
    XLS_ASSIGN_OR_RETURN(analyses->pairwise_vis,
                         PairwiseVisibilityAnalysis::Create(
                             *analyses->operand_vis, *analyses->general_vis));
    return analyses;
  }
};

TEST_F(PairwiseVisibilityAnalysisTest, BddVariablesAnalysisQueriesAndClears) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex v0 = bdd.NewVariable();
  BddNodeIndex v1 = bdd.NewVariable();
  BddNodeIndex v2 = bdd.NewVariable();
  BddNodeIndex expr = bdd.And(v0, bdd.Or(v2, bdd.Not(v0)));

  BddVariablesAnalysis analysis(&bdd);
  BddVariable var0 = bdd.GetNode(v0).variable;
  BddVariable var1 = bdd.GetNode(v1).variable;
  BddVariable var2 = bdd.GetNode(v2).variable;

  EXPECT_TRUE(analysis.Variables(BinaryDecisionDiagram::kInfeasible).empty());
  EXPECT_THAT(analysis.Variables(expr), ElementsAre(var0, var2));
  EXPECT_TRUE(analysis.UsesVariables(expr, {var2}));
  EXPECT_FALSE(analysis.UsesVariables(expr, {var1}));
  EXPECT_FALSE(analysis.UsesVariables(bdd.one(), {var0, var1, var2}));

  analysis.Clear();
  EXPECT_THAT(analysis.Variables(expr), ElementsAre(var0, var2));
}

TEST_F(PairwiseVisibilityAnalysisTest, PairwiseVisibilityReducesPathSize) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue opcode = fb.Param("opcode", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue mode_1 = fb.Param("mode_1", p->GetBitsType(1));
  BValue mode_2 = fb.Param("mode_2", p->GetBitsType(1));

  // Opcode 0: shll_1 = shll(a, b). Opcode 1: shll_2 = shll(b, a).
  // Both are also guarded by their own specific mode selects.
  BValue shll_1 = fb.Shll(a, b);
  BValue op0_res = fb.Select(fb.Eq(mode_1, fb.Literal(UBits(1, 1))),
                             /*cases=*/{shll_1, fb.Literal(UBits(0, 8))});
  BValue shll_2 = fb.Shll(b, a);
  BValue op1_res = fb.Select(fb.Eq(mode_2, fb.Literal(UBits(1, 1))),
                             /*cases=*/{shll_2, fb.Literal(UBits(0, 8))});
  BValue op0_filtered =
      fb.And(op0_res, fb.SignExtend(fb.Eq(opcode, fb.Literal(UBits(0, 2))), 8));
  BValue op1_filtered =
      fb.And(op1_res, fb.SignExtend(fb.Eq(opcode, fb.Literal(UBits(1, 2))), 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple({op0_filtered, op1_filtered})));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VisibilityAnalyses> vis,
                           CreateVisibilityAnalyses(f));

  Node* node1 = shll_1.node();
  Node* node2 = shll_2.node();

  BddNodeIndex gen_vis1 = vis->general_vis->GetInfo(node1)->visibility;
  BddNodeIndex gen_vis2 = vis->general_vis->GetInfo(node2)->visibility;
  auto [pair_vis1, pair_vis2] =
      vis->pairwise_vis->PairwiseVisibility(node1, node2);

  // Verify that pairwise visibility strictly reduces path size.
  EXPECT_LT(vis->bdd().path_count(pair_vis1), vis->bdd().path_count(gen_vis1));
  EXPECT_LT(vis->bdd().path_count(pair_vis2), vis->bdd().path_count(gen_vis2));

  // Verify that pairwise visibility expressions still prove mutual exclusivity.
  EXPECT_TRUE(vis->pairwise_vis->IsMutuallyExclusive(node1, node2));
  EXPECT_TRUE(vis->bdd().MutuallyExclusive(pair_vis1, pair_vis2));

  // Verify that the pair-specific expressions only use variables used by both.
  const std::vector<BddVariable>& vars_pair1 =
      vis->pairwise_vis->bdd_variables().Variables(pair_vis1);
  const std::vector<BddVariable>& vars_pair2 =
      vis->pairwise_vis->bdd_variables().Variables(pair_vis2);
  EXPECT_EQ(vars_pair1, vars_pair2);

  // Verify that pairwise visibility edges only include edges dependent on the
  // intersection variables (opcode filter via op0_filtered / op1_filtered) and
  // exclude operation-specific constraints.
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> edges1,
      vis->pairwise_vis->GetEdgesForVisibilityExpr(node1, node2));
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> edges2,
      vis->pairwise_vis->GetEdgesForVisibilityExpr(node2, node1));
  EXPECT_THAT(edges1,
              ElementsAre(Field(&OperandNode::node, op0_filtered.node())));
  EXPECT_THAT(edges2,
              ElementsAre(Field(&OperandNode::node, op1_filtered.node())));
}

TEST_F(PairwiseVisibilityAnalysisTest, PairwiseVisibilitySameNode) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue add_node = fb.Add(a, b);
  BValue sel_out = fb.Select(sel, /*cases=*/{add_node, a});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel_out));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VisibilityAnalyses> vis,
                           CreateVisibilityAnalyses(f));

  Node* node = add_node.node();
  auto [vis1, vis2] = vis->pairwise_vis->PairwiseVisibility(node, node);
  EXPECT_EQ(vis1, vis2);
  EXPECT_EQ(vis1, vis->general_vis->GetInfo(node)->visibility);
}

TEST_F(PairwiseVisibilityAnalysisTest,
       PairwiseVisibilityPrunesRedundantSharedVariableEdges) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  BValue opcode = fb.Param("opcode", p->GetBitsType(2));
  BValue shared_en = fb.Param("shared_en", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue zero_u8 = fb.Literal(UBits(0, 8));

  // Both shll_1 and shll_2 are gated by shared_en (so shared_en is in the
  // variable intersection), and then mutually exclusively filtered by opcode.
  BValue shll_1 = fb.Shll(a, b);
  BValue op0_en = fb.Select(shared_en, /*cases=*/{zero_u8, shll_1});
  BValue op0_filtered =
      fb.And(op0_en, fb.SignExtend(fb.Eq(opcode, fb.Literal(UBits(0, 2))), 8));

  BValue shll_2 = fb.Shll(b, a);
  BValue op1_en = fb.Select(shared_en, /*cases=*/{zero_u8, shll_2});
  BValue op1_filtered =
      fb.And(op1_en, fb.SignExtend(fb.Eq(opcode, fb.Literal(UBits(1, 2))), 8));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple({op0_filtered, op1_filtered})));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VisibilityAnalyses> vis,
                           CreateVisibilityAnalyses(f));

  // Although shared_en is in the BDD variable intersection of both nodes,
  // GetEdgesForMutuallyExclusiveVisibilityExpr prunes op0_en / op1_en because
  // op0_filtered / op1_filtered (opcode == 0 vs opcode == 1) suffices for
  // mutual exclusivity.
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> edges1,
      vis->pairwise_vis->GetEdgesForMutuallyExclusiveVisibilityExpr(
          shll_1.node(), {shll_2.node()}, /*max_edges_to_handle=*/-1));
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> edges2,
      vis->pairwise_vis->GetEdgesForMutuallyExclusiveVisibilityExpr(
          shll_2.node(), {shll_1.node()}, /*max_edges_to_handle=*/-1));
  EXPECT_THAT(edges1,
              ElementsAre(Field(&OperandNode::node, op0_filtered.node())));
  EXPECT_THAT(edges2,
              ElementsAre(Field(&OperandNode::node, op1_filtered.node())));
}

// Illustrates a limitation with how PairwiseVisibilityAnalysis filters edges
// based on variable intersection. This example is somewhat contrived, and the
// assumption is that such a construction would not commonly be observed.
TEST_F(PairwiseVisibilityAnalysisTest,
       PairwiseVisibilityMissesNecessaryConditionOnNonIntersectionVariable) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));

  // op1 is gated in series by two edges: e1: y, e2: !y || x
  // Together, Vis(op1) = y && (!y || x) = y && x.
  BValue op1 = fb.Shll(a, b);
  BValue op1_gated_by_y = fb.Select(y, /*cases=*/{zero, op1});
  BValue not_y_or_x = fb.Or(fb.Not(y), x);
  BValue op1_final = fb.Select(not_y_or_x, /*cases=*/{zero, op1_gated_by_y});

  // op2 is gated only by e3: !x; Vis(op2) = !x
  BValue op2 = fb.Shll(b, a);
  BValue op2_final = fb.Select(x, /*cases=*/{op2, zero});
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({op1_final, op2_final})));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VisibilityAnalyses> vis,
                           CreateVisibilityAnalyses(f));

  // Full visibility analysis keeps both e1 (y) and e2 (!y || x), simplifying
  // V(op1) to (y && x), which is mutually exclusive with V(op2) = !x.
  EXPECT_TRUE(vis->general_vis->IsMutuallyExclusive(op1.node(), op2.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> full_edges,
      vis->general_vis->GetEdgesForMutuallyExclusiveVisibilityExpr(
          op1.node(), {op2.node()}, /*max_edges_to_handle=*/-1));
  EXPECT_EQ(full_edges.size(), 2);

  // Limitation: The BDD variable intersection between op1 ({x, y}) and op2
  // ({x}) is {x}. Because e1 (op1 -> op1_gated_by_y) only depends on {y},
  // ComputeExcludedEdges excludes e1 even though y is coupled to x in e2
  // (!y || x). Without e1, pairwise V(op1) weakens to (!y || x), which cannot
  // prove mutual exclusivity with V(op2) = !x (both are true when x=0, y=0).
  EXPECT_FALSE(vis->pairwise_vis->IsMutuallyExclusive(op1.node(), op2.node()));
  XLS_ASSERT_OK_AND_ASSIGN(
      absl::flat_hash_set<OperandNode> pairwise_edges,
      vis->pairwise_vis->GetEdgesForMutuallyExclusiveVisibilityExpr(
          op1.node(), {op2.node()}, /*max_edges_to_handle=*/-1));
  EXPECT_TRUE(pairwise_edges.empty());
}

}  // namespace
}  // namespace xls
