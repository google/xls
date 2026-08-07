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

#include "xls/passes/folding_graph.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class FoldingGraphTest : public IrTestBase {
 protected:
  FoldingGraphTest() = default;
};

TEST_F(FoldingGraphTest, FoldingGraphIsOrdered) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue literal_0 = fb.Literal(UBits(0, 32));
  BValue literal_1 = fb.Literal(UBits(1, 32));
  BValue add = fb.Add(i, j);
  BValue sub = fb.Subtract(i, j);
  BValue mul = fb.UMul(i, j);
  BValue select = fb.PrioritySelect(
      fb.Concat({fb.Eq(op, literal_1), fb.Eq(op, literal_0)}), {add, sub}, mul);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;
  // NOTE: node order is reversed to ensure FoldingGraph does sorting.
  for (Node* one : {mul.node(), sub.node(), add.node()}) {
    for (Node* other : {sub.node(), mul.node(), add.node()}) {
      if (one != other) {
        foldable_actions.push_back(std::make_unique<BinaryFoldingAction>(
            one, other, /*from_edges=*/FoldingAction::VisibilityEdges(),
            /*to_edges=*/FoldingAction::VisibilityEdges(), 0.0));
      }
    }
  }
  FoldingGraph graph(f, std::move(foldable_actions));

  EXPECT_EQ(graph.GetNodes().size(), 3);
  EXPECT_THAT(graph.GetNodes(),
              ElementsAre(add.node(), sub.node(), mul.node()));
  EXPECT_EQ(graph.GetEdges().size(), 6);
  EXPECT_EQ(graph.GetEdges()[0]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[0]->GetTo(), sub.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetFrom(), add.node());
  EXPECT_EQ(graph.GetEdges()[1]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[2]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetFrom(), sub.node());
  EXPECT_EQ(graph.GetEdges()[3]->GetTo(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[4]->GetTo(), add.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetFrom(), mul.node());
  EXPECT_EQ(graph.GetEdges()[5]->GetTo(), sub.node());
}

TEST_F(FoldingGraphTest, WouldCommittingFoldingActionCreateDataCycle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));
  BValue w = fb.Param("w", p->GetBitsType(32));
  BValue add0 = fb.Add(x, y);
  BValue add1 = fb.Add(z, w);
  BValue shared_op_add = fb.Add(x, z);
  BValue dep = fb.Add(add0, z);
  BValue from_with_dep_lhs = fb.Add(dep, w);
  BValue from_with_dep_rhs = fb.Add(w, dep);
  BValue to_dependent_on_from = fb.Add(add1, z);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  FunctionBuilder fb2("other_func", p.get());
  BValue a = fb2.Param("a", p->GetBitsType(32));
  BValue b = fb2.Param("b", p->GetBitsType(32));
  BValue other_add = fb2.Add(a, b);
  XLS_ASSERT_OK(fb2.Build().status());

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));

  {
    // No cycle created when folding two independent nodes.
    BinaryFoldingAction fold(add0.node(), add1.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_FALSE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // No cycle created when to node depends on from node, but operands of from
    // node do not depend on to node.
    BinaryFoldingAction fold(add1.node(), to_dependent_on_from.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_FALSE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // No cycle created when from and to share an operand.
    BinaryFoldingAction fold(shared_op_add.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_FALSE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // Cycle created when from node depends directly on to node.
    BinaryFoldingAction fold(to_dependent_on_from.node(), add1.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_TRUE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // Cycle created when left operand of from node depends on to node.
    BinaryFoldingAction fold(from_with_dep_lhs.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_TRUE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // Cycle created when right operand of from node depends on to node.
    BinaryFoldingAction fold(from_with_dep_rhs.node(), add0.node(),
                             /*from_edges=*/{}, /*to_edges=*/{}, 0.0);
    EXPECT_TRUE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // No cycle created when to node is from a different function.
    BinaryFoldingAction fold(add0.node(), other_add.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_FALSE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
  {
    // No cycle created when from node is from a different function.
    BinaryFoldingAction fold(other_add.node(), add0.node(), /*from_edges=*/{},
                             /*to_edges=*/{}, 0.0);
    EXPECT_FALSE(WouldCommittingFoldingActionCreateDataCycle(nda, fold));
  }
}

TEST_F(FoldingGraphTest, WouldCommittingFoldingActionCreateVisibilityCycle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue sel_not = fb.Not(sel);
  BValue t = fb.Param("t", p->GetBitsType(32));
  BValue f_param = fb.Param("f", p->GetBitsType(32));
  BValue select = fb.Select(sel_not, t, f_param);
  BValue unrelated = fb.Param("unrelated", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.Build());

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));

  {
    // No cycle created when visibility edges don't depend on from/to nodes.
    BinaryFoldingAction fold(
        t.node(), unrelated.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(false));
  }
  {
    // Cycle created when from node is in from_edges dependency path.
    BinaryFoldingAction fold(
        sel.node(), unrelated.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when to node is in from_edges dependency path.
    BinaryFoldingAction fold(
        unrelated.node(), sel.node(),
        /*from_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())},
        /*to_edges=*/{}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when from node is in to_edges dependency path.
    BinaryFoldingAction fold(
        sel.node(), unrelated.node(),
        /*from_edges=*/{},
        /*to_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
  {
    // Cycle created when to node is in to_edges dependency path.
    BinaryFoldingAction fold(
        unrelated.node(), sel.node(),
        /*from_edges=*/{},
        /*to_edges=*/
        {OperandVisibilityAnalysis::OperandNode(t.node(), select.node())}, 0.0);
    EXPECT_THAT(WouldCommittingFoldingActionCreateVisibilityCycle(nda, fold),
                IsOkAndHolds(true));
  }
}

}  // namespace
}  // namespace xls
