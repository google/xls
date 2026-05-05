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

#include "xls/passes/visibility_expr_builder.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/visibility_analysis.h"
#include "xls/visualization/math_notation.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

class VisibilityExprBuilderTest : public IrTestBase {
 protected:
  absl::StatusOr<std::pair<Node*, VisibilityEstimator::AreaDelay>>
  BuildDefaultVisibilityExpr(
      Function* f, Node* node,
      std::vector<Node*> mutually_exclusive_requirements) {
    NodeForwardDependencyAnalysis nda;
    XLS_RETURN_IF_ERROR(nda.Attach(f).status());
    LazyPostDominatorAnalysis post_dom;
    XLS_RETURN_IF_ERROR(post_dom.Attach(f).status());
    std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
    XLS_RETURN_IF_ERROR(bdd_engine->Populate(f).status());
    XLS_ASSIGN_OR_RETURN(
        auto operand_visibility,
        OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
    XLS_ASSIGN_OR_RETURN(auto visibility,
                         VisibilityAnalysis::Create(
                             &operand_visibility, bdd_engine.get(), &post_dom));
    XLS_ASSIGN_OR_RETURN(auto conditional_edges,
                         visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(
                             node, mutually_exclusive_requirements, -1));
    VLOG(3) << "node: " << node->ToString() << "\n";
    VLOG(3) << "conditional_edges: " << conditional_edges.size() << "\n";
    for (auto edge : conditional_edges) {
      VLOG(3) << "edge: " << edge.operand->ToString() << " -> "
              << edge.node->ToString() << "\n";
    }
    auto last_node_id = f->nodes_reversed().begin()->id();
    XLS_ASSIGN_OR_RETURN(AreaEstimator * ae, GetAreaEstimator("unit"));
    XLS_ASSIGN_OR_RETURN(DelayEstimator * de, GetDelayEstimator("unit"));
    BitProvenanceAnalysis bpa;
    VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                  de);
    XLS_ASSIGN_OR_RETURN(Node * expr, estimator.BuildVisibilityIRExpr(
                                          f, node, conditional_edges));
    XLS_ASSIGN_OR_RETURN(
        auto area_delay,
        estimator.GetAreaAndDelayOfVisibilityExpr(node, conditional_edges));
    return std::make_pair(expr, area_delay);
  }
};

TEST_F(VisibilityExprBuilderTest, ExampleInFunctionHeaderComment) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));
  BValue op1 = fb.Param("op1", p->GetBitsType(4));
  BValue op2 = fb.Param("op2", p->GetBitsType(4));
  BValue select1 = fb.Select(op1, {x, y, x}, y);
  BValue lt1 = fb.ULt(op2, fb.Literal(UBits(5, 4)));
  BValue and1 = fb.And(x, fb.SignExtend(lt1, 32));
  BValue select2 = fb.Select(op1, {y, z, y}, and1);
  BValue ret = fb.Tuple({select1, select2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used,
                           BuildDefaultVisibilityExpr(f, x.node(), {}));
  std::pair<Node*, VisibilityEstimator::AreaDelay> is_y_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_y_used,
                           BuildDefaultVisibilityExpr(f, y.node(), {}));
  VLOG(3) << f->DumpIr();
  VLOG(3) << "is 'x' used:\n" << ToMathNotation(is_x_used.first);
  VLOG(3) << "is 'y' used:\n" << ToMathNotation(is_y_used.first);

  EXPECT_THAT(is_x_used.first,
              m::Or(m::Eq(m::Param("op1"), m::Literal(0)),
                    m::Eq(m::Param("op1"), m::Literal(2)),
                    m::And(m::ULt(m::Param("op2"), m::Literal(5)),
                           m::UGe(m::Param("op1"), m::Literal(3)))));
  // 5 instead of 6 because of flat OR gate consolidation.
  EXPECT_EQ(is_x_used.second.area, 5);
  EXPECT_EQ(is_x_used.second.delay, 3);
  EXPECT_THAT(is_y_used.first, m::Or(m::Eq(m::Param("op1"), m::Literal(1)),
                                     m::UGe(m::Param("op1"), m::Literal(3)),
                                     m::Eq(m::Param("op1"), m::Literal(0)),
                                     m::Eq(m::Param("op1"), m::Literal(2))));
  EXPECT_EQ(is_y_used.second.area, 5);
  EXPECT_EQ(is_y_used.second.delay, 2);

  // Now that the returned expression must be mutually exclusive with z's
  // visibility, it must condition on the selection criteria of 'select2'.
  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used_and_z_not;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used_and_z_not,
                           BuildDefaultVisibilityExpr(f, x.node(), {z.node()}));
  VLOG(3) << "is 'x' used and 'z' not used:\n"
          << ToMathNotation(is_x_used_and_z_not.first);
  EXPECT_THAT(is_x_used_and_z_not.first,
              m::Or(m::Eq(m::Param("op1"), m::Literal(0)),
                    m::Eq(m::Param("op1"), m::Literal(2)),
                    m::UGe(m::Param("op1"), m::Literal(3))));
  EXPECT_EQ(is_x_used_and_z_not.second.area, 4);
  EXPECT_EQ(is_x_used_and_z_not.second.delay, 2);
}

TEST_F(VisibilityExprBuilderTest, PrioritySelectOneHot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(3));
  BValue z = fb.Param("z", p->GetBitsType(3));
  BValue y_is_0 = fb.Eq(y, fb.Literal(UBits(0, 3)));
  BValue y_is_1 = fb.Eq(y, fb.Literal(UBits(1, 3)));
  BValue y_is_2 = fb.Eq(y, fb.Literal(UBits(2, 3)));
  BValue selectors = fb.Concat({y_is_0, y_is_1, y_is_2});
  BValue prio =
      fb.PrioritySelect(selectors, {z, x, x}, fb.Literal(UBits(0, 3)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prio));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used,
                           BuildDefaultVisibilityExpr(f, x.node(), {}));
  // NOTE: operands of the 'or' are sorted by id, so the y==0 comes first even
  // though the first case in the priority select corresponds to y==1 selector.
  ASSERT_THAT(is_x_used.first, m::Or(m::Eq(m::Param("y"), m::Literal(0)),
                                     m::Eq(m::Param("y"), m::Literal(1))));
  EXPECT_EQ(is_x_used.first->operand(0), y_is_0.node());
  EXPECT_EQ(is_x_used.first->operand(1), y_is_1.node());
}

TEST_F(VisibilityExprBuilderTest, PrioritySelectNotOneHot) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(3));
  BValue z = fb.Param("z", p->GetBitsType(3));
  BValue y_is_2 = fb.Eq(y, fb.Literal(UBits(2, 3)));
  BValue y_ge_1 = fb.UGe(y, fb.Literal(UBits(1, 3)));
  BValue y_ge_0 = fb.UGe(y, fb.Literal(UBits(0, 3)));
  BValue selectors = fb.Concat({y_ge_0, y_ge_1, y_is_2});
  BValue prio =
      fb.PrioritySelect(selectors, {x, x, z}, fb.Literal(UBits(0, 3)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prio));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used,
                           BuildDefaultVisibilityExpr(f, x.node(), {z.node()}));
  VLOG(1) << ToMathNotation(is_x_used.first);
  ASSERT_THAT(is_x_used.first,
              m::Or(m::Eq(m::Param("y"), m::Literal(2)),
                    m::Eq(m::BitSlice(selectors.node(), 0, 2), m::Literal(2))));
  EXPECT_EQ(is_x_used.first->operand(0), y_is_2.node());
}

TEST_F(VisibilityExprBuilderTest, Ors) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue bit1 = fb.BitSlice(x, 1, 1);
  BValue or_y = fb.Or(fb.SignExtend(bit1, 2), y);
  BValue bits12 = fb.BitSlice(x, 1, 2);
  BValue or_y2 = fb.Or(bits12, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({or_y, or_y2})));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_y_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_y_used,
                           BuildDefaultVisibilityExpr(f, y.node(), {}));
  EXPECT_THAT(is_y_used.first,
              m::Or(m::Not(m::BitSlice(m::Param("x"), 1, 1)),
                    m::Ne(m::BitSlice(m::Param("x"), 1, 2), m::Literal(3))));
}

TEST_F(VisibilityExprBuilderTest, Ands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue bits12 = fb.BitSlice(x, 1, 2);
  BValue and_y = fb.And(bits12, y);
  BValue bit1 = fb.BitSlice(x, 1, 1);
  BValue and_y2 = fb.And(fb.SignExtend(bit1, 2), and_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(and_y2));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_y_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_y_used,
                           BuildDefaultVisibilityExpr(f, y.node(), {}));
  EXPECT_THAT(is_y_used.first,
              m::And(m::Ne(m::BitSlice(m::Param("x"), 1, 2), m::Literal(0)),
                     m::BitSlice(m::Param("x"), 1, 1)));
}

TEST_F(VisibilityExprBuilderTest, FindsSourceOfOperandInComparison) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(5));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue bits1234 = fb.BitSlice(x, 1, 4);
  BValue bit3 = fb.BitSlice(bits1234, 2, 1);
  BValue y_and_bit3 = fb.And(y, fb.SignExtend(bit3, 4));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(y_and_bit3));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_y_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_y_used,
                           BuildDefaultVisibilityExpr(f, y.node(), {}));
  EXPECT_THAT(is_y_used.first, m::BitSlice(m::Param("x"), 3, 1));
}

TEST_F(VisibilityExprBuilderTest, NotAFunctionOfSelf) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(3));
  BValue and_x = fb.And(x, x);
  BValue or_y = fb.Or(y, and_x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(or_y));

  std::pair<Node*, VisibilityEstimator::AreaDelay> is_x_used;
  XLS_ASSERT_OK_AND_ASSIGN(is_x_used,
                           BuildDefaultVisibilityExpr(f, x.node(), {}));
  EXPECT_THAT(is_x_used.first, m::Ne(m::Param("y"), m::Literal(7)));
}

TEST_F(VisibilityExprBuilderTest, LivenessHalting) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));
  BValue op1 = fb.Param("op1", p->GetBitsType(4));
  BValue op2 = fb.Param("op2", p->GetBitsType(4));
  BValue select1 = fb.Select(op1, {x, y, x}, y);
  BValue lt1 = fb.ULt(op2, fb.Literal(UBits(5, 4)));
  BValue and1 = fb.And(x, fb.SignExtend(lt1, 32));
  BValue select2 = fb.Select(op1, {y, z, y}, and1);
  BValue ret = fb.Tuple({select1, select2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(x.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  // A callback that says 'and1' is not live!
  auto is_live_source = [&](Node* n) { return n != and1.node(); };

  XLS_ASSERT_OK_AND_ASSIGN(Node * expr,
                           estimator.BuildVisibilityIRExpr(
                               f, x.node(), conditional_edges, is_live_source));

  // Because and1 is ignored, we only accumulate its immediate condition (lt1 !=
  // 0) combined with the conditions from select1!
  EXPECT_THAT(expr, m::Or(m::ULt(m::Param("op2"), m::Literal(5)),
                          m::Eq(m::Param("op1"), m::Literal(0)),
                          m::Eq(m::Param("op1"), m::Literal(2))));
}

namespace {

int64_t CountAndTreeOperands(Node* n) {
  if (n->op() != Op::kAnd) {
    return 1;
  }
  int64_t total = 0;
  for (Node* operand : n->operands()) {
    total += CountAndTreeOperands(operand);
  }
  return total;
}

}  // namespace

TEST_F(VisibilityExprBuilderTest, AndPruning) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));

  BValue and_gate = fb.And({a, b, c, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(and_gate));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(y.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  // Mock slacks: we make 'c' have worst slack (10), then 'b' (20), then 'a'
  // (30).
  auto get_remaining_delay = [&](Node* n) -> int64_t {
    if (n == c.node()) {
      return 10;
    }
    if (n == b.node()) {
      return 20;
    }
    if (n == a.node()) {
      return 30;
    }

    // For the resulting AND tree, if it includes all three expected operands,
    // we return -1 (exceeds limit!). Otherwise, we return a positive slack.
    if (n->op() == Op::kAnd && CountAndTreeOperands(n) == 3) {
      return -1;
    }
    if (n->op() == Op::kAnd && CountAndTreeOperands(n) == 2) {
      return 5;
    }
    return 100;
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * expr, estimator.BuildVisibilityIRExpr(
                       f, y.node(), conditional_edges,
                       /*is_live_source=*/nullptr, get_remaining_delay));

  // The expected result should omit 'c'! So it should be AND(a, b)!
  EXPECT_THAT(expr, m::And(m::Param("a"), m::Param("b")));
}

TEST_F(VisibilityExprBuilderTest, OrPruning) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue bit1 = fb.BitSlice(x, 1, 1);
  BValue or_y = fb.Or(fb.SignExtend(bit1, 2), y);
  BValue bits12 = fb.BitSlice(x, 1, 2);
  BValue or_y2 = fb.Or(bits12, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({or_y, or_y2})));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(y.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  // Mock delay limit! If the OR node contains conditions from bits12 (which
  // corresponds to user 2), it will exceed timing slack limits!
  auto get_remaining_delay = [&](Node* n) -> int64_t {
    if (n->op() == Op::kOr) {
      for (Node* operand : n->operands()) {
        if (operand->Is<CompareOp>() && operand->operand(0) == bits12.node()) {
          return -1;
        }
      }
    }
    return 100;
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * expr, estimator.BuildVisibilityIRExpr(
                       f, y.node(), conditional_edges,
                       /*is_live_source=*/nullptr, get_remaining_delay));

  // The expected result should be true because the operation is pruned to
  // literal(1)!
  EXPECT_THAT(expr, m::Literal(1));
}

TEST_F(VisibilityExprBuilderTest, SelectPruning) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue sel = fb.Select(op, {x}, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sel));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(x.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  auto get_remaining_delay = [&](Node* n) -> int64_t {
    if (n->op() == Op::kOr || n->op() == Op::kEq) {
      return -1;
    }
    return 100;
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * expr, estimator.BuildVisibilityIRExpr(
                       f, x.node(), conditional_edges,
                       /*is_live_source=*/nullptr, get_remaining_delay));

  EXPECT_THAT(expr, m::Literal(1));
}

TEST_F(VisibilityExprBuilderTest, PrioritySelectPruning) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue prio = fb.PrioritySelect(sel, {x, y}, fb.Literal(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prio));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(x.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  auto get_remaining_delay = [&](Node* n) -> int64_t {
    if (n->op() == Op::kOr || n->op() == Op::kEq || n->op() == Op::kBitSlice) {
      return -1;
    }
    return 100;
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * expr, estimator.BuildVisibilityIRExpr(
                       f, x.node(), conditional_edges,
                       /*is_live_source=*/nullptr, get_remaining_delay));

  EXPECT_THAT(expr, m::Literal(1));
}

TEST_F(VisibilityExprBuilderTest, TargetStageEnforcement) {
  auto p = CreatePackage();
  ScheduledFunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue select1 = fb.Select(fb.Literal(UBits(1, 1)), {x}, y);
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledFunction * f,
                           fb.BuildWithReturnValue(select1));

  // Add empty stages to make it a scheduled function!
  f->AddEmptyStages(3);

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f).status());
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f).status());
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto conditional_edges,
      visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(x.node(), {}, -1));

  auto last_node_id = f->nodes_reversed().begin()->id();
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * ae, GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * de, GetDelayEstimator("unit"));
  BitProvenanceAnalysis bpa;
  VisibilityEstimator estimator(last_node_id, bdd_engine.get(), nda, bpa, ae,
                                de);

  int64_t kTargetStage = 2;
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * expr,
      estimator.BuildVisibilityIRExpr(
          f, x.node(), conditional_edges, /*is_live_source=*/nullptr,
          /*get_remaining_delay=*/nullptr, kTargetStage));
  (void)expr;

  // All nodes created during BuildVisibilityIRExpr should be assigned to stage
  // 2!
  for (Node* n : f->nodes()) {
    if (n->id() > last_node_id) {
      EXPECT_TRUE(f->IsStaged(n));
      XLS_ASSERT_OK_AND_ASSIGN(int64_t stage, f->GetStageIndex(n));
      EXPECT_EQ(stage, kTargetStage);
    }
  }
}

}  // namespace
}  // namespace xls
