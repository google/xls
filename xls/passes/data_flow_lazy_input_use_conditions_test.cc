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

#include "xls/passes/data_flow_lazy_input_use_conditions.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {

class DataFlowLazyInputUseConditionsTest : public IrTestBase {
 public:
  DataFlowLazyInputUseConditionsTest()
      : query_engine_(std::make_unique<PartialInfoQueryEngine>()) {}

  QueryEngine* query_engine() const { return query_engine_.get(); }

 private:
  std::unique_ptr<PartialInfoQueryEngine> query_engine_;
};

TEST_F(DataFlowLazyInputUseConditionsTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(x));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  ConditionsBySinkNode x_conds = {{x.node(), use_conditions.GetLiteralOne(f)}};
  SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
      use_conditions.GetInfo(x.node());
  EXPECT_EQ(x_tree, LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                        p->GetBitsType(32), x_conds));

  EXPECT_EQ(use_conditions.GetConditionsForNode(x.node()), x_conds);
}

TEST_F(DataFlowLazyInputUseConditionsTest, SimpleChange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));

  BValue ret_node = fb.Identity(x, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());
    EXPECT_EQ(x_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32),
                  ConditionsBySinkNode{
                      {ret_node.node(), use_conditions.GetLiteralOne(f)}}));
  }
  {
    SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
        use_conditions.GetInfo(y.node());

    EXPECT_EQ(y_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32), ConditionsBySinkNode{}));

    EXPECT_EQ(use_conditions.GetConditionsForNode(y.node()),
              ConditionsBySinkNode());
  }

  XLS_ASSERT_OK(x.node()->ReplaceUsesWith(y.node()));

  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());

    EXPECT_EQ(x_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32), ConditionsBySinkNode{}));
  }
  {
    SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
        use_conditions.GetInfo(y.node());
    EXPECT_EQ(y_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32),
                  ConditionsBySinkNode{
                      {ret_node.node(), use_conditions.GetLiteralOne(f)}}));
  }
}

TEST_F(DataFlowLazyInputUseConditionsTest, TupleIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));

  BValue t = fb.Tuple({x, y});
  BValue idx = fb.TupleIndex(t, /*idx=*/0);

  BValue ret_node = fb.Identity(idx, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  // TupleIndex is not explicitly implemented, so default behavior is expected,
  // which is to merge.
  SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
      use_conditions.GetInfo(x.node());
  EXPECT_EQ(x_tree,
            LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                p->GetBitsType(32),
                ConditionsBySinkNode{
                    {ret_node.node(), use_conditions.GetLiteralOne(f)}}));

  SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
      use_conditions.GetInfo(y.node());
  EXPECT_EQ(y_tree,
            LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                p->GetBitsType(32),
                ConditionsBySinkNode{
                    {ret_node.node(), use_conditions.GetLiteralOne(f)}}));
}

TEST_F(DataFlowLazyInputUseConditionsTest, Select) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Param("s", p->GetBitsType(2));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));

  BValue sel = fb.Select(s, /*cases=*/std::vector<BValue>{y, x},
                         /*default_value=*/z, SourceInfo(), "sel");

  BValue ret_node = fb.Identity(sel, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<ConditionsBySinkNode> sel_tree =
      use_conditions.GetInfo(sel.node());
  EXPECT_EQ(sel_tree,
            LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                p->GetBitsType(32),
                ConditionsBySinkNode{
                    {ret_node.node(), use_conditions.GetLiteralOne(f)}}));

  // Check that s has one unconditional sink ret_node
  {
    SharedLeafTypeTree<ConditionsBySinkNode> s_tree =
        use_conditions.GetInfo(s.node());

    EXPECT_EQ(s_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(2),
                  ConditionsBySinkNode{
                      {ret_node.node(), use_conditions.GetLiteralOne(f)}}));
  }

  // Check that x has one sink ret_node with condition s == 1
  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());
    ASSERT_EQ(x_tree.elements().size(), 1);
    const ConditionsBySinkNode& x_sinks = x_tree.elements().at(0);
    ASSERT_EQ(x_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *x_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_node = condition_node->operand(0);
    EXPECT_EQ(eq_node->op(), Op::kEq);
    EXPECT_EQ(eq_node->operand(0), s.node());
    EXPECT_EQ(eq_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_literal = eq_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_literal->value(), Value(UBits(1, 2)));
    EXPECT_EQ(condition_node->operand(1), use_conditions.GetLiteralOne(f));
  }

  // Check that y has one sink ret_node with condition s == 0
  {
    SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
        use_conditions.GetInfo(y.node());
    ASSERT_EQ(y_tree.elements().size(), 1);
    const ConditionsBySinkNode& y_sinks = y_tree.elements().at(0);
    ASSERT_EQ(y_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *y_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_node = condition_node->operand(0);
    EXPECT_EQ(eq_node->op(), Op::kEq);
    EXPECT_EQ(eq_node->operand(0), s.node());
    EXPECT_EQ(eq_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_literal = eq_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_literal->value(), Value(UBits(0, 2)));
    EXPECT_EQ(condition_node->operand(1), use_conditions.GetLiteralOne(f));
  }

  // Check that z has one sink ret_node with condition s >= 2
  {
    SharedLeafTypeTree<ConditionsBySinkNode> z_tree =
        use_conditions.GetInfo(z.node());
    ASSERT_EQ(z_tree.elements().size(), 1);
    const ConditionsBySinkNode& z_sinks = z_tree.elements().at(0);
    ASSERT_EQ(z_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *z_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_node = condition_node->operand(0);
    EXPECT_EQ(eq_node->op(), Op::kUGe);
    EXPECT_EQ(eq_node->operand(0), s.node());
    EXPECT_EQ(eq_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_literal = eq_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_literal->value(), Value(UBits(2, 2)));
    EXPECT_EQ(condition_node->operand(1), use_conditions.GetLiteralOne(f));
  }
}

TEST_F(DataFlowLazyInputUseConditionsTest, SelectWithLiteralSelector) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s = fb.Literal(Value(UBits(1, 2)));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));

  BValue sel = fb.Select(s, /*cases=*/std::vector<BValue>{y, x},
                         /*default_value=*/z, SourceInfo(), "sel");

  BValue ret_node = fb.Identity(sel, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  SharedLeafTypeTree<ConditionsBySinkNode> sel_tree =
      use_conditions.GetInfo(sel.node());
  EXPECT_EQ(sel_tree,
            LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                p->GetBitsType(32),
                ConditionsBySinkNode{
                    {ret_node.node(), use_conditions.GetLiteralOne(f)}}));

  // Check that s has no sinks, as it is a literal.
  {
    SharedLeafTypeTree<ConditionsBySinkNode> s_tree =
        use_conditions.GetInfo(s.node());

    EXPECT_EQ(s_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(2), ConditionsBySinkNode{}));
  }

  // Check that x has one unconditional sink ret_node.
  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());

    ASSERT_EQ(x_tree.elements().size(), 1);
    const ConditionsBySinkNode& x_sinks = x_tree.elements().at(0);
    ASSERT_EQ(x_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *x_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node, use_conditions.GetLiteralOne(f));
  }

  // Check that y has no sinks, as it is excluded by the literal selector.
  {
    SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
        use_conditions.GetInfo(y.node());

    EXPECT_EQ(y_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32), ConditionsBySinkNode{}));
  }

  // Check that z has no sinks, as it is excluded by the literal selector.
  {
    SharedLeafTypeTree<ConditionsBySinkNode> z_tree =
        use_conditions.GetInfo(z.node());

    EXPECT_EQ(z_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32), ConditionsBySinkNode{}));
  }

  XLS_ASSERT_OK(
      s.node()->ReplaceUsesWithNew<Literal>(Value(UBits(0, 2))).status());

  // Check that x has no sinks, as it is now excluded by the literal selector.
  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());

    EXPECT_EQ(x_tree,
              LeafTypeTree<ConditionsBySinkNode>::CreateSingleElementTree(
                  p->GetBitsType(32), ConditionsBySinkNode{}));
  }
}

TEST_F(DataFlowLazyInputUseConditionsTest, MultiSelectSeries) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s0 = fb.Param("s0", p->GetBitsType(1));
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));

  BValue sel0 =
      fb.Select(s0, /*on_true=*/x, /*on_false=*/y, SourceInfo(), "sel0");
  BValue sel1 =
      fb.Select(s1, /*on_true=*/sel0, /*on_false=*/z, SourceInfo(), "sel1");

  BValue ret_node = fb.Identity(sel1, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  // Check that z has one sink ret_node with condition s1 == 0
  {
    SharedLeafTypeTree<ConditionsBySinkNode> z_tree =
        use_conditions.GetInfo(z.node());
    ASSERT_EQ(z_tree.elements().size(), 1);
    const ConditionsBySinkNode& z_sinks = z_tree.elements().at(0);
    ASSERT_EQ(z_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *z_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_node = condition_node->operand(0);
    EXPECT_EQ(eq_node->op(), Op::kEq);
    EXPECT_EQ(eq_node->operand(0), s1.node());
    EXPECT_EQ(eq_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_literal = eq_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_literal->value(), Value(UBits(0, 1)));
    EXPECT_EQ(condition_node->operand(1), use_conditions.GetLiteralOne(f));
  }

  // Check that x has one sink ret_node with condition s1 == 1 and s0 == 0
  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());
    ASSERT_EQ(x_tree.elements().size(), 1);
    const ConditionsBySinkNode& x_sinks = x_tree.elements().at(0);
    ASSERT_EQ(x_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *x_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_s0_node = condition_node->operand(0);
    EXPECT_EQ(eq_s0_node->op(), Op::kEq);
    EXPECT_EQ(eq_s0_node->operand(0), s0.node());
    EXPECT_EQ(eq_s0_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_s0_literal = eq_s0_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_s0_literal->value(), Value(UBits(1, 1)));

    Node* inner_and = condition_node->operand(1);
    EXPECT_EQ(inner_and->op(), Op::kAnd);
    ASSERT_EQ(inner_and->operand_count(), 2);
    Node* eq_s1_node = inner_and->operand(0);
    EXPECT_EQ(eq_s1_node->op(), Op::kEq);
    EXPECT_EQ(eq_s1_node->operand(0), s1.node());
    EXPECT_EQ(eq_s1_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_s1_literal = eq_s1_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_s1_literal->value(), Value(UBits(1, 1)));
    EXPECT_EQ(inner_and->operand(1), use_conditions.GetLiteralOne(f));
  }

  // Check that y has one sink ret_node with condition s1 == 1 and s0 == 1
  {
    SharedLeafTypeTree<ConditionsBySinkNode> y_tree =
        use_conditions.GetInfo(x.node());
    ASSERT_EQ(y_tree.elements().size(), 1);
    const ConditionsBySinkNode& y_sinks = y_tree.elements().at(0);
    ASSERT_EQ(y_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *y_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kAnd);
    ASSERT_EQ(condition_node->operand_count(), 2);
    Node* eq_s0_node = condition_node->operand(0);
    EXPECT_EQ(eq_s0_node->op(), Op::kEq);
    EXPECT_EQ(eq_s0_node->operand(0), s0.node());
    EXPECT_EQ(eq_s0_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_s0_literal = eq_s0_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_s0_literal->value(), Value(UBits(1, 1)));

    Node* inner_and = condition_node->operand(1);
    EXPECT_EQ(inner_and->op(), Op::kAnd);
    ASSERT_EQ(inner_and->operand_count(), 2);
    Node* eq_s1_node = inner_and->operand(0);
    EXPECT_EQ(eq_s1_node->op(), Op::kEq);
    EXPECT_EQ(eq_s1_node->operand(0), s1.node());
    EXPECT_EQ(eq_s1_node->operand(1)->op(), Op::kLiteral);
    Literal* eq_s1_literal = eq_s1_node->operand(1)->As<Literal>();
    EXPECT_EQ(eq_s1_literal->value(), Value(UBits(1, 1)));
    EXPECT_EQ(inner_and->operand(1), use_conditions.GetLiteralOne(f));
  }
}

TEST_F(DataFlowLazyInputUseConditionsTest, MultiSelectParallel) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue s0 = fb.Param("s0", p->GetBitsType(1));
  BValue s1 = fb.Param("s1", p->GetBitsType(1));
  BValue s2 = fb.Param("s2", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue z = fb.Param("z", p->GetBitsType(32));

  BValue sel0 =
      fb.Select(s0, /*on_true=*/x, /*on_false=*/y, SourceInfo(), "sel0");
  BValue sel1 =
      fb.Select(s1, /*on_true=*/x, /*on_false=*/z, SourceInfo(), "sel1");
  BValue sel2 =
      fb.Select(s2, /*on_true=*/sel0, /*on_false=*/sel1, SourceInfo(), "sel2");

  BValue ret_node = fb.Identity(sel2, SourceInfo(), "ret_node");

  xls::Function* f = nullptr;
  XLS_ASSERT_OK_AND_ASSIGN(f, fb.BuildWithReturnValue(ret_node));

  DataFlowLazyInputUseConditions use_conditions;
  XLS_EXPECT_OK(use_conditions.Attach(f));
  use_conditions.set_query_engine(query_engine());
  XLS_ASSERT_OK(query_engine()->Populate(f).status());

  {
    SharedLeafTypeTree<ConditionsBySinkNode> x_tree =
        use_conditions.GetInfo(x.node());
    ASSERT_EQ(x_tree.elements().size(), 1);
    const ConditionsBySinkNode& x_sinks = x_tree.elements().at(0);

    // Check that x has one sink ret with condition:
    //   (s0 == 1 && s2 == 1) || (s1 == 1 && s2 == 0)
    ASSERT_EQ(x_sinks.size(), 1);
    const auto& [sink_node, condition_node] = *x_sinks.begin();
    EXPECT_EQ(sink_node, ret_node.node());
    EXPECT_EQ(condition_node->op(), Op::kOr);
    ASSERT_EQ(condition_node->operand_count(), 2);

    {
      Node* inner_and_sel0 = condition_node->operand(0);
      ASSERT_EQ(inner_and_sel0->operand_count(), 2);
      Node* eq_sel0 = inner_and_sel0->operand(0);
      EXPECT_EQ(eq_sel0->op(), Op::kEq);
      EXPECT_EQ(eq_sel0->operand(0), s0.node());
      EXPECT_EQ(eq_sel0->operand(1)->op(), Op::kLiteral);
      Literal* eq_sel0_literal = eq_sel0->operand(1)->As<Literal>();
      EXPECT_EQ(eq_sel0_literal->value(), Value(UBits(1, 1)));
      Node* inner_and_sel0_sel2 = inner_and_sel0->operand(1);
      EXPECT_EQ(inner_and_sel0_sel2->op(), Op::kAnd);
      Node* inner_eq_sel2 = inner_and_sel0_sel2->operand(0);
      EXPECT_EQ(inner_eq_sel2->op(), Op::kEq);
      EXPECT_EQ(inner_eq_sel2->operand(0), s2.node());
      EXPECT_EQ(inner_eq_sel2->operand(1)->op(), Op::kLiteral);
      Literal* inner_eq_sel2_literal = inner_eq_sel2->operand(1)->As<Literal>();
      EXPECT_EQ(inner_eq_sel2_literal->value(), Value(UBits(1, 1)));
    }

    {
      Node* inner_and_sel1 = condition_node->operand(1);
      ASSERT_EQ(inner_and_sel1->operand_count(), 2);
      Node* eq_sel1 = inner_and_sel1->operand(0);
      EXPECT_EQ(eq_sel1->op(), Op::kEq);
      EXPECT_EQ(eq_sel1->operand(0), s1.node());
      EXPECT_EQ(eq_sel1->operand(1)->op(), Op::kLiteral);
      Literal* eq_sel1_literal = eq_sel1->operand(1)->As<Literal>();
      EXPECT_EQ(eq_sel1_literal->value(), Value(UBits(1, 1)));
      Node* inner_and_sel1_sel2 = inner_and_sel1->operand(1);
      EXPECT_EQ(inner_and_sel1_sel2->op(), Op::kAnd);
      Node* inner_eq_sel2 = inner_and_sel1_sel2->operand(0);
      EXPECT_EQ(inner_eq_sel2->op(), Op::kEq);
      EXPECT_EQ(inner_eq_sel2->operand(0), s2.node());
      EXPECT_EQ(inner_eq_sel2->operand(1)->op(), Op::kLiteral);
      Literal* inner_eq_sel2_literal = inner_eq_sel2->operand(1)->As<Literal>();
      EXPECT_EQ(inner_eq_sel2_literal->value(), Value(UBits(0, 1)));
    }
  }
}

}  // namespace
}  // namespace xls
