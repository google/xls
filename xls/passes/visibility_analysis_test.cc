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

#include "xls/passes/visibility_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

class VisibilityAnalysisTest : public IrTestBase {
 protected:
  absl::StatusOr<BddNodeIndex> GetNodeBit(Node* node, uint32_t bit_index,
                                          BddQueryEngine& bdd_query_engine) {
    std::optional<BddNodeIndex> bit =
        bdd_query_engine.GetBddNodeOrVariable(TreeBitLocation(node, bit_index));
    if (!bit.has_value()) {
      return absl::InternalError(
          absl::StrCat("Bit not found: ", node->GetName(), bit_index));
    }
    return *bit;
  }

  absl::StatusOr<std::vector<BddNodeIndex>> GetNodeBits(
      Node* node, BddQueryEngine& bdd_query_engine) {
    std::vector<BddNodeIndex> bits;
    bits.reserve(node->BitCountOrDie());
    for (int i = 0; i < node->BitCountOrDie(); ++i) {
      XLS_ASSIGN_OR_RETURN(BddNodeIndex bit,
                           GetNodeBit(node, i, bdd_query_engine));
      bits.push_back(bit);
    }
    return bits;
  }

  absl::StatusOr<std::vector<SaturatingBddNodeIndex>> GetSaturatingNodeBits(
      Node* node, BddQueryEngine& bdd_query_engine) {
    std::vector<SaturatingBddNodeIndex> bits;
    bits.reserve(node->BitCountOrDie());
    for (int i = 0; i < node->BitCountOrDie(); ++i) {
      XLS_ASSIGN_OR_RETURN(BddNodeIndex bit,
                           GetNodeBit(node, i, bdd_query_engine));
      bits.push_back(bit);
    }
    return bits;
  }

  absl::StatusOr<BddNodeIndex> AllZeros(Node* node,
                                        BddQueryEngine& bdd_query_engine) {
    BddNodeIndex not_node = bdd_query_engine.bdd().one();
    for (int i = 0; i < node->BitCountOrDie(); ++i) {
      XLS_ASSIGN_OR_RETURN(BddNodeIndex bit,
                           GetNodeBit(node, i, bdd_query_engine));
      not_node =
          bdd_query_engine.bdd().And(not_node, bdd_query_engine.bdd().Not(bit));
    }
    return not_node;
  }

  absl::StatusOr<BddNodeIndex> AllOnes(Node* node,
                                       BddQueryEngine& bdd_query_engine) {
    BddNodeIndex all_ones = bdd_query_engine.bdd().one();
    for (int i = 0; i < node->BitCountOrDie(); ++i) {
      XLS_ASSIGN_OR_RETURN(BddNodeIndex bit,
                           GetNodeBit(node, i, bdd_query_engine));
      all_ones = bdd_query_engine.bdd().And(all_ones, bit);
    }
    return all_ones;
  }
};

class OperandVisibilityAnalysisTallyComputations
    : public OperandVisibilityAnalysis {
 public:
  OperandVisibilityAnalysisTallyComputations(
      const NodeForwardDependencyAnalysis* nda,
      const BddQueryEngine* bdd_query_engine)
      : OperandVisibilityAnalysis(
            nda, bdd_query_engine,
            OperandVisibilityAnalysis::kDefaultTermLimitForNodeToUserEdge) {}

  mutable int64_t computations_ = 0;

 protected:
  BddNodeIndex OperandVisibilityThroughNode(
      OperandVisibilityAnalysis::OperandNode& pair) const override {
    if (!pair_to_op_vis_.contains(pair)) {
      computations_++;
    }
    return OperandVisibilityAnalysis::OperandVisibilityThroughNode(pair);
  }
};

class VisibilityAnalysisTallyComputations : public VisibilityAnalysis {
 public:
  VisibilityAnalysisTallyComputations(
      const OperandVisibilityAnalysis* operand_vis,
      const BddQueryEngine* bdd_query_engine,
      const LazyPostDominatorAnalysis* post_dom_analysis)
      : VisibilityAnalysis(operand_vis, bdd_query_engine, post_dom_analysis) {}

  mutable int64_t computations_ = 0;

 protected:
  BddNodeIndex ComputeInfo(
      Node* node,
      absl::Span<const BddNodeIndex* const> user_infos) const override {
    computations_++;
    return VisibilityAnalysis::ComputeInfo(node, user_infos);
  }
};

TEST_F(VisibilityAnalysisTest, VisibilityCaches) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue ab = fb.And(a, b);
  BValue abc = fb.And(ab, c);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(abc));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  OperandVisibilityAnalysisTallyComputations op_vis(&nda, bdd_engine.get());
  XLS_ASSERT_OK(op_vis.Attach(f));
  std::unique_ptr<VisibilityAnalysisTallyComputations> visibility =
      std::make_unique<VisibilityAnalysisTallyComputations>(
          &op_vis, bdd_engine.get(), &post_dom);
  XLS_ASSERT_OK(visibility->Attach(f));

  XLS_ASSERT_OK_AND_ASSIGN(auto b_bit, GetNodeBit(b.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(auto c_bit, GetNodeBit(c.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(auto d_bit, GetNodeBit(d.node(), 0, *bdd_engine));

  BddNodeIndex abc_visible = *visibility->GetInfo(abc.node());
  EXPECT_EQ(abc_visible, bdd.one());
  EXPECT_EQ(op_vis.computations_, 0);
  EXPECT_EQ(visibility->computations_, 1);
  BddNodeIndex ab_visible = *visibility->GetInfo(ab.node());
  EXPECT_EQ(ab_visible, c_bit);
  EXPECT_EQ(op_vis.computations_, 1);
  EXPECT_EQ(visibility->computations_, 2);
  BddNodeIndex a_visible = *visibility->GetInfo(a.node());
  EXPECT_EQ(a_visible, bdd.And(b_bit, c_bit));
  EXPECT_EQ(op_vis.computations_, 2);
  EXPECT_EQ(visibility->computations_, 3);

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * abcd,
      f->MakeNode<NaryOp>(SourceInfo(),
                          std::vector<Node*>{abc.node(), d.node()}, Op::kAnd));
  XLS_ASSERT_OK(f->set_return_value(abcd));

  abc_visible = *visibility->GetInfo(abc.node());
  EXPECT_EQ(abc_visible, d_bit);
  EXPECT_EQ(op_vis.computations_, 3);
  // Recomputing visibility on 'abcd' and then 'abc' costs two computations:
  EXPECT_EQ(visibility->computations_, 5);
  BddNodeIndex abcd_visible = *visibility->GetInfo(abcd);
  EXPECT_EQ(abcd_visible, bdd.one());
  // op visibility is already cached for all 3 non-terminal nodes:
  EXPECT_EQ(op_vis.computations_, 3);
  EXPECT_EQ(visibility->computations_, 5);
  a_visible = *visibility->GetInfo(a.node());
  EXPECT_EQ(a_visible, bdd.And(bdd.And(b_bit, c_bit), d_bit));
  EXPECT_EQ(op_vis.computations_, 3);
  // Recomputing visibility on 'ab' and then 'a' costs two computations:
  EXPECT_EQ(visibility->computations_, 7);
}

TEST_F(VisibilityAnalysisTest, VisibilityInvalidates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue e = fb.Param("e", p->GetBitsType(1));
  BValue ab = fb.And(a, b);
  BValue abc = fb.And(ab, c);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(abc));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  XLS_ASSERT_OK_AND_ASSIGN(auto b_bit, GetNodeBit(b.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(auto c_bit, GetNodeBit(c.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(auto d_bit, GetNodeBit(d.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(auto e_bit, GetNodeBit(e.node(), 0, *bdd_engine));

  VLOG(3) << "b_bit: " << bdd.ToStringDnf(b_bit);
  VLOG(3) << "c_bit: " << bdd.ToStringDnf(c_bit);
  VLOG(3) << "d_bit: " << bdd.ToStringDnf(d_bit);
  VLOG(3) << "e_bit: " << bdd.ToStringDnf(e_bit);

  BddNodeIndex a_visible = *visibility->GetInfo(a.node());
  VLOG(3) << "a_visible 0: " << bdd.ToStringDnf(a_visible);
  EXPECT_EQ(a_visible, bdd.And(b_bit, c_bit));
  ab.node()->ReplaceOperand(b.node(), d.node());
  a_visible = *visibility->GetInfo(a.node());
  VLOG(3) << "a_visible 1: " << bdd.ToStringDnf(a_visible);
  EXPECT_EQ(a_visible, bdd.And(d_bit, c_bit));
  abc.node()->ReplaceOperand(c.node(), e.node());
  a_visible = *visibility->GetInfo(a.node());
  VLOG(3) << "a_visible 2: " << bdd.ToStringDnf(a_visible);
  EXPECT_EQ(a_visible, bdd.And(d_bit, e_bit));
}

TEST_F(VisibilityAnalysisTest, NodeImpactOnVisibilityViaBooleanGuards) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue op = fb.Param("op", p->GetBitsType(4));
  BValue and_guard = fb.And({x, op});
  BValue or_guard = fb.Or({x, fb.SignExtend(fb.BitSlice(op, 0, 1), 4)});
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f, fb.BuildWithReturnValue(fb.Tuple({and_guard, or_guard})));

  NodeImpactOnVisibilityAnalysis nia;
  XLS_ASSERT_OK(nia.Attach(f));
  EXPECT_EQ(nia.NodeImpactOnVisibility(x.node()), 2);
  EXPECT_EQ(nia.NodeImpactOnVisibility(op.node()), 2);
  EXPECT_EQ(nia.NodeImpactOnVisibility(and_guard.node()), 0);
  EXPECT_EQ(nia.NodeImpactOnVisibility(or_guard.node()), 0);
}

TEST_F(VisibilityAnalysisTest, NodeImpactOnVisibilityViaSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue op = fb.Param("op", p->GetBitsType(4));
  BValue select_guard =
      fb.Select(fb.ZeroExtend(op, 10), std::vector<BValue>{x, x, x, x}, x);
  BValue prio_guard = fb.PrioritySelect(
      fb.Concat({fb.Eq(fb.BitSlice(op, 0, 1), fb.Literal(UBits(1, 1))),
                 fb.Ne(fb.BitSlice(op, 1, 1), fb.Literal(UBits(0, 1)))}),
      {x, x}, x);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f, fb.BuildWithReturnValue(fb.Tuple({select_guard, prio_guard})));

  NodeImpactOnVisibilityAnalysis nia;
  XLS_ASSERT_OK(nia.Attach(f));
  EXPECT_EQ(nia.NodeImpactOnVisibility(x.node()), 0);
  // impact = num_cases + default, if any
  int64_t impact_via_select = 4 + 1;
  // impact = paths_to_selector * (num_cases + default)
  int64_t impact_via_priority_select = 2 * (2 + 1);
  EXPECT_EQ(nia.NodeImpactOnVisibility(op.node()),
            impact_via_select + impact_via_priority_select);
  EXPECT_EQ(nia.NodeImpactOnVisibility(select_guard.node()), 0);
  EXPECT_EQ(nia.NodeImpactOnVisibility(prio_guard.node()), 0);
}

TEST_F(VisibilityAnalysisTest, VisibilityThroughPrioritySelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue selector = fb.Param("y", p->GetBitsType(2));
  BValue add = fb.Add(x, fb.Literal(UBits(1, 4)));
  BValue select = fb.PrioritySelect(selector, {x, add}, x);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex and_visible = *visibility->GetInfo(add.node());

  // the selector bit determines the visibility of add
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex prev_case_bit,
                           GetNodeBit(selector.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex selector_bit,
                           GetNodeBit(selector.node(), 1, *bdd_engine));
  EXPECT_EQ(bdd.Implies(prev_case_bit, bdd.Not(and_visible)), bdd.one());
  EXPECT_EQ(bdd.Implies(bdd.Not(selector_bit), bdd.Not(and_visible)),
            bdd.one());
  EXPECT_EQ(
      bdd.Implies(bdd.And(bdd.Not(prev_case_bit), selector_bit), and_visible),
      bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityThroughSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue selector = fb.Param("y", p->GetBitsType(2));
  BValue add = fb.Add(x, fb.Literal(UBits(1, 4)));
  BValue sub = fb.Subtract(x, fb.Literal(UBits(1, 4)));
  BValue select = fb.Select(selector, {add, sub}, x);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex add_visible = *visibility->GetInfo(add.node());
  BddNodeIndex sub_visible = *visibility->GetInfo(sub.node());

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<BddNodeIndex> selector_bits,
                           GetNodeBits(selector.node(), *bdd_engine));
  EXPECT_EQ(
      bdd.Implies(bdd.And(bdd.Not(selector_bits[0]), bdd.Not(selector_bits[1])),
                  bdd.And(add_visible, bdd.Not(sub_visible))),
      bdd.one());
  EXPECT_EQ(bdd.Implies(bdd.And(selector_bits[0], bdd.Not(selector_bits[1])),
                        bdd.And(sub_visible, bdd.Not(add_visible))),
            bdd.one());
  EXPECT_EQ(
      bdd.Implies(selector_bits[1], bdd.Not(bdd.Or(add_visible, sub_visible))),
      bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityThroughAnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue z = fb.Param("z", p->GetBitsType(2));
  BValue and_xy = fb.And(x, y);
  BValue and_xyz = fb.And(and_xy, z);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(and_xyz));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex z_visible = *visibility->GetInfo(z.node());

  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex not_y, AllZeros(y.node(), *bdd_engine));
  EXPECT_EQ(bdd.Implies(not_y, bdd.Not(z_visible)), bdd.one());
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex not_x, AllZeros(y.node(), *bdd_engine));
  EXPECT_EQ(bdd.Implies(not_x, bdd.Not(z_visible)), bdd.one());

  // if x and y are both not all zeros, z is visible
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex x_bit_0,
                           GetNodeBit(x.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex y_bit_0,
                           GetNodeBit(y.node(), 0, *bdd_engine));
  EXPECT_EQ(bdd.Implies(bdd.And(x_bit_0, y_bit_0), z_visible), bdd.one());
  EXPECT_NE(bdd.Implies(x_bit_0, z_visible), bdd.one());
  EXPECT_NE(bdd.Implies(x_bit_0, bdd.Not(z_visible)), bdd.one());
  EXPECT_NE(bdd.Implies(y_bit_0, z_visible), bdd.one());
  EXPECT_NE(bdd.Implies(y_bit_0, bdd.Not(z_visible)), bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityHandlesIrrelevantUnknown) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue y = fb.Param("y", p->GetBitsType(2));
  BValue z = fb.Param("z", p->GetBitsType(2));
  BValue other = fb.Param("other", p->GetBitsType(2));
  BValue mul_y_other = fb.UMul(y, other);
  BValue or_xy = fb.Or(x, mul_y_other);
  BValue or_xyz = fb.Or({y, or_xy, z});
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(or_xyz));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex z_visible = *visibility->GetInfo(z.node());

  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex all_x, AllOnes(x.node(), *bdd_engine));
  EXPECT_EQ(bdd.Implies(all_x, bdd.Not(z_visible)), bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityTreatExpensiveConditionAsVariable) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue select = fb.PrioritySelect(op, {add}, x);
  BValue expensive_condition =
      fb.Or({fb.And({fb.UGe(x, fb.Literal(UBits(8, 4))),
                     fb.UGe(y, fb.Literal(UBits(8, 4)))}),
             fb.And({fb.ULt(x, fb.Literal(UBits(7, 4))),
                     fb.ULt(y, fb.Literal(UBits(7, 4)))})});
  BValue expensive_select = fb.Select(expensive_condition, select, x);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(expensive_select));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex add_visible = *visibility->GetInfo(add.node());

  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex op_bit,
                           GetNodeBit(op.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(
      BddNodeIndex expensive_condition_bit,
      GetNodeBit(expensive_condition.node(), 0, *bdd_engine));
  EXPECT_EQ(bdd.Implies(bdd.Not(op_bit), bdd.Not(add_visible)), bdd.one());
  EXPECT_EQ(bdd.Implies(bdd.And(op_bit, expensive_condition_bit), add_visible),
            bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityAvoidsSaturatingOnOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue unknown = fb.UMul(x, y);
  BValue lots_of_terms = fb.SignExtend(fb.ULe(x, fb.Literal(UBits(10, 4))), 4);
  BValue fewer_terms =
      fb.SignExtend(fb.ULe(fb.BitSlice(x, 0, 2), fb.Literal(UBits(2, 2))), 4);
  BValue simple = fb.SignExtend(op, 4);
  BValue use_add = fb.Or({add, unknown, lots_of_terms, simple, fewer_terms});
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(use_add));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();

  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex simple_bit,
                           GetNodeBit(simple.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex lots_of_terms_bit,
                           GetNodeBit(lots_of_terms.node(), 0, *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex fewer_terms_bit,
                           GetNodeBit(fewer_terms.node(), 0, *bdd_engine));

  // Enough terms to not saturate if visibility is a function of 'simple' and
  // 'fewer_terms'. Too few terms that will saturate if 'lots_of_terms' is
  // included. 'Unknown' is always ignored because it is fully unknown.
  int64_t term_limit = 4;
  ASSERT_TRUE(term_limit > bdd_engine->bdd().path_count(simple_bit));
  ASSERT_TRUE(bdd_engine->bdd().path_count(simple_bit) <
              bdd_engine->bdd().path_count(fewer_terms_bit));
  ASSERT_TRUE(bdd_engine->bdd().path_count(fewer_terms_bit) <
              bdd_engine->bdd().path_count(lots_of_terms_bit));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(term_limit, &nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex add_visible = *visibility->GetInfo(add.node());
  VLOG(3) << "add_visible: " << bdd.ToStringDnf(add_visible);
  VLOG(3) << "lots_of_terms: " << bdd.ToStringDnf(lots_of_terms_bit);
  VLOG(3) << "fewer_terms: " << bdd.ToStringDnf(fewer_terms_bit);
  VLOG(3) << "simple: " << bdd.ToStringDnf(simple_bit);

  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex op_bit,
                           GetNodeBit(op.node(), 0, *bdd_engine));

  EXPECT_EQ(bdd.Implies(op_bit, bdd.Not(add_visible)), bdd.one());
  EXPECT_EQ(bdd.Implies(fewer_terms_bit, bdd.Not(add_visible)), bdd.one());
  EXPECT_NE(bdd.Implies(bdd.Not(op_bit), add_visible), bdd.one());
  EXPECT_NE(bdd.Implies(bdd.Not(fewer_terms_bit), add_visible), bdd.one());
  EXPECT_EQ(bdd.Implies(bdd.And(bdd.Not(fewer_terms_bit), bdd.Not(op_bit)),
                        add_visible),
            bdd.one());
}

TEST_F(VisibilityAnalysisTest, VisibilityFallbackToPruningExpensiveEdges) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue selector = fb.Param("selector", p->GetBitsType(17));
  BValue x = fb.Param("x", p->GetBitsType(1));
  std::vector<BValue> x_selects;
  for (int i = 0; i < 16; ++i) {
    x_selects.push_back(
        fb.Select(fb.BitSlice(selector, i, 1), {fb.Literal(UBits(0, 1)), x}));
  }
  BValue complex = fb.And(x_selects);
  BValue simple = fb.And(complex, fb.BitSlice(selector, 16, 1));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(simple));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  // Enough to saturate the many selects on x, but not 'simple'
  int64_t path_limit = 4;
  std::unique_ptr<BddQueryEngine> bdd_engine =
      std::make_unique<BddQueryEngine>(path_limit, IsCheapForBdds);
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();

  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));
  BddNodeIndex x_visible = *visibility->GetInfo(x.node());
  BddNodeIndex complex_visible = *visibility->GetInfo(complex.node());
  EXPECT_EQ(x_visible, complex_visible);
  XLS_ASSERT_OK_AND_ASSIGN(BddNodeIndex last_selector_bit,
                           GetNodeBit(selector.node(), 16, *bdd_engine));
  EXPECT_TRUE(bdd.Implies(bdd.Not(last_selector_bit), bdd.Not(x_visible)));
}

TEST_F(VisibilityAnalysisTest, VisibilityFallbackToPostDominatorIfManyEdges) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // The necessary path count on the BDD to ensure edge pruning succeeds at
  // producing a conservative visibility expression more expressive than just
  // the post dominator's own visibility.
  int64_t num_selects = 10;
  int64_t max_edge_pruning = 5;
  int64_t bdd_can_path_remaining_edges = 150;
  int64_t bdd_cannot_path_remaining_edges = 100;

  BValue ops1 = fb.Param("op1", p->GetBitsType(num_selects));
  BValue ops2 = fb.Param("op2", p->GetBitsType(num_selects));
  BValue op_final = fb.Param("op_final", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  std::vector<BValue> selects;
  BValue reduced = y;
  // Add 'num_selects' edges from 'x', one edge between each select and the
  // reduction, and an edge between the last reduced value and function return.
  for (int i = 0; i < num_selects; ++i) {
    BValue select = fb.Select(fb.BitSlice(ops1, i, 1), {x}, y);
    selects.push_back(select);
    reduced = fb.Add(
        reduced, fb.And(select, fb.SignExtend(fb.BitSlice(ops2, i, 1), 16)));
  }
  BValue last_guard = fb.And(reduced, fb.SignExtend(op_final, 16));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(last_guard));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));

  std::unique_ptr<BddQueryEngine> bdd_engine_large =
      std::make_unique<BddQueryEngine>(bdd_can_path_remaining_edges,
                                       IsCheapForBdds);
  XLS_ASSERT_OK(bdd_engine_large->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility_large,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine_large.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility_large,
      VisibilityAnalysis::Create(&operand_visibility_large,
                                 bdd_engine_large.get(), &post_dom,
                                 max_edge_pruning));
  EXPECT_NE(*visibility_large->GetInfo(x.node()),
            *visibility_large->GetInfo(reduced.node()));
  EXPECT_TRUE(bdd_engine_large->bdd().Implies(
                  *visibility_large->GetInfo(x.node()),
                  *visibility_large->GetInfo(reduced.node())) ==
              bdd_engine_large->bdd().one());
  EXPECT_FALSE(bdd_engine_large->bdd().Implies(
                   *visibility_large->GetInfo(reduced.node()),
                   *visibility_large->GetInfo(x.node())) ==
               bdd_engine_large->bdd().one());

  // The conservative visibility expression should begin to include edges from
  // the last few selects; check implication on the 'ops' bits from the first of
  // these selects
  auto op1_bit_included = bdd_engine_large->GetBddNode(
      TreeBitLocation(ops1.node(), num_selects - max_edge_pruning));
  auto op2_bit_included = bdd_engine_large->GetBddNode(
      TreeBitLocation(ops2.node(), num_selects - max_edge_pruning));
  EXPECT_TRUE(bdd_engine_large->bdd().Implies(
      bdd_engine_large->bdd().And(op1_bit_included.value(),
                                  op2_bit_included.value()),
      *visibility_large->GetInfo(x.node())));

  std::unique_ptr<BddQueryEngine> bdd_engine_small =
      std::make_unique<BddQueryEngine>(bdd_cannot_path_remaining_edges,
                                       IsCheapForBdds);
  XLS_ASSERT_OK(bdd_engine_small->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility_small,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine_small.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility_small,
      VisibilityAnalysis::Create(&operand_visibility_small,
                                 bdd_engine_small.get(), &post_dom,
                                 max_edge_pruning));
  // Fell back to the post dominator's visibility:
  EXPECT_EQ(*visibility_small->GetInfo(x.node()),
            *visibility_small->GetInfo(reduced.node()));
}

TEST_F(VisibilityAnalysisTest, MutuallyExclusivePrioritySelectCases) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue sub = fb.Subtract(x, y);
  BValue select = fb.PrioritySelect(op, {add}, sub);
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  EXPECT_TRUE(visibility->IsMutuallyExclusive(add.node(), sub.node()));
}

TEST_F(VisibilityAnalysisTest, MutuallyExclusiveMultipleSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue sub = fb.Subtract(x, y);
  BValue select = fb.PrioritySelect(op, {add}, sub);
  BValue select2 = fb.Select(op, std::vector<BValue>{sub}, add);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f, fb.BuildWithReturnValue(fb.Tuple({select, select2})));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  EXPECT_TRUE(visibility->IsMutuallyExclusive(add.node(), sub.node()));
}

TEST_F(VisibilityAnalysisTest, VisibilityThroughManyAndsSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(16));
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue mul = fb.UMul(x, y);
  // visibility barrier: y must be non-negative
  BValue non_neg_y =
      fb.And(mul, fb.SignExtend(fb.Ne(y, fb.Literal(SBits(-1, 16))), 16));
  BValue rest = fb.BitSlice(non_neg_y, 0, 15);
  BValue last = fb.BitSlice(non_neg_y, 15, 1);
  BValue padded = fb.Concat({rest, fb.Literal(UBits(0, 1))});
  // visibility barrier: x[0:15] must not be 0
  BValue x_or_padded =
      fb.Select(fb.Eq(rest, fb.Literal(UBits(0, 15))), {x, padded});
  // noise, not a visibility barrier, just a jumbling of what is used
  BValue some_part =
      fb.Select(fb.Eq(last, fb.Literal(UBits(1, 1))), {non_neg_y, padded});
  // visibility barrier: upper and lower bounds
  BValue too_large = fb.SGe(some_part, fb.Literal(UBits(10000, 16)));
  BValue too_small = fb.SLe(some_part, fb.Literal(UBits(10, 16)));
  BValue bounded = fb.And(
      x_or_padded, fb.SignExtend(fb.Not(fb.Or(too_large, too_small)), 16));
  // visibility barrier: sign bit must be 0
  BValue sign_zero = fb.Select(fb.Eq(fb.Concat({fb.Literal(UBits(0, 9)), last}),
                                     fb.Literal(UBits(0, 10))),
                               {bounded, fb.Literal(UBits(10001, 16))});
  BValue mul_survived = fb.Concat({non_neg_y, sign_zero});

  // obfuscating later selections on the op by computing a different op
  BValue other_op_msb = fb.Or({fb.Eq(op, fb.Literal(UBits(100, 16))),
                               fb.Eq(op, fb.Literal(UBits(101, 16))),
                               fb.Eq(op, fb.Literal(UBits(102, 16)))});
  BValue other_op_lsbs =
      fb.PrioritySelect(fb.Concat({fb.Eq(op, fb.Literal(UBits(103, 16))),
                                   fb.Eq(op, fb.Literal(UBits(100, 16))),
                                   fb.Eq(op, fb.Literal(UBits(101, 16))),
                                   fb.Eq(op, fb.Literal(UBits(104, 16))),
                                   fb.Eq(op, fb.Literal(UBits(105, 16))),
                                   fb.Eq(op, fb.Literal(UBits(102, 16)))}),
                        {fb.Literal(UBits(0, 3)), fb.Literal(UBits(1, 3)),
                         fb.Literal(UBits(2, 3)), fb.Literal(UBits(3, 3)),
                         fb.Literal(UBits(0, 3)), fb.Literal(UBits(1, 3))},
                        fb.Literal(UBits(0, 3)));
  BValue other_op =
      fb.And(fb.Concat({other_op_msb, other_op_lsbs}),
             fb.SignExtend(fb.Ne(op, fb.Literal(UBits(106, 16))), 4));

  // visibility barrier: 'op' must be 103
  BValue lit0_32 = fb.Literal(UBits(0, 32));
  BValue lit1_4 = fb.Literal(UBits(1, 4));
  BValue lit103_16 = fb.Literal(UBits(103, 16));
  BValue prio_sel = fb.PrioritySelect(
      fb.Concat({fb.Eq(op, fb.Literal(UBits(104, 16))), fb.Eq(op, lit103_16),
                 fb.Eq(op, fb.Literal(UBits(102, 16)))}),
      {fb.Concat({x, y}), mul_survived, fb.Concat({y, x})}, lit0_32);
  // visibility barrier: 'other_op' must NOT be 0; if 'op' is 103, 'other_op'
  // is 0b0001 which suffices.
  BValue other_op_filtered =
      fb.Select(fb.Eq(other_op, fb.Literal(UBits(0, 4))), {prio_sel, lit0_32});
  // noise by using arrays and tuples
  BValue array = fb.Array({other_op_filtered, lit0_32}, p->GetBitsType(32));
  BValue use_mul =
      fb.And({mul_survived,
              fb.SignExtend(fb.Eq(other_op, fb.Literal(UBits(1, 4))), 32)});
  BValue mul2 = fb.UMul(x, fb.Add(y, fb.Literal(UBits(1234, 16))));
  BValue use_mul2 = fb.And(
      {mul2, fb.SignExtend(fb.Ne(other_op, fb.Literal(UBits(1, 4))), 16)});
  BValue tuple = fb.Tuple({fb.Literal(UBits(1, 1)), use_mul, use_mul2});
  BValue result = fb.Concat(
      {fb.ArrayIndex(array, {lit0_32}, true), fb.TupleIndex(tuple, 1)});

  BValue lit105_16 = fb.Literal(UBits(105, 16));
  XLS_ASSERT_OK_AND_ASSIGN(auto f, fb.BuildWithReturnValue(result));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  // assert visibility on later mul expression on 'op' value
  BddNodeIndex mul_survived_visible = *visibility->GetInfo(mul_survived.node());
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SaturatingBddNodeIndex> op_bits,
                           GetSaturatingNodeBits(op.node(), *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SaturatingBddNodeIndex> other_op_bits,
                           GetSaturatingNodeBits(other_op.node(), *bdd_engine));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SaturatingBddNodeIndex> lit103_16_bits,
      GetSaturatingNodeBits(lit103_16.node(), *bdd_engine));
  SaturatingBddNodeIndex op_is_103 =
      bdd_engine->evaluator().Equals(op_bits, lit103_16_bits);
  ASSERT_FALSE(HasTooManyPaths(op_is_103));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SaturatingBddNodeIndex> lit1_4_bits,
                           GetSaturatingNodeBits(lit1_4.node(), *bdd_engine));
  SaturatingBddNodeIndex other_op_is_1 =
      bdd_engine->evaluator().Equals(other_op_bits, lit1_4_bits);
  ASSERT_FALSE(HasTooManyPaths(other_op_is_1));
  EXPECT_EQ(bdd.Implies(ToBddNode(op_is_103), ToBddNode(other_op_is_1)),
            bdd.one());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<SaturatingBddNodeIndex> lit105_16_bits,
      GetSaturatingNodeBits(lit105_16.node(), *bdd_engine));
  SaturatingBddNodeIndex op_is_105 =
      bdd_engine->evaluator().Equals(op_bits, lit105_16_bits);
  ASSERT_FALSE(HasTooManyPaths(op_is_105));
  EXPECT_EQ(bdd.Implies(bdd.And(bdd.Not(ToBddNode(op_is_103)),
                                bdd.Not(ToBddNode(op_is_105))),
                        bdd.Not(ToBddNode(other_op_is_1))),
            bdd.one());

  EXPECT_EQ(bdd.Implies(bdd.And(bdd.Not(ToBddNode(op_is_103)),
                                bdd.Not(ToBddNode(op_is_105))),
                        bdd.Not(mul_survived_visible)),
            bdd.one());

  EXPECT_TRUE(visibility->IsMutuallyExclusive(mul.node(), mul2.node()));
}

TEST_F(VisibilityAnalysisTest, EdgesForVisibilityImpactingMutualExclusivity) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue z = fb.Param("z", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue or_to_ignore = fb.Or({add, z});
  BValue sub = fb.Subtract(x, y);
  BValue select = fb.PrioritySelect(op, {or_to_ignore}, sub);
  BValue select2 = fb.Select(op, std::vector<BValue>{sub}, or_to_ignore);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f, fb.BuildWithReturnValue(fb.Tuple({select, select2})));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode> edges;
  XLS_ASSERT_OK_AND_ASSIGN(
      edges, visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(
                 add.node(), {sub.node()}, -1));
  EXPECT_THAT(edges,
              UnorderedElementsAre(OperandVisibilityAnalysis::OperandNode(
                                       or_to_ignore.node(), select.node()),
                                   OperandVisibilityAnalysis::OperandNode(
                                       or_to_ignore.node(), select2.node())));
}

TEST_F(VisibilityAnalysisTest, EdgesForVisibilityPrunesLargerEdgesFirst) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(4));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  // Only the visibility condition from x_and_small is needed to prove mutual
  // exclusivity between x and y; it is in the middle to better test that edges
  // are sorted by complexity so that the larger edges are pruned first.
  BValue x_and_medium =
      fb.And(x, fb.SignExtend(fb.And({fb.UGe(op, fb.Literal(UBits(3, 4))),
                                      fb.ULe(op, fb.Literal(UBits(9, 4)))}),
                              4));
  BValue x_and_small = fb.And(
      x_and_medium, fb.SignExtend(fb.UGe(op, fb.Literal(UBits(3, 4))), 4));
  BValue x_and_large = fb.And(
      x_and_small, fb.SignExtend(fb.And({fb.UGe(op, fb.Literal(UBits(3, 4))),
                                         fb.ULe(op, fb.Literal(UBits(9, 4))),
                                         fb.Ne(op, fb.Literal(UBits(6, 4)))}),
                                 4));
  BValue y_and =
      fb.And(y, fb.SignExtend(fb.ULe(op, fb.Literal(UBits(2, 4))), 4));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f, fb.BuildWithReturnValue(fb.Tuple({x_and_large, y_and})));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  LazyPostDominatorAnalysis post_dom;
  XLS_ASSERT_OK(post_dom.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto visibility, VisibilityAnalysis::Create(&operand_visibility,
                                                  bdd_engine.get(), &post_dom));

  absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode> edges;
  XLS_ASSERT_OK_AND_ASSIGN(
      edges, visibility->GetEdgesForMutuallyExclusiveVisibilityExpr(
                 x.node(), {y.node()}, -1));
  EXPECT_THAT(edges,
              UnorderedElementsAre(OperandVisibilityAnalysis::OperandNode(
                  x_and_medium.node(), x_and_small.node())));
}

TEST_F(VisibilityAnalysisTest, SingleSelectVisibilityNotPostDominating) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(4));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue z = fb.Param("z", p->GetBitsType(4));
  BValue select =
      fb.PrioritySelect(fb.Concat({fb.Eq(op, fb.Literal(UBits(2, 4))),
                                   fb.Eq(op, fb.Literal(UBits(1, 4))),
                                   fb.Eq(op, fb.Literal(UBits(0, 4)))}),
                        {y, z, y}, x);
  BValue x_and_guard =
      fb.And({x, fb.SignExtend(fb.Eq(op, fb.Literal(UBits(3, 4))), 4)});
  BValue y_and_guard =
      fb.And({y, fb.SignExtend(fb.Eq(op, fb.Literal(UBits(2, 4))), 4)});
  XLS_ASSERT_OK_AND_ASSIGN(
      auto f,
      fb.BuildWithReturnValue(fb.Tuple({select, x_and_guard, y_and_guard})));

  NodeForwardDependencyAnalysis nda;
  XLS_ASSERT_OK(nda.Attach(f));
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto operand_visibility,
      OperandVisibilityAnalysis::Create(&nda, bdd_engine.get()));
  XLS_ASSERT_OK_AND_ASSIGN(auto visibility,
                           SingleSelectVisibilityAnalysis::Create(
                               &operand_visibility, &nda, bdd_engine.get()));

  auto x_vis = *visibility->GetInfo(x.node());
  EXPECT_EQ(x_vis.select, select.node());
  auto y_vis = *visibility->GetInfo(y.node());
  EXPECT_EQ(y_vis.select, select.node());
  EXPECT_TRUE(visibility->IsMutuallyExclusive(x.node(), y.node()));
}

}  // namespace
}  // namespace xls
