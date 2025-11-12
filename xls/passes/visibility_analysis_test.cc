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

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {

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

TEST_F(VisibilityAnalysisTest, VisibilityThroughPrioritySelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue selector = fb.Param("y", p->GetBitsType(2));
  BValue add = fb.Add(x, fb.Literal(UBits(1, 4)));
  BValue select = fb.PrioritySelect(selector, {x, add}, x);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());
  BddNodeIndex and_visible = *visibility.GetInfo(add.node());

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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());
  BddNodeIndex add_visible = *visibility.GetInfo(add.node());
  BddNodeIndex sub_visible = *visibility.GetInfo(sub.node());

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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(and_xyz));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());
  BddNodeIndex z_visible = *visibility.GetInfo(z.node());

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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(or_xyz));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());
  BddNodeIndex z_visible = *visibility.GetInfo(z.node());

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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(expensive_select));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  // Set term limit to significantly fewer terms than the expensive cond needs
  VisibilityAnalysis visibility(2, &nda, bdd_engine.get());
  BddNodeIndex add_visible = *visibility.GetInfo(add.node());

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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(use_add));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
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

  VisibilityAnalysis visibility(term_limit, &nda, bdd_engine.get());
  BddNodeIndex add_visible = *visibility.GetInfo(add.node());
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

TEST_F(VisibilityAnalysisTest, VisibilityAssumeAlwaysUsedIfTooManyUsers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue op = fb.Param("op", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue add = fb.Add(x, y);
  BValue select = fb.Select(op, {x}, add);
  for (int i = 0; i < 100; ++i) {
    select = fb.Select(op, {select}, add);
  }
  XLS_ASSERT_OK(fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;

  std::unique_ptr<BddQueryEngine> bdd_engine_small =
      std::make_unique<BddQueryEngine>(200, IsCheapForBdds);
  BinaryDecisionDiagram& bdd_small = bdd_engine_small->bdd();
  VisibilityAnalysis visibility_small(&nda, bdd_engine_small.get());
  EXPECT_EQ(*visibility_small.GetInfo(add.node()), bdd_small.one());

  std::unique_ptr<BddQueryEngine> bdd_engine_large =
      std::make_unique<BddQueryEngine>(201, IsCheapForBdds);
  BinaryDecisionDiagram& bdd_large = bdd_engine_large->bdd();
  VisibilityAnalysis visibility_large(&nda, bdd_engine_large.get());
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<BddNodeIndex> op_bits,
                           GetNodeBits(op.node(), *bdd_engine_large));
  EXPECT_NE(*visibility_large.GetInfo(add.node()), bdd_small.one());
  EXPECT_EQ(bdd_large.Implies(bdd_large.Not(op_bits[0]),
                              *visibility_large.GetInfo(add.node())),
            bdd_large.one());
  EXPECT_EQ(
      bdd_large.Implies(op_bits[0],
                        bdd_large.Not(*visibility_large.GetInfo(add.node()))),
      bdd_large.one());
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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(select));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());

  EXPECT_TRUE(visibility.IsMutuallyExclusive(add.node(), sub.node()));
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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(fb.Tuple({select, select2})));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());

  EXPECT_TRUE(visibility.IsMutuallyExclusive(add.node(), sub.node()));
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
  XLS_ASSERT_OK(fb.BuildWithReturnValue(result));

  NodeForwardDependencyAnalysis nda;
  std::unique_ptr<BddQueryEngine> bdd_engine = BddQueryEngine::MakeDefault();
  BinaryDecisionDiagram& bdd = bdd_engine->bdd();
  VisibilityAnalysis visibility(&nda, bdd_engine.get());

  // assert visibility on later mul expression on 'op' value
  BddNodeIndex mul_survived_visible = *visibility.GetInfo(mul_survived.node());
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

  EXPECT_TRUE(visibility.IsMutuallyExclusive(mul.node(), mul2.node()));
}

}  // namespace
}  // namespace xls
