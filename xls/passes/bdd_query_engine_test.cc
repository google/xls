// Copyright 2020 The XLS Authors
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

#include "xls/passes/bdd_query_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
#include "xls/solvers/prover_matchers.h"
#include "xls/solvers/solver.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

class BddQueryEngineTest : public IrTestBase {
 protected:
  // Convenience methods for testing implication, equality, and inverse for
  // single-bit node values.
  bool Implies(const QueryEngine& engine, Node* a, Node* b) {
    return engine.Implies(TreeBitLocation(a, 0), TreeBitLocation(b, 0));
  }
  bool KnownEquals(const QueryEngine& engine, Node* a, Node* b) {
    return engine.KnownEquals(TreeBitLocation(a, 0), TreeBitLocation(b, 0));
  }
  bool KnownNotEquals(const QueryEngine& engine, Node* a, Node* b) {
    return engine.KnownNotEquals(TreeBitLocation(a, 0), TreeBitLocation(b, 0));
  }
};

TEST_F(BddQueryEngineTest, EqualToPredicates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue x_eq_0 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_eq_0_2 = fb.Eq(x, fb.Literal(UBits(0, 8)));
  BValue x_ne_0 = fb.Not(x_eq_0);
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(7, 8)));
  BValue y_eq_42 = fb.Eq(y, fb.Literal(UBits(7, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  EXPECT_TRUE(query_engine.AtMostOneNodeTrue({}));
  EXPECT_FALSE(query_engine.AtMostOneBitTrue(x.node()));
  EXPECT_TRUE(query_engine.AtMostOneNodeTrue({x_eq_0.node(), x_eq_42.node()}));
  EXPECT_TRUE(query_engine.AtLeastOneNodeTrue({x_eq_0.node(), x_ne_0.node()}));

  EXPECT_TRUE(KnownEquals(query_engine, x_eq_0.node(), x_eq_0.node()));
  EXPECT_TRUE(KnownEquals(query_engine, x_eq_0.node(), x_eq_0_2.node()));
  EXPECT_FALSE(KnownNotEquals(query_engine, x_eq_0.node(), x_eq_0_2.node()));
  EXPECT_TRUE(KnownNotEquals(query_engine, x_eq_0.node(), x_ne_0.node()));

  EXPECT_TRUE(Implies(query_engine, x_eq_0.node(), x_eq_0.node()));
  EXPECT_TRUE(Implies(query_engine, x_eq_0.node(), x_eq_0_2.node()));
  EXPECT_FALSE(Implies(query_engine, x_eq_0.node(), x_eq_42.node()));

  // Unrelated values 'x' and 'y' should have no relationships.
  EXPECT_FALSE(Implies(query_engine, x_eq_42.node(), y_eq_42.node()));
  EXPECT_FALSE(KnownEquals(query_engine, x_eq_42.node(), y_eq_42.node()));
  EXPECT_FALSE(KnownNotEquals(query_engine, x_eq_42.node(), y_eq_42.node()));
  EXPECT_FALSE(
      query_engine.AtMostOneNodeTrue({x_eq_42.node(), y_eq_42.node()}));
  EXPECT_FALSE(
      query_engine.AtLeastOneNodeTrue({x_eq_42.node(), y_eq_42.node()}));
}

TEST_F(BddQueryEngineTest, VariousComparisonPredicates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_eq_42 = fb.Eq(x, fb.Literal(UBits(42, 32)));
  BValue x_lt_42 = fb.ULt(x, fb.Literal(UBits(42, 32)));
  BValue x_ge_20 = fb.UGe(x, fb.Literal(UBits(20, 32)));
  BValue x_lt_20 = fb.ULt(x, fb.Literal(UBits(20, 32)));
  BValue x_eq_7 = fb.Eq(x, fb.Literal(UBits(7, 32)));
  BValue x_eq_999 = fb.Eq(x, fb.Literal(UBits(999, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  EXPECT_TRUE(query_engine.AtMostOneNodeTrue(
      {x_eq_42.node(), x_eq_7.node(), x_eq_999.node()}));
  EXPECT_FALSE(
      query_engine.AtMostOneNodeTrue({x_lt_42.node(), x_ge_20.node()}));

  EXPECT_TRUE(
      query_engine.AtLeastOneNodeTrue({x_lt_42.node(), x_ge_20.node()}));
  EXPECT_TRUE(
      query_engine.AtLeastOneNodeTrue({x_ge_20.node(), x_lt_20.node()}));

  EXPECT_TRUE(Implies(query_engine, x_eq_7.node(), x_lt_42.node()));
  EXPECT_FALSE(Implies(query_engine, x_lt_42.node(), x_eq_7.node()));
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueSimple) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue x_not = fb.Not(x);
  BValue concat = fb.Concat({x, x_not});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result =
      query_engine.ImpliedNodeValue({{{x.node(), 0}, true}}, concat.node());
  ASSERT_TRUE(result.has_value());
  EXPECT_THAT(result.value().ToBitVector(), testing::ElementsAre(false, true));
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueComplex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue a_or_b = fb.Or(a, b);
  BValue a_and_b = fb.And(a, b);
  BValue c_and_d = fb.And(c, d);
  BValue c_xor_d = fb.Xor(c, d);
  BValue concat = fb.Concat({a_or_b, c_and_d});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue(
      {{{a_and_b.node(), 0}, true}, {{c_xor_d.node(), 0}, true}},
      concat.node());
  ASSERT_TRUE(result.has_value());
  EXPECT_THAT(result.value().ToBitVector(), testing::ElementsAre(false, true));
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueFalsePredice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue a_or_b = fb.Or(a, b);
  BValue a_and_b = fb.And(a, b);
  BValue a_xor_b = fb.Xor(a, b);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue(
      {{{a_and_b.node(), 0}, false}, {{a_or_b.node(), 0}, true}},
      a_xor_b.node());
  ASSERT_TRUE(result.has_value());
  EXPECT_THAT(result.value().ToBitVector(), testing::ElementsAre(true));
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueNoValueImpliedLogical) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue a_or_b = fb.Or(a, b);
  BValue a_and_b = fb.And(a, b);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue({{{a_or_b.node(), 0}, true}},
                                              a_and_b.node());
  EXPECT_FALSE(result.has_value());
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueNotImpliedUnrelated) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue d = fb.Param("d", p->GetBitsType(1));
  BValue a_or_b = fb.Or(a, b);
  BValue a_and_b = fb.And(a, b);
  BValue c_and_d = fb.And(c, d);
  BValue c_xor_d = fb.Xor(c, d);

  BValue q = fb.Param("q", p->GetBitsType(1));
  BValue concat = fb.Concat({a_or_b, c_and_d, q});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue(
      {{{a_and_b.node(), 0}, true}, {{c_xor_d.node(), 0}, true}},
      concat.node());
  EXPECT_FALSE(result.has_value());
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValueNoValueImpliedNonBit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue a_and_b = fb.And(a, b);
  BValue array = fb.Array({a, b}, a.node()->GetType());

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue({{{a_and_b.node(), 0}, true}},
                                              array.node());
  EXPECT_FALSE(result.has_value());
}

TEST_F(BddQueryEngineTest,
       BitValuesImplyNodeValueNoValueImpliedEmptyPredicate) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue({}, a.node());
  EXPECT_FALSE(result.has_value());
}

TEST_F(BddQueryEngineTest, ForceNodeToBeModeledAsVariable) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue x_not = fb.Not(x);
  BValue andop = fb.And(x, x_not);
  BValue orop = fb.Or(x, x_not);
  BValue my_one = fb.Literal(UBits(1, 1));
  BValue my_zero = fb.Literal(UBits(0, 1));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine(
      /*path_limit=*/0, [](const Node* n) { return n->op() != Op::kOr; });
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_FALSE(KnownEquals(query_engine, andop.node(), my_one.node()));
  EXPECT_TRUE(KnownEquals(query_engine, andop.node(), my_zero.node()));
  EXPECT_FALSE(KnownEquals(query_engine, orop.node(), my_one.node()));
  EXPECT_FALSE(KnownEquals(query_engine, orop.node(), my_zero.node()));
  BddQueryEngine query_engine_empty_op_set(/*path_limit=*/0);
  XLS_ASSERT_OK(query_engine_empty_op_set.Populate(f).status());
  EXPECT_FALSE(
      KnownEquals(query_engine_empty_op_set, andop.node(), my_one.node()));
  EXPECT_TRUE(
      KnownEquals(query_engine_empty_op_set, andop.node(), my_zero.node()));
  EXPECT_TRUE(
      KnownEquals(query_engine_empty_op_set, orop.node(), my_one.node()));
  EXPECT_FALSE(
      KnownEquals(query_engine_empty_op_set, orop.node(), my_zero.node()));
}

TEST_F(BddQueryEngineTest, BitValuesImplyNodeValuePredicateAlwaysFalse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue a_not = fb.Not(a);
  BValue orop = fb.Or(a, a_not);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto result = query_engine.ImpliedNodeValue(
      {{{a.node(), 0}, true}, {{a_not.node(), 0}, true}}, orop.node());
  EXPECT_FALSE(result.has_value());
}

TEST_F(BddQueryEngineTest, ImpliedNodeTernaryChecksPathLength) {
  auto p = CreatePackage();
  // Function found with fuzzing and would OOM the process when BDD query engine
  // is called with specific arguments.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(
                                             R"(
fn __sample__main(x0: bits[16], x1: bits[8], x2: bits[10], x3: bits[1]) -> (bits[13], bits[16], bits[16][6], bits[16], bits[16]) {
  literal.73: bits[6] = literal(value=0, id=73)
  concat.74: bits[16] = concat(literal.73, x2, id=74)
  x4: bits[16] = and(concat.74, x0, id=6, pos=[(0,4,32)])
  bit_slice.97: bits[10] = bit_slice(x4, start=0, width=10, id=97, pos=[(0,20,25)])
  literal.100: bits[10] = literal(value=2, id=100, pos=[(0,20,25)])
  x22: bits[16] = neg(x0, id=39, pos=[(0,21,23)])
  ugt.99: bits[1] = ugt(bit_slice.97, literal.100, id=99, pos=[(0,20,25)])
  literal.36: bits[16] = literal(value=2, id=36, pos=[(0,20,44)])
  bit_slice.49: bits[13] = bit_slice(x4, start=0, width=13, id=49)
  x23: bits[13] = bit_slice(x22, start=3, width=13, id=88, pos=[(0,22,26)])
  sel.78: bits[16] = sel(ugt.99, cases=[x4, literal.36], id=78, pos=[(0,20,25)])
  x28: bits[13] = and(bit_slice.49, x23, id=50, pos=[(0,26,33)])
  x12: bits[16][6] = array(x0, x4, x0, x0, x4, x0, id=92, pos=[(0,11,28)])
  x17: bits[16] = literal(value=0, id=9, pos=[(0,5,47)])
  x21: bits[16] = sel(sel.78, cases=[x0, x4], default=x0, id=96, pos=[(0,20,24)])
  ret tuple.51: (bits[13], bits[16], bits[16][6], bits[16], bits[16]) = tuple(x28, x0, x12, x17, x21, id=51, pos=[(0,27,8)])
}
)",
                                             p.get()));
  BddQueryEngine query_engine(/*path_limit=*/1024);
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  XLS_ASSERT_OK_AND_ASSIGN(Node * sel_node, f->GetNode("sel.78"));
  // NB The specific target is irrelevant.
  XLS_ASSERT_OK_AND_ASSIGN(Node * target, f->GetNode("x4"));
  std::vector<std::pair<TreeBitLocation, bool>> vals;
  // Set sel-node to 0x1.
  vals.reserve(sel_node->BitCountOrDie());
  vals.push_back({TreeBitLocation(sel_node, 0), true});
  for (int64_t i = 1; i < sel_node->BitCountOrDie(); ++i) {
    vals.push_back({TreeBitLocation(sel_node, i), false});
  }
  // This will blow up the path depth.
  EXPECT_EQ(query_engine.ImpliedNodeTernary(vals, target), std::nullopt)
      << "Expected failure to find result due to path-size explosion.";
}
TEST_F(BddQueryEngineTest, SpecializeGivenIntervalsHandlesSaturation) {
  // Found via fuzzing. When told to specialize on an interval set, if one
  // interval was saturated, it would be omitted from the check - making the
  // specialization assume more than was requested. In this case, it made a
  // value appear to be determined when it wasn't.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(
                                             R"(
fn __sample__main(x: bits[60]) -> bits[35] {
  bit_slice.1: bits[55] = bit_slice(x, start=5, width=55, id=1)
  literal.2: bits[60] = literal(value=459229419044003026, id=2)
  or_reduce.3: bits[1] = or_reduce(bit_slice.1, id=3)
  ult.4: bits[1] = ult(x, literal.2, id=4)
  literal.5: bits[60] = literal(value=0, id=5)
  literal.6: bits[60] = literal(value=465194495002459937, id=6)
  and.7: bits[1] = and(or_reduce.3, ult.4, id=7)
  eq.8: bits[1] = eq(x, literal.5, id=8)
  eq.9: bits[1] = eq(x, literal.6, id=9)
  literal.10: bits[1] = literal(value=0, id=10)
  or.11: bits[1] = or(and.7, eq.8, id=11)
  y: bits[3] = concat(eq.9, literal.10, or.11, id=12)
  ret z: bits[35] = bit_slice(x, start=0, width=35, id=13)
}
)",
                                             p.get()));
  BddQueryEngine query_engine(1024);
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  IntervalSet intervals = IntervalSet::Of(
      {Interval(UBits(1, 3), UBits(1, 3)), Interval(UBits(5, 3), UBits(5, 3))});
  XLS_ASSERT_OK_AND_ASSIGN(Node * specialized_node, f->GetNode("y"));
  std::unique_ptr<QueryEngine> specialized = query_engine.SpecializeGiven(
      {{specialized_node,
        ValueKnowledge{.intervals = IntervalSetTree::CreateSingleElementTree(
                           specialized_node->GetType(), intervals)}}});
  XLS_ASSERT_OK_AND_ASSIGN(Node * target, f->GetNode("z"));
  EXPECT_EQ(specialized->KnownValueAsBits(target), std::nullopt);
}

TEST_F(BddQueryEngineTest, DetectKeepLsbsLeavesHighBitsZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("x", p->GetBitsType(4));
  BValue ext = fb.ZeroExtend(a, 64, {}, "ext");
  BValue lead_zero = fb.Clz(ext, {}, "lead_zero");
  Node* lz_one_hot = lead_zero.node()->operand(0)->operand(0);
  ASSERT_THAT(lz_one_hot, m::OneHot());
  BValue rev_lz_one_hot =
      fb.Reverse(BValue(lz_one_hot, &fb), {}, "rev_lz_one_hot");
  BValue to_keep = fb.Subtract(fb.Literal(UBits(64, 7)),
                               fb.BitSlice(lead_zero, 0, 7), {}, "to_keep");
  BValue res = fb.And(fb.Param("inp", p->GetBitsType(64)),
                      fb.Not(fb.Shll(fb.Literal(Bits::AllOnes(64)), to_keep)));
  BValue top_zeros = fb.BitSlice(res, 4, 60);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  BddQueryEngine query_engine(1024);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      solvers::TryProve(f, top_zeros.node(), solvers::Predicate::EqualToZero(),
                        solvers::SolverLimit()));
  EXPECT_THAT(result, solvers::IsProvenTrue()) << f->DumpIr(
      solvers::CounterExampleAnnotator(std::get<solvers::ProvenFalse>(result),
                                       FormatPreference::kZeroPaddedHex));
  XLS_ASSERT_OK(query_engine.Populate(f).status());
  EXPECT_THAT(query_engine.KnownLeadingZeros(rev_lz_one_hot.node()),
              testing::Optional(60))
      << query_engine.GetTernary(lz_one_hot)->Get({});
  EXPECT_THAT(query_engine.ExactlyOneBitTrue(lz_one_hot), true);
  EXPECT_THAT(query_engine.KnownLeadingZeros(res.node()),
              testing::Optional(60));
}

TEST_F(BddQueryEngineTest, NestedSpecializeGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue both = fb.And(x, y);
  BValue not_both = fb.Not(both);
  BValue a = fb.Param("a", p->GetBitsType(4));
  BValue b = fb.Param("b", p->GetBitsType(4));
  BValue sum = fb.Add(a, b);
  BValue sel = fb.Select(x, {fb.Literal(UBits(0, 1)), y});
  fb.Tuple({both, not_both, sum, sel});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  BddQueryEngine query_engine;
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto bit_one = [&](Node* n) {
    return ValueKnowledge{
        .ternary = TernaryTree::CreateSingleElementTree(
            n->GetType(), TernaryVector{TernaryValue::kKnownOne})};
  };
  auto precise_u4 = [&](Node* n, uint64_t val) {
    return ValueKnowledge{
        .intervals = IntervalSetTree::CreateSingleElementTree(
            n->GetType(), IntervalSet::Precise(UBits(val, 4)))};
  };

  // Nested SpecializeGiven with ternary givens: outer assumes x == 1, inner
  // assumes y == 1. Both assumptions should hold in the nested engine without
  // mutating the outer engine.
  std::unique_ptr<QueryEngine> s1 =
      query_engine.SpecializeGiven({{x.node(), bit_one(x.node())}});
  EXPECT_TRUE(s1->IsAllOnes(x.node()));
  EXPECT_FALSE(s1->IsAllOnes(y.node()));
  EXPECT_FALSE(s1->IsAllOnes(both.node()));
  EXPECT_FALSE(s1->IsAllZeros(not_both.node()));

  std::unique_ptr<QueryEngine> s2 =
      s1->SpecializeGiven({{y.node(), bit_one(y.node())}});
  EXPECT_TRUE(s2->IsAllOnes(x.node()));
  EXPECT_TRUE(s2->IsAllOnes(y.node()));
  EXPECT_TRUE(s2->IsAllOnes(both.node()));
  EXPECT_TRUE(s2->IsAllZeros(not_both.node()));
  // Outer specialized engine must remain unaffected.
  EXPECT_FALSE(s1->IsAllOnes(y.node()));
  EXPECT_FALSE(s1->IsAllOnes(both.node()));

  // Nested SpecializeGiven with interval givens: outer assumes a == 3, inner
  // assumes b == 4, so a + b == 7.
  std::unique_ptr<QueryEngine> i1 =
      query_engine.SpecializeGiven({{a.node(), precise_u4(a.node(), 3)}});
  EXPECT_EQ(i1->KnownValueAsBits(sum.node()), std::nullopt);
  std::unique_ptr<QueryEngine> i2 =
      i1->SpecializeGiven({{b.node(), precise_u4(b.node(), 4)}});
  EXPECT_EQ(i2->KnownValueAsBits(a.node()), UBits(3, 4));
  EXPECT_EQ(i2->KnownValueAsBits(b.node()), UBits(4, 4));
  EXPECT_EQ(i2->KnownValueAsBits(sum.node()), UBits(7, 4));

  // Combining SpecializeGivenPredicate and SpecializeGiven in both orders.
  PredicateState arm1(sel.node()->As<Select>(), /*arm_index=*/1);
  std::unique_ptr<QueryEngine> pred_then_given =
      query_engine.SpecializeGivenPredicate({arm1})->SpecializeGiven(
          {{y.node(), bit_one(y.node())}});
  EXPECT_TRUE(pred_then_given->IsAllOnes(x.node()));
  EXPECT_TRUE(pred_then_given->IsAllOnes(y.node()));
  EXPECT_TRUE(pred_then_given->IsAllOnes(both.node()));

  std::unique_ptr<QueryEngine> given_then_pred =
      query_engine.SpecializeGiven({{y.node(), bit_one(y.node())}})
          ->SpecializeGivenPredicate({arm1});
  EXPECT_TRUE(given_then_pred->IsAllOnes(x.node()));
  EXPECT_TRUE(given_then_pred->IsAllOnes(y.node()));
  EXPECT_TRUE(given_then_pred->IsAllOnes(both.node()));

  // Contradictory nested givens (both via conflicting bits and via an empty
  // interval set) produce an unsatisfiable assumption, and further nesting
  // stays unsatisfiable.
  std::unique_ptr<QueryEngine> conflicting_bits = s1->SpecializeGiven(
      {{x.node(),
        ValueKnowledge{
            .ternary = TernaryTree::CreateSingleElementTree(
                x.GetType(), TernaryVector{TernaryValue::kKnownZero})}}});
  EXPECT_TRUE(conflicting_bits->IsAllZeros(y.node()));
  EXPECT_TRUE(conflicting_bits->IsAllOnes(y.node()));

  std::unique_ptr<QueryEngine> empty_interval = s1->SpecializeGiven(
      {{a.node(),
        ValueKnowledge{.intervals = IntervalSetTree::CreateSingleElementTree(
                           a.GetType(), IntervalSet(4))}}});
  EXPECT_TRUE(empty_interval->IsAllZeros(y.node()));
  EXPECT_TRUE(empty_interval->IsAllOnes(y.node()));
  std::unique_ptr<QueryEngine> after_empty =
      empty_interval->SpecializeGiven({{y.node(), bit_one(y.node())}});
  EXPECT_TRUE(after_empty->IsAllZeros(y.node()));
  EXPECT_TRUE(after_empty->IsAllOnes(y.node()));
}

TEST_F(BddQueryEngineTest, NestedSpecializeGivenRespectsPathLimit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Param("z", p->GetBitsType(1));
  BValue w = fb.Param("w", p->GetBitsType(1));
  BValue extra = fb.Param("extra", p->GetBitsType(1));
  // e1 = x ^ y (4 paths), e2 = z ^ w (4 paths); e1 & e2 has 10 paths > 8.
  BValue e1 = fb.Xor(x, y);
  BValue e2 = fb.Xor(z, w);
  fb.Tuple({e1, e2, extra});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  BddQueryEngine query_engine(/*path_limit=*/8);
  XLS_ASSERT_OK(query_engine.Populate(f).status());

  auto bit_one = [&](Node* n) {
    return ValueKnowledge{
        .ternary = TernaryTree::CreateSingleElementTree(
            n->GetType(), TernaryVector{TernaryValue::kKnownOne})};
  };

  std::unique_ptr<QueryEngine> s1 =
      query_engine.SpecializeGiven({{e1.node(), bit_one(e1.node())}});
  EXPECT_TRUE(s1->IsAllOnes(e1.node()));

  // Conjoining e2 into e1 exceeds path_limit=8, so e2 is skipped while the
  // smaller given `extra == 1` is still conjoined.
  std::unique_ptr<QueryEngine> s2 = s1->SpecializeGiven(
      {{e2.node(), bit_one(e2.node())}, {extra.node(), bit_one(extra.node())}});
  EXPECT_TRUE(s2->IsAllOnes(e1.node()));
  EXPECT_FALSE(s2->IsAllOnes(e2.node()));
  EXPECT_TRUE(s2->IsAllOnes(extra.node()));
}

}  // namespace
}  // namespace xls
