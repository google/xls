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

#include "xls/passes/concat_simplification_pass.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_cse_pass.h"
#include "xls/passes/bdd_simplification_pass.h"
#include "xls/passes/bit_slice_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ConcatSimplificationPassTest : public IrTestBase {
 protected:
  ConcatSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    bool changed = true;
    bool any_concat_changed = false;
    while (changed) {
      changed = false;
      PassResults results;
      XLS_ASSIGN_OR_RETURN(bool concat_changed,
                           ConcatSimplificationPass().RunOnFunctionBase(
                               f, OptimizationPassOptions(), &results));
      changed = changed || concat_changed;
      any_concat_changed = any_concat_changed || concat_changed;

      // Run other passes to clean things up.
      XLS_ASSIGN_OR_RETURN(bool dce_changed,
                           DeadCodeEliminationPass().RunOnFunctionBase(
                               f, OptimizationPassOptions(), &results));
      changed = changed || dce_changed;
      XLS_ASSIGN_OR_RETURN(bool slice_changed,
                           BitSliceSimplificationPass().RunOnFunctionBase(
                               f, OptimizationPassOptions(), &results));
      changed = changed || slice_changed;

      XLS_ASSIGN_OR_RETURN(bool cse_changed,
                           BddCsePass().RunOnFunctionBase(
                               f, OptimizationPassOptions(), &results));
      changed = changed || cse_changed;
    }

    // Return whether concat simplification changed anything.
    return any_concat_changed;
  }
};

TEST_F(ConcatSimplificationPassTest, TrivialConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn TrivialConcat(x: bits[42]) -> bits[42] {
        ret concat: bits[42] = concat(x)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_TRUE(f->return_value()->Is<Param>());
}

TEST_F(ConcatSimplificationPassTest, TowerOfConcats) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn TowerOfConcats(x: bits[42]) -> bits[42] {
        concat.1: bits[42] = concat(x)
        concat.2: bits[42] = concat(concat.1)
        concat.3: bits[42] = concat(concat.2)
        ret concat.4: bits[42] = concat(concat.3)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 5);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_TRUE(f->return_value()->Is<Param>());
}

TEST_F(ConcatSimplificationPassTest, TreeOfConcats) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn TreeOfConcats(a: bits[16], b: bits[1], c: bits[4], d: bits[7]) -> bits[28] {
        concat.1: bits[5] = concat(b, c)
        concat.2: bits[21] = concat(a, concat.1)
        concat.3: bits[21] = concat(concat.2)
        ret concat.4: bits[28] = concat(concat.3, d)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 5);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  ASSERT_EQ(f->return_value()->operand_count(), 4);
  ASSERT_TRUE(f->return_value()->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(0)->GetName(), "a");

  ASSERT_TRUE(f->return_value()->operand(1)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(1)->GetName(), "b");

  ASSERT_TRUE(f->return_value()->operand(2)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(2)->GetName(), "c");

  ASSERT_TRUE(f->return_value()->operand(3)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(3)->GetName(), "d");
}

TEST_F(ConcatSimplificationPassTest, TreeOfConcatsOfSameValue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn TreeOfConcatsOfSameValue(a: bits[8]) -> bits[48] {
        concat.1: bits[16] = concat(a, a)
        concat.2: bits[24] = concat(a, concat.1)
        concat.3: bits[40] = concat(a, concat.2, a)
        ret concat.4: bits[48] = concat(concat.3, a)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 5);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  ASSERT_EQ(f->return_value()->operand_count(), 6);
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_TRUE(f->return_value()->operand(i)->Is<Param>());
    EXPECT_EQ(f->return_value()->operand(i)->GetName(), "a");
  }
}

TEST_F(ConcatSimplificationPassTest, ConsecutiveLiteralOperandsOfAConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn ConsecutiveLiteralOperandsOfAConcat(a: bits[16], b: bits[1]) -> bits[33] {
        literal.1: bits[2] = literal(value=1)
        literal.2: bits[2] = literal(value=2)
        literal.3: bits[4] = literal(value=0xa)
        literal.4: bits[4] = literal(value=0xb)
        literal.5: bits[4] = literal(value=0xc)
        ret concat.6: bits[33] = concat(literal.1, literal.2, a, literal.3, b, literal.4, literal.5)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 6);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  ASSERT_EQ(f->return_value()->operand_count(), 5);
  ASSERT_TRUE(f->return_value()->operand(0)->Is<Literal>());
  EXPECT_EQ(f->return_value()->operand(0)->As<Literal>()->value().bits(),
            UBits(6, 4));

  ASSERT_TRUE(f->return_value()->operand(1)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(1)->GetName(), "a");

  ASSERT_TRUE(f->return_value()->operand(2)->Is<Literal>());
  EXPECT_EQ(f->return_value()->operand(2)->As<Literal>()->value().bits(),
            UBits(0xa, 4));

  ASSERT_TRUE(f->return_value()->operand(3)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(3)->GetName(), "b");

  ASSERT_TRUE(f->return_value()->operand(4)->Is<Literal>());
  EXPECT_EQ(f->return_value()->operand(4)->As<Literal>()->value().bits(),
            UBits(0xbc, 8));
}

TEST_F(ConcatSimplificationPassTest, NotOfConcat) {
  /*
   * Bit-range --> |----|
   * a-cat ranges: |----|--|-|
   * union ranges: |----|--|-|
   *   upper bits <-----------------> lower bits
   */
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var1 = fb.Param("a_var1", p->GetBitsType(20));
  BValue a_var2 = fb.Param("a_var2", p->GetBitsType(10));
  BValue a_var3 = fb.Param("a_var3", p->GetBitsType(5));
  BValue a_cat = fb.Concat({a_var1, a_var2, a_var3});

  fb.Not(a_cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Not(m::Param("a_var1")), m::Not(m::Param("a_var2")),
                        m::Not(m::Param("a_var3"))));
}

TEST_F(ConcatSimplificationPassTest, XorOfConcat) {
  /*
   * Bit-range --> |----|
   * a-cat ranges: |--|---|
   * b_cat ranges: |--|---|
   * union ranges: |--|---|
   *   upper bits <-----------------> lower bits
   */
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var1 = fb.Param("a_var1", p->GetBitsType(8));
  BValue a_var2 = fb.Param("a_var2", p->GetBitsType(10));
  BValue b_var1 = fb.Param("b_var1", p->GetBitsType(8));
  BValue b_var2 = fb.Param("b_var2", p->GetBitsType(10));
  BValue a_cat = fb.Concat({a_var1, a_var2});
  BValue b_cat = fb.Concat({b_var1, b_var2});

  fb.Xor(a_cat, b_cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Xor(m::Param("a_var1"), m::Param("b_var1")),
                        m::Xor(m::Param("a_var2"), m::Param("b_var2"))));
}

TEST_F(ConcatSimplificationPassTest, XorOfConcatSplitRange) {
  /*
   * Bit-range --> |----|
   * a-cat ranges: |--|--|--|
   * b_cat ranges: |-----|--|
   * union ranges: |--|--|--|
   *   upper bits <-----------------> lower bits
   */
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var1 = fb.Param("a_var1", p->GetBitsType(5));
  BValue a_var2 = fb.Param("a_var2", p->GetBitsType(5));
  BValue a_var3 = fb.Param("a_var3", p->GetBitsType(5));
  BValue b_var1 = fb.Param("b_var1", p->GetBitsType(10));
  BValue b_var2 = fb.Param("b_var2", p->GetBitsType(5));
  BValue a_cat = fb.Concat({a_var1, a_var2, a_var3});
  BValue b_cat = fb.Concat({b_var1, b_var2});

  fb.Xor(a_cat, b_cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::Xor(m::Param("a_var1"), m::BitSlice(m::Param("b_var1"),
                                                 /*start=*/5, /*width=*/5)),
          m::Xor(m::Param("a_var2"), m::BitSlice(m::Param("b_var1"),
                                                 /*start=*/0, /*width=*/5)),
          m::Xor(m::Param("a_var3"), m::Param("b_var2"))));
}

TEST_F(ConcatSimplificationPassTest, XorOfConcatsSkewedRanges) {
  /*
   * Bit-range --> |----|
   * a-cat ranges: |----|--|
   * b_cat ranges: |--|----|
   * union ranges: |--|-|--|
   *   upper bits <-----------------> lower bits
   */
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var1 = fb.Param("a_var1", p->GetBitsType(8));
  BValue a_var2 = fb.Param("a_var2", p->GetBitsType(4));
  BValue b_var1 = fb.Param("b_var1", p->GetBitsType(4));
  BValue b_var2 = fb.Param("b_var2", p->GetBitsType(8));
  BValue a_cat = fb.Concat({a_var1, a_var2});
  BValue b_cat = fb.Concat({b_var1, b_var2});

  fb.Xor(a_cat, b_cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Concat(m::Xor(m::BitSlice(m::Param("a_var1"),
                                           /*start=*/4, /*width=*/4),
                               m::Param("b_var1")),
                        m::Xor(m::BitSlice(m::Param("a_var1"),
                                           /*start=*/0, /*width=*/4),
                               m::BitSlice(m::Param("b_var2"),
                                           /*start=*/4, /*width=*/4)),
                        m::Xor(m::Param("a_var2"),
                               m::BitSlice(m::Param("b_var2"),
                                           /*start=*/0, /*width=*/4))));
}

TEST_F(ConcatSimplificationPassTest, XorOfConcatZeroBitInput) {
  /*
   * Bit-range --> |----|
   * a-cat ranges: |--||--|--|
   * b_cat ranges: |------|--|
   * union ranges: |--|---|--|
   *   upper bits <-----------------> lower bits
   */
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var1 = fb.Param("a_var1", p->GetBitsType(5));
  BValue a_var2 = fb.Param("a_var2", p->GetBitsType(0));
  BValue a_var3 = fb.Param("a_var3", p->GetBitsType(5));
  BValue a_var4 = fb.Param("a_var4", p->GetBitsType(5));
  BValue b_var1 = fb.Param("b_var1", p->GetBitsType(10));
  BValue b_var2 = fb.Param("b_var2", p->GetBitsType(5));
  BValue a_cat = fb.Concat({a_var1, a_var2, a_var3, a_var4});
  BValue b_cat = fb.Concat({b_var1, b_var2});

  fb.Xor(a_cat, b_cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::Xor(m::Param("a_var1"), m::BitSlice(m::Param("b_var1"),
                                                 /*start=*/5, /*width=*/5)),
          m::Xor(m::Param("a_var3"), m::BitSlice(m::Param("b_var1"),
                                                 /*start=*/0, /*width=*/5)),
          m::Xor(m::Param("a_var4"), m::Param("b_var2"))));
}

TEST_F(ConcatSimplificationPassTest, EqOfConcatDistributes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[5], y: bits[10]) -> bits[1] {
  concat.1: bits[15] = concat(x, y)
  literal.2: bits[15] = literal(value=0)
  ret eq.3: bits[1] = eq(concat.1, literal.2)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Eq(m::Param("x"), m::BitSlice(m::Literal(0), 10, 5)),
                     m::Eq(m::Param("y"), m::BitSlice(m::Literal(0), 0, 10))));
}

TEST_F(ConcatSimplificationPassTest, EqOfConcatDistributes3Pieces) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[1], y: bits[2], z: bits[3]) -> bits[1] {
  concat.1: bits[6] = concat(x, y, z)
  literal.2: bits[6] = literal(value=0)
  ret eq.3: bits[1] = eq(concat.1, literal.2)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::And(m::And(m::Eq(m::Param("x"), m::BitSlice(m::Literal(0), 5, 1)),
                    m::Eq(m::Param("y"), m::BitSlice(m::Literal(0), 3, 2))),
             m::Eq(m::Param("z"), m::BitSlice(m::Literal(0), 0, 3))));
}

TEST_F(ConcatSimplificationPassTest, NeOfConcatDistributes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[5], y: bits[10]) -> bits[1] {
  concat.1: bits[15] = concat(x, y)
  literal.2: bits[15] = literal(value=0)
  ret result: bits[1] = ne(concat.1, literal.2)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Or(m::Ne(m::Param("x"), m::BitSlice(10, 5)),
                    m::Ne(m::Param("y"), m::BitSlice(0, 10))));
}

TEST_F(ConcatSimplificationPassTest, ConcatOnRhs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[5], y: bits[10]) -> bits[1] {
  concat.1: bits[15] = concat(x, y)
  literal.2: bits[15] = literal(value=0)
  ret result: bits[1] = eq(literal.2, concat.1)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::And(m::Eq(m::Param("x"), m::BitSlice(10, 5)),
                     m::Eq(m::Param("y"), m::BitSlice(0, 10))));
}

TEST_F(ConcatSimplificationPassTest, ReverseConcatenationOfBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[1], y: bits[1], z: bits[1]) -> bits[4] {
  concat.1: bits[3] = concat(x, y, z)
  reverse.2: bits[3] = reverse(concat.1)
  ret result: bits[4] = one_hot(reverse.2, lsb_prio=true)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHot(m::Concat(m::Param("z"), m::Param("y"), m::Param("x"))));
}

TEST_F(ConcatSimplificationPassTest,
       ReverseConcatenationOfBitsWithReverseAsRoot) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[1], y: bits[1], z: bits[1]) -> bits[3] {
  concat: bits[3] = concat(x, y, z)
  ret rev: bits[3] = reverse(concat)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("z"), m::Param("y"), m::Param("x")));
}

TEST_F(ConcatSimplificationPassTest,
       ReverseConcatenationOfBitsWithOtherUnreversibleUser) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Param("z", p->GetBitsType(1));
  BValue cat = fb.Concat({x, y, z});
  BValue reverse = fb.Reverse(cat);
  fb.Xor(reverse, cat);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest, ReverseConcatenationOfBitsWithReduction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Param("z", p->GetBitsType(1));
  BValue cat = fb.Concat({x, y, z});
  BValue reduce = fb.OrReduce(cat);
  BValue reverse = fb.Reverse(cat);
  BValue extend = fb.SignExtend(reduce, 3);
  fb.Xor(reverse, extend);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Xor(m::Concat(m::Param("z"), m::Param("y"), m::Param("x")),
             m::SignExt(m::Or(m::Param("x"), m::Param("y"), m::Param("z")))));
}

TEST_F(ConcatSimplificationPassTest,
       ReverseConcatenationOfBitsWithReductionDifferentOrderOfUsers) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Param("z", p->GetBitsType(1));
  BValue cat = fb.Concat({x, y, z});
  BValue reverse = fb.Reverse(cat);
  BValue reduce = fb.OrReduce(cat);
  BValue extend = fb.SignExtend(reduce, 3);
  fb.Xor(reverse, extend);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(
      f->return_value(),
      m::Xor(m::Concat(m::Param("z"), m::Param("y"), m::Param("x")),
             m::SignExt(m::Or(m::Param("x"), m::Param("y"), m::Param("z")))));
}

TEST_F(ConcatSimplificationPassTest, ReverseConcatenationMultiReverse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  BValue z = fb.Param("z", p->GetBitsType(1));
  BValue cat = fb.Concat({x, y, z});
  BValue reverse_a = fb.Reverse(cat);
  BValue reverse_b = fb.Reverse(cat);
  fb.Concat({reverse_a, reverse_b});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("z"), m::Param("y"), m::Param("x"),
                        m::Param("z"), m::Param("y"), m::Param("x")));
}

TEST_F(ConcatSimplificationPassTest, ReverseConcatenationOfMultiBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[2], y: bits[2], z: bits[2]) -> bits[7] {
  concat.1: bits[6] = concat(x, y, z)
  reverse.2: bits[6] = reverse(concat.1)
  ret result: bits[7] = one_hot(reverse.2, lsb_prio=true)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::OneHot(m::Concat(m::Reverse(m::Param("z")), m::Reverse(m::Param("y")),
                          m::Reverse(m::Param("x")))));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesBeginning) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(bit_slice.4, bit_slice.3, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), 0, 4), m::Param("y")));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesMiddle) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[8] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[8] = concat(y, bit_slice.4, bit_slice.3, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("y"), m::BitSlice(m::Param("x"), 0, 4),
                        m::Param("y")));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesEnd) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(y, bit_slice.4, bit_slice.3, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Param("y"), m::BitSlice(m::Param("x"), 0, 4)));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesAll) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[4] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[4] = concat(bit_slice.4, bit_slice.3, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::BitSlice(m::Param("x"), 0, 4));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesMultipleSlices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[8] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  bit_slice.5: bits[2] = bit_slice(x, start=4, width=2, pos=[(0,2,15)])
  ret result: bits[8] = concat(bit_slice.5, bit_slice.4, bit_slice.3, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Concat(m::Param("x"), m::Param("y")));
}

TEST_F(ConcatSimplificationPassTest,
       MergeConcatenationOfSlicesMultipleSliceSeries) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[8], y: bits[2]) -> bits[10] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  bit_slice.5: bits[2] = bit_slice(x, start=4, width=2, pos=[(0,2,15)])
  bit_slice.6: bits[2] = bit_slice(x, start=6, width=2, pos=[(0,2,15)])
  ret result: bits[10] = concat(bit_slice.6, bit_slice.5, y, bit_slice.4, bit_slice.3, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), 4, 4), m::Param("y"),
                        m::BitSlice(m::Param("x"), 0, 4)));
}

TEST_F(ConcatSimplificationPassTest, MergeConcatenationOfSlicesNonConsecutive) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(bit_slice.4, y, bit_slice.3, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest,
       MergeConcatenationOfSlicesNotConsecutiveReverseOrder) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(bit_slice.3, bit_slice.4, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest,
       MergeConcatenationOfSlicesNonConsecutiveBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=3, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(bit_slice.4, bit_slice.3, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest,
       MergeConcatenationOfSlicesOverlappingBits) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[6] {
  bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=1, width=2, pos=[(0,2,15)])
  ret result: bits[6] = concat(bit_slice.4, bit_slice.3, y, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest,
       MergeConcatenationOfSlicesDifferentNodeSources) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
  fn __merge_slice__main(x: bits[6], y: bits[2]) -> bits[4] {
  bit_slice.3: bits[2] = bit_slice(y, start=0, width=2, pos=[(0,1,15)])
  bit_slice.4: bits[2] = bit_slice(x, start=2, width=2, pos=[(0,2,15)])
  ret result: bits[4] = concat(bit_slice.4, bit_slice.3, pos=[(0,3,26)])
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ConcatSimplificationPassTest, BypassConcatReductionOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var = fb.Param("a_var", p->GetBitsType(1));
  BValue b_var = fb.Param("b_var", p->GetBitsType(1));
  BValue cat = fb.Concat({a_var, b_var});
  fb.OrReduce(cat);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::Param("a_var"), m::Param("b_var")));
}

TEST_F(ConcatSimplificationPassTest, BypassConcatReductionXor) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var = fb.Param("a_var", p->GetBitsType(1));
  BValue b_var = fb.Param("b_var", p->GetBitsType(1));
  BValue cat = fb.Concat({a_var, b_var});
  fb.XorReduce(cat);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Xor(m::Param("a_var"), m::Param("b_var")));
}

TEST_F(ConcatSimplificationPassTest, BypassConcatReductionAnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var = fb.Param("a_var", p->GetBitsType(1));
  BValue b_var = fb.Param("b_var", p->GetBitsType(1));
  BValue cat = fb.Concat({a_var, b_var});
  fb.AndReduce(cat);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::And(m::Param("a_var"), m::Param("b_var")));
}

TEST_F(ConcatSimplificationPassTest, BypassConcatReductionMultibit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a_var = fb.Param("a_var", p->GetBitsType(8));
  BValue b_var = fb.Param("b_var", p->GetBitsType(8));
  BValue cat = fb.Concat({a_var, b_var});
  fb.OrReduce(cat);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Or(m::OrReduce(m::Param("a_var")),
                                       m::OrReduce(m::Param("b_var"))));
}

TEST_F(ConcatSimplificationPassTest, ZeroElementConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn __sample__x5(x: bits[32]) -> bits[32] {
  zero_concat: bits[0] = concat()
  ret result: bits[32] = concat(zero_concat, x)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ConcatSimplificationPassTest, OrOfConcatWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(16));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue concat = fb.Concat({a, b, c});
  fb.Or(concat, fb.Literal(UBits(0x1234567, 28)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Or(m::Param("a"), m::Literal(0x12)),
                        m::Or(m::Param("b"), m::Literal(0x3456)),
                        m::Or(m::Param("c"), m::Literal(0x7))));
}

TEST_F(ConcatSimplificationPassTest, OrOfConcatWithLiteralAndSomethingElse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(16));
  BValue c = fb.Param("c", p->GetBitsType(4));
  BValue d = fb.Param("d", p->GetBitsType(28));
  BValue concat = fb.Concat({a, b, c});
  fb.Or({concat, d, fb.Literal(UBits(0x1234567, 28))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
