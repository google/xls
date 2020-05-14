// Copyright 2020 Google LLC
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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_interpreter.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ConcatSimplificationPassTest : public IrTestBase {
 protected:
  ConcatSimplificationPassTest() = default;

  xabsl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed, ConcatSimplificationPass().RunOnFunction(
                                           f, PassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunction(f, PassOptions(), &results)
                            .status());
    // Return whether concat simplification changed anything.
    return changed;
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
  for (int64 i = 0; i < 6; ++i) {
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
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn ConsecutiveLiteralOperandsOfAConcat(a: bits[16], b: bits[10], c: bits[7]) -> bits[33] {
        concat.1: bits[33] = concat(a, b, c)
        ret not.2: bits[33] = not(concat.1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 5);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  ASSERT_EQ(f->return_value()->operand_count(), 3);

  ASSERT_EQ(f->return_value()->operand(0)->op(), Op::kNot);
  EXPECT_TRUE(f->return_value()->operand(0)->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->GetName(), "a");

  ASSERT_EQ(f->return_value()->operand(1)->op(), Op::kNot);
  EXPECT_TRUE(f->return_value()->operand(1)->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(1)->operand(0)->GetName(), "b");

  ASSERT_EQ(f->return_value()->operand(2)->op(), Op::kNot);
  EXPECT_TRUE(f->return_value()->operand(2)->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(2)->operand(0)->GetName(), "c");
}

TEST_F(ConcatSimplificationPassTest, XorOfConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn ConsecutiveLiteralOperandsOfAConcat(a: bits[8], b: bits[10], c: bits[8], d: bits[10]) -> bits[18] {
        concat.1: bits[18] = concat(a, b)
        concat.2: bits[18] = concat(c, d)
        ret xor.3: bits[18] = xor(concat.1, concat.2)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  ASSERT_EQ(f->return_value()->operand_count(), 2);

  ASSERT_EQ(f->return_value()->operand(0)->op(), Op::kXor);
  EXPECT_TRUE(f->return_value()->operand(0)->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->GetName(), "a");
  EXPECT_TRUE(f->return_value()->operand(0)->operand(1)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(0)->operand(1)->GetName(), "c");

  ASSERT_EQ(f->return_value()->operand(1)->op(), Op::kXor);
  EXPECT_TRUE(f->return_value()->operand(1)->operand(0)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(1)->operand(0)->GetName(), "b");
  EXPECT_TRUE(f->return_value()->operand(1)->operand(1)->Is<Param>());
  EXPECT_EQ(f->return_value()->operand(1)->operand(1)->GetName(), "d");
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
  EXPECT_EQ(f->DumpIr(), R"(fn f(x: bits[5], y: bits[10]) -> bits[1] {
  literal.2: bits[15] = literal(value=0)
  bit_slice.6: bits[5] = bit_slice(literal.2, start=10, width=5)
  bit_slice.8: bits[10] = bit_slice(literal.2, start=0, width=10)
  eq.7: bits[1] = eq(x, bit_slice.6)
  eq.9: bits[1] = eq(y, bit_slice.8)
  ret and.10: bits[1] = and(eq.7, eq.9)
}
)");
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
  EXPECT_EQ(f->DumpIr(),
            R"(fn f(x: bits[1], y: bits[2], z: bits[3]) -> bits[1] {
  literal.2: bits[6] = literal(value=0)
  bit_slice.7: bits[1] = bit_slice(literal.2, start=5, width=1)
  bit_slice.9: bits[2] = bit_slice(literal.2, start=3, width=2)
  eq.8: bits[1] = eq(x, bit_slice.7)
  eq.10: bits[1] = eq(y, bit_slice.9)
  bit_slice.12: bits[3] = bit_slice(literal.2, start=0, width=3)
  and.11: bits[1] = and(eq.8, eq.10)
  eq.13: bits[1] = eq(z, bit_slice.12)
  ret and.14: bits[1] = and(and.11, eq.13)
}
)");
}

TEST_F(ConcatSimplificationPassTest, NeOfConcatDistributes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[5], y: bits[10]) -> bits[1] {
  concat.1: bits[15] = concat(x, y)
  literal.2: bits[15] = literal(value=0)
  ret eq.3: bits[1] = ne(concat.1, literal.2)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->DumpIr(), R"(fn f(x: bits[5], y: bits[10]) -> bits[1] {
  literal.2: bits[15] = literal(value=0)
  bit_slice.6: bits[5] = bit_slice(literal.2, start=10, width=5)
  bit_slice.8: bits[10] = bit_slice(literal.2, start=0, width=10)
  ne.7: bits[1] = ne(x, bit_slice.6)
  ne.9: bits[1] = ne(y, bit_slice.8)
  ret or.10: bits[1] = or(ne.7, ne.9)
}
)");
}

TEST_F(ConcatSimplificationPassTest, ConcatOnRhs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[5], y: bits[10]) -> bits[1] {
  concat.1: bits[15] = concat(x, y)
  literal.2: bits[15] = literal(value=0)
  ret eq.3: bits[1] = eq(literal.2, concat.1)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->DumpIr(), R"(fn f(x: bits[5], y: bits[10]) -> bits[1] {
  literal.2: bits[15] = literal(value=0)
  bit_slice.6: bits[5] = bit_slice(literal.2, start=10, width=5)
  bit_slice.8: bits[10] = bit_slice(literal.2, start=0, width=10)
  eq.7: bits[1] = eq(x, bit_slice.6)
  eq.9: bits[1] = eq(y, bit_slice.8)
  ret and.10: bits[1] = and(eq.7, eq.9)
}
)");
}

}  // namespace
}  // namespace xls
