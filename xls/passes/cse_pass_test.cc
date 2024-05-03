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

#include "xls/passes/cse_pass.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

TEST(FixedPointOfSPO, Simple) {
  absl::flat_hash_map<std::string, std::string> spo;
  spo["foo"] = "bar";
  spo["bar"] = "baz";
  spo["qux"] = "qux";
  spo["a"] = "b";

  absl::flat_hash_map<std::string, std::string> result = FixedPointOfSPO(spo);
  EXPECT_EQ(result.at("foo"), "baz");
  EXPECT_EQ(result.at("bar"), "baz");
  EXPECT_EQ(result.at("qux"), "qux");
  EXPECT_EQ(result.at("a"), "b");
}

class CsePassTest : public IrTestBase {
 protected:
  CsePassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed,
        CsePass().RunOnFunctionBase(f, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
            .status());
    // Return whether cse changed anything.
    return changed;
  }
};

TEST_F(CsePassTest, SingleLiteralNoChange) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn single_literal() -> bits[2] {
        ret one: bits[2] = literal(value=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 1);
}

TEST_F(CsePassTest, TwoIdenticalLiterals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn IdenticalLiterals() -> (bits[2], bits[2]) {
        literal.1: bits[2] = literal(value=1)
        literal.2: bits[2] = literal(value=1)
        ret tuple.3: (bits[2], bits[2]) = tuple(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_EQ(f->return_value()->operand(0), f->return_value()->operand(1));
}

TEST_F(CsePassTest, NontrivialCommonSubexpressions) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn nontrivial(x: bits[8], y: bits[8], z: bits[8]) -> bits[8] {
        and.1: bits[8] = and(x, y)
        neg.2: bits[8] = neg(and.1)
        or.3: bits[8] = or(neg.2, z)

        and.4: bits[8] = and(x, y)
        neg.5: bits[8] = neg(and.4)
        or.6: bits[8] = or(neg.5, z)

        ret add.7: bits[8] = add(or.3, or.6)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 10);
  EXPECT_EQ(FindNode("x", f)->users().size(), 2);
  EXPECT_EQ(FindNode("y", f)->users().size(), 2);

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_EQ(f->return_value()->operand(0), f->return_value()->operand(1));
  EXPECT_THAT(
      f->return_value(),
      m::Add(m::Or(m::Neg(m::And(m::Param("x"), m::Param("y"))), m::Param("z")),
             m::Or()));
}

TEST_F(CsePassTest, CountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package CountedFor

fn body(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y)
}

fn same_as_body(a: bits[11], b: bits[11]) -> bits[11] {
  ret add.102: bits[11] = add(a, b)
}

top fn main(x: bits[11]) -> (bits[11], bits[11], bits[11]) {
  counted_for.6: bits[11] = counted_for(x, trip_count=7, stride=1, body=body)
  counted_for.7: bits[11] = counted_for(x, trip_count=7, stride=1, body=body)
  counted_for.8: bits[11] = counted_for(x, trip_count=7, stride=1, body=same_as_body)
  ret tuple.4: (bits[11], bits[11], bits[11]) = tuple(counted_for.6, counted_for.7, counted_for.8)
}
)"));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, p->GetTopAsFunction());
  EXPECT_EQ(entry->node_count(), 5);

  EXPECT_THAT(Run(entry), IsOkAndHolds(true));

  EXPECT_EQ(entry->node_count(), 3);
  EXPECT_EQ(entry->return_value()->operand(0),
            entry->return_value()->operand(1));
  EXPECT_EQ(entry->return_value()->operand(0),
            entry->return_value()->operand(2));
}

TEST_F(CsePassTest, BitSliceConcat) {
  // Note: in constructing this test case we're careful to put a literal value
  // with a different bitwidth ahead of the other literals in topological order
  // (as a regression test for not expanding the node bucket for a distinct
  // value).
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package bit_slice_concat

top fn main(x: bits[11]) -> (bits[11], bits[11], bits[11]) {
  bit_slice.2: bits[10] = bit_slice(x, start=1, width=10)
  literal.42: bits[32] = literal(value=0)
  concat.43: bits[42] = concat(literal.42, bit_slice.2)
  bit_slice.44: bits[10] = bit_slice(concat.43, start=0, width=10)
  literal.3: bits[1] = literal(value=0, pos=[(1,2,3)])
  literal.4: bits[1] = literal(value=0, pos=[(4,5,6)])
  literal.5: bits[1] = literal(value=0, pos=[(7,8,9)])
  concat.6: bits[11] = concat(bit_slice.44, literal.3, pos=[(10,11,12)])
  concat.7: bits[11] = concat(bit_slice.44, literal.4, pos=[(11,12,13)])
  concat.8: bits[11] = concat(bit_slice.44, literal.5, pos=[(12,13,14)])
  ret tuple.9: (bits[11], bits[11], bits[11]) = tuple(concat.6, concat.7, concat.8)
}
)"));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, p->GetTopAsFunction());

  EXPECT_THAT(Run(entry), IsOkAndHolds(true));

  EXPECT_EQ(entry->return_value()->operand(0)->op(), Op::kConcat);
  EXPECT_EQ(entry->return_value()->operand(0),
            entry->return_value()->operand(1));
  EXPECT_EQ(entry->return_value()->operand(0),
            entry->return_value()->operand(2));
}

TEST_F(CsePassTest, CommutativeOperands) {
  // Commutative operations can be equivalent irrespective of operand order.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue z = fb.Param("z", u32);
  BValue and_xyz = fb.AddNaryOp(Op::kAnd, {x, y, z});
  BValue and_xzy = fb.AddNaryOp(Op::kAnd, {x, z, y});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Add(and_xyz, and_xzy)));

  EXPECT_NE(f->return_value()->operand(0), f->return_value()->operand(1));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->return_value()->operand(0), f->return_value()->operand(1));

  EXPECT_THAT(f->return_value(), m::Add(m::And(), m::And()));
}

TEST_F(CsePassTest, NonCommutativeOperands) {
  // Operand order is significant for non-commutative operations.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue z = fb.Param("z", u32);
  BValue concat_xyz = fb.Concat({x, y, z});
  BValue concat_xzy = fb.Concat({x, z, y});
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Add(concat_xyz, concat_xzy)));

  EXPECT_NE(f->return_value()->operand(0), f->return_value()->operand(1));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_NE(f->return_value()->operand(0), f->return_value()->operand(1));
}

}  // namespace
}  // namespace xls
