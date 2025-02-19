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

#include "xls/passes/constant_folding_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class ConstantFoldingPassTest : public IrTestBase {
 protected:
  ConstantFoldingPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         ConstantFoldingPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results, &context));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results, &context)
            .status());
    // Return whether constant folding changed anything.
    return changed;
  }
};

TEST_F(ConstantFoldingPassTest, SingleLiteralNoChange) {
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

TEST_F(ConstantFoldingPassTest, AddOfTwoLiterals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn IdenticalLiterals() -> bits[8] {
        literal.1: bits[8] = literal(value=42)
        literal.2: bits[8] = literal(value=123)
        ret add.3: bits[8] = add(literal.1, literal.2)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  ASSERT_TRUE(f->return_value()->Is<Literal>());
  EXPECT_EQ(f->return_value()->As<Literal>()->value(), Value(UBits(165, 8)));
}

TEST_F(ConstantFoldingPassTest, AddWithOnlyOneLiteral) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn IdenticalLiterals(x: bits[8]) -> bits[8] {
        literal.1: bits[8] = literal(value=42)
        ret add.2: bits[8] = add(literal.1, x)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 3);
}

TEST_F(ConstantFoldingPassTest, CountedFor) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package CountedFor

fn body(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y)
}

top fn main() -> bits[11] {
  literal.1: bits[11] = literal(value=0)
  ret counted_for.2: bits[11] = counted_for(literal.1, trip_count=7, stride=1, body=body)
}
)"));

  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, p->GetTopAsFunction());
  EXPECT_EQ(entry->node_count(), 2);
  EXPECT_THAT(Run(entry), IsOkAndHolds(true));
  EXPECT_EQ(entry->node_count(), 1);
  ASSERT_TRUE(entry->return_value()->Is<Literal>());
  EXPECT_EQ(entry->return_value()->As<Literal>()->value(),
            Value(UBits(21, 11)));
}

TEST_F(ConstantFoldingPassTest, EmptyTuple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn empty_tuple() -> () {
        ret result: () = tuple()
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Tuple());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(Value::Tuple({})));
}

TEST_F(ConstantFoldingPassTest, GateOpFalse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn gate_op() -> bits[32] {
        cond: bits[1] = literal(value=0)
        data: bits[32] = literal(value=42)
        ret result: bits[32] = gate(cond, data)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Gate(m::Literal(), m::Literal()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 32)));
}

TEST_F(ConstantFoldingPassTest, GateOpTrue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn gate_op() -> bits[32] {
        cond: bits[1] = literal(value=1)
        data: bits[32] = literal(value=42)
        ret result: bits[32] = gate(cond, data)
     }
  )",
                                                       p.get()));
  EXPECT_THAT(f->return_value(), m::Gate(m::Literal(), m::Literal()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(42, 32)));
}

}  // namespace
}  // namespace xls
