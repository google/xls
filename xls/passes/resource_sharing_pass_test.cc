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
#include "xls/passes/resource_sharing_pass.h"

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

class ResourceSharingPassTest : public IrTestBase {
 protected:
  ResourceSharingPassTest() = default;

  absl::StatusOr<bool> Run(Function *f) {
    PassResults results;
    OptimizationContext context;

    // Enable resource sharing
    OptimizationPassOptions opts{};
    opts.enable_resource_sharing = true;

    // Run the select lifting pass.
    XLS_ASSIGN_OR_RETURN(bool changed, ResourceSharingPass().RunOnFunctionBase(
                                           f, opts, &results, context));

    // Return whether select lifting changed anything.
    return changed;
  }
};

TEST_F(ResourceSharingPassTest, MergeSingleUnsignedMultiplication) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32] id=34, i: bits[32] id=35, j: bits[32] id=36, k: bits[32] id=37, z: bits[32] id=38) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31, id=62)
  literal.63: bits[32] = literal(value=1, id=63)
  literal.64: bits[32] = literal(value=0, id=64, pos=[(0,16,4)])
  or_reduce.65: bits[1] = or_reduce(bit_slice.62, id=65)
  eq.66: bits[1] = eq(op, literal.63, id=66)
  eq.67: bits[1] = eq(op, literal.64, id=67)
  t: bits[32] = umul(k, z, id=68, pos=[(0,7,10), (0,17,17)])
  literal.69: bits[32] = literal(value=4294967295, id=69, pos=[(0,9,2), (0,17,17)])
  after_all.39: token = after_all(id=39)
  not.80: bits[1] = not(or_reduce.65, id=80)
  concat.71: bits[2] = concat(eq.66, eq.67, id=71)
  umul.72: bits[32] = umul(i, j, id=72, pos=[(0,3,2), (0,16,17)])
  add.73: bits[32] = add(t, literal.69, id=73, pos=[(0,9,2), (0,17,17)])
  ret v: bits[32] = priority_sel(concat.71, cases=[umul.72, add.73], default=literal.64, id=75)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint32_t number_of_umul = 0;
  for (Node *node : f->nodes()) {
    if (!node->Is<ArithOp>()) {
      continue;
    }
    Node *arith_node = node->As<ArithOp>();
    if (arith_node->op() == Op::kUMul) {
      number_of_umul++;
    }
  }
  EXPECT_EQ(number_of_umul, 1);
}

TEST_F(ResourceSharingPassTest, MergeSingleSignedMultiplication) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32] id=34, i: bits[32] id=35, j: bits[32] id=36, k: bits[32] id=37, z: bits[32] id=38) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31, id=62)
  literal.63: bits[32] = literal(value=1, id=63)
  literal.64: bits[32] = literal(value=0, id=64, pos=[(0,16,4)])
  or_reduce.65: bits[1] = or_reduce(bit_slice.62, id=65)
  eq.66: bits[1] = eq(op, literal.63, id=66)
  eq.67: bits[1] = eq(op, literal.64, id=67)
  t: bits[32] = smul(k, z, id=68, pos=[(0,7,10), (0,17,17)])
  literal.69: bits[32] = literal(value=4294967295, id=69, pos=[(0,9,2), (0,17,17)])
  after_all.39: token = after_all(id=39)
  not.80: bits[1] = not(or_reduce.65, id=80)
  concat.71: bits[2] = concat(eq.66, eq.67, id=71)
  smul.72: bits[32] = smul(i, j, id=72, pos=[(0,3,2), (0,16,17)])
  add.73: bits[32] = add(t, literal.69, id=73, pos=[(0,9,2), (0,17,17)])
  ret v: bits[32] = priority_sel(concat.71, cases=[smul.72, add.73], default=literal.64, id=75)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint32_t number_of_smul = 0;
  for (Node *node : f->nodes()) {
    if (!node->Is<ArithOp>()) {
      continue;
    }
    Node *arith_node = node->As<ArithOp>();
    if (arith_node->op() == Op::kSMul) {
      number_of_smul++;
    }
  }
  EXPECT_EQ(number_of_smul, 1);
}

}  // namespace

}  // namespace xls
