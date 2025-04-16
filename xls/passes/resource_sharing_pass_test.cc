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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
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

void InterpretAndCheck(Function *f, const std::vector<uint32_t> &inputs,
                       uint32_t expected_output) {
  // Translate the inputs to IR values
  std::vector<Value> IR_inputs;
  for (uint32_t input : inputs) {
    IR_inputs.push_back(Value(UBits(input, 32)));
  }

  // Interpret the function with the specified inputs
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                           InterpretFunction(f, IR_inputs));

  // Check the output
  EXPECT_EQ(r.value, Value(UBits(expected_output, 32)));
}

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

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
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

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeMultipleUnsignedMultiplications) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32] id=54, i: bits[32] id=55, j: bits[32] id=56, k: bits[32] id=57, z: bits[32] id=58, w: bits[32] id=59, p: bits[32] id=60, o: bits[32] id=61, y: bits[32] id=62) -> bits[32] {
  t__2: bits[32] = umul(o, y, id=110, pos=[(0,19,10), (0,31,17)])
  bit_slice.96: bits[30] = bit_slice(op, start=2, width=30, id=96)
  literal.97: bits[32] = literal(value=3, id=97)
  literal.98: bits[32] = literal(value=2, id=98, pos=[(0,30,4)])
  literal.99: bits[32] = literal(value=1, id=99, pos=[(0,29,4)])
  literal.100: bits[32] = literal(value=0, id=100, pos=[(0,28,4)])
  bit_slice.125: bits[31] = bit_slice(t__2, start=1, width=31, id=125, pos=[(0,21,2), (0,31,17)])
  literal.130: bits[31] = literal(value=1, id=130, pos=[(0,21,2), (0,31,17)])
  or_reduce.101: bits[1] = or_reduce(bit_slice.96, id=101)
  eq.102: bits[1] = eq(op, literal.97, id=102)
  eq.103: bits[1] = eq(op, literal.98, id=103)
  eq.104: bits[1] = eq(op, literal.99, id=104)
  eq.105: bits[1] = eq(op, literal.100, id=105)
  t: bits[32] = umul(k, z, id=106, pos=[(0,7,10), (0,29,17)])
  literal.107: bits[32] = literal(value=4294967295, id=107, pos=[(0,9,2), (0,29,17)])
  t__1: bits[32] = umul(w, p, id=108, pos=[(0,13,10), (0,30,17)])
  add.127: bits[31] = add(bit_slice.125, literal.130, id=127, pos=[(0,21,2), (0,31,17)])
  bit_slice.128: bits[1] = bit_slice(t__2, start=0, width=1, id=128, pos=[(0,21,2), (0,31,17)])
  after_all.63: token = after_all(id=63)
  not.124: bits[1] = not(or_reduce.101, id=124)
  concat.113: bits[4] = concat(eq.102, eq.103, eq.104, eq.105, id=113)
  umul.114: bits[32] = umul(i, j, id=114, pos=[(0,3,2), (0,28,17)])
  add.115: bits[32] = add(t, literal.107, id=115, pos=[(0,9,2), (0,29,17)])
  add.116: bits[32] = add(t__1, literal.99, id=116, pos=[(0,15,2), (0,30,17)])
  concat.129: bits[32] = concat(add.127, bit_slice.128, id=129, pos=[(0,21,2), (0,31,17)])
  ret v: bits[32] = priority_sel(concat.113, cases=[umul.114, add.115, add.116, concat.129], default=literal.100, id=119)
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

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {3, 0, 0, 0, 0, 0, 0, 2, 3}, 8);
  InterpretAndCheck(f, {2, 0, 0, 0, 0, 2, 3, 0, 0}, 7);
  InterpretAndCheck(f, {1, 0, 0, 2, 3, 0, 0, 0, 0}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0, 0, 0, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeMultipleSignedMultiplications) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32] id=54, i: bits[32] id=55, j: bits[32] id=56, k: bits[32] id=57, z: bits[32] id=58, w: bits[32] id=59, p: bits[32] id=60, o: bits[32] id=61, y: bits[32] id=62) -> bits[32] {
  t__2: bits[32] = smul(o, y, id=110, pos=[(0,19,10), (0,31,17)])
  bit_slice.96: bits[30] = bit_slice(op, start=2, width=30, id=96)
  literal.97: bits[32] = literal(value=3, id=97)
  literal.98: bits[32] = literal(value=2, id=98, pos=[(0,30,4)])
  literal.99: bits[32] = literal(value=1, id=99, pos=[(0,29,4)])
  literal.100: bits[32] = literal(value=0, id=100, pos=[(0,28,4)])
  bit_slice.125: bits[31] = bit_slice(t__2, start=1, width=31, id=125, pos=[(0,21,2), (0,31,17)])
  literal.130: bits[31] = literal(value=1, id=130, pos=[(0,21,2), (0,31,17)])
  or_reduce.101: bits[1] = or_reduce(bit_slice.96, id=101)
  eq.102: bits[1] = eq(op, literal.97, id=102)
  eq.103: bits[1] = eq(op, literal.98, id=103)
  eq.104: bits[1] = eq(op, literal.99, id=104)
  eq.105: bits[1] = eq(op, literal.100, id=105)
  t: bits[32] = smul(k, z, id=106, pos=[(0,7,10), (0,29,17)])
  literal.107: bits[32] = literal(value=4294967295, id=107, pos=[(0,9,2), (0,29,17)])
  t__1: bits[32] = smul(w, p, id=108, pos=[(0,13,10), (0,30,17)])
  add.127: bits[31] = add(bit_slice.125, literal.130, id=127, pos=[(0,21,2), (0,31,17)])
  bit_slice.128: bits[1] = bit_slice(t__2, start=0, width=1, id=128, pos=[(0,21,2), (0,31,17)])
  after_all.63: token = after_all(id=63)
  not.124: bits[1] = not(or_reduce.101, id=124)
  concat.113: bits[4] = concat(eq.102, eq.103, eq.104, eq.105, id=113)
  smul.114: bits[32] = smul(i, j, id=114, pos=[(0,3,2), (0,28,17)])
  add.115: bits[32] = add(t, literal.107, id=115, pos=[(0,9,2), (0,29,17)])
  add.116: bits[32] = add(t__1, literal.99, id=116, pos=[(0,15,2), (0,30,17)])
  concat.129: bits[32] = concat(add.127, bit_slice.128, id=129, pos=[(0,21,2), (0,31,17)])
  ret v: bits[32] = priority_sel(concat.113, cases=[smul.114, add.115, add.116, concat.129], default=literal.100, id=119)
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

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {3, 0, 0, 0, 0, 0, 0, 2, 3}, 8);
  InterpretAndCheck(f, {2, 0, 0, 0, 0, 2, 3, 0, 0}, 7);
  InterpretAndCheck(f, {1, 0, 0, 2, 3, 0, 0, 0, 0}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0, 0, 0, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, NotPossibleMergeMultiplication) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32] id=37, i: bits[32] id=38, j: bits[32] id=39, k: bits[32] id=40, z: bits[32] id=41, w: bits[32] id=42, p: bits[32] id=43, o: bits[32] id=44, y: bits[32] id=45) -> bits[32] {
  literal.66: bits[32] = literal(value=0, id=66, pos=[(0,30,4)])
  t: bits[32] = umul(o, y, id=70, pos=[(0,19,10), (0,27,15)])
  eq.67: bits[1] = eq(op, literal.66, id=67)
  bit_slice.84: bits[31] = bit_slice(t, start=1, width=31, id=84, pos=[(0,21,2), (0,27,15)])
  literal.89: bits[31] = literal(value=1, id=89, pos=[(0,21,2), (0,27,15)])
  umul.69: bits[32] = umul(i, j, id=69, pos=[(0,3,2), (0,30,17)])
  sign_ext.82: bits[32] = sign_ext(eq.67, new_bit_count=32, id=82)
  add.86: bits[31] = add(bit_slice.84, literal.89, id=86, pos=[(0,21,2), (0,27,15)])
  bit_slice.87: bits[1] = bit_slice(t, start=0, width=1, id=87, pos=[(0,21,2), (0,27,15)])
  after_all.46: token = after_all(id=46)
  v: bits[32] = and(umul.69, sign_ext.82, id=83)
  v2: bits[32] = concat(add.86, bit_slice.87, id=88, pos=[(0,21,2), (0,27,15)])
  ret add.76: bits[32] = add(v, v2, id=76, pos=[(0,35,2)])
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

}  // namespace

}  // namespace xls
