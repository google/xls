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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "cppitertools/zip.hpp"
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

uint64_t NumberOfMultiplications(Function *f) {
  uint64_t c = 0;
  for (Node *node : f->nodes()) {
    if (node->OpIn({Op::kSMul, Op::kUMul})) {
      c++;
    }
  }

  return c;
}

void InterpretAndCheck(Function *f, const std::vector<int32_t> &inputs,
                       const std::vector<uint32_t> &input_bitwidths,
                       int32_t expected_output) {
  // Translate the inputs to IR values
  std::vector<Value> IR_inputs;
  for (auto [input, input_bitwidth] : iter::zip(inputs, input_bitwidths)) {
    IR_inputs.push_back(Value(SBits(input, input_bitwidth)));
  }

  // Interpret the function with the specified inputs
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                           InterpretFunction(f, IR_inputs));

  // Check the output
  EXPECT_EQ(r.value, Value(UBits(expected_output, 32)));
}

void InterpretAndCheck(Function *f, const std::vector<int32_t> &inputs,
                       int32_t expected_output) {
  // Prepare the list of default bitwidths, one per input
  std::vector<uint32_t> bitwidths(inputs.size(), 32);

  // Run the IR and check the result
  InterpretAndCheck(f, inputs, bitwidths, expected_output);
}

TEST_F(ResourceSharingPassTest, MergeSingleUnsignedMultiplication) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);

  // Create the IR body
  BValue mulIJ = fb.UMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));
  BValue add = fb.Add(mulKZ, kNeg1);
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {mulIJ, add}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeSingleSignedMultiplication) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);

  // Create the IR body
  BValue mulIJ = fb.SMul(i, j);
  BValue mulKZ = fb.SMul(k, z);
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));
  BValue add = fb.Add(mulKZ, kNeg1);
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {mulIJ, add}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeMultipleUnsignedMultiplications) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);
  BValue w = fb.Param("w", u32_type);
  BValue p = fb.Param("p", u32_type);
  BValue o = fb.Param("o", u32_type);
  BValue y = fb.Param("y", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue k1_31bits = fb.Literal(UBits(1, 31));
  BValue k2 = fb.Literal(UBits(2, 32));
  BValue k3 = fb.Literal(UBits(3, 32));
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue result0 = fb.UMul(i, j);

  BValue mulKZ = fb.UMul(k, z);
  BValue result1 = fb.Add(mulKZ, kNeg1);

  BValue mulWP = fb.UMul(w, p);
  BValue result2 = fb.Add(mulWP, k1);

  BValue mulOY = fb.UMul(o, y);
  BValue mulOY_0_1 = fb.BitSlice(mulOY, 0, 1);
  BValue mulOY_1_32 = fb.BitSlice(mulOY, 1, 31);

  BValue add0 = fb.Add(mulOY_1_32, k1_31bits);
  BValue result3 = fb.Concat({add0, mulOY_0_1});

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue isOp2 = fb.Eq(op, k2);
  BValue isOp3 = fb.Eq(op, k3);
  BValue selector = fb.Concat({isOp3, isOp2, isOp1, isOp0});
  BValue select =
      fb.PrioritySelect(selector, {result0, result1, result2, result3}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {3, 0, 0, 0, 0, 0, 0, 2, 3}, 8);
  InterpretAndCheck(f, {2, 0, 0, 0, 0, 2, 3, 0, 0}, 7);
  InterpretAndCheck(f, {1, 0, 0, 2, 3, 0, 0, 0, 0}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0, 0, 0, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeMultipleSignedMultiplications) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);
  BValue w = fb.Param("w", u32_type);
  BValue p = fb.Param("p", u32_type);
  BValue o = fb.Param("o", u32_type);
  BValue y = fb.Param("y", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue k1_31bits = fb.Literal(UBits(1, 31));
  BValue k2 = fb.Literal(UBits(2, 32));
  BValue k3 = fb.Literal(UBits(3, 32));
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue result0 = fb.SMul(i, j);

  BValue mulKZ = fb.SMul(k, z);
  BValue result1 = fb.Add(mulKZ, kNeg1);

  BValue mulWP = fb.SMul(w, p);
  BValue result2 = fb.Add(mulWP, k1);

  BValue mulOY = fb.SMul(o, y);
  BValue mulOY_0_1 = fb.BitSlice(mulOY, 0, 1);
  BValue mulOY_1_32 = fb.BitSlice(mulOY, 1, 31);

  BValue add0 = fb.Add(mulOY_1_32, k1_31bits);
  BValue result3 = fb.Concat({add0, mulOY_0_1});

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue isOp2 = fb.Eq(op, k2);
  BValue isOp3 = fb.Eq(op, k3);
  BValue selector = fb.Concat({isOp3, isOp2, isOp1, isOp0});
  BValue select =
      fb.PrioritySelect(selector, {result0, result1, result2, result3}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {3, 0, 0, 0, 0, 0, 0, 2, 3}, 8);
  InterpretAndCheck(f, {2, 0, 0, 0, 0, 2, 3, 0, 0}, 7);
  InterpretAndCheck(f, {1, 0, 0, 2, 3, 0, 0, 0, 0}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0, 0, 0, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding0) {
  // The next IR has been generated from the following DSLX code:
  /*
  pub fn my_main_function (op: u32, i: u32, j: u32, k: u32, z: u32) -> u32{
    let v0 = i * j;
    let v1 = (k * z) - u32:1;
    let v = match op {
      u32:0 => v0,
      u32:1 => v1,
      _ => fail!("unsupported_operation", zero!<u32>()),
    };
    v + v0 + v1
  }
  */

  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue result0 = fb.UMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue result1 = fb.Add(mulKZ, kNeg1);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {result0, result1}, k0);

  // Step 3: post-select computation
  BValue post_select_add0 = fb.Add(select, result0);
  BValue post_select_add1 = fb.Add(post_select_add0, result1);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(post_select_add1));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding1) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue o = fb.Param("o", u32_type);
  BValue y = fb.Param("y", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1_31bits = fb.Literal(UBits(1, 31));

  // Step 1: results
  BValue mulIJ = fb.UMul(i, j);

  BValue mulOY = fb.UMul(o, y);
  BValue mulOY_0_1 = fb.BitSlice(mulOY, 0, 1);
  BValue mulOY_1_32 = fb.BitSlice(mulOY, 1, 31);

  BValue add0 = fb.Add(mulOY_1_32, k1_31bits);
  BValue concat = fb.Concat({add0, mulOY_0_1});

  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp0_32bits = fb.SignExtend(isOp0, 32);
  BValue and0 = fb.And(mulIJ, isOp0_32bits);

  BValue add1 = fb.Add(and0, concat);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add1));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding2) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u46_type = package->GetBitsType(46);

  // Create the parameters of the IR function
  BValue x0 = fb.Param("x0", u46_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 46));
  BValue k42 = fb.Literal(UBits(42, 46));

  // Step 1: results
  BValue mul_x0_k42 = fb.UMul(x0, k42, 92);
  BValue mul_x0_x0 = fb.UMul(x0, x0);
  BValue mul_x0_k42_91_1 = fb.BitSlice(mul_x0_k42, 91, 1);
  BValue mul_x0_x0_41_2 = fb.BitSlice(mul_x0_x0, 41, 2);
  BValue x3 = fb.ZeroExtend(mul_x0_k42_91_1, 46);

  // Step 2: select the result to return
  BValue select = fb.PrioritySelect(mul_x0_x0_41_2, {mul_x0_x0, x3}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding3) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue result0 = fb.SMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue result1 = fb.Add(mulKZ, kNeg1);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {result0, result1}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding4) {
  // Create the function builder
  auto package = CreatePackage();
  FunctionBuilder fb(TestName(), package.get());

  // Fetch the types
  Type* u2_type = package->GetBitsType(2);
  Type* u32_type = package->GetBitsType(32);

  // Create the parameters of the IR function
  BValue x = fb.Param("x", u2_type);
  BValue y = fb.Param("y", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k4 = fb.Literal(UBits(4, 16));
  BValue k8 = fb.Literal(UBits(8, 32));

  // Step 1: results
  BValue y_0_15 = fb.BitSlice(y, 0, 16);
  BValue y_16_31 = fb.BitSlice(y, 16, 16);
  BValue mul0 = fb.UMul(y_16_31, k4);
  BValue mul1 = fb.UMul(y_0_15, k8, 16);
  BValue muls = fb.Concat({mul0, mul1});
  BValue mul2 = fb.UMul(y, k8);

  // Step 2: select the result to return
  BValue select = fb.PrioritySelect(x, {muls, mul2}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation to be applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {-2 /* bits = 10 */, 3}, {2, 32}, 24);
  InterpretAndCheck(f,
                    {
                        1,      // bits = 01
                        131074  // "10" for the most significant 16 bits,
                                // and "10" for the least significant 16 bits
                    },
                    {2, 32},
                    524288    // 2 times 4 stored in the top 16 bits
                        + 16  // 2 times 8
  );
}

TEST_F(ResourceSharingPassTest,
       MergeMultiplicationsThatAreNotPostDominatedBySelect) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue z = fb.Param("z", u32_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue mulIJ = fb.UMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue add = fb.Add(mulKZ, kNeg1);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {mulIJ, add}, k0);

  // Step 3: post-select computation
  BValue not0 = fb.Not(isOp0);
  BValue not0Extended = fb.SignExtend(not0, 32);
  BValue post_select_and = fb.And(add, not0Extended);
  BValue post_select_add = fb.Add(select, post_select_and);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(post_select_add));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 10);
}

TEST_F(ResourceSharingPassTest,
       MergeSingleUnsignedMultiplicationDifferentBitwidths) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u16_type = p->GetBitsType(16);
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u16_type);
  BValue z = fb.Param("z", u16_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue k42 = fb.Literal(UBits(42, 16));

  // Step 1: results
  BValue mulIJ = fb.UMul(i, j, 32);
  BValue mulKZ = fb.UMul(k, z, 16);
  BValue add = fb.Add(mulKZ, k42);
  BValue addExtended = fb.ZeroExtend(add, 32);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {mulIJ, addExtended}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  // Unfortunately, ScopedVerifyEquivalence timed out and therefore we did not
  // rely on it.
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, {32, 32, 32, 16, 16}, 48);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, {32, 32, 32, 16, 16}, 6);
}

TEST_F(ResourceSharingPassTest,
       MergeSingleUnsignedMultiplicationDifferentBitwidths2) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  Type* u16_type = p->GetBitsType(16);
  Type* u32_type = p->GetBitsType(32);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", u32_type);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u16_type);
  BValue z = fb.Param("z", u16_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue k42 = fb.Literal(UBits(42, 16));

  // Step 1: results
  BValue mulIJ = fb.SMul(i, j, 32);
  BValue mulKZ = fb.SMul(k, z, 16);
  BValue add = fb.Add(mulKZ, k42);
  BValue addExtended = fb.ZeroExtend(add, 32);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {mulIJ, addExtended}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, -3}, {32, 32, 32, 16, 16}, 36);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, {32, 32, 32, 16, 16}, 6);
}

}  // namespace

}  // namespace xls
