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
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
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
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace xls {

namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::z3::ScopedVerifyEquivalence;

class ResourceSharingPassTest : public IrTestBase {
 protected:
  ResourceSharingPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;

    // Enable resource sharing
    OptimizationPassOptions opts{};
    opts.enable_resource_sharing = true;
    opts.force_resource_sharing = true;

    // Run the select lifting pass.
    XLS_ASSIGN_OR_RETURN(bool changed, ResourceSharingPass().RunOnFunctionBase(
                                           f, opts, &results, context));

    // Return whether select lifting changed anything.
    return changed;
  }
};

uint64_t NumberOfNodes(Function* f, absl::Span<const Op> node_types) {
  uint64_t c = 0;
  for (Node* node : f->nodes()) {
    if (node->OpIn(node_types)) {
      c++;
    }
  }

  return c;
}

uint64_t NumberOfAdders(Function* f) {
  return NumberOfNodes(f, {Op::kAdd, Op::kSub});
}

uint64_t NumberOfSelects(Function* f) {
  return NumberOfNodes(f, {Op::kPrioritySel, Op::kOneHotSel, Op::kSel});
}

uint64_t NumberOfMultiplications(Function* f) {
  return NumberOfNodes(f, {Op::kSMul, Op::kUMul});
}

uint64_t NumberOfShifts(Function* f) {
  return NumberOfNodes(f, {Op::kShll, Op::kShra, Op::kShrl});
}

void InterpretAndCheck(Function* f, const std::vector<int32_t>& inputs,
                       const std::vector<uint32_t>& input_bitwidths,
                       int32_t expected_output,
                       int32_t expected_output_bitwidth) {
  // Translate the inputs to IR values
  std::vector<Value> IR_inputs;
  for (auto [input, input_bitwidth] : iter::zip(inputs, input_bitwidths)) {
    IR_inputs.push_back(Value(SBits(input, input_bitwidth)));
  }

  // Interpret the function with the specified inputs
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> r,
                           InterpretFunction(f, IR_inputs));

  // Check the output
  EXPECT_EQ(r.value, Value(UBits(expected_output, expected_output_bitwidth)));
}

void InterpretAndCheck(Function* f, const std::vector<int32_t>& inputs,
                       int32_t expected_output) {
  // Prepare the list of default bitwidths, one per input
  std::vector<uint32_t> bitwidths(inputs.size(), 32);

  // Run the IR and check the result
  InterpretAndCheck(f, inputs, bitwidths, expected_output, 32);
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeSingleUnsignedMultiplication2) {
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
  BValue o = fb.Param("o", u32_type);
  BValue l = fb.Param("l", u32_type);

  // Create the IR body
  BValue mulIJ = fb.UMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue mulOL = fb.UMul(o, l);
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));
  BValue add = fb.Add(mulKZ, kNeg1);
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue k2 = fb.Literal(UBits(2, 32));
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue isOp2 = fb.Eq(op, k2);
  BValue sub_selector = fb.Concat({isOp1, isOp0});
  BValue selector = fb.Concat({isOp2, sub_selector});
  BValue select = fb.PrioritySelect(selector, {mulIJ, add, mulOL}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3, 0, 0}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0, 0, 0}, 6);
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 5);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
}

TEST_F(ResourceSharingPassTest, MergeSingleSignedMultiplicationInDefaultCase) {
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
  BValue selector = fb.Eq(op, fb.Literal(UBits(0, 32)));
  BValue select = fb.PrioritySelect(selector, {mulIJ}, add);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
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
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {-2 /* bits = 10 */, 3}, {2, 32}, 24, 32);
  InterpretAndCheck(f,
                    {
                        1,      // bits = 01
                        131074  // "10" for the most significant 16 bits,
                                // and "10" for the least significant 16 bits
                    },
                    {2, 32},
                    524288     // 2 times 4 stored in the top 16 bits
                        + 16,  // 2 times 8
                    32);
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, 10);
}

TEST_F(ResourceSharingPassTest,
       MergeMultiplicationsNotPostDominatedBySelectUsingDefaultCase) {
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
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));

  // Step 1: results
  BValue mulIJ = fb.UMul(i, j);
  BValue mulKZ = fb.UMul(k, z);
  BValue add = fb.Add(mulKZ, kNeg1);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue select = fb.PrioritySelect(isOp0, {mulIJ}, add);

  // Step 3: post-select computation
  // BValue not0 = fb.Not(isOp0);
  // BValue not0Extended = fb.SignExtend(not0, 32);
  // BValue post_select_and = fb.And(add, not0Extended);
  // BValue post_select_add = fb.Add(select, post_select_and);
  BValue is0Extended = fb.SignExtend(isOp0, 32);
  BValue post_select_or = fb.Or(add, is0Extended);
  BValue post_select_add = fb.Add(select, post_select_or);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(post_select_add));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  // InterpretAndCheck(f, {0, 2, 3, 0, 0}, 6);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, 5);
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, 3}, {32, 32, 32, 16, 16}, 48, 32);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, {32, 32, 32, 16, 16}, 6, 32);
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
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_muls = NumberOfMultiplications(f);
  EXPECT_EQ(number_of_muls, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 2, -3}, {32, 32, 32, 16, 16}, 36, 32);
  InterpretAndCheck(f, {0, 2, 3, 0, 0}, {32, 32, 32, 16, 16}, 6, 32);
}

TEST_F(ResourceSharingPassTest, MergeAdds) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t uint_bitwidth = 32;
  Type* uint_type = p->GetBitsType(uint_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_type);
  BValue i = fb.Param("i", uint_type);
  BValue j = fb.Param("j", uint_type);
  BValue k = fb.Param("k", uint_type);
  BValue z = fb.Param("z", uint_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, uint_bitwidth));
  BValue k1 = fb.Literal(UBits(1, uint_bitwidth));
  BValue k2 = fb.Literal(UBits(2, uint_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Add(i, j);
  BValue addKZ = fb.Add(k, z);
  BValue mulKZ = fb.UMul(addKZ, k2);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, mulKZ}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {0, 3, 4, 2, 124},
                    {uint_bitwidth, uint_bitwidth, uint_bitwidth, uint_bitwidth,
                     uint_bitwidth},
                    7, uint_bitwidth);
  InterpretAndCheck(f, {1, 3, 4, 2, 124},
                    {uint_bitwidth, uint_bitwidth, uint_bitwidth, uint_bitwidth,
                     uint_bitwidth},
                    252, uint_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeAddsWithDifferentBitwidths) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t uint_small_bitwidth = 31;
  uint32_t uint_large_bitwidth = 32;
  Type* uint_small_type = p->GetBitsType(uint_small_bitwidth);
  Type* uint_large_type = p->GetBitsType(uint_large_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_large_type);
  BValue i = fb.Param("i", uint_large_type);
  BValue j = fb.Param("j", uint_large_type);
  BValue k = fb.Param("k", uint_small_type);
  BValue z = fb.Param("z", uint_small_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, uint_large_bitwidth));
  BValue k1 = fb.Literal(UBits(1, uint_large_bitwidth));
  BValue k2 = fb.Literal(UBits(2, uint_small_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Add(i, j);
  BValue addKZ = fb.Add(k, z);
  BValue mulKZ_30bits = fb.UMul(addKZ, k2, uint_small_bitwidth);
  BValue mulKZ = fb.ZeroExtend(mulKZ_30bits, uint_large_bitwidth);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, mulKZ}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(
      f, {0, 3, 4, 2, 124},
      {uint_large_bitwidth, uint_large_bitwidth, uint_large_bitwidth,
       uint_small_bitwidth, uint_small_bitwidth},
      7, uint_large_bitwidth);
  InterpretAndCheck(
      f, {1, 3, 4, 2, 124},
      {uint_large_bitwidth, uint_large_bitwidth, uint_large_bitwidth,
       uint_small_bitwidth, uint_small_bitwidth},
      252, uint_large_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeSubs) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t int_bitwidth = 36;
  Type* uint_type = p->GetBitsType(int_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_type);
  BValue i = fb.Param("i", uint_type);
  BValue j = fb.Param("j", uint_type);
  BValue k = fb.Param("k", uint_type);
  BValue z = fb.Param("z", uint_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, int_bitwidth));
  BValue k1 = fb.Literal(UBits(1, int_bitwidth));
  BValue k2 = fb.Literal(UBits(2, int_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Subtract(i, j);
  BValue addKZ = fb.Subtract(k, z);
  BValue mulKZ = fb.UMul(addKZ, k2);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, mulKZ}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one subtraction in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(
      f, {0, 5, 2, 2, 124},
      {int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth}, 3,
      int_bitwidth);
  InterpretAndCheck(
      f, {1, 4, 1, 124, 120},
      {int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth}, 8,
      int_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeSubs2) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t int_bitwidth = 36;
  Type* uint_type = p->GetBitsType(int_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_type);
  BValue i = fb.Param("i", uint_type);
  BValue j = fb.Param("j", uint_type);
  BValue k = fb.Param("k", uint_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, int_bitwidth));
  BValue k1 = fb.Literal(UBits(1, int_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Subtract(i, j);
  BValue addIK = fb.Subtract(i, k);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, addIK}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one subtraction in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the result function to have only two priority selects, one for
  // the selection of the operation, and one for the second input (the first
  // input is in common between the two subtractions so we don't need a select
  // for it).
  uint64_t number_of_selects = NumberOfSelects(f);
  EXPECT_EQ(number_of_selects, 2);

  // We expect the result function to include a single concat node and no
  // bitslice nodes.
  uint64_t number_of_concats = NumberOfNodes(f, {Op::kConcat});
  EXPECT_EQ(number_of_concats, 1);
  uint64_t number_of_bitslices = NumberOfNodes(f, {Op::kBitSlice});
  EXPECT_EQ(number_of_bitslices, 0);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(
      f, {0, 5, 2, 3},
      {int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth}, 3,
      int_bitwidth);
  InterpretAndCheck(
      f, {1, 5, 2, 3},
      {int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth, int_bitwidth}, 2,
      int_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeSubsWithDifferentBitwidths) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t uint_small_bitwidth = 33;
  uint32_t uint_large_bitwidth = 36;
  Type* uint_small_type = p->GetBitsType(uint_small_bitwidth);
  Type* uint_large_type = p->GetBitsType(uint_large_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_large_type);
  BValue i = fb.Param("i", uint_large_type);
  BValue j = fb.Param("j", uint_large_type);
  BValue k = fb.Param("k", uint_small_type);
  BValue z = fb.Param("z", uint_small_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, uint_large_bitwidth));
  BValue k1 = fb.Literal(UBits(1, uint_large_bitwidth));
  BValue k2 = fb.Literal(UBits(2, uint_small_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Subtract(i, j);
  BValue addKZ = fb.Subtract(k, z);
  BValue mulKZ_small_bitwidths = fb.UMul(addKZ, k2, uint_small_bitwidth);
  BValue mulKZ = fb.ZeroExtend(mulKZ_small_bitwidths, uint_large_bitwidth);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, mulKZ}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(
      f, {0, 5, 2, 2, 124},
      {uint_large_bitwidth, uint_large_bitwidth, uint_large_bitwidth,
       uint_small_bitwidth, uint_small_bitwidth},
      3, uint_large_bitwidth);
  InterpretAndCheck(
      f, {1, 4, 1, 124, 120},
      {uint_large_bitwidth, uint_large_bitwidth, uint_large_bitwidth,
       uint_small_bitwidth, uint_small_bitwidth},
      8, uint_large_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeAddsAndSubs) {
  // Create the function builder
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());

  // Fetch the types
  uint32_t uint_bitwidth = 33;
  Type* uint_type = p->GetBitsType(uint_bitwidth);

  // Create the parameters of the IR function
  BValue op = fb.Param("op", uint_type);
  BValue i = fb.Param("i", uint_type);
  BValue j = fb.Param("j", uint_type);
  BValue k = fb.Param("k", uint_type);
  BValue z = fb.Param("z", uint_type);

  // Create the IR body
  //
  // Step 0: constants
  BValue k0 = fb.Literal(UBits(0, uint_bitwidth));
  BValue k1 = fb.Literal(UBits(1, uint_bitwidth));
  BValue k2 = fb.Literal(UBits(2, uint_bitwidth));

  // Step 1: results
  BValue addIJ = fb.Add(i, j);
  BValue negZ = fb.Negate(z);
  BValue addKZ = fb.Subtract(k, negZ);
  BValue mulKZ = fb.UMul(addKZ, k2);

  // Step 2: select the result to return
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {addIJ, mulKZ}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_adders = NumberOfAdders(f);
  EXPECT_EQ(number_of_adders, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {0, 5, 2, 2, -124},
                    {uint_bitwidth, uint_bitwidth, uint_bitwidth, uint_bitwidth,
                     uint_bitwidth},
                    7, uint_bitwidth);
  InterpretAndCheck(f, {1, 4, 1, 124, -120},
                    {uint_bitwidth, uint_bitwidth, uint_bitwidth, uint_bitwidth,
                     uint_bitwidth},
                    8, uint_bitwidth);
}

TEST_F(ResourceSharingPassTest, MergeShift) {
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
  BValue shiftIJ = fb.Shll(i, j);
  BValue shiftKZ = fb.Shll(k, z);
  BValue kNeg1 = fb.Literal(UBits(4294967295, 32));
  BValue add = fb.Add(shiftKZ, kNeg1);
  BValue k0 = fb.Literal(UBits(0, 32));
  BValue k1 = fb.Literal(UBits(1, 32));
  BValue isOp0 = fb.Eq(op, k0);
  BValue isOp1 = fb.Eq(op, k1);
  BValue selector = fb.Concat({isOp1, isOp0});
  BValue select = fb.PrioritySelect(selector, {shiftIJ, add}, k0);

  // Create the function
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(select));

  // We expect the transformation successfully completed and it returned true
  ScopedVerifyEquivalence check_equivalent(f, absl::Seconds(100));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  // We expect the result function has only one multiplication in its body
  uint64_t number_of_shifts = NumberOfShifts(f);
  EXPECT_EQ(number_of_shifts, 1);

  // We expect the resource sharing optimization to have preserved the
  // inputs/outputs pairs we know to be valid.
  InterpretAndCheck(f, {1, 0, 0, 4, 1}, 7);
  InterpretAndCheck(f, {0, 8, 1, 0, 0}, 16);
}

void IrFuzzResourceSharing(FuzzPackageWithArgs fuzz_package_with_args) {
  ResourceSharingPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzResourceSharing)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace

}  // namespace xls
