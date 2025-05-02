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
#include "absl/log/check.h"
#include "absl/log/log.h"
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
  // The next IR has been generated from the following DSLX code:
  /*
    pub fn my_main_function (op: u32, i: u32, j: u32, k: u32, z: u32) -> u32{
      let v = match op {
        u32:0 => op_0(i, j),
        u32:1 => op_1(k, z),
        _ => fail!("unsupported_operation", zero!<u32>()),
      };
      v
    }
  */
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32]) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  literal.63: bits[32] = literal(value=1)
  literal.64: bits[32] = literal(value=0)
  or_reduce.65: bits[1] = or_reduce(bit_slice.62)
  eq.66: bits[1] = eq(op, literal.63)
  eq.67: bits[1] = eq(op, literal.64)
  t: bits[32] = umul(k, z)
  literal.69: bits[32] = literal(value=4294967295)
  after_all.39: token = after_all()
  not.80: bits[1] = not(or_reduce.65)
  concat.71: bits[2] = concat(eq.66, eq.67)
  umul.72: bits[32] = umul(i, j)
  add.73: bits[32] = add(t, literal.69)
  ret v: bits[32] = priority_sel(concat.71, cases=[umul.72, add.73], default=literal.64)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32]) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  literal.63: bits[32] = literal(value=1)
  literal.64: bits[32] = literal(value=0)
  or_reduce.65: bits[1] = or_reduce(bit_slice.62)
  eq.66: bits[1] = eq(op, literal.63)
  eq.67: bits[1] = eq(op, literal.64)
  t: bits[32] = smul(k, z)
  literal.69: bits[32] = literal(value=4294967295)
  after_all.39: token = after_all()
  not.80: bits[1] = not(or_reduce.65)
  concat.71: bits[2] = concat(eq.66, eq.67)
  smul.72: bits[32] = smul(i, j)
  add.73: bits[32] = add(t, literal.69)
  ret v: bits[32] = priority_sel(concat.71, cases=[smul.72, add.73], default=literal.64)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32], w: bits[32], p: bits[32], o: bits[32], y: bits[32]) -> bits[32] {
  t__2: bits[32] = umul(o, y)
  bit_slice.96: bits[30] = bit_slice(op, start=2, width=30)
  literal.97: bits[32] = literal(value=3)
  literal.98: bits[32] = literal(value=2)
  literal.99: bits[32] = literal(value=1)
  literal.100: bits[32] = literal(value=0)
  bit_slice.125: bits[31] = bit_slice(t__2, start=1, width=31)
  literal.130: bits[31] = literal(value=1)
  or_reduce.101: bits[1] = or_reduce(bit_slice.96)
  eq.102: bits[1] = eq(op, literal.97)
  eq.103: bits[1] = eq(op, literal.98)
  eq.104: bits[1] = eq(op, literal.99)
  eq.105: bits[1] = eq(op, literal.100)
  t: bits[32] = umul(k, z)
  literal.107: bits[32] = literal(value=4294967295)
  t__1: bits[32] = umul(w, p)
  add.127: bits[31] = add(bit_slice.125, literal.130)
  bit_slice.128: bits[1] = bit_slice(t__2, start=0, width=1)
  after_all.63: token = after_all()
  not.124: bits[1] = not(or_reduce.101)
  concat.113: bits[4] = concat(eq.102, eq.103, eq.104, eq.105)
  umul.114: bits[32] = umul(i, j)
  add.115: bits[32] = add(t, literal.107)
  add.116: bits[32] = add(t__1, literal.99)
  concat.129: bits[32] = concat(add.127, bit_slice.128)
  ret v: bits[32] = priority_sel(concat.113, cases=[umul.114, add.115, add.116, concat.129], default=literal.100)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32], w: bits[32], p: bits[32], o: bits[32], y: bits[32]) -> bits[32] {
  t__2: bits[32] = smul(o, y)
  bit_slice.96: bits[30] = bit_slice(op, start=2, width=30)
  literal.97: bits[32] = literal(value=3)
  literal.98: bits[32] = literal(value=2)
  literal.99: bits[32] = literal(value=1)
  literal.100: bits[32] = literal(value=0)
  bit_slice.125: bits[31] = bit_slice(t__2, start=1, width=31)
  literal.130: bits[31] = literal(value=1)
  or_reduce.101: bits[1] = or_reduce(bit_slice.96)
  eq.102: bits[1] = eq(op, literal.97)
  eq.103: bits[1] = eq(op, literal.98)
  eq.104: bits[1] = eq(op, literal.99)
  eq.105: bits[1] = eq(op, literal.100)
  t: bits[32] = smul(k, z)
  literal.107: bits[32] = literal(value=4294967295)
  t__1: bits[32] = smul(w, p)
  add.127: bits[31] = add(bit_slice.125, literal.130)
  bit_slice.128: bits[1] = bit_slice(t__2, start=0, width=1)
  after_all.63: token = after_all()
  not.124: bits[1] = not(or_reduce.101)
  concat.113: bits[4] = concat(eq.102, eq.103, eq.104, eq.105)
  smul.114: bits[32] = smul(i, j)
  add.115: bits[32] = add(t, literal.107)
  add.116: bits[32] = add(t__1, literal.99)
  concat.129: bits[32] = concat(add.127, bit_slice.128)
  ret v: bits[32] = priority_sel(concat.113, cases=[smul.114, add.115, add.116, concat.129], default=literal.100)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32]) -> bits[32] {
  literal.56: bits[32] = literal(value=1)
  literal.57: bits[32] = literal(value=0)
  eq.58: bits[1] = eq(op, literal.56)
  eq.59: bits[1] = eq(op, literal.57)
  umul.60: bits[32] = umul(k, z)
  literal.61: bits[32] = literal(value=4294967295)
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  concat.63: bits[2] = concat(eq.58, eq.59)
  v0: bits[32] = umul(i, j)
  v1: bits[32] = add(umul.60, literal.61)
  or_reduce.66: bits[1] = or_reduce(bit_slice.62)
  v: bits[32] = priority_sel(concat.63, cases=[v0, v1], default=literal.57)
  after_all.35: token = after_all()
  not.76: bits[1] = not(or_reduce.66)
  add.69: bits[32] = add(v, v0)
  ret add.71: bits[32] = add(add.69, v1)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding1) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32], w: bits[32], p: bits[32], o: bits[32], y: bits[32]) -> bits[32] {
  literal.66: bits[32] = literal(value=0)
  t: bits[32] = umul(o, y)
  eq.67: bits[1] = eq(op, literal.66)
  bit_slice.84: bits[31] = bit_slice(t, start=1, width=31)
  literal.89: bits[31] = literal(value=1)
  umul.69: bits[32] = umul(i, j)
  sign_ext.82: bits[32] = sign_ext(eq.67, new_bit_count=32)
  add.86: bits[31] = add(bit_slice.84, literal.89)
  bit_slice.87: bits[1] = bit_slice(t, start=0, width=1)
  after_all.46: token = after_all()
  v: bits[32] = and(umul.69, sign_ext.82)
  v2: bits[32] = concat(add.86, bit_slice.87)
  ret add.76: bits[32] = add(v, v2)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding2) {
  const std::string program = R"(
fn __sample__main(x0: bits[46] id=1) -> bits[46] {
  literal.141: bits[46] = literal(value=42)
  umul.142: bits[92] = umul(literal.141, x0)
  x1: bits[46] = umul(x0, x0)
  bit_slice.155: bits[1] = bit_slice(umul.142, start=91, width=1)
  x14: bits[2] = bit_slice(x1, start=41, width=2)
  x3: bits[46] = zero_ext(bit_slice.155, new_bit_count=46)
  x2: bits[46] = literal(value=0)
  ret x19: bits[46] = priority_sel(x14, cases=[x1, x3], default=x2)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding3) {
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32]) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  literal.63: bits[32] = literal(value=1)
  literal.64: bits[32] = literal(value=0)
  or_reduce.65: bits[1] = or_reduce(bit_slice.62)
  eq.66: bits[1] = eq(op, literal.63)
  eq.67: bits[1] = eq(op, literal.64)
  t: bits[32] = umul(k, z)
  literal.69: bits[32] = literal(value=4294967295)
  after_all.39: token = after_all()
  not.80: bits[1] = not(or_reduce.65)
  concat.71: bits[2] = concat(eq.66, eq.67)
  smul.72: bits[32] = smul(i, j)
  add.73: bits[32] = add(t, literal.69)
  ret v: bits[32] = priority_sel(concat.71, cases=[smul.72, add.73], default=literal.64)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

  // We expect the transformation to be not applicable
  EXPECT_THAT(Run(f), absl_testing::IsOkAndHolds(false));
}

TEST_F(ResourceSharingPassTest, NotPossibleFolding4) {
  const std::string program = R"(
    fn function_0(x: bits[2], y: bits[32]) -> bits[32] {
      bit_slice.3: bits[16] = bit_slice(y, start=16, width=16)
      literal.4: bits[16] = literal(value=4)
      literal.8: bits[32] = literal(value=8)
      bit_slice.5: bits[16] = bit_slice(y, start=0, width=16)
      umul.6: bits[16] = umul(bit_slice.3, literal.4)
      umul.7: bits[16] = umul(bit_slice.5, literal.8)
      concat.9: bits[32] = concat(umul.6, umul.7)
      umul.10: bits[32] = umul(y, literal.8)
      literal.11: bits[32] = literal(value=0)
      ret priority_sel.12: bits[32] = priority_sel(x, cases=[concat.9, umul.10], default=literal.11)
    })";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[32], z: bits[32]) -> bits[32] {
  literal.72: bits[32] = literal(value=0)
  literal.71: bits[32] = literal(value=1)
  eq.75: bits[1] = eq(op, literal.72)
  bit_slice.73: bits[31] = bit_slice(op, start=1, width=31)
  eq.74: bits[1] = eq(op, literal.71)
  t: bits[32] = umul(k, z)
  literal.77: bits[32] = literal(value=4294967295)
  not.92: bits[1] = not(eq.75)
  or_reduce.78: bits[1] = or_reduce(bit_slice.73)
  concat.79: bits[2] = concat(eq.74, eq.75)
  umul.80: bits[32] = umul(i, j)
  v1: bits[32] = add(t, literal.77)
  sign_ext.93: bits[32] = sign_ext(not.92, new_bit_count=32)
  after_all.44: token = after_all()
  not.91: bits[1] = not(or_reduce.78)
  v: bits[32] = priority_sel(concat.79, cases=[umul.80, v1], default=literal.72)
  v1__1: bits[32] = and(v1, sign_ext.93)
  ret add.86: bits[32] = add(v, v1__1)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[16], z: bits[16]) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  literal.63: bits[32] = literal(value=1)
  literal.64: bits[32] = literal(value=0)
  or_reduce.65: bits[1] = or_reduce(bit_slice.62)
  eq.66: bits[1] = eq(op, literal.63)
  eq.67: bits[1] = eq(op, literal.64)
  t: bits[16] = umul(k, z)
  literal.69: bits[16] = literal(value=42)
  after_all.39: token = after_all()
  not.80: bits[1] = not(or_reduce.65)
  concat.71: bits[2] = concat(eq.66, eq.67)
  umul.72: bits[32] = umul(i, j)
  add.73: bits[16] = add(t, literal.69)
  zero_ext.74: bits[32] = zero_ext(add.73, new_bit_count=32)
  ret v: bits[32] = priority_sel(concat.71, cases=[umul.72, zero_ext.74], default=literal.64)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
  const std::string program = R"(
fn __main_function__my_main_function(op: bits[32], i: bits[32], j: bits[32], k: bits[16], z: bits[16]) -> bits[32] {
  bit_slice.62: bits[31] = bit_slice(op, start=1, width=31)
  literal.63: bits[32] = literal(value=1)
  literal.64: bits[32] = literal(value=0)
  or_reduce.65: bits[1] = or_reduce(bit_slice.62)
  eq.66: bits[1] = eq(op, literal.63)
  eq.67: bits[1] = eq(op, literal.64)
  t: bits[16] = smul(k, z)
  literal.69: bits[16] = literal(value=42)
  after_all.39: token = after_all()
  not.80: bits[1] = not(or_reduce.65)
  concat.71: bits[2] = concat(eq.66, eq.67)
  smul.72: bits[32] = smul(i, j)
  add.73: bits[16] = add(t, literal.69)
  zero_ext.74: bits[32] = zero_ext(add.73, new_bit_count=32)
  ret v: bits[32] = priority_sel(concat.71, cases=[smul.72, zero_ext.74], default=literal.64)
})";

  // Create the function
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(program, p.get()));

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
