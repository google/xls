// Copyright 2026 The XLS Authors
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

#include "xls/passes/bitwise_recombination_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::ScopedVerifyEquivalence;

class BitwiseRecombinationPassTest : public IrTestBase {
 protected:
  BitwiseRecombinationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    OptimizationContext context;
    OptimizationPassOptions options;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         BitwiseRecombinationPass().RunOnFunctionBase(
                             f, options, &results, context));
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunctionBase(f, options, &results, context)
                            .status());
    return changed;
  }
};

TEST_F(BitwiseRecombinationPassTest, SliceWithInvertedSlice) {
  // Pattern: {x[15:8], ~x[7:0]}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/8);
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/8);
  BValue not_s0 = fb.Not(s0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({s1, not_s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, SliceWithZeroes) {
  // Pattern: {8b'0, x[15:8]}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({zero, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, SliceWithOnes) {
  // Pattern: {8b'1, x[15:8]}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({ones, s1})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, InsertedZeroesAndOnes) {
  // Pattern: {4b'0, x[15:12], 4b'1, x[11:8], 4b'0}; coarse enough to be
  // preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s1 = fb.BitSlice(x, /*start=*/12, /*width=*/4);
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s0 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({zero, s1, ones, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, ZerosAndOnesInMiddle) {
  // Pattern: {x[19:16], 4b'0, x[15:12], 4b'1, x[3:0]}; coarse enough to be
  // preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue s2 = fb.BitSlice(x, /*start=*/16, /*width=*/4);
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({s2, zero, s1, ones, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, OnesAndInvertedSlices) {
  // Pattern: {4b'1, x[15:12], ~x[11:8], x[7:4], 4b'0}; coarse enough to be
  // preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s2 = fb.BitSlice(x, /*start=*/12, /*width=*/4);
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue not_s1 = fb.Not(s1);
  BValue s0 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  BValue zero = fb.Literal(UBits(0, 4));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({ones, s2, not_s1, s0, zero})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, OnesAndInvertedSlicesInMiddle) {
  // Pattern: {x[19:16], 4b'1, x[15:8], ~x[7:4], 4b'0, x[3:0]}; coarse enough to
  // be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue s3 = fb.BitSlice(x, /*start=*/16, /*width=*/4);
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s2 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue s1 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  BValue not_s1 = fb.Not(s1);
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({s3, ones, s2, not_s1, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, ZeroesAndInvertedSlices) {
  // Pattern: {4b'0, x[15:8], ~x[7:4], x[3:0]}; coarse enough to be
  // preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s2 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue s1 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  BValue not_s1 = fb.Not(s1);
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({zero, s2, not_s1, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, ZeroesAndInvertedSlicesInMiddle) {
  // Pattern: {x[19:16], 4b'0, x[15:8], ~x[7:4], x[3:0]}; coarse enough to be
  // preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(20));
  BValue s3 = fb.BitSlice(x, /*start=*/16, /*width=*/4);
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s2 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue s1 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  BValue not_s1 = fb.Not(s1);
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({s3, zero, s2, not_s1, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, ComplexMix) {
  // Pattern: {4b'1, x[15:12], 4b'0, x[11:8], 4b'1, x[7:4], 4b'0}.
  // Complex, but the masked option isn't actually any better.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s2 = fb.BitSlice(x, /*start=*/12, /*width=*/4);
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue s0 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({ones, s2, zero, s1, ones, s0, zero})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, PureInvertedRun) {
  // Pattern: {~x[15:8], ~x[7:0]}; can be merged to a single inverted run with
  // no concat.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/8);
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/8);
  BValue not_s1 = fb.Not(s1);
  BValue not_s0 = fb.Not(s0);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({not_s1, not_s0})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(BitwiseRecombinationPassTest, InvertedAndOnesWithoutRaw) {
  // Pattern: {8b'1, ~x[7:0]}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/8);
  BValue not_s0 = fb.Not(s0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({ones, not_s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, InvertedAndZerosWithoutRaw) {
  // Pattern: {8b'0, ~x[7:0]}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/8);
  BValue not_s0 = fb.Not(s0);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({zero, not_s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, SingleOperandConcat) {
  // Pattern: {x[15:0]}; trivial concat, we can remove the concat.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({x})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(BitwiseRecombinationPassTest, InvertedBitfieldDepositWithoutRaw) {
  // Pattern: {4b'1, ~x[7:4], 4b'0}; coarse enough to be preserved.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(12));
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s1 = fb.BitSlice(x, /*start=*/4, /*width=*/4);
  BValue not_s1 = fb.Not(s1);
  BValue zero = fb.Literal(UBits(0, 4));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({ones, not_s1, zero})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, AlternatingBitsWithZeroPadding) {
  // Pattern: {x[15], ~x[14], x[13], ~x[12], ..., ~x[8], 8'b0}.
  // Can be cleaned up by recognizing the 8-bit alternating negation pattern &
  // replacing with XOR.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue b15 = fb.BitSlice(x, /*start=*/15, /*width=*/1);
  BValue b14 = fb.BitSlice(x, /*start=*/14, /*width=*/1);
  BValue not_b14 = fb.Not(b14);
  BValue b13 = fb.BitSlice(x, /*start=*/13, /*width=*/1);
  BValue b12 = fb.BitSlice(x, /*start=*/12, /*width=*/1);
  BValue not_b12 = fb.Not(b12);
  BValue b11 = fb.BitSlice(x, /*start=*/11, /*width=*/1);
  BValue b10 = fb.BitSlice(x, /*start=*/10, /*width=*/1);
  BValue not_b10 = fb.Not(b10);
  BValue b9 = fb.BitSlice(x, /*start=*/9, /*width=*/1);
  BValue b8 = fb.BitSlice(x, /*start=*/8, /*width=*/1);
  BValue not_b8 = fb.Not(b8);
  BValue zero = fb.Literal(UBits(0, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat(
          {b15, not_b14, b13, not_b12, b11, not_b10, b9, not_b8, zero})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Xor(m::BitSlice(m::Param("x"), /*start=*/8, /*width=*/8),
                       m::Literal(UBits(0x55, 8))),
                m::Literal(UBits(0, 8))));
}

TEST_F(BitwiseRecombinationPassTest, CoarseSlicesPreservedOverChainedMask) {
  // Pattern: {x[31:16], ~x[15:8], 8'b0}.
  // Coarse enough that adding masking ends up making it worse.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue s2 = fb.BitSlice(x, /*start=*/16, /*width=*/16);
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/8);
  BValue not_s1 = fb.Not(s1);
  BValue zero = fb.Literal(UBits(0, 8));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({s2, not_s1, zero})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, StructPreservation) {
  // 5 coarse, non-nibble-aligned fields:
  // {x[63:51], 7'b0, x[43:29], 11'b0, x[17:0]}
  // An ugly monolithic 64-bit mask is rejected; all 5 coarse fields are kept.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(64));
  BValue s4 = fb.BitSlice(x, /*start=*/51, /*width=*/13);
  BValue zero1 = fb.Literal(UBits(0, 7));
  BValue s2 = fb.BitSlice(x, /*start=*/29, /*width=*/15);
  BValue zero2 = fb.Literal(UBits(0, 11));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/18);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({s4, zero1, s2, zero2, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(BitwiseRecombinationPassTest, AlternatingBitsWithOneBitZeroPadding) {
  // Pattern: {x[15], ~x[14], x[13], ~x[12], ..., ~x[2], x[1], 1'b0}
  // The best we can do is an alternating-negation mask on the top 15 bits:
  // {x[15:1] ^ 15'h2aaa, 1'b0}.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue b15 = fb.BitSlice(x, /*start=*/15, /*width=*/1);
  BValue b14 = fb.BitSlice(x, /*start=*/14, /*width=*/1);
  BValue not_b14 = fb.Not(b14);
  BValue b13 = fb.BitSlice(x, /*start=*/13, /*width=*/1);
  BValue b12 = fb.BitSlice(x, /*start=*/12, /*width=*/1);
  BValue not_b12 = fb.Not(b12);
  BValue b11 = fb.BitSlice(x, /*start=*/11, /*width=*/1);
  BValue b10 = fb.BitSlice(x, /*start=*/10, /*width=*/1);
  BValue not_b10 = fb.Not(b10);
  BValue b9 = fb.BitSlice(x, /*start=*/9, /*width=*/1);
  BValue b8 = fb.BitSlice(x, /*start=*/8, /*width=*/1);
  BValue not_b8 = fb.Not(b8);
  BValue b7 = fb.BitSlice(x, /*start=*/7, /*width=*/1);
  BValue b6 = fb.BitSlice(x, /*start=*/6, /*width=*/1);
  BValue not_b6 = fb.Not(b6);
  BValue b5 = fb.BitSlice(x, /*start=*/5, /*width=*/1);
  BValue b4 = fb.BitSlice(x, /*start=*/4, /*width=*/1);
  BValue not_b4 = fb.Not(b4);
  BValue b3 = fb.BitSlice(x, /*start=*/3, /*width=*/1);
  BValue b2 = fb.BitSlice(x, /*start=*/2, /*width=*/1);
  BValue not_b2 = fb.Not(b2);
  BValue b1 = fb.BitSlice(x, /*start=*/1, /*width=*/1);
  BValue zero = fb.Literal(UBits(0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat(
                        {b15, not_b14, b13, not_b12, b11, not_b10, b9, not_b8,
                         b7, not_b6, b5, not_b4, b3, not_b2, b1, zero})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Xor(m::BitSlice(m::Param("x"), /*start=*/1, /*width=*/15),
                       m::Literal(UBits(0x2aaa, 15))),
                m::Literal(UBits(0, 1))));
}

TEST_F(BitwiseRecombinationPassTest, AlternatingBitsCleanCutoff) {
  // Pattern: {~x[15], x[14], ~x[13], x[12], ..., x[2], ~x[1], 1'b0}
  // The best we can do is an alternating-negation mask on the top 15 bits:
  // {x[15:1] ^ 15'h5555, 1'b0}.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue b15 = fb.BitSlice(x, /*start=*/15, /*width=*/1);
  BValue not_b15 = fb.Not(b15);
  BValue b14 = fb.BitSlice(x, /*start=*/14, /*width=*/1);
  BValue b13 = fb.BitSlice(x, /*start=*/13, /*width=*/1);
  BValue not_b13 = fb.Not(b13);
  BValue b12 = fb.BitSlice(x, /*start=*/12, /*width=*/1);
  BValue b11 = fb.BitSlice(x, /*start=*/11, /*width=*/1);
  BValue not_b11 = fb.Not(b11);
  BValue b10 = fb.BitSlice(x, /*start=*/10, /*width=*/1);
  BValue b9 = fb.BitSlice(x, /*start=*/9, /*width=*/1);
  BValue not_b9 = fb.Not(b9);
  BValue b8 = fb.BitSlice(x, /*start=*/8, /*width=*/1);
  BValue b7 = fb.BitSlice(x, /*start=*/7, /*width=*/1);
  BValue not_b7 = fb.Not(b7);
  BValue b6 = fb.BitSlice(x, /*start=*/6, /*width=*/1);
  BValue b5 = fb.BitSlice(x, /*start=*/5, /*width=*/1);
  BValue not_b5 = fb.Not(b5);
  BValue b4 = fb.BitSlice(x, /*start=*/4, /*width=*/1);
  BValue b3 = fb.BitSlice(x, /*start=*/3, /*width=*/1);
  BValue not_b3 = fb.Not(b3);
  BValue b2 = fb.BitSlice(x, /*start=*/2, /*width=*/1);
  BValue b1 = fb.BitSlice(x, /*start=*/1, /*width=*/1);
  BValue not_b1 = fb.Not(b1);
  BValue zero = fb.Literal(UBits(0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat(
                        {not_b15, b14, not_b13, b12, not_b11, b10, not_b9, b8,
                         not_b7, b6, not_b5, b4, not_b3, b2, not_b1, zero})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Xor(m::BitSlice(m::Param("x"), /*start=*/1, /*width=*/15),
                       m::Literal(UBits(0x5555, 15))),
                m::Literal(UBits(0, 1))));
}

TEST_F(BitwiseRecombinationPassTest, MultipleBitSplinterInversion) {
  // Pattern: {~x[3], ~x[2], x[1], ~x[0]}.
  // Even with adjacent inverted slices ~x[3:2] fusing, a single XOR mask x ^
  // 4'hd works out better.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue b3 = fb.BitSlice(x, /*start=*/3, /*width=*/1);
  BValue not_b3 = fb.Not(b3);
  BValue b2 = fb.BitSlice(x, /*start=*/2, /*width=*/1);
  BValue not_b2 = fb.Not(b2);
  BValue b1 = fb.BitSlice(x, /*start=*/1, /*width=*/1);
  BValue b0 = fb.BitSlice(x, /*start=*/0, /*width=*/1);
  BValue not_b0 = fb.Not(b0);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Concat({not_b3, not_b2, b1, not_b0})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Xor(m::Param("x"), m::Literal(UBits(0xd, 4))));
}

TEST_F(BitwiseRecombinationPassTest, CoarseBitfieldDepositPreserved) {
  // Pattern: {4b'f, x[11:8], 4b'0, x[3:0]}.
  // Requires a 2-gate chained bitfield deposit: (x & 16'h0f0f) | 16'hf000.
  // With coarse nibble wiring (separate cost = 4), synthesizing the 2-gate
  // chained mask is rejected.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue ones = fb.Literal(UBits(0xf, 4));
  BValue s1 = fb.BitSlice(x, /*start=*/8, /*width=*/4);
  BValue zero = fb.Literal(UBits(0, 4));
  BValue s0 = fb.BitSlice(x, /*start=*/0, /*width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Concat({ones, s1, zero, s0})));

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
