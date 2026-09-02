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

#include "xls/passes/bitwise_simplification_pass.h"

#include <cstdint>

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

class BitwiseSimplificationPassTest : public IrTestBase {
 protected:
  BitwiseSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f, int64_t opt_level = 3) {
    PassResults results;
    OptimizationContext context;
    OptimizationPassOptions options;
    options.opt_level = opt_level;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         BitwiseSimplificationPass().RunOnFunctionBase(
                             f, options, &results, context));
    // Clean up dead nodes.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunctionBase(f, options, &results, context)
                            .status());
    return changed;
  }
};

TEST_F(BitwiseSimplificationPassTest, AndWithMaskSplitsEnabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0x00ff)
  ret and.1: bits[16] = and(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithMaskSplitsDisabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0x00ff)
  ret and.1: bits[16] = and(x, mask)
}
  )",
                                                       p.get()));
  // SplitsEnabled requires opt_level >= 3, and this mask doesn't fully
  // collapse.
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::And());
}

TEST_F(BitwiseSimplificationPassTest, OrWithMaskSplitsEnabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0xff00)
  ret or.1: bits[16] = or(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(Bits::AllOnes(8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, OrWithMaskSplitsDisabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0xff00)
  ret or.1: bits[16] = or(x, mask)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Or());
}

TEST_F(BitwiseSimplificationPassTest, AndWithMultipleRuns) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8]) -> bits[8] {
  mask: bits[8] = literal(value=0b1010_1100)
  ret and.1: bits[8] = and(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), /*start=*/7, /*width=*/1),
                        m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/5, /*width=*/1),
                        m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/2),
                        m::Literal(UBits(0, 2))));
}

TEST_F(BitwiseSimplificationPassTest, OrWithMultipleRuns) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8]) -> bits[8] {
  mask: bits[8] = literal(value=0b0101_0011)
  ret or.1: bits[8] = or(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::BitSlice(m::Param("x"), /*start=*/7, /*width=*/1),
                        m::Literal(Bits::AllOnes(1)),
                        m::BitSlice(m::Param("x"), /*start=*/5, /*width=*/1),
                        m::Literal(Bits::AllOnes(1)),
                        m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/2),
                        m::Literal(Bits::AllOnes(2))));
}

TEST_F(BitwiseSimplificationPassTest, AndWithMultipleNonLiterals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16], y: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0x00ff)
  ret and.1: bits[16] = and(x, y, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Literal(UBits(0, 8)),
                m::And(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8),
                       m::BitSlice(m::Param("y"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, OrWithMultipleNonLiterals) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16], y: bits[16]) -> bits[16] {
  mask: bits[16] = literal(value=0xff00)
  ret or.1: bits[16] = or(x, y, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Literal(Bits::AllOnes(8)),
                m::Or(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8),
                      m::BitSlice(m::Param("y"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, AndWithMultipleLiterals) {
  auto p = CreatePackage();
  // mask1 & mask2 = 0x0fff & 0x00ff = 0x00ff
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  mask1: bits[16] = literal(value=0x0fff)
  mask2: bits[16] = literal(value=0x00ff)
  ret and.1: bits[16] = and(x, mask1, mask2)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithNonLiteralConstantOperand) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[16]) -> bits[16] {
  hi: bits[8] = literal(value=0x00)
  lo: bits[8] = literal(value=0xff)
  mask: bits[16] = concat(hi, lo)
  ret and.1: bits[16] = and(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, FullCollapseToAbsorbingSplitsDisabled) {
  auto p = CreatePackage();
  // Neither operand is constant, but their bits cancel everywhere:
  // x has upper 8 bits 0, y has lower 8 bits 0.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[8], b: bits[8]) -> bits[16] {
  zero: bits[8] = literal(value=0)
  x: bits[16] = concat(zero, a)
  y: bits[16] = concat(b, zero)
  ret and.1: bits[16] = and(x, y)
}
  )",
                                                       p.get()));
  // Full collapse works even with opt_level = 2 (splits disabled).
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 16)));
}

TEST_F(BitwiseSimplificationPassTest, OrFullCollapseToAbsorbingSplitsDisabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[8], b: bits[8]) -> bits[16] {
  ones: bits[8] = literal(value=0xff)
  x: bits[16] = concat(ones, a)
  y: bits[16] = concat(b, ones)
  ret or.1: bits[16] = or(x, y)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(Bits::AllOnes(16)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithPartiallyKnownNonConstantOperand) {
  auto p = CreatePackage();
  // x has upper 8 bits 0, lower 8 bits dynamic. y is completely dynamic.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[8], y: bits[16]) -> bits[16] {
  zero: bits[8] = literal(value=0)
  x: bits[16] = concat(zero, a)
  ret and.1: bits[16] = and(x, y)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Literal(UBits(0, 8)),
                m::And(m::BitSlice(m::Name("x"), /*start=*/0, /*width=*/8),
                       m::BitSlice(m::Param("y"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, UniformMaskCollapses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8]) -> bits[8] {
  all_zeros: bits[8] = literal(value=0)
  ret and.1: bits[8] = and(x, all_zeros)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));

  auto p2 = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, ParseFunction(R"(
fn f(x: bits[8]) -> bits[8] {
  all_ones: bits[8] = literal(value=0xff)
  ret or.1: bits[8] = or(x, all_ones)
}
  )",
                                                        p2.get()));
  EXPECT_THAT(Run(f2, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f2->return_value(), m::Literal(Bits::AllOnes(8)));
}

TEST_F(BitwiseSimplificationPassTest, NoKnownBitsDoesNotChange) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8], y: bits[8]) -> bits[8] {
  ret and.1: bits[8] = and(x, y)
}
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(false));
}

TEST_F(BitwiseSimplificationPassTest, BitwiseMultiplexingTransformsSlices) {
  auto p = CreatePackage();
  // (a & 0x00ff) | (b & 0xff00)
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[16], b: bits[16]) -> bits[16] {
  mask_a: bits[16] = literal(value=0x00ff)
  mask_b: bits[16] = literal(value=0xff00)
  and_a: bits[16] = and(a, mask_a)
  and_b: bits[16] = and(b, mask_b)
  ret or.1: bits[16] = or(and_a, and_b)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Or(m::Concat(m::Literal(UBits(0, 8)),
                      m::BitSlice(m::Param("a"), /*start=*/0, /*width=*/8)),
            m::Concat(m::BitSlice(m::Param("b"), /*start=*/8, /*width=*/8),
                      m::Literal(UBits(0, 8)))));
}

TEST_F(BitwiseSimplificationPassTest,
       AndUnforcedSliceAllNonConstantOperandsIdentity) {
  auto p = CreatePackage();
  // x has: [23:16]=a, [15:8]=0,    [7:0]=0xff
  // y has: [23:16]=0, [15:8]=b,    [7:0]=0xff
  // [23:16] is forced to 0 (by y).
  // [15:8]  is forced to 0 (by x).
  // [7:0]   is unforced, but both x and y are known 0xff, so all non-constant
  //         operands drop out and the identity value (0xff) is emitted.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[8], b: bits[8]) -> bits[24] {
  zero: bits[8] = literal(value=0)
  ones: bits[8] = literal(value=0xff)
  x: bits[24] = concat(a, zero, ones)
  y: bits[24] = concat(zero, b, ones)
  ret and.1: bits[24] = and(x, y)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  // Full collapse to literal works even with splits disabled (opt_level = 2).
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0x0000ff, 24)));
}

TEST_F(BitwiseSimplificationPassTest,
       OrUnforcedSliceAllNonConstantOperandsIdentity) {
  auto p = CreatePackage();
  // x has: [23:16]=a,    [15:8]=0xff, [7:0]=0
  // y has: [23:16]=0xff, [15:8]=b,    [7:0]=0
  // [23:16] is forced to 1 (by y).
  // [15:8]  is forced to 1 (by x).
  // [7:0]   is unforced, but both x and y are known 0, so all non-constant
  //         operands drop out and the identity value (0) is emitted.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(a: bits[8], b: bits[8]) -> bits[24] {
  ones: bits[8] = literal(value=0xff)
  zero: bits[8] = literal(value=0)
  x: bits[24] = concat(a, ones, zero)
  y: bits[24] = concat(ones, b, zero)
  ret or.1: bits[24] = or(x, y)
}
  )",
                                                       p.get()));
  // Full collapse to literal works even with splits disabled (opt_level = 2).
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0xffff00, 24)));
}

TEST_F(BitwiseSimplificationPassTest, AlternatingBitMask) {
  auto p = CreatePackage();
  // x & 0x55: every other bit zeroed.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
fn f(x: bits[8]) -> bits[8] {
  mask: bits[8] = literal(value=0x55)
  ret and.1: bits[8] = and(x, mask)
}
  )",
                                                       p.get()));
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/6, /*width=*/1),
                        m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/4, /*width=*/1),
                        m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/1),
                        m::Literal(UBits(0, 1)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/1)));
}

}  // namespace
}  // namespace xls
