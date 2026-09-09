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
#include "xls/ir/source_location.h"
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
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithMaskSplitsDisabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, mask)));

  // SplitsEnabled requires opt_level >= 3, and this mask doesn't fully
  // collapse.
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::And());
}

TEST_F(BitwiseSimplificationPassTest, OrWithMaskSplitsEnabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xff00, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(Bits::AllOnes(8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, OrWithMaskSplitsDisabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xff00, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(x, mask)));

  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Or());
}

TEST_F(BitwiseSimplificationPassTest, AndWithMultipleRuns) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue mask = fb.Literal(UBits(0b1010'1100, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, mask)));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue mask = fb.Literal(UBits(0b0101'0011, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(x, mask)));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And({x, y, mask})));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xff00, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or({x, y, mask})));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask1 = fb.Literal(UBits(0x0fff, 16));
  BValue mask2 = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And({x, mask1, mask2})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 8)),
                        m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithNonLiteralConstantOperand) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue hi = fb.Literal(UBits(0x00, 8));
  BValue lo = fb.Literal(UBits(0xff, 8));
  BValue mask = fb.Concat({hi, lo});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, mask)));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue x = fb.Concat({zero, a});
  BValue y = fb.Concat({b, zero});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.And(x, y)));

  // Full collapse works even with opt_level = 2 (splits disabled).
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 16)));
}

TEST_F(BitwiseSimplificationPassTest, OrFullCollapseToAbsorbingSplitsDisabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue x = fb.Concat({ones, a});
  BValue y = fb.Concat({b, ones});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Or(x, y)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(Bits::AllOnes(16)));
}

TEST_F(BitwiseSimplificationPassTest, AndWithPartiallyKnownNonConstantOperand) {
  auto p = CreatePackage();
  // x has upper 8 bits 0, lower 8 bits dynamic. y is completely dynamic.
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue x = fb.Concat({zero, a}, /*loc=*/SourceInfo(), /*name=*/"x");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.And(x, y)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Literal(UBits(0, 8)),
                m::And(m::BitSlice(m::Name("x"), /*start=*/0, /*width=*/8),
                       m::BitSlice(m::Param("y"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, UniformAndMaskCollapses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, zero)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_F(BitwiseSimplificationPassTest, UniformOrMaskCollapses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(x, ones)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(Bits::AllOnes(8)));
}

TEST_F(BitwiseSimplificationPassTest, NoKnownBitsDoesNotChange) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.And(x, y)));

  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(false));
}

TEST_F(BitwiseSimplificationPassTest, BitwiseMultiplexingTransformsSlices) {
  auto p = CreatePackage();
  // (a & 0x00ff) | (b & 0xff00)
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(16));
  BValue b = fb.Param("b", p->GetBitsType(16));
  BValue mask_a = fb.Literal(UBits(0x00ff, 16));
  BValue mask_b = fb.Literal(UBits(0xff00, 16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Or(fb.And(a, mask_a), fb.And(b, mask_b))));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue x = fb.Concat({a, zero, ones});
  BValue y = fb.Concat({zero, b, ones});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.And(x, y)));

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
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue x = fb.Concat({a, ones, zero});
  BValue y = fb.Concat({ones, b, zero});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Or(x, y)));

  // Full collapse to literal works even with splits disabled (opt_level = 2).
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0xffff00, 24)));
}

TEST_F(BitwiseSimplificationPassTest,
       AndUnforcedSliceAllNonConstantOperandsIdentitySplitsEnabled) {
  auto p = CreatePackage();
  // x has: [31:24]=c, [23:16]=a, [15:8]=0,    [7:0]=0xff
  // y has: [31:24]=d, [23:16]=0, [15:8]=b,    [7:0]=0xff
  // [31:24] is dynamic (c & d), so the expression does not fully collapse.
  // [23:16] is forced to 0 (by y).
  // [15:8]  is forced to 0 (by x).
  // [7:0]   is unforced, but both x and y are known 0xff, so all non-constant
  //         operands drop out and the identity value (0xff) is emitted.
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue x = fb.Concat({c, a, zero, ones}, /*loc=*/SourceInfo(), /*name=*/"x");
  BValue y = fb.Concat({d, zero, b, ones}, /*loc=*/SourceInfo(), /*name=*/"y");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.And(x, y)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::And(m::BitSlice(m::Name("x"), /*start=*/24, /*width=*/8),
                       m::BitSlice(m::Name("y"), /*start=*/24, /*width=*/8)),
                m::Literal(UBits(0, 16)), m::Literal(UBits(0xff, 8))));
}

TEST_F(BitwiseSimplificationPassTest,
       OrUnforcedSliceAllNonConstantOperandsIdentitySplitsEnabled) {
  auto p = CreatePackage();
  // x has: [31:24]=c, [23:16]=a,    [15:8]=0xff, [7:0]=0
  // y has: [31:24]=d, [23:16]=0xff, [15:8]=b,    [7:0]=0
  // [31:24] is dynamic (c | d), so the expression does not fully collapse.
  // [23:16] is forced to 1 (by y).
  // [15:8]  is forced to 1 (by x).
  // [7:0]   is unforced, but both x and y are known 0, so all non-constant
  //         operands drop out and the identity value (0) is emitted.
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(8));
  BValue b = fb.Param("b", p->GetBitsType(8));
  BValue c = fb.Param("c", p->GetBitsType(8));
  BValue d = fb.Param("d", p->GetBitsType(8));
  BValue zero = fb.Literal(UBits(0, 8));
  BValue ones = fb.Literal(UBits(0xff, 8));
  BValue x = fb.Concat({c, a, ones, zero}, /*loc=*/SourceInfo(), /*name=*/"x");
  BValue y = fb.Concat({d, ones, b, zero}, /*loc=*/SourceInfo(), /*name=*/"y");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Or(x, y)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Or(m::BitSlice(m::Name("x"), /*start=*/24, /*width=*/8),
                      m::BitSlice(m::Name("y"), /*start=*/24, /*width=*/8)),
                m::Literal(Bits::AllOnes(16)), m::Literal(UBits(0, 8))));
}

TEST_F(BitwiseSimplificationPassTest, AlternatingBitMask) {
  auto p = CreatePackage();
  // x & 0x55: every other bit zeroed.
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue mask = fb.Literal(UBits(0x55, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.And(x, mask)));

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

TEST_F(BitwiseSimplificationPassTest, XorWithMaskSplitsEnabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::BitSlice(m::Param("x"), /*start=*/8, /*width=*/8),
                m::Not(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, XorWithMaskSplitsDisabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0x00ff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  // SplitsEnabled requires opt_level >= 3, and this mask doesn't fully
  // collapse to Not.
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Xor());
}

TEST_F(BitwiseSimplificationPassTest, XorWithAllOnesSplitsDisabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xffff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  ScopedVerifyEquivalence sve(f);
  // Full collapse to Not works even with splits disabled (opt_level = 2).
  EXPECT_THAT(Run(f, /*opt_level=*/2), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(BitwiseSimplificationPassTest, XorWithAllOnesSplitsEnabled) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xffff, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Not(m::Param("x")));
}

TEST_F(BitwiseSimplificationPassTest, XorSignBitToggle) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue mask = fb.Literal(UBits(0x8000'0000, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::Not(m::BitSlice(m::Param("x"), /*start=*/31, /*width=*/1)),
                m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/31)));
}

TEST_F(BitwiseSimplificationPassTest, XorMultipleOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue mask = fb.Literal(UBits(0xff00, 16));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor({x, y, mask})));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(
          m::Not(m::Xor(m::BitSlice(m::Param("x"), /*start=*/8, /*width=*/8),
                        m::BitSlice(m::Param("y"), /*start=*/8, /*width=*/8))),
          m::Xor(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/8),
                 m::BitSlice(m::Param("y"), /*start=*/0, /*width=*/8))));
}

TEST_F(BitwiseSimplificationPassTest, XorAlternatingMask) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue mask = fb.Literal(UBits(0x55, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(x, mask)));

  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(Run(f, /*opt_level=*/3), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::BitSlice(m::Param("x"), /*start=*/7, /*width=*/1),
                m::Not(m::BitSlice(m::Param("x"), /*start=*/6, /*width=*/1)),
                m::BitSlice(m::Param("x"), /*start=*/5, /*width=*/1),
                m::Not(m::BitSlice(m::Param("x"), /*start=*/4, /*width=*/1)),
                m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1),
                m::Not(m::BitSlice(m::Param("x"), /*start=*/2, /*width=*/1)),
                m::BitSlice(m::Param("x"), /*start=*/1, /*width=*/1),
                m::Not(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/1))));
}

}  // namespace
}  // namespace xls
