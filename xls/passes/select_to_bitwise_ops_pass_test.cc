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

#include "xls/passes/select_to_bitwise_ops_pass.h"

#include <optional>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class SelectToBitwiseOpsPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(FunctionBase* f,
                           const OptimizationPassOptions& options = {}) {
    PassResults results;
    OptimizationContext context;
    return SelectToBitwiseOpsPass().RunOnFunctionBase(f, options, &results,
                                                      context);
  }
};

TEST_F(SelectToBitwiseOpsPassTest, SelectWithConstantXorCaseCrcShift) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue feedback = fb.Param("feedback", p->GetBitsType(1));
  BValue crc = fb.Param("crc", p->GetBitsType(16));
  BValue shifted = fb.Shll(crc, fb.Literal(UBits(1, 16)));
  BValue poly_xor = fb.Xor(shifted, fb.Literal(UBits(0x1021, 16)));
  BValue result = fb.Select(feedback, {shifted, poly_xor});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // The masked-xor strength reduction:
  //   (crc << 1) ^ (sign_ext(feedback, 16) & 0x1021)
  EXPECT_THAT(f->return_value(), m::Xor(m::Shll(m::Param("crc"), m::Literal(1)),
                                        m::And(m::SignExt(m::Param("feedback")),
                                               m::Literal("bits[16]:0x1021"))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithConstantXorCaseReversed) {
  // Same strength reduction with the xor arm in the on-false slot.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue xored = fb.Xor(x, fb.Literal(UBits(0x5a, 8)));
  BValue result = fb.Select(selector, {xored, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // The xor arm is the on-false case here, so the mask condition is the
  // complemented selector:
  //   x ^ (sign_ext(~p, 8) & 0x5a)
  EXPECT_THAT(f->return_value(),
              m::Xor(m::Param("x"), m::And(m::SignExt(m::Not(m::Param("p"))),
                                           m::Literal("bits[8]:0x5a"))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithConstantXorCaseDefaulted) {
  // A select with one explicit case and a default value is still a binary mux
  // when the selector is a single bit; the same strength reduction applies.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue xored = fb.Xor(x, fb.Literal(UBits(0x5a, 8)));
  BValue result =
      fb.Select(selector, {x}, /*default_value=*/std::optional<BValue>(xored));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Xor(m::Param("x"), m::And(m::SignExt(m::Param("p")),
                                           m::Literal("bits[8]:0x5a"))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithNonConstantXorCaseUnchanged) {
  // If the xor term is not fully known, the rewrite does not apply.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue xored = fb.Xor(x, y);
  BValue result = fb.Select(selector, {x, xored});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithConstantXorCaseOneBit) {
  // For a 1-bit result the masked xor uses the (complemented) selector directly
  // as the mask, without sign-extension.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue xored = fb.Xor(x, fb.Literal(UBits(1, 1)));
  BValue result = fb.Select(selector, {xored, x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // The xor arm is the on-false case, so the mask is the complemented selector:
  //   x ^ (~p & 1)
  EXPECT_THAT(
      f->return_value(),
      m::Xor(m::Param("x"), m::And(m::Not(m::Param("p")), m::Literal(1))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithSplitXorCase) {
  // Regression test for a binary select whose xor arm is not a literal
  // `xor(x, c)` node, because `concat_simp` (or similar) already split the wide
  // xor limb into per-slice pieces:
  //
  //   sel(p, [concat(x_hi, 0), concat(xor(x_hi, c_hi), xor(0, c_lo))])
  //
  // i.e. the arm structure of an unrolled CRC/shift loop after concat
  // simplification. The masked-xor rewrite still applies because the arms
  // provably differ by the fully-known constant 0x1021 (the CRC polynomial).
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue slice = fb.BitSlice(x, /*start=*/0, /*width=*/15);
  BValue shifted = fb.Concat({slice, fb.Literal(UBits(0, 1))});
  BValue xored = fb.Concat(
      {fb.Xor(slice, fb.Literal(UBits(2064, 15))), fb.Literal(UBits(1, 1))});
  BValue result = fb.Select(selector, {shifted, xored});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Xor(m::Concat(m::BitSlice(m::Param("x"), /*start=*/0, /*width=*/15),
                       m::Literal(0)),
             m::And(m::SignExt(m::Param("p")), m::Literal("bits[16]:0x1021"))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithSplitXorCaseOnFalse) {
  // Same as SelectWithSplitXorCase but with the decomposed xor limb in the
  // on-false case; the masked-xor rewrite is orientation-invariant.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue slice = fb.BitSlice(x, /*start=*/0, /*width=*/15);
  BValue shifted = fb.Concat({slice, fb.Literal(UBits(0, 1))});
  BValue xored = fb.Concat(
      {fb.Xor(slice, fb.Literal(UBits(2064, 15))), fb.Literal(UBits(1, 1))});
  BValue result = fb.Select(selector, {xored, shifted});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  solvers::ScopedVerifyEquivalence stays_equivalent{f};
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  // `sel = on_false ^ (sign_ext(p, W) & delta)` with delta = on_false ^ on_true
  // = 0x1021; the on-false arm is itself the concat/xor limb.
  EXPECT_THAT(
      f->return_value(),
      m::Xor(m::Concat(m::Xor(m::BitSlice(m::Param("x"), /*start=*/0,
                                          /*width=*/15),
                              m::Literal("bits[15]:0x810")),
                       m::Literal(1)),
             m::And(m::SignExt(m::Param("p")), m::Literal("bits[16]:0x1021"))));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectWithUnresolvableSplitXorCase) {
  // If the corresponding pieces of the two arms do not provably differ by a
  // fully-known constant, the masked-xor rewrite is not applied.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(16));
  BValue arm_a = fb.Concat({fb.BitSlice(x, /*start=*/0, /*width=*/8),
                            fb.BitSlice(x, /*start=*/8, /*width=*/8)});
  BValue arm_b = fb.Concat({fb.BitSlice(y, /*start=*/0, /*width=*/8),
                            fb.BitSlice(y, /*start=*/8, /*width=*/8)});
  BValue result = fb.Select(selector, {arm_a, arm_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(SelectToBitwiseOpsPassTest, SelectNotConvertedBelowSplitsOptLevel) {
  // The rewrite is gated on SplitsEnabled; at lower optimization levels the
  // select is left alone so other value-based optimizations can still see it.
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue selector = fb.Param("p", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue xored = fb.Xor(x, fb.Literal(UBits(0x5a, 8)));
  BValue result = fb.Select(selector, {x, xored});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  EXPECT_THAT(Run(f, OptimizationPassOptions().WithOptLevel(2)),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
