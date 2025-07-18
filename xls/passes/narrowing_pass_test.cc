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

#include "xls/passes/narrowing_pass.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/scoped_vlog_level.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl::ScopedMockLog;
using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::z3::ScopedVerifyEquivalence;
using ::xls::solvers::z3::ScopedVerifyProcEquivalence;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::HasSubstr;

// The test is parameterized on whether to use range analysis or not.
class NarrowingPassTestBase : public IrTestBase {
 protected:
  NarrowingPassTestBase() = default;

  virtual NarrowingPass::AnalysisType analysis() const = 0;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationPassOptions options;
    options.convert_array_index_to_select = 2;
    OptimizationContext context;
    return NarrowingPass(analysis()).Run(p, options, &results, context);
  }
};

class NarrowingPassTest
    : public NarrowingPassTestBase,
      public ::testing::WithParamInterface<NarrowingPass::AnalysisType> {
 protected:
  NarrowingPass::AnalysisType analysis() const override { return GetParam(); }
};

class ContextNarrowingPassTest : public NarrowingPassTestBase {
 protected:
  NarrowingPass::AnalysisType analysis() const override {
    return NarrowingPass::AnalysisType::kRangeWithContext;
  }
};

// This was found by fuzzing. The range-query-engine deep inside of context-qe
// had a bug where it incorrectly check-failed instead of returning unknown.
TEST_P(NarrowingPassTest, RequestsUntrackedNode) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto param = fb.Param("param", p->GetBitsType(1));
  fb.Add(fb.UMod(param, param), param);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(
      Run(p.get()),
      IsOkAndHolds(analysis() != NarrowingPass::AnalysisType::kTernary));
}

TEST_P(NarrowingPassTest, NarrowSub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(4));
  auto y = fb.Param("y", p->GetBitsType(4));
  // NB The ternary qe only passes because stateless can see the extends.
  auto x_wide = fb.ZeroExtend(x, 32);
  // y_wide is always larger than x
  auto y_wide = fb.ZeroExtend(fb.Concat({fb.Literal(UBits(1, 1)), y}), 32);
  fb.Subtract(y_wide, x_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(AllOf(m::Sub(_, _), m::Type("bits[5]"))));
}

// This is just an edge case where the operands to the subtract are exactly '0'
// or '-1'.
TEST_P(NarrowingPassTest, NarrowSubAllSignBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  auto y = fb.Param("y", p->GetBitsType(1));
  // NB The ternary qe only passes because stateless can see the extends.
  auto x_wide = fb.SignExtend(x, 32);
  auto y_wide = fb.SignExtend(y, 32);
  fb.Subtract(y_wide, x_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Sub(_, _), m::Type("bits[2]"))));
}

// This is just an edge case where the operands to the subtract are exactly '0'
// or '-1'.
TEST_P(NarrowingPassTest, NarrowAddAllSignBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  auto y = fb.Param("y", p->GetBitsType(1));
  // NB The ternary qe only passes because stateless can see the extends.
  auto x_wide = fb.SignExtend(x, 32);
  auto y_wide = fb.SignExtend(y, 32);
  fb.Add(y_wide, x_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Add(_, _), m::Type("bits[2]"))));
}

TEST_P(NarrowingPassTest, NarrowSubKnownNegativeKeepsSignBits) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  auto x_wide = fb.ZeroExtend(fb.Concat({fb.Literal(UBits(1, 1)), x}), 32);
  auto y = fb.Literal(UBits(0, 32));
  fb.Subtract(y, x_wide);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Sub(_, _), m::Type("bits[3]"))));
}

TEST_P(NarrowingPassTest, AddDoNotNarrowToSameBitWidth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto x_wide = fb.SignExtend(x, 33);
  auto y = fb.Literal(SBits(-1, 33));
  fb.Add(x_wide, y);
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, NarrowableSubNegative) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(4));
  auto y = fb.Param("y", p->GetBitsType(4));
  // NB The ternary qe only passes because stateless can see the extends.
  // x_wide is [0, 15]
  auto x_wide = fb.ZeroExtend(x, 32);
  // y_wide is always larger than x (range [16, 31])
  auto y_wide = fb.ZeroExtend(fb.Concat({fb.Literal(UBits(1, 1)), y}), 32);
  fb.Subtract(x_wide, y_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Sub(_, _), m::Type("bits[6]"))));
}

TEST_P(NarrowingPassTest, NarrowableSubNegativeGeneric) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(4));
  auto y = fb.Param("y", p->GetBitsType(4));
  // NB The ternary qe only passes because stateless can see the extends.
  auto x_wide = fb.ZeroExtend(x, 32);
  auto y_wide = fb.ZeroExtend(y, 32);
  fb.Subtract(x_wide, y_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Sub(_, _), m::Type("bits[5]"))));
}

TEST_P(NarrowingPassTest, NarrowableSubPositive) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(4));
  auto y = fb.Param("y", p->GetBitsType(4));
  auto x_wide = fb.ZeroExtend(x, 32);
  // NB The range analysis is able to see much better
  // y_wide is always larger than x
  auto y_narrow = fb.Concat({fb.Literal(UBits(1, 1)), y});
  auto y_wide = fb.ZeroExtend(y_narrow, 32);
  fb.Subtract(y_wide, x_wide);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ZeroExt(m::Sub(y_narrow.node(),
                        m::BitSlice(x_wide.node(), /*start=*/0, /*width=*/5))));
}

TEST_P(NarrowingPassTest, NarrowableNegOneBit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // -1 * -b0000000X
  auto param = fb.Param("x", p->GetBitsType(1));
  auto ext = fb.ZeroExtend(param, 32);
  fb.Negate(ext);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(m::BitSlice(ext.node(), /*start=*/0, /*width=*/1)));
}

TEST_P(NarrowingPassTest, NarrowableNegThreeBit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // -1 * -b00000XXX
  auto param = fb.Param("x", p->GetBitsType(3));
  auto ext = fb.ZeroExtend(param, 32);
  fb.Negate(ext);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SignExt(m::Neg(m::BitSlice(ext.node(), /*start=*/0, /*width=*/4))));
}

TEST_P(NarrowingPassTest, UnnarrowableShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shra(fb.Param("in", p->GetBitsType(32)),
          fb.Param("amt", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Shra(m::Param("in"), m::Param("amt")));
}

TEST_P(NarrowingPassTest, NarrowableShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(
      fb.Param("in", p->GetBitsType(32)),
      fb.ZeroExtend(fb.Param("amt", p->GetBitsType(3)), /*new_bit_count=*/123));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shll(m::Param("in"), m::BitSlice(/*start=*/0, /*width=*/3)));
}

TEST_P(NarrowingPassTest, ShiftWithKnownZeroShiftAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)), fb.Literal(UBits(0, 27)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("in"));
}

TEST_P(NarrowingPassTest, ShiftWithKnownOnePrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)),
          fb.Concat({fb.Literal(UBits(0b111, 3)),
                     fb.Param("amt", p->GetBitsType(2))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false)) << f->DumpIr();
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("in"), m::Concat()));
}

TEST_P(NarrowingPassTest, ShiftWithKnownZeroPrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)),
          fb.Concat({fb.Literal(UBits(0b000, 3)),
                     fb.Param("amt", p->GetBitsType(2))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shll(m::Param("in"), m::BitSlice(/*start=*/0, /*width=*/2)));
}

TEST_P(NarrowingPassTest, ShraNarrowValueAndAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue shift_val = fb.Param("val", p->GetBitsType(3));
  BValue shift_amnt = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit signed value
  BValue shift_val_big =
      fb.Add(fb.SignExtend(shift_val, 256),
             fb.SignExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits.
  BValue shift_amnt_big =
      fb.And(fb.SignExtend(shift_amnt, 256), fb.Literal(UBits(0b1111, 256)));
  fb.Shra(shift_val_big, shift_amnt_big);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Shra(m::Type("bits[6]"), m::Type("bits[4]")),
                               m::Type("bits[6]"))));
}

TEST_P(NarrowingPassTest, ShrlNarrowValueAndAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue shift_val = fb.Param("val", p->GetBitsType(3));
  BValue shift_amnt = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit unsigned value
  BValue shift_val_big =
      fb.Add(fb.ZeroExtend(shift_val, 256),
             fb.ZeroExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits.
  BValue shift_amnt_big =
      fb.And(fb.SignExtend(shift_amnt, 256), fb.Literal(UBits(0b1111, 256)));
  fb.Shrl(shift_val_big, shift_amnt_big);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(AllOf(m::Shrl(m::Type("bits[6]"), m::Type("bits[4]")),
                               m::Type("bits[6]"))));
}

TEST_P(NarrowingPassTest, ShllKnownLeadingNarrowValueAndAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue shift_val = fb.Param("val", p->GetBitsType(3));
  BValue shift_amnt = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit unsigned value
  BValue shift_val_big =
      fb.Add(fb.ZeroExtend(shift_val, 256),
             fb.ZeroExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits max of 15.
  BValue shift_amnt_big =
      fb.And(fb.SignExtend(shift_amnt, 256), fb.Literal(UBits(0b1111, 256)));
  fb.Shll(shift_val_big, shift_amnt_big);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(),
                        AllOf(m::Shll(m::Type("bits[21]"), m::Type("bits[4]")),
                              m::Type("bits[21]"))));
}

TEST_P(NarrowingPassTest, ShllUnknownLeadingNarrowValueAndAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue shift_val = fb.Param("val", p->GetBitsType(3));
  BValue shift_amnt = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit signed value
  BValue shift_val_big =
      fb.Add(fb.SignExtend(shift_val, 256),
             fb.SignExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits max of 15.
  BValue shift_amnt_big =
      fb.And(fb.SignExtend(shift_amnt, 256), fb.Literal(UBits(0b1111, 256)));
  fb.Shll(shift_val_big, shift_amnt_big);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Shll(m::Type("bits[21]"), m::Type("bits[4]")),
                               m::Type("bits[21]"))));
}

TEST_P(NarrowingPassTest, DynamicBitSliceNarrowerThanResult) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue slice_val = fb.Param("val", p->GetBitsType(3));
  BValue slice_start = fb.Literal(UBits(1, 12));
  fb.DynamicBitSlice(slice_val, slice_start, /*width=*/3);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::BitSlice(m::Type("bits[3]"), 1, 2)));
}

TEST_P(NarrowingPassTest, DynamicBitSliceSmallStart) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue slice_val = fb.Literal(UBits(6, 4));
  BValue slice_start = fb.Param("amnt", p->GetBitsType(1));
  fb.DynamicBitSlice(slice_val, slice_start, /*width=*/3);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::DynamicBitSlice(m::Type("bits[3]"), m::Param("amnt")));
}

TEST_P(NarrowingPassTest, DynamicBitSliceStartDomain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue slice_val = fb.Param("val", p->GetBitsType(256));
  // Max 7
  BValue slice_start = fb.Param("amnt", p->GetBitsType(3));
  fb.DynamicBitSlice(slice_val, slice_start, /*width=*/2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::DynamicBitSlice(m::Type("bits[9]"), m::Param("amnt")));
}

TEST_P(NarrowingPassTest, DynamicBitSliceLeadingZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue slice_val = fb.Param("val", p->GetBitsType(3));
  BValue slice_start = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit unsigned value
  BValue slice_val_big =
      fb.Add(fb.ZeroExtend(slice_val, 256),
             fb.ZeroExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits max of 15.
  BValue slice_start_big =
      fb.And(fb.SignExtend(slice_start, 256), fb.Literal(UBits(0b1111, 256)));
  // Only actually 5 bits are even possibly interesting, rest are 0s
  fb.DynamicBitSlice(slice_val_big, slice_start_big, /*width=*/2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::DynamicBitSlice(m::Type("bits[6]"), m::Type("bits[4]")));
}

TEST_P(NarrowingPassTest, DynamicBitSliceLeadingZerosTooBig) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue slice_val = fb.Param("val", p->GetBitsType(3));
  BValue slice_start = fb.Param("amnt", p->GetBitsType(3));
  // Max 6-bit unsigned value
  BValue slice_val_big =
      fb.Add(fb.ZeroExtend(slice_val, 256),
             fb.ZeroExtend(fb.Param("add_val", p->GetBitsType(5)), 256));
  // 4 bits max of 15.
  BValue slice_start_big =
      fb.And(fb.SignExtend(slice_start, 256), fb.Literal(UBits(0b1111, 256)));
  // Only actually 5 bits are even possibly interesting, rest are 0s
  fb.DynamicBitSlice(slice_val_big, slice_start_big, /*width=*/8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ZeroExt(m::DynamicBitSlice(
                                     m::Type("bits[6]"), m::Type("bits[4]"))));
}

TEST_P(NarrowingPassTest, DecodeWithKnownZeroIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Literal(UBits(0, 13)), /*width=*/27);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(/*value=*/1, /*width=*/27));
}

TEST_P(NarrowingPassTest, DecodeWithKnownOnePrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat(
      {fb.Literal(UBits(0b111, 3)), fb.Param("amt", p->GetBitsType(2))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false)) << f->DumpIr();
  EXPECT_THAT(f->return_value(), m::Decode(m::Concat()));
}

TEST_P(NarrowingPassTest, DecodeWithKnownZeroPrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Decode(fb.Concat(
      {fb.Literal(UBits(0b000, 3)), fb.Param("amt", p->GetBitsType(2))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::Decode(m::BitSlice(/*start=*/0, /*width=*/2))));
}

TEST_P(NarrowingPassTest, NarrowableArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                {fb.ZeroExtend(fb.Param("idx", p->GetBitsType(8)),
                               /*new_bit_count=*/123)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param("a"), /*indices=*/{
                                m::BitSlice(/*start=*/0, /*width=*/8)}));
}

TEST_P(NarrowingPassTest, LiteralArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(4, p->GetBitsType(32)));
  fb.ArrayIndex(a, {fb.Literal(Value(UBits(0, 32)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(UBits(0, 2))}));
}

TEST_P(NarrowingPassTest, LiteralArrayIndex3d) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* array_type = p->GetArrayType(
      42, p->GetArrayType(4, p->GetArrayType(3, p->GetBitsType(32))));
  BValue a = fb.Param("a", array_type);
  fb.ArrayIndex(
      a, {fb.Literal(Value(UBits(0, 32))), fb.Literal(Value(UBits(5, 16))),
          fb.Literal(Value(UBits(1, 64)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Param("a"), /*indices=*/{
                                                   m::Literal(UBits(0, 6)),
                                                   m::Literal(UBits(4, 3)),
                                                   m::Literal(UBits(1, 2))}));
}

TEST_P(NarrowingPassTest, LiteralArrayIndexOddNumberOfElements) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(5, p->GetBitsType(32)));
  fb.ArrayIndex(a, {fb.Literal(Value(UBits(0, 32)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(UBits(0, 3))}));
}

TEST_P(NarrowingPassTest, NonzeroLiteralArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                {fb.Literal(Value(UBits(0x0f, 8)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(UBits(0x0f, 6))}));
}

TEST_P(NarrowingPassTest, OutofBoundsLiteralArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                {fb.Literal(Value(UBits(123, 64)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(UBits(42, 6))}));
}

TEST_P(NarrowingPassTest, NarrowableArrayIndexAllZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                {fb.And(fb.Param("idx", p->GetBitsType(8)),
                        fb.Literal(Value(UBits(0, 8))))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(UBits(0, 6))}));
}

TEST_P(NarrowingPassTest, MultiplyWiderThanSumOfOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  fb.SMul(fb.Param("lhs", u8), fb.Param("rhs", u8), /*result_width=*/42);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      AllOf(m::Type("bits[42]"),
            m::SignExt(AllOf(m::Type("bits[16]"),
                             m::SMul(m::Param("lhs"), m::Param("rhs"))))));
}

TEST_P(NarrowingPassTest, MultiplyOperandsWiderThanResult) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u42 = p->GetBitsType(42);
  fb.UMul(fb.Param("lhs", u17), fb.Param("rhs", u42), /*result_width=*/9);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UMul(AllOf(m::Type("bits[9]"),
                    m::BitSlice(m::Param("lhs"), /*start=*/0, /*width=*/9)),
              AllOf(m::Type("bits[9]"),
                    m::BitSlice(m::Param("rhs"), /*start=*/0, /*width=*/9))));
}

TEST_P(NarrowingPassTest, ExtendedUMulOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  fb.UMul(fb.ZeroExtend(fb.Param("lhs", u17), 32),
          fb.SignExtend(fb.Param("rhs", u17), 54),
          /*result_width=*/62);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Only the zero-extend should have been elided.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::Param("lhs"), m::SignExt(m::Param("rhs"))));
}

TEST_P(NarrowingPassTest, ExtendedUMulSignAgnosticToUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  fb.UMul(fb.ZeroExtend(fb.Param("lhs", u17), 21),
          fb.SignExtend(fb.Param("rhs", u20), 21),
          /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::Param("lhs"), m::SignExt(m::Param("rhs"))));
}

TEST_P(NarrowingPassTest, ExtendedUMulSignAgnosticToSMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  fb.UMul(fb.ZeroExtend(fb.Param("lhs", u20), 21),
          fb.SignExtend(fb.Param("rhs", u17), 21),
          /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The sign-extended operand should be sliced down to the sign bit, and the
  // product switched to an SMul.
  EXPECT_THAT(f->return_value(),
              m::SMul(m::ZeroExt(m::Param("lhs")), m::Param("rhs")));
}

TEST_P(NarrowingPassTest, ExtendedSMulOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  fb.SMul(fb.ZeroExtend(fb.Param("lhs", u17), 21),
          fb.SignExtend(fb.Param("rhs", u17), 23),
          /*result_width=*/29);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(f->return_value(),
              m::SMul(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                  /*width=*/18),
                      m::Param("rhs")));
}

TEST_P(NarrowingPassTest, ExtendedSMulSignAgnosticToUMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  fb.SMul(fb.ZeroExtend(fb.Param("lhs", u17), 21),
          fb.SignExtend(fb.Param("rhs", u20), 21),
          /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down, and the product switched
  // to a UMul.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::Param("lhs"), m::SignExt(m::Param("rhs"))));
}

TEST_P(NarrowingPassTest, ExtendedSMulSignAgnosticToSMul) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  fb.SMul(fb.ZeroExtend(fb.Param("lhs", u20), 21),
          fb.SignExtend(fb.Param("rhs", u17), 21),
          /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The sign-extended operand should be sliced down to the sign bit.
  EXPECT_THAT(f->return_value(),
              m::SMul(m::ZeroExt(m::Param("lhs")), m::Param("rhs")));
}

TEST_P(NarrowingPassTest, PartialMultiplyWiderThanSumOfOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue smulp =
      fb.SMulp(fb.Param("lhs", u8), fb.Param("rhs", u8), /*result_width=*/42);
  BValue product = fb.Add(fb.TupleIndex(smulp, 0), fb.TupleIndex(smulp, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(product));
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_THAT(f->return_value(),
              AllOf(m::Type("bits[42]"), m::SignExt(m::Type("bits[16]"))));
  EXPECT_THAT(
      f->nodes(),
      AllOf(
          // Smulp
          Contains(AllOf(m::Type("(bits[16], bits[16])"),
                         m::SMulp(m::Type("bits[8]"), m::Type("bits[8]")))),
          // Two tuple indices
          Contains(m::TupleIndex(std::optional<int64_t>(0))),
          Contains(m::TupleIndex(1)),
          // Sum
          Contains(AllOf(m::Type("bits[16]"),
                         m::Add(m::Type("bits[16]"), m::Type("bits[16]")))),
          // Return value
          Contains(
              AllOf(m::Type("bits[42]"), m::SignExt(m::Type("bits[16]"))))));
}

TEST_P(NarrowingPassTest, PartialMultiplyOperandsWiderThanResult) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u42 = p->GetBitsType(42);
  fb.UMulp(fb.Param("lhs", u17), fb.Param("rhs", u42), /*result_width=*/9);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UMulp(AllOf(m::Type("bits[9]"),
                     m::BitSlice(m::Param("lhs"), /*start=*/0, /*width=*/9)),
               AllOf(m::Type("bits[9]"),
                     m::BitSlice(m::Param("rhs"), /*start=*/0, /*width=*/9))));
}

TEST_P(NarrowingPassTest, ExtendedUMulpOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  BValue mul = fb.UMulp(fb.ZeroExtend(fb.Param("lhs", u17), 32),
                        fb.SignExtend(fb.Param("rhs", u17), 54),
                        /*result_width=*/62);
  fb.Add(fb.TupleIndex(mul, 0), fb.TupleIndex(mul, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Only the zero-extend should have been elided.
  auto mul_match = m::UMulp(m::Param("lhs"), m::SignExt(m::Param("rhs")));
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(mul_match, 0), m::TupleIndex(mul_match, 1)));
}

TEST_P(NarrowingPassTest, ExtendedUMulpSignAgnosticToUMulp) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  BValue prod = fb.UMulp(fb.ZeroExtend(fb.Param("lhs", u17), 21),
                         fb.SignExtend(fb.Param("rhs", u20), 21),
                         /*result_width=*/21);
  fb.Add(fb.TupleIndex(prod, 0), fb.TupleIndex(prod, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down.
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(m::UMulp(m::Param("lhs"),
                                            m::SignExt(m::Param("rhs")))),
                     m::TupleIndex()));
}

TEST_P(NarrowingPassTest, ExtendedUMulpSignAgnosticToSMulp) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  BValue prod = fb.UMulp(fb.ZeroExtend(fb.Param("lhs", u20), 21),
                         fb.SignExtend(fb.Param("rhs", u17), 21),
                         /*result_width=*/21);
  fb.Add(fb.TupleIndex(prod, 0), fb.TupleIndex(prod, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The sign-extended operand should be sliced down to the sign bit, and the
  // product switched to an SMulp.
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(m::SMulp(m::ZeroExt(m::Param("lhs")),
                                            m::Param("rhs"))),
                     m::TupleIndex()));
}

TEST_P(NarrowingPassTest, ExtendedSMulpOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  BValue mul = fb.SMulp(fb.ZeroExtend(fb.Param("lhs", u17), 21),
                        fb.SignExtend(fb.Param("rhs", u17), 23),
                        /*result_width=*/29);
  fb.Add(fb.TupleIndex(mul, 0), fb.TupleIndex(mul, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  auto mul_match =
      m::SMulp(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                           /*width=*/18),
               m::Param("rhs"));
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(mul_match, 0), m::TupleIndex(mul_match, 1)));
}

TEST_P(NarrowingPassTest, ExtendedSMulpSignAgnosticToUMulp) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  BValue prod = fb.SMulp(fb.ZeroExtend(fb.Param("lhs", u17), 21),
                         fb.SignExtend(fb.Param("rhs", u20), 21),
                         /*result_width=*/21);
  fb.Add(fb.TupleIndex(prod, 0), fb.TupleIndex(prod, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down, and the product switched
  // to a UMulp.
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(m::UMulp(m::Param("lhs"),
                                            m::SignExt(m::Param("rhs")))),
                     m::TupleIndex()));
}

TEST_P(NarrowingPassTest, ExtendedSMulpSignAgnosticToSMulp) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u20 = p->GetBitsType(20);
  BValue prod = fb.SMulp(fb.ZeroExtend(fb.Param("lhs", u20), 21),
                         fb.SignExtend(fb.Param("rhs", u17), 21),
                         /*result_width=*/21);
  fb.Add(fb.TupleIndex(prod, 0), fb.TupleIndex(prod, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The sign-extended operand should be sliced down to the sign bit.
  EXPECT_THAT(f->return_value(),
              m::Add(m::TupleIndex(m::SMulp(m::ZeroExt(m::Param("lhs")),
                                            m::Param("rhs"))),
                     m::TupleIndex()));
}

TEST_P(NarrowingPassTest, UMulTrailingZerosRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(20));
  BValue r = fb.Param("r", p->GetBitsType(16));
  BValue r_wide = fb.Shll(fb.ZeroExtend(r, 20), fb.Literal(UBits(4, 3)));
  fb.UMul(l, r_wide, 40);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::UMul(l.node(),
                        m::BitSlice(r_wide.node(), /*start=*/4, /*width=*/16)),
                m::Literal(UBits(0, 4))));
}

TEST_P(NarrowingPassTest, UMulTrailingZerosLeft) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(16));
  BValue r = fb.Param("r", p->GetBitsType(20));
  BValue l_wide = fb.Shll(fb.ZeroExtend(l, 20), fb.Literal(UBits(4, 3)));
  fb.UMul(l_wide, r, 40);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::UMul(m::BitSlice(l_wide.node(), /*start=*/4, /*width=*/16),
                        r.node()),
                m::Literal(UBits(0, 4))));
}

TEST_P(NarrowingPassTest, UMulTrailingZerosBoth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(4));
  BValue l_wide = fb.Shll(fb.ZeroExtend(l, 10), fb.Literal(UBits(6, 3)));
  BValue r_wide = fb.Shll(fb.ZeroExtend(r, 10), fb.Literal(UBits(6, 3)));
  fb.UMul(l_wide, r_wide, 20);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::UMul(m::BitSlice(l_wide.node(), /*start=*/6, /*width=*/4),
                        m::BitSlice(r_wide.node(), /*start=*/6, /*width=*/4)),
                m::Literal(UBits(0, 12))));
}

TEST_P(NarrowingPassTest, SMulTrailingZerosRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(20));
  BValue r = fb.Param("r", p->GetBitsType(16));
  BValue r_wide = fb.Shll(fb.ZeroExtend(r, 20), fb.Literal(UBits(4, 3)));
  fb.SMul(l, r_wide, 40);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::SMul(l.node(),
                        m::BitSlice(r_wide.node(), /*start=*/4, /*width=*/16)),
                m::Literal(UBits(0, 4))));
}

TEST_P(NarrowingPassTest, SMulTrailingZerosLeft) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(16));
  BValue r = fb.Param("r", p->GetBitsType(20));
  BValue l_wide = fb.Shll(fb.ZeroExtend(l, 20), fb.Literal(UBits(4, 3)));
  fb.SMul(l_wide, r, 40);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::SMul(m::BitSlice(l_wide.node(), /*start=*/4, /*width=*/16),
                        r.node()),
                m::Literal(UBits(0, 4))));
}

TEST_P(NarrowingPassTest, SMulTrailingZerosBoth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue l = fb.Param("l", p->GetBitsType(4));
  BValue r = fb.Param("r", p->GetBitsType(4));
  BValue l_wide = fb.Shll(fb.ZeroExtend(l, 10), fb.Literal(UBits(6, 3)));
  BValue r_wide = fb.Shll(fb.ZeroExtend(r, 10), fb.Literal(UBits(6, 3)));
  fb.SMul(l_wide, r_wide, 20);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::SMul(m::BitSlice(l_wide.node(), /*start=*/6, /*width=*/4),
                        m::BitSlice(r_wide.node(), /*start=*/6, /*width=*/4)),
                m::Literal(UBits(0, 12))));
}

TEST_P(NarrowingPassTest, UnsignedCompareWithKnownZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u23 = p->GetBitsType(23);
  fb.UGt(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.ZeroExtend(fb.Param("rhs", u23), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::UGt(m::Type(u23), m::Type(u23)));
}

TEST_P(NarrowingPassTest, LeadingZerosOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.UGt(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UGt(
          m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0, /*width=*/23),
          m::BitSlice(m::ZeroExt(m::Param("rhs")), /*start=*/0, /*width=*/23)));
}

TEST_P(NarrowingPassTest, LeadingZerosOneOneSideOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Leading zeros on only one side of the unsigned comparison should not result
  // in any transformation.
  fb.UGt(fb.Param("lhs", p->GetBitsType(42)),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, MatchedLeadingBitsOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ULe(fb.Concat({fb.Literal(UBits(0b001101, 6)),
                    fb.Param("lhs", p->GetBitsType(13))}),
         fb.Concat({fb.Literal(UBits(0b0011, 4)),
                    fb.Param("rhs", p->GetBitsType(15))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ULe(m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15),
                     m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15)));
}

TEST_P(NarrowingPassTest, MatchedLeadingAndTrailingBitsOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Four matching leading bits and three matching trailing bits.
  fb.ULe(fb.Concat({fb.Literal(UBits(0b001101, 6)),
                    fb.Param("lhs", p->GetBitsType(13)),
                    fb.Literal(UBits(0b10101, 5))}),
         fb.Concat({fb.Literal(UBits(0b0011, 4)),
                    fb.Param("rhs", p->GetBitsType(15)),
                    fb.Literal(UBits(0b11101, 5))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ULe(m::BitSlice(m::Concat(), /*start=*/3, /*width=*/17),
                     m::BitSlice(m::Concat(), /*start=*/3, /*width=*/17)));
}

TEST_P(NarrowingPassTest, LeadingZerosSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SGt(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // Signed comparison with zero-extended operands becomes an unsigned
      // comparison.
      m::UGt(
          m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0, /*width=*/23),
          m::BitSlice(m::ZeroExt(m::Param("rhs")), /*start=*/0, /*width=*/23)));
}

TEST_P(NarrowingPassTest, LeadingOnesOfSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SLe(fb.Concat({fb.Literal(UBits(0b111101, 6)),
                    fb.Param("lhs", p->GetBitsType(12))}),
         fb.Concat({fb.Literal(UBits(0b111, 3)),
                    fb.Param("rhs", p->GetBitsType(15))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // Signed comparison with known equal MSB becomes unsigned comparison.
      m::ULe(m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15),
             m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15)));
}

TEST_P(NarrowingPassTest, SignExtendedOperandsOfSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SGe(fb.SignExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.SignExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SGe(
          m::BitSlice(m::SignExt(m::Param("lhs")), /*start=*/0, /*width=*/23),
          m::BitSlice(m::SignExt(m::Param("rhs")), /*start=*/0, /*width=*/23)));
}

TEST_P(NarrowingPassTest, SignedCompareSignExtendComparedWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_ge_lit = fb.SGe(fb.SignExtend(fb.Param("x", p->GetBitsType(17)), 42),
                           fb.Literal(UBits(0, 42)));
  BValue lit_ge_x =
      fb.SGe(fb.Literal(UBits(0, 42)),
             fb.SignExtend(fb.Param("y", p->GetBitsType(17)), 42));
  fb.Concat({x_ge_lit, lit_ge_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      m::Concat(m::SGe(m::BitSlice(m::SignExt(m::Param("x")),
                                   /*start=*/0, /*width=*/17),
                       m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17)),
                m::SGe(m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17),
                       m::BitSlice(m::SignExt(m::Param("y")), /*start=*/0,
                                   /*width=*/17))));
}

TEST_P(NarrowingPassTest, SignedCompareZeroExtendComparedWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_lit = fb.SGt(fb.ZeroExtend(fb.Param("x", p->GetBitsType(17)), 42),
                           fb.Literal(UBits(0, 42)));
  BValue lit_lt_x =
      fb.SLt(fb.Literal(UBits(0, 42)),
             fb.ZeroExtend(fb.Param("y", p->GetBitsType(17)), 42));
  fb.Concat({x_gt_lit, lit_lt_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      // Signed comparison with zero-extended operands becomes unsigned
      // comparison.
      m::Concat(m::UGt(m::BitSlice(m::ZeroExt(m::Param("x")),
                                   /*start=*/0, /*width=*/17),
                       m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17)),
                m::ULt(m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17),
                       m::BitSlice(m::ZeroExt(m::Param("y")), /*start=*/0,
                                   /*width=*/17))));
}

TEST_P(NarrowingPassTest, SignedCompareZeroExtendComparedWithSignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_y = fb.SGt(fb.ZeroExtend(fb.Param("x", p->GetBitsType(17)), 42),
                         fb.SignExtend(fb.Param("y", p->GetBitsType(19)), 42));
  BValue a_lt_b = fb.SLt(fb.ZeroExtend(fb.Param("a", p->GetBitsType(19)), 42),
                         fb.SignExtend(fb.Param("b", p->GetBitsType(17)), 42));
  fb.Concat({x_gt_y, a_lt_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      m::Concat(m::SGt(m::BitSlice(m::ZeroExt(m::Param("x")),
                                   /*start=*/0, /*width=*/19),
                       m::BitSlice(m::SignExt(m::Param("y")), /*start=*/0,
                                   /*width=*/19)),
                m::SLt(m::BitSlice(m::ZeroExt(m::Param("a")),
                                   /*start=*/0, /*width=*/20),
                       m::BitSlice(m::SignExt(m::Param("b")), /*start=*/0,
                                   /*width=*/20))));
}

TEST_P(NarrowingPassTest, SignExtendAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.SignExtend(fb.Param("foo", p->GetBitsType(8)), 128),
         fb.Literal(UBits(1, 128)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::SignExt(AllOf(m::Add(_, _), m::Type("bits[9]"))),
                    m::Type("bits[128]")));
}

TEST_P(NarrowingPassTest, SignExtendAddTree) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u3 = p->GetBitsType(3);
  fb.Add(fb.Add(fb.SignExtend(fb.Param("a", u3), 32),
                fb.SignExtend(fb.Param("b", u3), 32)),
         fb.Add(fb.SignExtend(fb.Param("c", u3), 32),
                fb.SignExtend(fb.Param("d", u3), 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::SignExt(AllOf(m::Add(_, _), m::Type("bits[5]"))),
                    m::Type("bits[32]")));
}

TEST_P(NarrowingPassTest, AddNegativeNumber) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.ZeroExtend(fb.Param("foo", p->GetBitsType(8)), 128),
         fb.Concat(
             {fb.Literal(SBits(-1, 124)), fb.Param("bar", p->GetBitsType(4))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::SignExt(AllOf(m::Add(_, _), m::Type("bits[9]"))),
                    m::Type("bits[128]")));
}

TEST_P(NarrowingPassTest, UnsignedCompareSignExtendComparedWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_lit = fb.UGt(fb.SignExtend(fb.Param("x", p->GetBitsType(17)), 42),
                           fb.Literal(UBits(0, 42)));
  BValue lit_lt_x =
      fb.ULt(fb.Literal(UBits(0, 42)),
             fb.SignExtend(fb.Param("y", p->GetBitsType(17)), 42));
  fb.Concat({x_gt_lit, lit_lt_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      m::Concat(m::UGt(m::BitSlice(m::SignExt(m::Param("x")),
                                   /*start=*/0, /*width=*/17),
                       m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17)),
                m::ULt(m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17),
                       m::BitSlice(m::SignExt(m::Param("y")), /*start=*/0,
                                   /*width=*/17))));
}

TEST_P(NarrowingPassTest, UnsignedCompareZeroExtendComparedWithLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_lit = fb.UGt(fb.ZeroExtend(fb.Param("x", p->GetBitsType(17)), 42),
                           fb.Literal(UBits(0, 42)));
  BValue lit_lt_x =
      fb.ULt(fb.Literal(UBits(0, 42)),
             fb.ZeroExtend(fb.Param("y", p->GetBitsType(17)), 42));
  fb.Concat({x_gt_lit, lit_lt_x});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      m::Concat(m::UGt(m::BitSlice(m::ZeroExt(m::Param("x")),
                                   /*start=*/0, /*width=*/17),
                       m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17)),
                m::ULt(m::BitSlice(m::Literal(0), /*start=*/0, /*width=*/17),
                       m::BitSlice(m::ZeroExt(m::Param("y")), /*start=*/0,
                                   /*width=*/17))));
}

TEST_P(NarrowingPassTest, UnsignedCompareZeroExtendComparedWithSignExtend) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_y = fb.UGt(fb.ZeroExtend(fb.Param("x", p->GetBitsType(17)), 42),
                         fb.SignExtend(fb.Param("y", p->GetBitsType(19)), 42));
  BValue a_lt_b = fb.ULt(fb.ZeroExtend(fb.Param("a", p->GetBitsType(19)), 42),
                         fb.SignExtend(fb.Param("b", p->GetBitsType(17)), 42));
  fb.Concat({x_gt_y, a_lt_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // The bit-slices would be removed later by the optimization pipeline.
      m::Concat(m::UGt(m::BitSlice(m::ZeroExt(m::Param("x")),
                                   /*start=*/0, /*width=*/19),
                       m::BitSlice(m::SignExt(m::Param("y")), /*start=*/0,
                                   /*width=*/19)),
                m::ULt(m::BitSlice(m::ZeroExt(m::Param("a")),
                                   /*start=*/0, /*width=*/20),
                       m::BitSlice(m::SignExt(m::Param("b")), /*start=*/0,
                                   /*width=*/20))));
}

TEST_P(NarrowingPassTest, SignedCompareWithKnownMSBs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x_gt_y = fb.SGt(fb.Concat({fb.Literal(Value(UBits(0, 1))),
                                    fb.Param("x", p->GetBitsType(17))}),
                         fb.Literal(Value(UBits(0, 18))));
  BValue lit_lt_a = fb.SLt(fb.Literal(Value(UBits(0x80000ull, 20))),
                           fb.Concat({fb.Literal(Value(UBits(2, 2))),
                                      fb.Param("a", p->GetBitsType(18))}));
  BValue lit_lt_b = fb.SLt(fb.Literal(Value(UBits(0x80000ull, 20))),
                           fb.Concat({fb.Literal(Value(UBits(1, 1))),
                                      fb.Param("b", p->GetBitsType(19))}));
  fb.Concat({x_gt_y, lit_lt_a, lit_lt_b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence stays_equivalent{f};
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      // Signed comparisons with known-equal MSBs become unsigned comparisons.
      m::Concat(m::UGt(m::BitSlice(m::Concat(m::Literal(0), m::Param("x")),
                                   /*start=*/0, /*width=*/17),
                       m::BitSlice(m::Literal(0))),
                m::ULt(m::BitSlice(m::Literal(0x80000ull)),
                       m::BitSlice(m::Concat(m::Literal(2), m::Param("a")),
                                   /*start=*/0, /*width=*/18)),
                m::ULt(m::BitSlice(m::Literal(0x80000ull)),
                       m::BitSlice(m::Concat(m::Literal(1), m::Param("b")),
                                   /*start=*/0,
                                   /*width=*/19))));
}

TEST_P(NarrowingPassTest, AddWithLeadingZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(10)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(30)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::Add(m::BitSlice(m::ZeroExt(m::Param("lhs")),
                                            /*start=*/0, /*width=*/31),
                                m::BitSlice(m::ZeroExt(m::Param("rhs")),
                                            /*start=*/0, /*width=*/31))));
}

TEST_P(NarrowingPassTest, AddWithOnlyOneOperandLeadingZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Param("lhs", p->GetBitsType(42)),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(30)), 42));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, ArrayIndexWithAllSameValue) {
  // An array index that, under range analysis, is determined to always index
  // into positions in the array that all have the same precisely known value,
  // should be replaced with a literal containing that value.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue index = fb.Select(fb.Param("selector", p->GetBitsType(2)),
                           {
                               fb.Literal(Value(UBits(4, 4))),
                               fb.Literal(Value(UBits(7, 4))),
                               fb.Literal(Value(UBits(2, 4))),
                               fb.Literal(Value(UBits(6, 4))),
                           },
                           std::nullopt);
  BValue array = fb.Array(
      {
          fb.Literal(Value(UBits(500, 32))),
          fb.Literal(Value(UBits(501, 32))),
          fb.Literal(Value(UBits(600, 32))),
          fb.Literal(Value(UBits(503, 32))),
          fb.Literal(Value(UBits(600, 32))),
          fb.Literal(Value(UBits(505, 32))),
          fb.Literal(Value(UBits(600, 32))),
          fb.Literal(Value(UBits(600, 32))),
          fb.Literal(Value(UBits(508, 32))),
          fb.Literal(Value(UBits(509, 32))),
      },
      p->GetBitsType(32));
  fb.ArrayIndex(array, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  if (analysis() == NarrowingPass::AnalysisType::kRange) {
    ScopedVerifyEquivalence sve(f);
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(), m::Literal(600));
  }
}

TEST_P(NarrowingPassTest, ConvertArrayIndexToSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue index = fb.Select(
      fb.Param("s", p->GetBitsType(1)),
      /*cases=*/
      {fb.Literal(Value(UBits(3, 4))), fb.Literal(Value(UBits(7, 4)))},
      /*default_value=*/std::nullopt);
  BValue array = fb.Literal(Value::ArrayOrDie({
      Value(UBits(0, 4)),
      Value(UBits(1, 4)),
      Value(UBits(2, 4)),
      Value(UBits(3, 4)),
      Value(UBits(4, 4)),
      Value(UBits(5, 4)),
      Value(UBits(6, 4)),
      Value(UBits(7, 4)),
      Value(UBits(8, 4)),
      Value(UBits(9, 4)),
  }));
  fb.ArrayIndex(array, /*indices=*/{index});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  if (analysis() == NarrowingPass::AnalysisType::kRange) {
    ScopedVerifyEquivalence sve(f);
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(),
                m::Select(m::And(m::Eq(index.node(), m::Literal(7))),
                          /*cases=*/
                          {
                              m::ArrayIndex(array.node(), {m::Literal(3)}),
                              m::ArrayIndex(array.node(), {m::Literal(7)}),
                          },
                          /*default_value=*/std::nullopt));
  }
}

TEST_P(NarrowingPassTest, SideEffectfulButNarrowable) {
  auto p = CreatePackage();
  Block* to_instantiate = nullptr;
  {
    BlockBuilder bb("to_instantiate", p.get());
    bb.OutputPort("output", bb.Literal(UBits(0, 0)));
    XLS_ASSERT_OK_AND_ASSIGN(to_instantiate, bb.Build());
  }
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * instantiation,
      bb.block()->AddBlockInstantiation("instantiation", to_instantiate));
  bb.InstantiationOutput(instantiation, "output");
  XLS_ASSERT_OK(bb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, KnownZeroValueGateRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("x", p->GetBitsType(1));
  BValue known_zero = fb.Literal(UBits(0, 8));
  fb.Gate(param, known_zero);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_P(NarrowingPassTest, KnownNonZeroGateConditionRemovedIfConstant) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Literal(UBits(12, 8));
  BValue known_nonzero = fb.Literal(UBits(1, 1));
  fb.Gate(known_nonzero, param);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(f->return_value(), m::Literal(UBits(12, 8)));
}

TEST_P(NarrowingPassTest, KnownZeroGateConditionRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("x", p->GetBitsType(8));
  BValue known_zero = fb.Literal(UBits(0, 1));
  fb.Gate(known_zero, param);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_P(NarrowingPassTest, NarrowableArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("s", p->GetBitsType(4));
  BValue array = fb.Literal(Value::ArrayOrDie({
      Value(UBits(1, 32)),
      Value(UBits(3, 32)),
      Value(UBits(5, 32)),
      Value(UBits(7, 32)),
      Value(UBits(11, 32)),
      Value(UBits(13, 32)),
      Value(UBits(15, 32)),
      Value(UBits((uint64_t{1U} << 20) + 1, 32)),
  }));
  fb.ArrayIndex(array, /*indices=*/{param});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Concat(m::Literal(UBits(0, 11)),
                        m::BitSlice(m::ArrayIndex(m::Literal(), {_}), 3, 1),
                        m::Literal(UBits(0, 16)),
                        m::BitSlice(m::ArrayIndex(m::Literal(), {_}), 0, 3),
                        m::Literal(UBits(1, 1))));
}

TEST_P(NarrowingPassTest, EliminableArray) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("s", p->GetBitsType(7));
  std::vector<Value> array_elements(118, Value(UBits(0, 8)));
  BValue array = fb.Literal(Value::ArrayOrDie(array_elements));
  fb.ArrayIndex(array, /*indices=*/{param});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_F(ContextNarrowingPassTest, ExactMatch) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(64));
  BValue lit_10 = fb.Literal(UBits(10, 64));
  BValue add_10 = fb.Add(param, lit_10);
  BValue result = fb.Select(param, {lit_10, add_10, lit_10, add_10}, lit_10);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_EQ(f->return_value(), result.node());
  ASSERT_THAT(result.node(),
              m::Select(param.node(),
                        {m::Literal(UBits(10, 64)), m::Literal(UBits(11, 64)),
                         m::Literal(UBits(10, 64)), m::Literal(UBits(13, 64))},
                        m::Literal(UBits(10, 64))))
      << f->DumpIr();
}

TEST_F(ContextNarrowingPassTest, ExactMatchWithEq) {
  auto p = CreatePackage();
  // fn (x) { if (x == 10) { x + 10 } else { 13 } }
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(64));
  BValue lit_10 = fb.Literal(UBits(10, 64));
  BValue add_10 = fb.Add(param, lit_10);
  BValue result =
      fb.Select(fb.Eq(param, lit_10), {fb.Literal(UBits(13, 64)), add_10});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_EQ(f->return_value(), result.node());
  ASSERT_THAT(result.node(), m::Select(m::Eq(m::Param(), m::Literal()),
                                       {m::Literal(UBits(13, 64)),
                                        m::Literal(UBits(20, 64))}));
}

TEST_F(ContextNarrowingPassTest, ExactMatchWithEq2) {
  auto p = CreatePackage();
  // fn (x) { if (x == 10) { x + 10 } else { x } }
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(64));
  BValue lit_10 = fb.Literal(UBits(10, 64));
  BValue add_10 = fb.Add(param, lit_10);
  BValue result = fb.Select(fb.Eq(param, lit_10), {param, add_10});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_EQ(f->return_value(), result.node());
  ASSERT_THAT(result.node(),
              m::Select(m::Eq(m::Param(), m::Literal()),
                        {param.node(), m::Literal(UBits(20, 64))}));
}

TEST_F(ContextNarrowingPassTest, MaxSizeShift) {
  auto p = CreatePackage();
  // fn (x:u64, y:u64) { if (x < 4) { y << x } else { y } }
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(64));
  BValue y = fb.Param("y", p->GetBitsType(64));
  BValue lit_4 = fb.Literal(UBits(4, 64));
  BValue shift_l = fb.Shll(y, x);
  BValue cond = fb.ULt(x, lit_4);
  BValue result = fb.Select(cond, {y, shift_l});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_EQ(f->return_value(), result.node());
  ASSERT_THAT(
      result.node(),
      m::Select(m::ULt(m::Param("x"), m::Literal(UBits(4, 64))),
                {m::Param("y"),
                 m::Shll(m::Param("y"), m::BitSlice(m::Param("x"), 0, 2))}));
}

TEST_F(ContextNarrowingPassTest, KnownSmallAdd) {
  auto p = CreatePackage();
  // fn (x:u64, y:u4) { if (x < 4) { (y as u64) + x } else { x } }
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(64));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue lit_4 = fb.Literal(UBits(4, 64));
  BValue add = fb.Add(fb.ZeroExtend(y, 64), x);
  BValue cond = fb.ULt(x, lit_4);
  BValue result = fb.Select(cond, {x, add});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_EQ(f->return_value(), result.node());
  ASSERT_THAT(result.node(),
              m::Select(m::ULt(m::Param("x"), m::Literal(UBits(4, 64))),
                        {m::Param("x"), m::ZeroExt()}))
      << f->DumpIr();
}

TEST_P(NarrowingPassTest, FullNegativeNarrowLit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue v = fb.PrioritySelect(sel, {fb.Literal(UBits(127, 8))},
                               fb.Literal(UBits(255, 8)));
  BValue nv = fb.Not(v);
  fb.UMul(nv, nv);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_P(NarrowingPassTest, FullNegativeNarrow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue lb = fb.Param("lb", p->GetBitsType(1));
  BValue rb = fb.Param("rb", p->GetBitsType(1));
  BValue v = fb.PrioritySelect(sel, {fb.Concat({lb, fb.Literal(SBits(-1, 7))})},
                               fb.Concat({rb, fb.Literal(SBits(-1, 7))}));
  BValue nv = fb.Not(v);
  fb.UMul(nv, nv);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 8)));
}

TEST_P(NarrowingPassTest, AnalysisLogOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.ZeroExtend(fb.Param("foo", p->GetBitsType(8)), 128),
         fb.Literal(UBits(1, 128)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedSetVlogLevel vlog("narrowing_pass", 3);
  ScopedVerifyEquivalence stays_equivalent{f};
  ScopedMockLog log;
  EXPECT_CALL(
      log, Log(absl::LogSeverity::kInfo, _, HasSubstr("literal shrinkable")));
  log.StartCapturingLogs();
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  log.StopCapturingLogs();
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::Add(m::BitSlice(m::ZeroExt(m::Param("foo"))),
                                m::BitSlice(m::Literal()))));
}

TEST_P(NarrowingPassTest, TracksUpdates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(2));
  // decoded is 0b00...0XXXX
  BValue decoded = fb.Decode(fb.ZeroExtend(param, 8), 256);
  // Should be able to do a 4 or 5-bit adder (need range analysis to see that 4
  // bits is safe)
  fb.Add(fb.Literal(UBits(1, 256)), decoded);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // TODO(allight): We should be able to recognize this as only needing 4 bit
  // add for range analyses.
  EXPECT_THAT(
      f->return_value(),
      m::ZeroExt(AllOf(m::Type("bits[5]"),
                       m::Add(_, m::BitSlice(m::ZeroExt(AllOf(
                                     m::Type("bits[4]"), m::Decode())))))));
}

TEST_P(NarrowingPassTest, NarrowingDoesNotReplaceLiteralWithItself) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Select(fb.Param("foo", p->GetBitsType(1)),
            fb.Param("param", p->GetBitsType(16)), fb.Literal(UBits(12, 16)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, ProcStateInformationIsUsed) {
  if (analysis() == NarrowingPass::AnalysisType::kTernary) {
    GTEST_SKIP() << "Ternary narrowing does not take into account proc-state "
                    "information";
  }
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan, p->CreateSingleValueChannel("chan", ChannelOps::kSendOnly,
                                             p->GetBitsType(1)));
  ProcBuilder pb(TestName(), p.get());
  BValue state_read = pb.StateElement("foo", UBits(0, 64));
  // The sent value is always true.
  BValue snd = pb.Send(chan, pb.Literal(Value::Token()),
                       pb.ULt(state_read, pb.Literal(UBits(12, 64))));
  BValue incr =
      pb.Next(state_read, pb.Add(state_read, pb.Literal(UBits(1, 64))),
              pb.ULt(state_read, pb.Literal(UBits(10, 64))));
  BValue rst = pb.Next(state_read, pb.Literal(UBits(0, 64)),
                       pb.UGe(state_read, pb.Literal(UBits(10, 64))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * pr, pb.Build());
  // Narrowing won't actually mess with the state variables themselves so we can
  // ensure they remain consistent.
  ScopedVerifyProcEquivalence sve(pr, /*activation_count=*/25,
                                  /*include_state=*/true);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(snd.node(),
              m::Send(m::Literal(Value::Token()), m::Literal(UBits(1, 1))));
  EXPECT_THAT(incr.node(),
              m::Next(m::StateRead(), m::ZeroExt(m::Add()),
                      m::ULt(m::Type("bits[4]"), m::Type("bits[4]"))));
  EXPECT_THAT(rst.node(),
              m::Next(m::StateRead(), m::Literal(),
                      m::UGe(m::Type("bits[4]"), m::Type("bits[4]"))));
}

TEST_P(NarrowingPassTest, ArrayBoundsContextual) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue idx = fb.Param("idx", p->GetBitsType(16));
  fb.ArrayIndex(fb.Param("arr", p->GetArrayType(10, p->GetBitsType(32))),
                {fb.Select(fb.ULt(idx, fb.Literal(UBits(10, 16))),
                           {fb.Literal(UBits(0, 16)), idx})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  bool is_context_analysis =
      analysis() == NarrowingPass::AnalysisType::kRangeWithContext;
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(is_context_analysis));
  if (is_context_analysis) {
    EXPECT_THAT(f->return_value(), m::ArrayIndex(_, {_}, m::AssumedInBounds()));
  }
}

TEST_P(NarrowingPassTest, ArrayBoundsProof) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue idx = fb.Param("idx", p->GetBitsType(3));
  // 5 bits
  fb.ArrayIndex(fb.Param("arr", p->GetArrayType(32, p->GetBitsType(32))),
                // Definitely less than 24 even with ternary.
                {fb.Add(fb.Literal(UBits(16, 32)), fb.ZeroExtend(idx, 32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(_, {_}, m::AssumedInBounds()));
}

TEST_P(NarrowingPassTest, NarrowingSliceDoesntOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.DynamicBitSlice(fb.Literal(UBits(0, 1487)),
                     fb.Literal(Bits::AllOnes(1975)), 1487);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 1487)));
}

TEST_P(NarrowingPassTest, ShiftNoOverflow) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue amnt = fb.Param("amnt", p->GetBitsType(63));
  BValue to_shift = fb.Param("shifter", p->GetBitsType(100));
  fb.Shll(to_shift, amnt);
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_P(NarrowingPassTest, UninterestingBitSlice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue param = fb.Param("param", p->GetBitsType(11));
  BValue zero = fb.Literal(UBits(0, 258));
  BValue bit_slice = fb.BitSlice(param, 0, 4);
  BValue dynamic_bit_slice = fb.DynamicBitSlice(zero, bit_slice, 243);
  fb.Tuple({dynamic_bit_slice});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Literal(Value::Tuple({Value(UBits(0, 243))})));
}

INSTANTIATE_TEST_SUITE_P(
    NarrowingPassTestInstantiation, NarrowingPassTest,
    ::testing::Values(NarrowingPass::AnalysisType::kTernary,
                      NarrowingPass::AnalysisType::kRange,
                      NarrowingPass::AnalysisType::kRangeWithContext),
    ::testing::PrintToStringParamName());

void IrFuzzNarrowingTernary(
    const PackageAndTestParams& package_and_test_params) {
  NarrowingPass pass(NarrowingPass::AnalysisType::kTernary);
  OptimizationPassChangesOutputs(package_and_test_params, pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNarrowingTernary)
    .WithDomains(IrFuzzDomainWithParams(/*param_set_count=*/10));

void IrFuzzNarrowingRange(const PackageAndTestParams& package_and_test_params) {
  NarrowingPass pass(NarrowingPass::AnalysisType::kRange);
  OptimizationPassChangesOutputs(package_and_test_params, pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNarrowingRange)
    .WithDomains(IrFuzzDomainWithParams(/*param_set_count=*/10));

void IrFuzzNarrowingRangeWithContext(
    const PackageAndTestParams& package_and_test_params) {
  NarrowingPass pass(NarrowingPass::AnalysisType::kRangeWithContext);
  OptimizationPassChangesOutputs(package_and_test_params, pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNarrowingRangeWithContext)
    .WithDomains(IrFuzzDomainWithParams(/*param_set_count=*/10));

}  // namespace
}  // namespace xls
