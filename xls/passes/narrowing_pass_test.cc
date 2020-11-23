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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::AllOf;

class NarrowingPassTest : public IrTestBase {
 protected:
  NarrowingPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return NarrowingPass().Run(p, PassOptions(), &results);
  }
};

TEST_F(NarrowingPassTest, UnnarrowableShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shra(fb.Param("in", p->GetBitsType(32)),
          fb.Param("amt", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Shra(m::Param("in"), m::Param("amt")));
}

TEST_F(NarrowingPassTest, NarrowableShift) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(
      fb.Param("in", p->GetBitsType(32)),
      fb.ZeroExtend(fb.Param("amt", p->GetBitsType(3)), /*new_bit_count=*/123));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shll(m::Param("in"), m::BitSlice(/*start=*/0, /*width=*/3)));
}

TEST_F(NarrowingPassTest, ShiftWithKnownZeroShiftAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)), fb.Literal(UBits(0, 27)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("in"));
}

TEST_F(NarrowingPassTest, ShiftWithKnownOnePrefix) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)),
          fb.Concat({fb.Literal(UBits(0b111000, 6)),
                     fb.Param("amt", p->GetBitsType(3))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("in"), m::Concat()));
}

TEST_F(NarrowingPassTest, NarrowableArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(
      fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
      fb.ZeroExtend(fb.Param("idx", p->GetBitsType(8)), /*new_bit_count=*/123));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), /*indices=*/{
                                     m::BitSlice(/*start=*/0, /*width=*/8)}));
}

TEST_F(NarrowingPassTest, NarrowableArrayIndexAllZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                fb.And(fb.Param("idx", p->GetBitsType(8)),
                       fb.Literal(Value(UBits(0, 8)))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), /*indices=*/{m::Literal(0)}));
}

TEST_F(NarrowingPassTest, LiteralArrayIndex) {
  // A literal array index should not be substituted.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                fb.Literal(Value(UBits(0x0f, 8))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(
      f->return_value(),
      m::MultiArrayIndex(m::Param("a"), /*indices=*/{m::Literal(0x0f)}));
}

TEST_F(NarrowingPassTest, MultiplyWiderThanSumOfOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  fb.SMul(fb.Param("lhs", u8), fb.Param("rhs", u8), /*result_width=*/42);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      AllOf(m::Type("bits[42]"),
            m::SignExt(AllOf(m::Type("bits[16]"),
                             m::SMul(m::Param("lhs"), m::Param("rhs"))))));
}

TEST_F(NarrowingPassTest, MultiplyOperandsWiderThanResult) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u42 = p->GetBitsType(42);
  fb.UMul(fb.Param("lhs", u17), fb.Param("rhs", u42), /*result_width=*/9);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UMul(AllOf(m::Type("bits[9]"),
                    m::BitSlice(m::Param("lhs"), /*start=*/0, /*width=*/9)),
              AllOf(m::Type("bits[9]"),
                    m::BitSlice(m::Param("rhs"), /*start=*/0, /*width=*/9))));
}

TEST_F(NarrowingPassTest, ExtendedUMulOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  fb.UMul(fb.ZeroExtend(fb.Param("lhs", u17), 32),
          fb.SignExtend(fb.Param("rhs", u17), 54),
          /*result_width=*/62);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Only the zero-extend should have been elided.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                  /*width=*/17),
                      m::SignExt(m::Param("rhs"))));
}

TEST_F(NarrowingPassTest, ExtendedSMulOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  fb.SMul(fb.ZeroExtend(fb.Param("lhs", u17), 21),
          fb.SignExtend(fb.Param("rhs", u17), 23),
          /*result_width=*/29);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(f->return_value(),
              m::SMul(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                  /*width=*/18),
                      m::Param("rhs")));
}

TEST_F(NarrowingPassTest, LeadingZerosOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.UGt(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UGt(
          m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0, /*width=*/23),
          m::BitSlice(m::ZeroExt(m::Param("rhs")), /*start=*/0, /*width=*/23)));
}

TEST_F(NarrowingPassTest, LeadingZerosOneOneSideOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  // Leading zeros on only one side of the unsigned comparison should not result
  // in any transformation.
  fb.UGt(fb.Param("lhs", p->GetBitsType(42)),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(NarrowingPassTest, MatchedLeadingBitsOfUnsignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ULe(fb.Concat({fb.Literal(UBits(0b001101, 6)),
                    fb.Param("lhs", p->GetBitsType(13))}),
         fb.Concat({fb.Literal(UBits(0b0011, 4)),
                    fb.Param("rhs", p->GetBitsType(15))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ULe(m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15),
                     m::BitSlice(m::Concat(), /*start=*/0, /*width=*/15)));
}

TEST_F(NarrowingPassTest, MatchedLeadingAndTrailingBitsOfUnsignedCompare) {
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
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ULe(m::BitSlice(m::Concat(), /*start=*/3, /*width=*/17),
                     m::BitSlice(m::Concat(), /*start=*/3, /*width=*/17)));
}

TEST_F(NarrowingPassTest, LeadingZerosSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SGt(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SGt(
          m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0, /*width=*/24),
          m::BitSlice(m::ZeroExt(m::Param("rhs")), /*start=*/0, /*width=*/24)));
}

TEST_F(NarrowingPassTest, LeadingOnesOfSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SLe(fb.Concat({fb.Literal(UBits(0b111101, 6)),
                    fb.Param("lhs", p->GetBitsType(12))}),
         fb.Concat({fb.Literal(UBits(0b111, 3)),
                    fb.Param("rhs", p->GetBitsType(15))}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SLe(m::BitSlice(m::Concat(), /*start=*/0, /*width=*/16),
                     m::BitSlice(m::Concat(), /*start=*/0, /*width=*/16)));
}

TEST_F(NarrowingPassTest, SignExtendedOperandsOfSignedCompare) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.SGe(fb.SignExtend(fb.Param("lhs", p->GetBitsType(17)), 42),
         fb.SignExtend(fb.Param("rhs", p->GetBitsType(23)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SGe(
          m::BitSlice(m::SignExt(m::Param("lhs")), /*start=*/0, /*width=*/23),
          m::BitSlice(m::SignExt(m::Param("rhs")), /*start=*/0, /*width=*/23)));
}

TEST_F(NarrowingPassTest, CompareOfIdenticalLiterals) {
  // Identical literals should not be narrowed by this pass. Those are handled
  // elsewhere.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.UGt(fb.Literal(Value(UBits(42, 32))), fb.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(NarrowingPassTest, AddWithLeadingZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.ZeroExtend(fb.Param("lhs", p->GetBitsType(10)), 42),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(30)), 42));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(m::Add(m::BitSlice(m::ZeroExt(m::Param("lhs")),
                                            /*start=*/0, /*width=*/31),
                                m::BitSlice(m::ZeroExt(m::Param("rhs")),
                                            /*start=*/0, /*width=*/31))));
}

TEST_F(NarrowingPassTest, AddWithOnlyOneOperandLeadingZeros) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Param("lhs", p->GetBitsType(42)),
         fb.ZeroExtend(fb.Param("rhs", p->GetBitsType(30)), 42));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(NarrowingPassTest, AddWithAllZeroOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Add(fb.Literal(Value(UBits(0, 16))),
         fb.Literal(Value(UBits(0, 16))));
  // There shouldn't be narrowing because this special case of all known zero is
  // handled elsewhere in the pipeline.
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
