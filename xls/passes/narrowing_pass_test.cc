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

// The test is parameterized on whether to use range analysis or not.
class NarrowingPassTest : public IrTestBase,
                          public testing::WithParamInterface<bool> {
 protected:
  NarrowingPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    PassOptions options;
    options.convert_array_index_to_select = 2;
    return NarrowingPass(/*use_range_analysis=*/GetParam())
        .Run(p, options, &results);
  }
};

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
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Shll(m::Param("in"), m::BitSlice(/*start=*/0, /*width=*/3)));
}

TEST_P(NarrowingPassTest, ShiftWithKnownZeroShiftAmount) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Shll(fb.Param("in", p->GetBitsType(32)), fb.Literal(UBits(0, 27)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
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
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Shll(m::Param("in"), m::Concat()));
}

TEST_P(NarrowingPassTest, NarrowableArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.ArrayIndex(fb.Param("a", p->GetArrayType(42, p->GetBitsType(32))),
                {fb.ZeroExtend(fb.Param("idx", p->GetBitsType(8)),
                               /*new_bit_count=*/123)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
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
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Only the zero-extend should have been elided.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                  /*width=*/17),
                      m::SignExt(m::Param("rhs"))));
}

TEST_P(NarrowingPassTest, ExtendedSMulOperands) {
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

TEST_P(NarrowingPassTest, ExtendedSMulSignAgnostic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u21 = p->GetBitsType(21);
  fb.SMul(fb.ZeroExtend(fb.Param("lhs", u17), 21), fb.Param("rhs", u21),
          /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(f->return_value(),
              m::UMul(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                  /*width=*/17),
                      m::Param("rhs")));
}

TEST_P(NarrowingPassTest, PartialMultiplyWiderThanSumOfOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  fb.SMulp(fb.Param("lhs", u8), fb.Param("rhs", u8), /*result_width=*/42);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              AllOf(m::Type("(bits[42], bits[42])"),
                    m::Tuple(m::SignExt(m::Type("bits[16]")),
                             m::SignExt(m::Type("bits[16]")))));
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
  fb.UMulp(fb.ZeroExtend(fb.Param("lhs", u17), 32),
           fb.SignExtend(fb.Param("rhs", u17), 54),
           /*result_width=*/62);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // Only the zero-extend should have been elided.
  EXPECT_THAT(f->return_value(),
              m::UMulp(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                   /*width=*/17),
                       m::SignExt(m::Param("rhs"))));
}

TEST_P(NarrowingPassTest, ExtendedSMulpOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  fb.SMulp(fb.ZeroExtend(fb.Param("lhs", u17), 21),
           fb.SignExtend(fb.Param("rhs", u17), 23),
           /*result_width=*/29);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(f->return_value(),
              m::SMulp(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                   /*width=*/18),
                       m::Param("rhs")));
}

TEST_P(NarrowingPassTest, ExtendedSMulpSignAgnostic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u17 = p->GetBitsType(17);
  Type* u21 = p->GetBitsType(21);
  fb.SMulp(fb.ZeroExtend(fb.Param("lhs", u17), 21), fb.Param("rhs", u21),
           /*result_width=*/21);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  // The zero-extended operand should be sliced down to the (zero) sign bit.
  EXPECT_THAT(f->return_value(),
              m::UMulp(m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0,
                                   /*width=*/17),
                       m::Param("rhs")));
}

TEST_P(NarrowingPassTest, LeadingZerosOfUnsignedCompare) {
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
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SGt(
          m::BitSlice(m::ZeroExt(m::Param("lhs")), /*start=*/0, /*width=*/24),
          m::BitSlice(m::ZeroExt(m::Param("rhs")), /*start=*/0, /*width=*/24)));
}

TEST_P(NarrowingPassTest, LeadingOnesOfSignedCompare) {
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

TEST_P(NarrowingPassTest, SignExtendedOperandsOfSignedCompare) {
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

TEST_P(NarrowingPassTest, AddWithLeadingZeros) {
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
  if (/*use_range_analysis=*/GetParam()) {
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
      {fb.Literal(Value(UBits(3, 5))), fb.Literal(Value(UBits(7, 5)))},
      /*default=*/std::nullopt);
  BValue array = fb.Literal(Value::ArrayOrDie({
      Value(UBits(0, 20)),
      Value(UBits(1, 20)),
      Value(UBits(2, 20)),
      Value(UBits(3, 20)),
      Value(UBits(4, 20)),
      Value(UBits(5, 20)),
      Value(UBits(6, 20)),
      Value(UBits(7, 20)),
      Value(UBits(8, 20)),
      Value(UBits(9, 20)),
  }));
  fb.ArrayIndex(array, /*indices=*/{index});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  if (/*use_range_analysis=*/GetParam()) {
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
    EXPECT_THAT(f->return_value(),
                m::Select(m::And(m::Eq(index.node(), m::Literal(7))),
                          /*cases=*/
                          {
                              m::ArrayIndex(array.node(), {m::Literal(3)}),
                              m::ArrayIndex(array.node(), {m::Literal(7)}),
                          },
                          /*default=*/std::nullopt));
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

INSTANTIATE_TEST_SUITE_P(
    NarrowingPassTestInstantiation, NarrowingPassTest,
    testing::Values(false, true),
    [](const testing::TestParamInfo<NarrowingPassTest::ParamType>& info) {
      return info.param ? "with_range_analysis" : "without_range_analysis";
    });

}  // namespace
}  // namespace xls
