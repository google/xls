// Copyright 2021 The XLS Authors
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

#include "xls/dslx/interp_value.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;

TEST(InterpValueTest, FormatU8) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/8, /*value=*/0xff);
  EXPECT_EQ(ff.ToString(), "u8:255");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kHex), "0xff");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kSignedDecimal),
            "-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kUnsignedDecimal),
            "255");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kBinary),
            "0b1111_1111");
}

TEST(InterpValueTest, FormatS8) {
  auto ff = InterpValue::MakeSBits(/*bit_count=*/8, /*value=*/-1);
  EXPECT_EQ(ff.ToString(), "s8:-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kHex), "0xff");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kUnsignedDecimal),
            "255");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kSignedDecimal),
            "-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kBinary),
            "0b1111_1111");
}

TEST(InterpValueTest, BitsEquivalence) {
  auto a = InterpValue::MakeUBits(/*bit_count=*/4, /*value=*/4);
  EXPECT_EQ(a, a);
  auto b = InterpValue::MakeUBits(/*bit_count=*/4, /*value=*/5);
  EXPECT_EQ(b, b);
  EXPECT_NE(a, b);
}

TEST(InterpValueTest, FlattenArrayOfBits) {
  auto a = InterpValue::MakeUBits(/*bit_count=*/12, /*value=*/0xf00);
  auto b = InterpValue::MakeUBits(/*bit_count=*/12, /*value=*/0xba5);
  auto array = InterpValue::MakeArray({a, b});
  auto o = array->Flatten();
  EXPECT_THAT(o->GetBitCount(), IsOkAndHolds(24));
  EXPECT_THAT(o->GetBitValueUnsigned(), IsOkAndHolds(0xf00ba5));
}

TEST(InterpValueTest, BitwiseNegateAllBitsSet) {
  auto v = InterpValue::MakeUBits(/*bit_count=*/3, 0x7);
  auto expected = InterpValue::MakeUBits(/*bit_count=*/3, 0);
  EXPECT_TRUE(v.BitwiseNegate().value().Eq(expected));
}

TEST(InterpValueTest, BitwiseNegateLowBitUnset) {
  auto v = InterpValue::MakeUBits(/*bit_count=*/3, 0x6);
  auto expected = InterpValue::MakeUBits(/*bit_count=*/3, 1);
  EXPECT_TRUE(v.BitwiseNegate().value().Eq(expected));
}

TEST(InterpValueTest, BitwiseNegateMiddleBitUnset) {
  auto v = InterpValue::MakeUBits(/*bit_count=*/3, 0x5);
  auto expected = InterpValue::MakeUBits(/*bit_count=*/3, 0x2);
  EXPECT_TRUE(v.BitwiseNegate().value().Eq(expected));
}

TEST(InterpValueTest, BitwiseNegateHighBitUnset) {
  auto v = InterpValue::MakeUBits(/*bit_count=*/3, 0x3);
  auto expected = InterpValue::MakeUBits(/*bit_count=*/3, 0x4);
  EXPECT_TRUE(v.BitwiseNegate().value().Eq(expected));
}

TEST(InterpValueTest, LessThan) {
  auto uf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  auto sf = InterpValue::MakeSBits(/*bit_count=*/4, -1);

  auto uzero = InterpValue::MakeUBits(4, 0);
  auto szero = InterpValue::MakeSBits(4, 0);

  auto true_value = InterpValue::MakeBool(true);
  auto false_value = InterpValue::MakeBool(false);

  EXPECT_THAT(uf.Gt(uzero), IsOkAndHolds(true_value));
  EXPECT_THAT(uf.Lt(uzero), IsOkAndHolds(false_value));
  EXPECT_THAT(sf.Gt(szero), IsOkAndHolds(false_value));
  EXPECT_THAT(sf.Lt(szero), IsOkAndHolds(true_value));
}

TEST(InterpValueTest, Negate) {
  auto uone = InterpValue::MakeUBits(/*bit_count=*/4, 1);
  auto uf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  EXPECT_THAT(uone.ArithmeticNegate(), IsOkAndHolds(uf));

  auto sone = InterpValue::MakeUBits(/*bit_count=*/4, 1);
  auto sf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  EXPECT_THAT(sone.ArithmeticNegate(), IsOkAndHolds(sf));
}

TEST(InterpValueTest, SampleOps) {
  auto sample_ops = [](InterpValue x) -> InterpValue {
    return x.Shrl(x)
        .value()
        .BitwiseXor(x)
        .value()
        .Shra(x)
        .value()
        .BitwiseOr(x)
        .value()
        .BitwiseAnd(x)
        .value()
        .BitwiseNegate()
        .value()
        .ArithmeticNegate()
        .value()
        .Sub(x)
        .value();
  };

  auto uone = InterpValue::MakeUBits(/*bit_count=*/4, 5);
  auto uzero = InterpValue::MakeUBits(/*bit_count=*/4, 1);
  EXPECT_EQ(uzero, sample_ops(uone));

  auto sone = InterpValue::MakeUBits(/*bit_count=*/4, 5);
  auto szero = InterpValue::MakeUBits(/*bit_count=*/4, 1);
  EXPECT_EQ(szero, sample_ops(sone));
}

TEST(InterpValueTest, ArrayOfU32HumanStr) {
  auto array =
      InterpValue::MakeArray({InterpValue::MakeU32(2), InterpValue::MakeU32(3),
                              InterpValue::MakeU32(4)});
  EXPECT_EQ(array->ToHumanString(), "[2, 3, 4]");
}

TEST(InterpValueTest, TestPredicates) {
  auto false_value = InterpValue::MakeBool(false);
  EXPECT_TRUE(false_value.IsFalse());
  EXPECT_FALSE(false_value.IsTrue());

  auto true_value = InterpValue::MakeBool(true);
  EXPECT_TRUE(true_value.IsTrue());
  EXPECT_FALSE(true_value.IsFalse());

  // All-zero-bits is not considered the "false" value, has to be single bit.
  EXPECT_FALSE(InterpValue::MakeU32(0).IsFalse());
  // Ditto, all-one-bits is not true, has to be single bit.
  EXPECT_FALSE(InterpValue::MakeU32(-1U).IsTrue());
  EXPECT_FALSE(InterpValue::MakeU32(1).IsTrue());
}

TEST(InterpValueTest, FormatNilTupleWrongElementCount) {
  auto tuple = InterpValue::MakeTuple({});

  std::vector<StructFormatDescriptor::Element> elements;
  elements.emplace_back(StructFormatDescriptor::Element{
      "x",
      std::make_unique<LeafValueFormatDescriptor>(FormatPreference::kHex)});
  StructFormatDescriptor fmt_desc{"MyStruct", std::move(elements)};
  ASSERT_THAT(tuple.ToFormattedString(fmt_desc),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Number of tuple elements (0)")));
}

TEST(InterpValueTest, FormatFlatStructViaDescriptor) {
  auto uf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  auto sf = InterpValue::MakeSBits(/*bit_count=*/4, -1);
  auto tuple = InterpValue::MakeTuple({uf, sf});

  std::vector<StructFormatDescriptor::Element> elements;
  elements.emplace_back(StructFormatDescriptor::Element{
      "x",
      std::make_unique<LeafValueFormatDescriptor>(FormatPreference::kHex)});
  elements.emplace_back(StructFormatDescriptor::Element{
      "y", std::make_unique<LeafValueFormatDescriptor>(
               FormatPreference::kSignedDecimal)});
  StructFormatDescriptor fmt_desc{"MyStruct", std::move(elements)};
  XLS_ASSERT_OK_AND_ASSIGN(std::string s, tuple.ToFormattedString(fmt_desc));
  EXPECT_EQ(s, R"(MyStruct {
  x: 0xf,
  y: -1
})");
}

TEST(InterpValueTest, FormatNestedStructViaDescriptor) {
  auto uf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  auto sf = InterpValue::MakeSBits(/*bit_count=*/4, -1);
  auto inner = InterpValue::MakeTuple({uf, sf});

  std::vector<StructFormatDescriptor::Element> elements;
  elements.emplace_back(StructFormatDescriptor::Element{
      "x",
      std::make_unique<LeafValueFormatDescriptor>(FormatPreference::kHex)});
  elements.emplace_back(StructFormatDescriptor::Element{
      "y", std::make_unique<LeafValueFormatDescriptor>(
               FormatPreference::kSignedDecimal)});
  auto inner_fmt_desc = std::make_unique<StructFormatDescriptor>(
      "InnerStruct", std::move(elements));

  auto outer = InterpValue::MakeTuple(
      {inner, InterpValue::MakeUBits(/*bit_count=*/32, 42)});

  std::vector<StructFormatDescriptor::Element> outer_elements;
  outer_elements.push_back(StructFormatDescriptor::Element{
      .field_name = "a", .fmt = std::move(inner_fmt_desc)});
  outer_elements.push_back(StructFormatDescriptor::Element{
      "b", std::make_unique<LeafValueFormatDescriptor>(
               FormatPreference::kUnsignedDecimal)});
  StructFormatDescriptor outer_fmt_desc{"OuterStruct",
                                        std::move(outer_elements)};

  XLS_ASSERT_OK_AND_ASSIGN(std::string s,
                           outer.ToFormattedString(outer_fmt_desc));
  EXPECT_EQ(s, R"(OuterStruct {
  a: InnerStruct {
    x: 0xf,
    y: -1
  },
  b: 42
})");
}

TEST(InterpValueTest, MakeMaxValue) {
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/false, 0).ToString(),
            "u0:0");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/true, 0).ToString(),
            "s0:0");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/false, 1).ToString(),
            "u1:1");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/true, 1).ToString(),
            "s1:0");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/false, 2).ToString(),
            "u2:3");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/true, 2).ToString(),
            "s2:1");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/false, 16).ToString(),
            "u16:65535");
  EXPECT_EQ(InterpValue::MakeMaxValue(/*is_signed=*/true, 16).ToString(),
            "s16:32767");
}

}  // namespace
}  // namespace xls::dslx
