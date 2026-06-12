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
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;

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
  EXPECT_EQ(ff.ToString(/*humanize=*/true), "-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/false), "s8:-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kHex), "0xff");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kUnsignedDecimal),
            "255");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kSignedDecimal),
            "-1");
  EXPECT_EQ(ff.ToString(/*humanize=*/true, FormatPreference::kBinary),
            "0b1111_1111");
}

TEST(InterpValueTest, FormatArray) {
  auto a = InterpValue::MakeUBits(/*bit_count=*/12, /*value=*/0xf00);
  auto b = InterpValue::MakeUBits(/*bit_count=*/12, /*value=*/0xba5);
  auto array = InterpValue::MakeArray({a, b}).value();
  EXPECT_EQ(array.ToString(/*humanize=*/true), "[3840, 2981]");
  EXPECT_EQ(array.ToString(/*humanize=*/false), "[u12:3840, u12:2981]");
  EXPECT_EQ(array.ToString(/*humanize=*/true, FormatPreference::kHex),
            "[0xf00, 0xba5]");
  EXPECT_EQ(
      array.ToString(/*humanize=*/true, FormatPreference::kUnsignedDecimal),
      "[3840, 2981]");
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

TEST(InterpValueTest, ConcatRange) {
  auto a = InterpValue::MakeUBits(/*bit_count=*/4, 1);
  auto b = InterpValue::MakeUBits(/*bit_count=*/4, 2);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue range_a, InterpValue::MakeRange({a}));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue array_b, InterpValue::MakeArray({b}));

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue concat, range_a.Concat(array_b));
  // Concatenating ranges don't produce a range, it produces an array.
  EXPECT_FALSE(concat.is_range());
  EXPECT_TRUE(concat.IsArray());

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue concat2, array_b.Concat(range_a));
  EXPECT_FALSE(concat2.is_range());
  EXPECT_TRUE(concat2.IsArray());
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

TEST(InterpValueTest, Array1DUpdate) {
  auto array = InterpValue::MakeArray(
      {InterpValue::MakeU32(1), InterpValue::MakeU32(2)});
  auto index = InterpValue::MakeU8(1);
  EXPECT_EQ(array->Update(index, InterpValue::MakeU32(4))->ToHumanString(),
            "[1, 4]");
}

TEST(InterpValueTest, Array2DUpdate) {
  auto array =
      InterpValue::MakeArray({InterpValue::MakeArray({InterpValue::MakeU32(1),
                                                      InterpValue::MakeU32(2)})
                                  .value(),
                              InterpValue::MakeArray({InterpValue::MakeU32(3),
                                                      InterpValue::MakeU32(4)})
                                  .value()});
  auto indices =
      InterpValue::MakeTuple({InterpValue::MakeU8(1), InterpValue::MakeU32(0)});
  EXPECT_EQ(array->Update(indices, InterpValue::MakeU32(4))->ToHumanString(),
            "[[1, 2], [4, 4]]");
}

TEST(InterpValueTest, Array2DUpdateEmptyIndices) {
  auto array =
      InterpValue::MakeArray({InterpValue::MakeArray({InterpValue::MakeU32(1),
                                                      InterpValue::MakeU32(2)})
                                  .value(),
                              InterpValue::MakeArray({InterpValue::MakeU32(3),
                                                      InterpValue::MakeU32(4)})
                                  .value()});
  auto indices = InterpValue::MakeTuple({});
  auto value =
      InterpValue::MakeArray({InterpValue::MakeArray({InterpValue::MakeU32(4),
                                                      InterpValue::MakeU32(3)})
                                  .value(),
                              InterpValue::MakeArray({InterpValue::MakeU32(2),
                                                      InterpValue::MakeU32(1)})
                                  .value()})
          .value();
  EXPECT_EQ(array->Update(indices, value)->ToHumanString(), "[[4, 3], [2, 1]]");
}

TEST(InterpValueTest, Array2DUpdateArrayWrongDimension) {
  auto array =
      InterpValue::MakeArray({InterpValue::MakeArray({InterpValue::MakeU32(1),
                                                      InterpValue::MakeU32(2)})
                                  .value(),
                              InterpValue::MakeArray({InterpValue::MakeU32(3),
                                                      InterpValue::MakeU32(4)})
                                  .value()});
  auto indices =
      InterpValue::MakeTuple({InterpValue::MakeU8(1), InterpValue::MakeU8(0),
                              InterpValue::MakeU32(0)});
  EXPECT_THAT(array->Update(indices, InterpValue::MakeU32(4)),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Update of non-array element")));
}

TEST(InterpValueTest, Array2DUpdateArrayOutOfBounds) {
  auto array =
      InterpValue::MakeArray({InterpValue::MakeArray({InterpValue::MakeU32(1),
                                                      InterpValue::MakeU32(2)})
                                  .value(),
                              InterpValue::MakeArray({InterpValue::MakeU32(3),
                                                      InterpValue::MakeU32(4)})
                                  .value()});
  auto indices =
      InterpValue::MakeTuple({InterpValue::MakeU8(2), InterpValue::MakeU8(0)});
  EXPECT_THAT(array->Update(indices, InterpValue::MakeU32(4)),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Update index 2 is out of bounds")));
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

  std::vector<ValueFormatDescriptor> elements;
  ValueFormatDescriptor fmt_desc = ValueFormatDescriptor::MakeStruct(
      "MyStruct", {"x"},
      {ValueFormatDescriptor::MakeLeafValue(FormatPreference::kHex)});
  ASSERT_THAT(tuple.ToFormattedString(fmt_desc),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Number of tuple elements (0)")));
}

TEST(InterpValueTest, FormatFlatStructViaDescriptor) {
  auto uf = InterpValue::MakeUBits(/*bit_count=*/4, 0xf);
  auto sf = InterpValue::MakeSBits(/*bit_count=*/4, -1);
  auto tuple = InterpValue::MakeTuple({uf, sf});

  ValueFormatDescriptor fmt_desc = ValueFormatDescriptor::MakeStruct(
      "MyStruct", {"x", "y"},
      {ValueFormatDescriptor::MakeLeafValue(FormatPreference::kHex),
       ValueFormatDescriptor::MakeLeafValue(FormatPreference::kSignedDecimal)});
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
  auto outer = InterpValue::MakeTuple(
      {inner, InterpValue::MakeUBits(/*bit_count=*/32, 42)});

  ValueFormatDescriptor inner_fmt_desc = ValueFormatDescriptor::MakeStruct(
      "InnerStruct", {"x", "y"},
      {ValueFormatDescriptor::MakeLeafValue(FormatPreference::kHex),
       ValueFormatDescriptor::MakeLeafValue(FormatPreference::kSignedDecimal)});
  ValueFormatDescriptor outer_fmt_desc = ValueFormatDescriptor::MakeStruct(
      "OuterStruct", {"a", "b"},
      {inner_fmt_desc,
       ValueFormatDescriptor::MakeLeafValue(FormatPreference::kSignedDecimal)});
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

TEST(InterpValueTest, FormatEnum) {
  constexpr std::string_view kProgram = R"(enum MyEnum : u32 {
    FOO = 0,
    BAR = 1,
})";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * enum_def,
                           module->GetMemberOrError<EnumDef>("MyEnum"));
  InterpValue foo = InterpValue::MakeEnum(UBits(0, 32), false, enum_def);
  InterpValue bar = InterpValue::MakeEnum(UBits(1, 32), false, enum_def);
  EXPECT_EQ(foo.ToString(), "MyEnum:0");
  EXPECT_EQ(bar.ToString(), "MyEnum:1");

  ValueFormatDescriptor fmt_desc = ValueFormatDescriptor::MakeEnum(
      "MyEnum", absl::flat_hash_map<Bits, std::string>(
                    {{UBits(0, 32), "FOO"}, {UBits(1, 32), "BAR"}}));
  EXPECT_EQ(foo.ToFormattedString(fmt_desc).value(), "MyEnum::FOO  // u32:0");
  EXPECT_EQ(bar.ToFormattedString(fmt_desc).value(), "MyEnum::BAR  // u32:1");
}

TEST(InterpValueTest, AsProtoBits) {
  InterpValue iv = InterpValue::MakeU32(0xdeadbeef);
  XLS_ASSERT_OK_AND_ASSIGN(xls::ValueProto proto, iv.AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value round_trip, xls::Value::FromProto(proto));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value expected, iv.ConvertToIr());
  EXPECT_EQ(round_trip, expected);
}

TEST(InterpValueTest, AsProtoArray) {
  InterpValue a = InterpValue::MakeU8(1);
  InterpValue b = InterpValue::MakeU8(2);
  InterpValue c = InterpValue::MakeU8(3);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue array,
                           InterpValue::MakeArray({a, b, c}));

  XLS_ASSERT_OK_AND_ASSIGN(xls::ValueProto proto, array.AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value round_trip, xls::Value::FromProto(proto));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value expected, array.ConvertToIr());
  EXPECT_EQ(round_trip, expected);
}

TEST(InterpValueTest, AsProtoTuple) {
  InterpValue x = InterpValue::MakeU8(255);
  InterpValue y = InterpValue::MakeBool(true);
  InterpValue tup = InterpValue::MakeTuple({x, y});

  XLS_ASSERT_OK_AND_ASSIGN(xls::ValueProto proto, tup.AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value round_trip, xls::Value::FromProto(proto));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value expected, tup.ConvertToIr());
  EXPECT_EQ(round_trip, expected);
}

TEST(InterpValueTest, AsProtoSignedBits) {
  // -2 in 8-bit two's complement.
  InterpValue s = InterpValue::MakeSBits(/*bit_count=*/8, /*value=*/-2);
  XLS_ASSERT_OK_AND_ASSIGN(xls::ValueProto proto, s.AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value round_trip, xls::Value::FromProto(proto));
  // Note: Proto loses signedness;  SBits constructs a twos-complement
  // representation of the signed value.
  EXPECT_EQ(round_trip, xls::Value(SBits(-2, 8)));
}

TEST(InterpValueTest, AsProtoEnum) {
  // Define a tiny enum and create an InterpValue::Enum.
  constexpr std::string_view kProgram = R"(enum MyEnum : u3 {
    A = 0,
    B = 5,
  })";

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(auto module, ParseModule(kProgram, "fake_path.x",
                                                    "the_module", file_table));
  XLS_ASSERT_OK_AND_ASSIGN(EnumDef * enum_def,
                           module->GetMemberOrError<EnumDef>("MyEnum"));

  Bits bits_val = UBits(/*value=*/5, /*bit_count=*/3);
  InterpValue enum_value =
      InterpValue::MakeEnum(bits_val, /*is_signed=*/false, enum_def);

  XLS_ASSERT_OK_AND_ASSIGN(xls::ValueProto proto, enum_value.AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(xls::Value round_trip, xls::Value::FromProto(proto));
  // Expected IR value is just the raw bits (enum metadata is not preserved).
  EXPECT_EQ(round_trip, xls::Value(UBits(5, 3)));
}

TEST(InterpValueTest, RangeGetLengthWideBitsCrash) {
  // Test exclusive range with bit-width < 63 does not crash.
  InterpValue start = InterpValue::MakeUBits(32, 0);
  InterpValue end = InterpValue::MakeUBits(32, 10);
  InterpValue range =
      InterpValue::MakeSymbolicRange(start, end, /*inclusive=*/false);
  EXPECT_TRUE(range.is_range());
  EXPECT_THAT(range.GetLength(), IsOkAndHolds(10));
}

TEST(InterpValueTest, RangeGetLengthWideBitsCrashInclusive) {
  // Test inclusive range with bit-width < 63 does not crash.
  InterpValue start = InterpValue::MakeUBits(32, 0);
  InterpValue end = InterpValue::MakeUBits(32, 10);
  InterpValue range =
      InterpValue::MakeSymbolicRange(start, end, /*inclusive=*/true);
  EXPECT_TRUE(range.is_range());
  EXPECT_THAT(range.GetLength(), IsOkAndHolds(11));
}

TEST(InterpValueTest, ChannelArrayBasicProperties) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  auto chan_ref2 = InterpValue::MakeChannelReference(ChannelDirection::kOut, 2);

  auto chan_arr = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1, chan_ref2});

  EXPECT_TRUE(chan_arr.IsChannelArray());
  EXPECT_FALSE(chan_arr.IsArray());
  EXPECT_FALSE(chan_arr.IsChannelReference());

  EXPECT_THAT(chan_arr.GetLength(), IsOkAndHolds(2));

  XLS_ASSERT_OK_AND_ASSIGN(const InterpValue::ChannelArray* ca,
                           chan_arr.GetChannelArray());
  EXPECT_EQ(ca->channel_array_id(), 42);
  EXPECT_EQ(ca->definer(), nullptr);
  EXPECT_EQ(ca->elements().size(), 2);
  EXPECT_EQ(ca->elements()[0], chan_ref1);
  EXPECT_EQ(ca->elements()[1], chan_ref2);

  XLS_ASSERT_OK_AND_ASSIGN(const std::vector<InterpValue>* values,
                           chan_arr.GetValues());
  EXPECT_EQ(values->size(), 2);
  EXPECT_EQ((*values)[0], chan_ref1);
}

TEST(InterpValueTest, ChannelArrayIndexing) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  auto chan_ref2 = InterpValue::MakeChannelReference(ChannelDirection::kOut, 2);

  auto chan_arr = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1, chan_ref2});

  EXPECT_THAT(chan_arr.Index(0), IsOkAndHolds(chan_ref1));
  EXPECT_THAT(chan_arr.Index(1), IsOkAndHolds(chan_ref2));
  EXPECT_FALSE(chan_arr.Index(2).ok());
}

TEST(InterpValueTest, ChannelArrayEqualityAndComparison) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  auto chan_ref2 = InterpValue::MakeChannelReference(ChannelDirection::kOut, 2);

  auto chan_arr = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1, chan_ref2});

  auto chan_arr_same_id = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1});
  auto chan_arr_diff_id = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 43, nullptr, {chan_ref1, chan_ref2});
  EXPECT_EQ(chan_arr, chan_arr_same_id);
  EXPECT_NE(chan_arr, chan_arr_diff_id);

  EXPECT_LT(chan_arr, chan_arr_diff_id);
  EXPECT_FALSE(chan_arr_diff_id < chan_arr);

  auto reg_array = InterpValue::MakeArray({chan_ref1}).value();
  EXPECT_LT(reg_array, chan_arr);
  EXPECT_FALSE(chan_arr < reg_array);
}

TEST(InterpValueTest, ChannelArrayToString) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  auto chan_ref2 = InterpValue::MakeChannelReference(ChannelDirection::kOut, 2);

  auto chan_arr = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1, chan_ref2});

  EXPECT_EQ(chan_arr.ToString(),
            "channel_array(in, channel_array_id=42, definer=none, "
            "elements=[channel_reference(in, "
            "channel_instance_id=1), "
            "channel_reference(out, channel_instance_id=2)])");
}

TEST(InterpValueTest, ChannelArrayConvertToIr) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);
  auto chan_ref2 = InterpValue::MakeChannelReference(ChannelDirection::kOut, 2);

  auto chan_arr = InterpValue::MakeChannelArray(
      ChannelDirection::kIn, 42, nullptr, {chan_ref1, chan_ref2});

  EXPECT_FALSE(chan_arr.ConvertToIr().ok());
}

TEST(InterpValueTest, ChannelArrayValidationMismatchedTypesDeathTest) {
  auto chan_ref1 = InterpValue::MakeChannelReference(ChannelDirection::kIn, 1);

  auto bits_val = InterpValue::MakeU32(1);
  EXPECT_DEATH(InterpValue::MakeChannelArray(ChannelDirection::kIn, 1, nullptr,
                                             {chan_ref1, bits_val}),
               "ChannelArray elements must be either all ChannelArray or all "
               "ChannelReference");

  auto inner_arr = InterpValue::MakeChannelArray(ChannelDirection::kIn, 2,
                                                 nullptr, {chan_ref1});
  auto nested_arr = InterpValue::MakeChannelArray(ChannelDirection::kIn, 3,
                                                  nullptr, {inner_arr});
  EXPECT_TRUE(nested_arr.IsChannelArray());

  EXPECT_DEATH(InterpValue::MakeChannelArray(ChannelDirection::kIn, 4, nullptr,
                                             {inner_arr, chan_ref1}),
               "ChannelArray elements must be either all ChannelArray or all "
               "ChannelReference");
}

}  // namespace
}  // namespace xls::dslx
