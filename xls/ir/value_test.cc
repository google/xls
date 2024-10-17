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

#include "xls/ir/value.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/xls_value.pb.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::HasSubstr;
using ::xls::proto_testing::EqualsProto;

TEST(ValueTest, ToHumanString) {
  Value bits_value(UBits(42, 33));
  EXPECT_EQ(bits_value.ToHumanString(), "42");

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value,
                           Value::Array({Value(UBits(3, 8)), Value(UBits(4, 8)),
                                         Value(UBits(5, 8))}));
  EXPECT_EQ(array_value.ToHumanString(), "[3, 4, 5]");

  XLS_ASSERT_OK_AND_ASSIGN(Value nested_array_value,
                           Value::Array({array_value, array_value}));
  EXPECT_EQ(nested_array_value.ToHumanString(), "[[3, 4, 5], [3, 4, 5]]");

  Value tuple_value =
      Value::Tuple({array_value, Value(UBits(42, 8)), Value(UBits(123, 8))});
  EXPECT_EQ(tuple_value.ToHumanString(), "([3, 4, 5], 42, 123)");

  Value token_value = Value::Token();
  EXPECT_EQ(token_value.ToHumanString(), "token");
}

TEST(ValueTest, ToString) {
  Value bits_value(UBits(42, 33));
  EXPECT_EQ(bits_value.ToString(), "bits[33]:42");

  XLS_ASSERT_OK_AND_ASSIGN(Value array_value,
                           Value::Array({Value(UBits(3, 8)), Value(UBits(4, 8)),
                                         Value(UBits(5, 8))}));
  EXPECT_EQ(array_value.ToString(), "[bits[8]:3, bits[8]:4, bits[8]:5]");

  XLS_ASSERT_OK_AND_ASSIGN(Value nested_array_value,
                           Value::Array({array_value, array_value}));
  EXPECT_EQ(
      nested_array_value.ToString(),
      "[[bits[8]:3, bits[8]:4, bits[8]:5], [bits[8]:3, bits[8]:4, bits[8]:5]]");

  Value tuple_value =
      Value::Tuple({array_value, Value(UBits(42, 17)), Value(UBits(123, 33))});
  EXPECT_EQ(tuple_value.ToString(),
            "([bits[8]:3, bits[8]:4, bits[8]:5], bits[17]:42, bits[33]:123)");

  Value token_value = Value::Token();
  EXPECT_EQ(token_value.ToHumanString(), "token");
}

TEST(ValueTest, SameTypeAs) {
  Value b1(UBits(42, 33));
  Value b2(UBits(42, 10));
  Value b3(UBits(0, 33));
  EXPECT_TRUE(b1.SameTypeAs(b1));
  EXPECT_FALSE(b1.SameTypeAs(b2));
  EXPECT_TRUE(b1.SameTypeAs(b3));

  Value tuple1 = Value::Tuple({b1, b2});
  Value tuple2 = Value::Tuple({b1, b2});
  Value tuple3 = Value::Tuple({b1, b2, b3});
  EXPECT_TRUE(tuple1.SameTypeAs(tuple1));
  EXPECT_TRUE(tuple1.SameTypeAs(tuple2));
  EXPECT_FALSE(tuple1.SameTypeAs(tuple3));

  XLS_ASSERT_OK_AND_ASSIGN(Value array1, Value::Array({b1, b3}));
  XLS_ASSERT_OK_AND_ASSIGN(Value array2, Value::Array({b3, b1}));
  XLS_ASSERT_OK_AND_ASSIGN(Value array3, Value::Array({b1, b3, b3}));
  EXPECT_TRUE(array1.SameTypeAs(array1));
  EXPECT_TRUE(array1.SameTypeAs(array2));
  EXPECT_FALSE(array1.SameTypeAs(array3));

  Value token_a = Value::Token();
  Value token_b = Value::Token();
  EXPECT_TRUE(token_a.SameTypeAs(token_b));

  EXPECT_FALSE(b1.SameTypeAs(tuple1));
  EXPECT_FALSE(b1.SameTypeAs(array1));
  EXPECT_FALSE(b1.SameTypeAs(token_a));
}

TEST(ValueTest, IsAllZeroOnes) {
  EXPECT_TRUE(Value(UBits(0, 0)).IsAllZeros());
  EXPECT_TRUE(Value(UBits(0, 0)).IsAllOnes());

  EXPECT_TRUE(Value(UBits(0, 1)).IsAllZeros());
  EXPECT_FALSE(Value(UBits(0, 1)).IsAllOnes());

  EXPECT_FALSE(Value(UBits(1, 1)).IsAllZeros());
  EXPECT_TRUE(Value(UBits(1, 1)).IsAllOnes());

  EXPECT_TRUE(Value(UBits(0, 8)).IsAllZeros());
  EXPECT_FALSE(Value(UBits(0, 8)).IsAllOnes());

  EXPECT_FALSE(Value(UBits(255, 8)).IsAllZeros());
  EXPECT_TRUE(Value(UBits(255, 8)).IsAllOnes());

  EXPECT_FALSE(Value(UBits(123, 32)).IsAllZeros());
  EXPECT_FALSE(Value(UBits(123, 32)).IsAllOnes());

  EXPECT_TRUE(Value::ArrayOrDie({Value(UBits(0, 32)), Value(UBits(0, 32)),
                                 Value(UBits(0, 32))})
                  .IsAllZeros());
  EXPECT_FALSE(Value::ArrayOrDie({Value(UBits(0, 32)), Value(UBits(1234, 32)),
                                  Value(UBits(0, 32))})
                   .IsAllZeros());

  EXPECT_TRUE(Value::ArrayOrDie({Value(UBits(127, 7)), Value(UBits(127, 7)),
                                 Value(UBits(127, 7))})
                  .IsAllOnes());
  EXPECT_FALSE(Value::ArrayOrDie({Value(UBits(127, 7)), Value(UBits(126, 7)),
                                  Value(UBits(127, 7))})
                   .IsAllOnes());

  EXPECT_TRUE(Value::Tuple({}).IsAllZeros());
  EXPECT_TRUE(Value::Tuple({}).IsAllOnes());

  EXPECT_TRUE(
      Value::Tuple({Value(UBits(0, 3)), Value(UBits(0, 7)), Value(UBits(0, 1))})
          .IsAllZeros());
  EXPECT_FALSE(
      Value::Tuple({Value(UBits(0, 3)), Value(UBits(1, 7)), Value(UBits(0, 1))})
          .IsAllZeros());

  EXPECT_TRUE(Value::Tuple({Value(UBits(7, 3)), Value(UBits(127, 7)),
                            Value(UBits(1, 1))})
                  .IsAllOnes());
  EXPECT_FALSE(Value::Tuple({Value(UBits(7, 3)), Value(UBits(121, 7)),
                             Value(UBits(1, 1))})
                   .IsAllOnes());
}

TEST(ValueTest, XBitsArray) {
  Value v0;

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBitsArray({0}, 1));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 1);
  EXPECT_EQ(v0.element(0).ToString(), "bits[1]:0");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBitsArray({1}, 1));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 1);
  EXPECT_EQ(v0.element(0).ToString(), "bits[1]:1");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::SBitsArray({-1}, 2));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 1);
  EXPECT_EQ(v0.element(0).ToString(), "bits[2]:3");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::SBitsArray({1}, 2));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 1);
  EXPECT_EQ(v0.element(0).ToString(), "bits[2]:1");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBitsArray({0, 1}, 4));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 2);
  EXPECT_EQ(v0.element(0).ToString(), "bits[4]:0");
  EXPECT_EQ(v0.element(1).ToString(), "bits[4]:1");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBitsArray({1, 2, 3}, 4));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 3);
  EXPECT_EQ(v0.element(0).ToString(), "bits[4]:1");
  EXPECT_EQ(v0.element(1).ToString(), "bits[4]:2");
  EXPECT_EQ(v0.element(2).ToString(), "bits[4]:3");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::SBitsArray({-1, 0}, 4));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 2);
  EXPECT_EQ(v0.element(0).ToString(), "bits[4]:15");
  EXPECT_EQ(v0.element(1).ToString(), "bits[4]:0");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::SBitsArray({1, 2, 3, 4}, 4));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 4);
  EXPECT_EQ(v0.element(0).ToString(), "bits[4]:1");
  EXPECT_EQ(v0.element(1).ToString(), "bits[4]:2");
  EXPECT_EQ(v0.element(2).ToString(), "bits[4]:3");
  EXPECT_EQ(v0.element(3).ToString(), "bits[4]:4");
}

TEST(ValueTest, XBits2DArray) {
  Value v0;

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBits2DArray({{0}, {1}}, 1));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 2);
  EXPECT_TRUE(v0.element(0).IsArray());
  EXPECT_TRUE(v0.element(1).IsArray());
  EXPECT_EQ(v0.element(0).size(), 1);
  EXPECT_EQ(v0.element(1).size(), 1);
  EXPECT_EQ(v0.element(0).element(0).ToString(), "bits[1]:0");
  EXPECT_EQ(v0.element(1).element(0).ToString(), "bits[1]:1");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::UBits2DArray({{0, 1}, {2, 3}}, 2));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 2);
  EXPECT_TRUE(v0.element(0).IsArray());
  EXPECT_TRUE(v0.element(1).IsArray());
  EXPECT_EQ(v0.element(0).size(), 2);
  EXPECT_EQ(v0.element(1).size(), 2);
  EXPECT_EQ(v0.element(0).element(0).ToString(), "bits[2]:0");
  EXPECT_EQ(v0.element(0).element(1).ToString(), "bits[2]:1");
  EXPECT_EQ(v0.element(1).element(0).ToString(), "bits[2]:2");
  EXPECT_EQ(v0.element(1).element(1).ToString(), "bits[2]:3");

  XLS_ASSERT_OK_AND_ASSIGN(v0, Value::SBits2DArray({{0}, {1}, {-1}}, 2));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 3);
  EXPECT_TRUE(v0.element(0).IsArray());
  EXPECT_TRUE(v0.element(1).IsArray());
  EXPECT_TRUE(v0.element(2).IsArray());
  EXPECT_EQ(v0.element(0).size(), 1);
  EXPECT_EQ(v0.element(1).size(), 1);
  EXPECT_EQ(v0.element(2).size(), 1);
  EXPECT_EQ(v0.element(0).element(0).ToString(), "bits[2]:0");
  EXPECT_EQ(v0.element(1).element(0).ToString(), "bits[2]:1");
  EXPECT_EQ(v0.element(2).element(0).ToString(), "bits[2]:3");

  XLS_ASSERT_OK_AND_ASSIGN(v0,
                           Value::SBits2DArray({{0, 1}, {2, 3}, {4, 5}}, 4));
  EXPECT_TRUE(v0.IsArray());
  EXPECT_EQ(v0.size(), 3);
  EXPECT_TRUE(v0.element(0).IsArray());
  EXPECT_TRUE(v0.element(1).IsArray());
  EXPECT_TRUE(v0.element(2).IsArray());
  EXPECT_EQ(v0.element(0).size(), 2);
  EXPECT_EQ(v0.element(1).size(), 2);
  EXPECT_EQ(v0.element(2).size(), 2);
  EXPECT_EQ(v0.element(0).element(0).ToString(), "bits[4]:0");
  EXPECT_EQ(v0.element(0).element(1).ToString(), "bits[4]:1");
  EXPECT_EQ(v0.element(1).element(0).ToString(), "bits[4]:2");
  EXPECT_EQ(v0.element(1).element(1).ToString(), "bits[4]:3");
  EXPECT_EQ(v0.element(2).element(0).ToString(), "bits[4]:4");
  EXPECT_EQ(v0.element(2).element(1).ToString(), "bits[4]:5");
}

TEST(ValueTest, XBitsArrayEmpty) {
  // Test empty array values
  auto v0 = Value::UBitsArray(absl::Span<uint64_t>({}), 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::SBitsArray(absl::Span<int64_t>({}), 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::UBits2DArray(absl::Span<const absl::Span<const uint64_t>>({}), 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::SBits2DArray(absl::Span<const absl::Span<const int64_t>>({}), 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  // Test partially empty values (2D Arrays)
  v0 = Value::UBits2DArray(
      absl::Span<const absl::Span<const uint64_t>>({{}, {1}}), 2);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::SBits2DArray(
      absl::Span<const absl::Span<const int64_t>>({{}, {1}}), 2);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::UBits2DArray(
      absl::Span<const absl::Span<const uint64_t>>({{1}, {}}), 2);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");

  v0 = Value::SBits2DArray(
      absl::Span<const absl::Span<const int64_t>>({{1}, {}}), 2);
  EXPECT_FALSE(v0.ok());
  EXPECT_EQ(v0.status().message(), "Empty array Values are not supported.");
}

TEST(ValueTest, XBitsArrayWrongSizes) {
  // Test bit sizes
  auto v0 = Value::UBitsArray({2}, 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(), HasSubstr("Value 0x2 requires 2 bits"));

  v0 = Value::SBitsArray({1}, 1);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(), HasSubstr("Value 0x1 requires 2 bits"));

  // Test partially arrays sizes (2D Arrays)
  v0 = Value::UBits2DArray({{1, 2}, {3}}, 4);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(),
              HasSubstr("elements of arrays should have consistent size."));

  v0 = Value::SBits2DArray({{1, 2}, {3}}, 4);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(),
              HasSubstr("elements of arrays should have consistent size."));

  v0 = Value::UBits2DArray({{1}, {2, 3}}, 4);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(),
              HasSubstr("elements of arrays should have consistent size."));

  v0 = Value::SBits2DArray({{1}, {2, 3}}, 4);
  EXPECT_FALSE(v0.ok());
  EXPECT_THAT(v0.status().message(),
              HasSubstr("elements of arrays should have consistent size."));
}

TEST(ValueTest, ToProtoBits) {
  {
    std::string_view expected_txt = R"pb(
      bits { bit_count: 8, data: "A" }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(UBits('A', 8));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }

  // endianness
  {
    std::string_view expected_txt = R"pb(
      bits { bit_count: 24, data: "ABC" }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(UBits(0x434241, 24));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }

  // >64 bits
  {
    std::string_view expected_txt = R"pb(
      bits { bit_count: 256, data: "ABCDHIJKLMNOPQRSTUVWXYZABCDEFHIJ" }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value v, Parser::ParseTypedValue(
                     "bits[256]:0x4a49_4846_4544_4342_415a_5958_5756_5554_5352_"
                     "5150_4f4e_4d4c_4b4a_4948_4443_4241"));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }

  // half-byte
  {
    std::string_view expected_txt = R"pb(
      bits { bit_count: 7, data: "A" }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(UBits(0x41, 7));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
}

TEST(ValueTest, ToProtoToken) {
  std::string_view expected_txt = R"pb(
    token {}
  )pb";
  ValueProto expected;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
  Value v = Value::Token();

  EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
}

TEST(ValueTest, ToProtoTuple) {
  // standard
  {
    std::string_view expected_txt = R"pb(
      tuple {
        elements { bits { bit_count: 8, data: "A" } }
        elements { bits { bit_count: 7, data: "B" } }
      }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(Value::Tuple({Value(UBits('A', 8)), Value(UBits('B', 7))}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
  // empty
  {
    std::string_view expected_txt = R"pb(
      tuple {}
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(Value::Tuple({}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
  // tuple in tuple
  {
    std::string_view expected_txt = R"pb(
      tuple {
        elements { tuple { elements { bits { bit_count: 8, data: "A" } } } }
        elements { bits { bit_count: 8, data: "B" } }
      }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    Value v(Value::Tuple(
        {Value::Tuple({Value(UBits('A', 8))}), Value(UBits('B', 8))}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
}

TEST(ValueTest, ToProtoArray) {
  // standard
  {
    std::string_view expected_txt = R"pb(
      array {
        elements { bits { bit_count: 8, data: "A" } }
        elements { bits { bit_count: 8, data: "B" } }
      }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value v, Value::Array({Value(UBits('A', 8)), Value(UBits('B', 8))}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
  // tuple in array
  {
    std::string_view expected_txt = R"pb(
      array {
        elements { tuple { elements { bits { bit_count: 8, data: "A" } } } }
        elements { tuple { elements { bits { bit_count: 8, data: "B" } } } }
      }
    )pb";
    ValueProto expected;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(expected_txt, &expected));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value v, Value::Array({Value::Tuple({Value(UBits('A', 8))}),
                               Value::Tuple({Value(UBits('B', 8))})}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(expected)));
  }
}

TEST(ValueTest, FromProtoBits) {
  // normal
  {
    std::string_view source_txt = R"pb(
      bits { bit_count: 8, data: "A" }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(UBits(0x41, 8));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }

  // partial byte
  {
    std::string_view source_txt = R"pb(
      bits { bit_count: 7, data: "A" }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(UBits(0x41, 7));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }
  // Extend with zeros
  {
    std::string_view source_txt = R"pb(
      bits { bit_count: 256, data: "A" }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(UBits(0x41, 256));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }

  // ignore extra
  {
    std::string_view source_txt = R"pb(
      bits { bit_count: 8, data: "AZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ" }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(UBits(0x41, 8));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }
}

TEST(ValueTest, FromProtoToken) {
  std::string_view source_txt = R"pb(
    token {}
  )pb";
  ValueProto source;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
  Value v = Value::Token();

  EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
}

TEST(ValueTest, FromProtoTuple) {
  // standard
  {
    std::string_view source_txt = R"pb(
      tuple {
        elements { bits { bit_count: 8, data: "A" } }
        elements { bits { bit_count: 7, data: "B" } }
      }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(Value::Tuple({Value(UBits('A', 8)), Value(UBits('B', 7))}));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }
  // empty
  {
    std::string_view source_txt = R"pb(
      tuple {}
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(Value::Tuple({}));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }
  // tuple in tuple
  {
    std::string_view source_txt = R"pb(
      tuple {
        elements { tuple { elements { bits { bit_count: 8, data: "A" } } } }
        elements { bits { bit_count: 8, data: "B" } }
      }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    Value v(Value::Tuple(
        {Value::Tuple({Value(UBits('A', 8))}), Value(UBits('B', 8))}));

    EXPECT_THAT(Value::FromProto(source), IsOkAndHolds(v));
  }
}

TEST(ValueTest, FromProtoArray) {
  // standard
  {
    std::string_view source_txt = R"pb(
      array {
        elements { bits { bit_count: 8, data: "A" } }
        elements { bits { bit_count: 8, data: "B" } }
      }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value v, Value::Array({Value(UBits('A', 8)), Value(UBits('B', 8))}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(source)));
  }
  // tuple in array
  {
    std::string_view source_txt = R"pb(
      array {
        elements { tuple { elements { bits { bit_count: 8, data: "A" } } } }
        elements { tuple { elements { bits { bit_count: 8, data: "B" } } } }
      }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    XLS_ASSERT_OK_AND_ASSIGN(
        Value v, Value::Array({Value::Tuple({Value(UBits('A', 8))}),
                               Value::Tuple({Value(UBits('B', 8))})}));

    EXPECT_THAT(v.AsProto(), IsOkAndHolds(EqualsProto(source)));
  }
  // No empty
  {
    std::string_view source_txt = R"pb(
      array {}
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    EXPECT_THAT(Value::FromProto(source),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }
  // No type-mismatch
  {
    std::string_view source_txt = R"pb(
      array {
        elements { tuple { elements { bits { bit_count: 8, data: "A" } } } }
        elements { tuple { elements { bits { bit_count: 7, data: "B" } } } }
      }
    )pb";
    ValueProto source;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(source_txt, &source));
    EXPECT_THAT(Value::FromProto(source),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

void ProtoValueRoundTripWorks(const ValueProto& v) {
  auto value = Value::FromProto(v, /*max_bit_size=*/1 << 16);
  if (!value.ok()) {
    // Don't bother with protos with invalid arrays and such.
    return;
  }

  XLS_ASSERT_OK_AND_ASSIGN(ValueProto proto, value->AsProto());
  XLS_ASSERT_OK_AND_ASSIGN(Value restored, Value::FromProto(proto));
  XLS_ASSERT_OK_AND_ASSIGN(ValueProto proto2, restored.AsProto());
  ASSERT_THAT(proto2, EqualsProto(proto));
  ASSERT_EQ(*value, restored);
}

FUZZ_TEST(ValueProto, ProtoValueRoundTripWorks)
    .WithDomains(fuzztest::Arbitrary<ValueProto>());
}  // namespace

}  // namespace xls
