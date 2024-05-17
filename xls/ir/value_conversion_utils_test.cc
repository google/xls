// Copyright 2022 The XLS Authors
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

#include "xls/ir/value_conversion_utils.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;

TEST(ValueConversionUtils, ConvertCppToXlsValueSignedIntegrals) {
  BitsType type(10);
  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<int64_t>(42)),
              IsOkAndHolds(Value(SBits(42, 10))));

  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<int32_t>(42)),
              IsOkAndHolds(Value(SBits(42, 10))));

  ArrayType invalid_type(2, &type);
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_type, static_cast<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for integral input value")));

  BitsType invalid_bit_count(2);
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_bit_count, static_cast<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueUnsignedIntegrals) {
  BitsType type(10);
  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<uint64_t>(42)),
              IsOkAndHolds(Value(UBits(42, 10))));

  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<uint32_t>(42)),
              IsOkAndHolds(Value(UBits(42, 10))));

  ArrayType invalid_type(2, &type);
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_type, static_cast<uint64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for integral input value")));

  BitsType invalid_bit_count(2);
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_bit_count, static_cast<uint64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_golden,
      Value::Array({Value(UBits(64, 10)), Value(UBits(42, 10))}));
  Package p("package");
  ArrayType* array_type = p.GetArrayType(2, p.GetBitsType(10));
  EXPECT_THAT(xls::ConvertToXlsValue(array_type, std::vector<int64_t>{64, 42}),
              IsOkAndHolds(array_golden));

  TupleType invalid_type({p.GetBitsType(10), p.GetBitsType(10)});
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_type, std::vector<int64_t>{64, 42}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Invalid type conversion for an absl::Span")));

  ArrayType size_mismatch_type(10, p.GetBitsType(10));
  EXPECT_THAT(
      xls::ConvertToXlsValue(&size_mismatch_type, std::vector{64, 42}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Array size mismatch between conversion type and value")));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueTuple) {
  Package p("package");
  Type* tuple_type = p.GetTupleType({p.GetBitsType(10)});
  EXPECT_THAT(xls::ConvertToXlsValue(tuple_type, std::make_tuple<int64_t>(42)),
              IsOkAndHolds(Value::Tuple({Value(UBits(42, 10))})));

  ArrayType invalid_type(10, p.GetBitsType(10));
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_type, std::make_tuple<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid type conversion for a std::tuple")));

  TupleType size_mismatch_type({p.GetBitsType(10), p.GetBitsType(10)});
  EXPECT_THAT(
      xls::ConvertToXlsValue(&size_mismatch_type, std::make_tuple<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Tuple size mismatch between conversion type and value")));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueEmptyTuple) {
  TupleType tuple_type({});
  EXPECT_THAT(xls::ConvertToXlsValue(&tuple_type, std::make_tuple()),
              IsOkAndHolds(Value::Tuple({})));

  BitsType invalid_type(10);
  EXPECT_THAT(
      xls::ConvertToXlsValue(&invalid_type, std::make_tuple()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid type conversion for a std::tuple")));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueTupleWithArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value element1,
      Value::Array({Value(UBits(42, 10)), Value(UBits(64, 10))}));
  Package p("package");
  TupleType tuple_type(
      {p.GetBitsType(10), p.GetArrayType(2, p.GetBitsType(10))});
  absl::StatusOr<Value> result_or = xls::ConvertToXlsValue(
      &tuple_type, std::make_tuple<uint64_t, absl::Span<const uint64_t>>(
                       64, std::vector<uint64_t>{42, 64}));
  EXPECT_THAT(result_or,
              IsOkAndHolds(Value::Tuple({Value(UBits(64, 10)), element1})));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueTupleWithValueAsElement) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value element1,
      Value::Array({Value(UBits(42, 10)), Value(UBits(64, 10))}));
  Package p("package");
  TupleType tuple_type(
      {p.GetBitsType(10), p.GetArrayType(2, p.GetBitsType(10))});
  absl::StatusOr<Value> result_or =
      xls::ConvertToXlsValue(&tuple_type, std::make_tuple(64, element1));
  EXPECT_THAT(result_or,
              IsOkAndHolds(Value::Tuple({Value(UBits(64, 10)), element1})));
}

struct UserStruct {
  int64_t integer_value;
  bool boolean_value;
};

bool operator==(const UserStruct& lhs, const UserStruct& rhs) {
  return lhs.integer_value == rhs.integer_value &&
         lhs.boolean_value == rhs.boolean_value;
}

absl::StatusOr<xls::Value> ConvertToXlsValue(const Type* type,
                                             const UserStruct& user_struct) {
  return xls::ConvertToXlsValue(
      type, std::make_tuple(user_struct.integer_value,
                            static_cast<uint64_t>(user_struct.boolean_value)));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueUserStruct) {
  Package p("package");
  TupleType* user_struct_type =
      p.GetTupleType({p.GetBitsType(10), p.GetBitsType(1)});
  EXPECT_THAT(
      ConvertToXlsValue(user_struct_type, UserStruct{42, false}),
      IsOkAndHolds(Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)})));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueArrayOfUserStruct) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_golden,
      Value::Array({Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)}),
                    Value::Tuple({Value(UBits(64, 10)), Value::Bool(true)})}));
  Package p("package");
  ArrayType* array_type =
      p.GetArrayType(2, p.GetTupleType({p.GetBitsType(10), p.GetBitsType(1)}));
  EXPECT_THAT(
      xls::ConvertToXlsValue(
          array_type, std::vector{UserStruct{42, false}, UserStruct{64, true}}),
      IsOkAndHolds(array_golden));
}

TEST(ValueConversionUtils, ConvertCppToXlsValueNegativeValues) {
  BitsType type(10);
  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<int64_t>(-1)),
              IsOkAndHolds(Value(SBits(-1, 10))));

  EXPECT_THAT(xls::ConvertToXlsValue(&type, static_cast<uint64_t>(-1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppBool) {
  bool cpp_value;
  cpp_value = false;
  Value value_true = Value::Bool(true);
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value_true, cpp_value));
  EXPECT_TRUE(cpp_value);
  cpp_value = true;
  Value value_false = Value::Bool(false);
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value_false, cpp_value));
  EXPECT_FALSE(cpp_value);

  Value invalid_type = Value::Tuple({value_true});
  EXPECT_THAT(
      xls::ConvertFromXlsValue(invalid_type, cpp_value),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid type conversion for bool input.")));

  Value invalid_bit_count(SBits(1024, 12));
  EXPECT_THAT(xls::ConvertFromXlsValue(invalid_bit_count, cpp_value),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Value does not fit in bool type.")));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppSignedIntegrals) {
  Value value(SBits(42, 10));
  int64_t cpp_value = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value));
  EXPECT_EQ(cpp_value, 42);
  int32_t cpp_value_32 = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value_32));
  EXPECT_EQ(cpp_value_32, 42);

  Value invalid_type = Value::Tuple({value});
  EXPECT_THAT(
      xls::ConvertFromXlsValue(invalid_type, cpp_value),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for signed integral input.")));

  int8_t cpp_value_8 = 0;
  Value invalid_bit_count(SBits(1024, 12));
  EXPECT_THAT(xls::ConvertFromXlsValue(invalid_bit_count, cpp_value_8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Value does not fit in signed integral type.")));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppUnsignedIntegrals) {
  Value value(UBits(42, 10));
  uint64_t cpp_value = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value));
  EXPECT_EQ(cpp_value, 42);
  uint32_t cpp_value_32 = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value_32));
  EXPECT_EQ(cpp_value_32, 42);

  Value invalid_type = Value::Tuple({value});
  EXPECT_THAT(
      xls::ConvertFromXlsValue(invalid_type, cpp_value),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for unsigned integral input.")));

  uint8_t cpp_value_8 = 0;
  Value invalid_bit_count(UBits(1024, 12));
  EXPECT_THAT(xls::ConvertFromXlsValue(invalid_bit_count, cpp_value_8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Value does not fit in unsigned integral type.")));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array, Value::Array({Value(UBits(64, 10)), Value(UBits(42, 10))}));
  std::vector<uint64_t> cpp_array;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(array, cpp_array));
  EXPECT_THAT(cpp_array, ElementsAre(64, 42));

  Value invalid_type = Value::Tuple({Value(UBits(64, 10))});
  EXPECT_THAT(xls::ConvertFromXlsValue(invalid_type, cpp_array),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Invalid type conversion for std::vector input.")));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppTuple) {
  Value tuple = Value::Tuple({Value(UBits(42, 10))});
  int64_t cpp_value = 0;
  std::tuple<int64_t&> cpp_tuple(cpp_value);
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(tuple, cpp_tuple));
  EXPECT_EQ(cpp_value, 42);

  XLS_ASSERT_OK_AND_ASSIGN(
      Value invalid_type,
      Value::Array({Value(UBits(64, 10)), Value(UBits(42, 10))}));
  EXPECT_THAT(xls::ConvertFromXlsValue(invalid_type, cpp_tuple),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Invalid type conversion for std::tuple input.")));

  Value size_mismatch =
      Value::Tuple({Value(UBits(42, 10)), Value(UBits(64, 10))});
  EXPECT_THAT(
      xls::ConvertFromXlsValue(size_mismatch, cpp_tuple),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Tuple size mismatch between conversion type and value.")));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppEmptyTuple) {
  Value tuple = Value::Tuple({});
  std::tuple<> cpp_tuple;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(tuple, cpp_tuple));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppTupleWithArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value element1,
      Value::Array({Value(UBits(42, 10)), Value(UBits(64, 10))}));
  Value tuple = Value::Tuple({Value(UBits(64, 10)), element1});
  int64_t cpp_value = 0;
  std::vector<int64_t> cpp_array;
  std::tuple<int64_t&, std::vector<int64_t>&> cpp_tuple(cpp_value, cpp_array);
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(tuple, cpp_tuple));
  EXPECT_EQ(cpp_value, 64);
  EXPECT_THAT(cpp_array, ElementsAre(42, 64));
}

absl::Status ConvertFromXlsValue(const xls::Value& user_struct_value,
                                 UserStruct& user_struct) {
  return xls::ConvertFromXlsValue(
      user_struct_value,
      std::tuple<int64_t&, bool&>(user_struct.integer_value,
                                  user_struct.boolean_value));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppUserStruct) {
  Value user_struct_value =
      Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)});
  UserStruct user_struct_cpp;
  XLS_EXPECT_OK(ConvertFromXlsValue(user_struct_value, user_struct_cpp));
  EXPECT_EQ(user_struct_cpp, (UserStruct{42, false}));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppArrayOfUserStruct) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_value,
      Value::Array({Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)}),
                    Value::Tuple({Value(UBits(64, 10)), Value::Bool(true)})}));
  std::vector<UserStruct> array_cpp;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(array_value, array_cpp));
  EXPECT_THAT(array_cpp,
              ElementsAre(UserStruct{42, false}, UserStruct{64, true}));
}

TEST(ValueConversionUtils, ConvertXlsValueToCppNegativeValues) {
  Value value = Value(SBits(-1, 10));
  int64_t cpp_value = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value));
  EXPECT_EQ(cpp_value, -1);
  uint64_t cpp_value_unsigned = 0;
  XLS_EXPECT_OK(xls::ConvertFromXlsValue(value, cpp_value_unsigned));
  // Note that a 10-bit signed value of -1 is equal to 1023 when converted to a
  // 10-bit unsigned value.
  EXPECT_EQ(cpp_value_unsigned, 1023);
}

}  // namespace
}  // namespace xls
