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

#include "xls/ir/conversion_utils.h"

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;

TEST(ConversionUtils, ConvertSignedIntegrals) {
  BitsType type(10);
  EXPECT_THAT(xls::Convert(&type, static_cast<int64_t>(42)),
              IsOkAndHolds(Value(SBits(42, 10))));

  EXPECT_THAT(xls::Convert(&type, static_cast<int32_t>(42)),
              IsOkAndHolds(Value(SBits(42, 10))));

  ArrayType invalid_type(2, &type);
  EXPECT_THAT(
      xls::Convert(&invalid_type, static_cast<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for integral input value")));

  BitsType invalid_bit_count(2);
  EXPECT_THAT(xls::Convert(&invalid_bit_count, static_cast<int64_t>(42)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ConversionUtils, ConvertUnsignedIntegrals) {
  BitsType type(10);
  EXPECT_THAT(xls::Convert(&type, static_cast<uint64_t>(42)),
              IsOkAndHolds(Value(UBits(42, 10))));

  EXPECT_THAT(xls::Convert(&type, static_cast<uint32_t>(42)),
              IsOkAndHolds(Value(UBits(42, 10))));

  ArrayType invalid_type(2, &type);
  EXPECT_THAT(
      xls::Convert(&invalid_type, static_cast<uint64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Invalid type conversion for integral input value")));

  BitsType invalid_bit_count(2);
  EXPECT_THAT(xls::Convert(&invalid_bit_count, static_cast<uint64_t>(42)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ConversionUtils, ConvertArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_golden,
      Value::Array({Value(UBits(64, 10)), Value(UBits(42, 10))}));
  Package p("package");
  ArrayType* array_type = p.GetArrayType(2, p.GetBitsType(10));
  EXPECT_THAT(xls::Convert(array_type, std::vector<int64_t>{64, 42}),
              IsOkAndHolds(array_golden));

  TupleType invalid_type({p.GetBitsType(10), p.GetBitsType(10)});
  EXPECT_THAT(xls::Convert(&invalid_type, std::vector<int64_t>{64, 42}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr(
                           "Invalid type conversion for an absl::Span")));

  ArrayType size_mismatch_type(10, p.GetBitsType(10));
  EXPECT_THAT(
      xls::Convert(&size_mismatch_type, std::vector{64, 42}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Array size mismatch between conversion type and value")));
}

TEST(ConversionUtils, ConvertTuple) {
  Package p("package");
  Type* tuple_type = p.GetTupleType({p.GetBitsType(10)});
  EXPECT_THAT(xls::Convert(tuple_type, std::make_tuple<int64_t>(42)),
              IsOkAndHolds(Value::Tuple({Value(UBits(42, 10))})));

  ArrayType invalid_type(10, p.GetBitsType(10));
  EXPECT_THAT(
      xls::Convert(&invalid_type, std::make_tuple<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid type conversion for a std::tuple")));

  TupleType size_mismatch_type({p.GetBitsType(10), p.GetBitsType(10)});
  EXPECT_THAT(
      xls::Convert<int64_t>(&size_mismatch_type, std::make_tuple<int64_t>(42)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr(
                   "Tuple size mismatch between conversion type and value")));
}

TEST(ConversionUtils, ConvertEmptyTuple) {
  TupleType tuple_type({});
  EXPECT_THAT(xls::Convert(&tuple_type, std::make_tuple()),
              IsOkAndHolds(Value::Tuple({})));

  BitsType invalid_type(10);
  EXPECT_THAT(
      xls::Convert(&invalid_type, std::make_tuple()),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("Invalid type conversion for a std::tuple")));
}

TEST(ConversionUtils, ConvertTupleWithArray) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value element1,
      Value::Array({Value(UBits(42, 10)), Value(UBits(64, 10))}));
  Package p("package");
  TupleType tuple_type(
      {p.GetBitsType(10), p.GetArrayType(2, p.GetBitsType(10))});
  absl::StatusOr<Value> result_or = xls::Convert(
      &tuple_type, std::make_tuple<uint64_t, absl::Span<const uint64_t>>(
                       64, std::vector<uint64_t>{42, 64}));
  EXPECT_THAT(result_or,
              IsOkAndHolds(Value::Tuple({Value(UBits(64, 10)), element1})));
}

struct UserStruct {
  int64_t integer_value;
  bool boolean_value;
};

absl::StatusOr<xls::Value> Convert(const Type* type,
                                   const UserStruct& user_struct) {
  return xls::Convert(
      type, std::make_tuple(user_struct.integer_value,
                            static_cast<uint64_t>(user_struct.boolean_value)));
}

TEST(ConversionUtils, ConvertUserStruct) {
  Package p("package");
  TupleType* user_struct_type =
      p.GetTupleType({p.GetBitsType(10), p.GetBitsType(1)});
  EXPECT_THAT(
      Convert(user_struct_type, UserStruct{42, false}),
      IsOkAndHolds(Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)})));
}

TEST(ConversionUtils, ConvertArrayOfUserStruct) {
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_golden,
      Value::Array({Value::Tuple({Value(UBits(42, 10)), Value::Bool(false)}),
                    Value::Tuple({Value(UBits(64, 10)), Value::Bool(true)})}));
  Package p("package");
  ArrayType* array_type =
      p.GetArrayType(2, p.GetTupleType({p.GetBitsType(10), p.GetBitsType(1)}));
  EXPECT_THAT(
      xls::Convert<UserStruct>(
          array_type, std::vector{UserStruct{42, false}, UserStruct{64, true}}),
      IsOkAndHolds(array_golden));
}

TEST(ConversionUtils, ConvertNegativeValues) {
  BitsType type(10);
  EXPECT_THAT(xls::Convert(&type, static_cast<int64_t>(-1)),
              IsOkAndHolds(Value(SBits(-1, 10))));

  EXPECT_THAT(xls::Convert(&type, static_cast<uint64_t>(-1)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xls
