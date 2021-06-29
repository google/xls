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

#include "xls/dslx/evaluate.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;

TEST(CppEvaluateTest, EmptyArrayCompatibility) {
  auto empty_array_of_nil = absl::make_unique<ArrayType>(
      ConcreteType::MakeUnit(), ConcreteTypeDim::CreateU32(0));
  auto empty_array_of_u32 = absl::make_unique<ArrayType>(
      BitsType::MakeU32(), ConcreteTypeDim::CreateU32(0));
  auto empty_array_value = InterpValue::MakeArray(/*elements=*/{}).value();

  EXPECT_THAT(ValueCompatibleWithType(*empty_array_of_nil, empty_array_value),
              IsOkAndHolds(true));
  EXPECT_THAT(ValueCompatibleWithType(*empty_array_of_u32, empty_array_value),
              IsOkAndHolds(true));
}

TEST(CppEvaluateTest, DiffNumberOfElementsVs0Compatibility) {
  auto empty_array_of_nil = absl::make_unique<ArrayType>(
      ConcreteType::MakeUnit(), ConcreteTypeDim::CreateU32(0));
  auto array_of_1_value =
      InterpValue::MakeArray(/*elements=*/{InterpValue::MakeU32(1)}).value();

  EXPECT_THAT(ValueCompatibleWithType(*empty_array_of_nil, array_of_1_value),
              IsOkAndHolds(false));
}

TEST(CppEvaluateTest, TupleSizeIncompatibilityVsUnit) {
  auto nil_tuple = ConcreteType::MakeUnit();
  auto tuple_of_1 =
      InterpValue::MakeTuple(/*members=*/{InterpValue::MakeU32(1)});

  EXPECT_THAT(ValueCompatibleWithType(*nil_tuple, tuple_of_1),
              IsOkAndHolds(false));
}

TEST(CppEvaluateTest, TupleSizeIncompatibilityUnitVs1Element) {
  std::vector<std::unique_ptr<ConcreteType>> members;
  members.push_back(BitsType::MakeU32());
  auto tuple_of_u32_type = absl::make_unique<TupleType>(std::move(members));
  auto nil_tuple_value = InterpValue::MakeTuple(/*members=*/{});

  EXPECT_THAT(ValueCompatibleWithType(*tuple_of_u32_type, nil_tuple_value),
              IsOkAndHolds(false));
}

TEST(CppEvaluateTest, BitsValueSignednessAcceptance) {
  auto s1_type = absl::make_unique<BitsType>(true, 1);
  auto u1_type = BitsType::MakeU1();
  auto u1_value = InterpValue::MakeUBits(1, 0);
  auto s1_value = InterpValue::MakeSBits(1, 0);

  std::vector<std::tuple<ConcreteType*, InterpValue, bool>> examples = {
      {s1_type.get(), s1_value, true},
      {s1_type.get(), u1_value, false},
      {u1_type.get(), u1_value, true},
      {u1_type.get(), s1_value, false},
  };

  for (auto [type, value, want] : examples) {
    EXPECT_THAT(ConcreteTypeAcceptsValue(*type, value), IsOkAndHolds(want));
  }
}

}  // namespace
}  // namespace xls::dslx
