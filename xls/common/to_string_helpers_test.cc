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

#include "xls/common/to_string_helpers.h"

#include <deque>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xls {
namespace {

using testing::StrEq;

TEST(ToStringHelpersTest, ToStringBool) {
  EXPECT_THAT(::xls::ToString(true), StrEq("true"));
  EXPECT_THAT(::xls::ToString(false), StrEq("false"));
}

template <typename T>
class ToStringHelpersIntegralTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ToStringHelpersIntegralTypedTest);

TYPED_TEST_P(ToStringHelpersIntegralTypedTest, CompareIntegrals) {
  EXPECT_THAT(::xls::ToString(static_cast<TypeParam>(42)), StrEq("42"));
}

REGISTER_TYPED_TEST_SUITE_P(ToStringHelpersIntegralTypedTest, CompareIntegrals);

using MyIntegralTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t,
                                         uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Integrals, ToStringHelpersIntegralTypedTest,
                               MyIntegralTypes);

template <typename T>
class ToStringHelpersContainerTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ToStringHelpersContainerTypedTest);

TYPED_TEST_P(ToStringHelpersContainerTypedTest, Compare1DContainerIntegral) {
  EXPECT_THAT(::xls::ToString(TypeParam{42, 1}), StrEq("[ 42, 1 ]"));
}

REGISTER_TYPED_TEST_SUITE_P(ToStringHelpersContainerTypedTest,
                            Compare1DContainerIntegral);

// Test a single integral type per container type, since the integral types were
// thoroughly tested in a previous tests.
using My1DContainerIntegralTypes =
    ::testing::Types<absl::Span<const int8_t>, std::vector<int8_t>,
                     std::deque<int8_t>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Integrals, ToStringHelpersContainerTypedTest,
                               My1DContainerIntegralTypes);

// Test a single container type for a multi-dimensional containers, since the
// container types were thoroughly tested in a previous tests.
TEST(ToStringHelpersTest, Compare2DVectorIntegral) {
  EXPECT_THAT(
      ::xls::ToString(std::vector<std::vector<int64_t>>{{42, 42}, {42, 42}}),
      StrEq("[ [ 42, 42 ], [ 42, 42 ] ]"));
  EXPECT_THAT(
      ::xls::ToString(std::vector<std::deque<int64_t>>{{42, 42}, {42, 42}}),
      StrEq("[ [ 42, 42 ], [ 42, 42 ] ]"));
}

struct UserDefinedType {
  int64_t integer;
  bool boolean;
  std::string ToString() const {
    return absl::StrFormat("{integer: %s, boolean: %s}",
                           ::xls::ToString(integer), ::xls::ToString(boolean));
  }
};

TEST(ToStringHelpersTest, ToStringUserDefinedType) {
  EXPECT_THAT((UserDefinedType{42, true}).ToString(),
              StrEq("{integer: 42, boolean: true}"));
}

TEST(ToStringHelpersTest, ToStringSpanUserDefinedType) {
  EXPECT_THAT(
      ::xls::ToString(absl::Span<const UserDefinedType>{
          UserDefinedType{42, true}, UserDefinedType{1, false}}),
      StrEq("[ {integer: 42, boolean: true}, {integer: 1, boolean: false} ]"));
}

}  // namespace
}  // namespace xls
