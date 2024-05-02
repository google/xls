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

#include "xls/common/comparison_utils.h"

#include <cstdint>
#include <deque>
#include <list>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"

namespace xls {
namespace {

using testing::HasSubstr;
using testing::IsEmpty;

TEST(ComparisonUtilsTest, CompareBool) {
  EXPECT_THAT(::xls::Compare("boolean", true, true), IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("boolean", true, false),
      HasSubstr("Element boolean differ: expected (true), got (false)."));
}

template <typename T>
class ComparisonUtilsIntegralTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ComparisonUtilsIntegralTypedTest);

TYPED_TEST_P(ComparisonUtilsIntegralTypedTest, CompareIntegrals) {
  EXPECT_THAT(::xls::Compare("Integral value", static_cast<TypeParam>(8),
                             static_cast<TypeParam>(8)),
              IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("Integral value", static_cast<TypeParam>(8),
                     static_cast<TypeParam>(42)),
      HasSubstr("Element Integral value differ: expected (8), got (42)."));
}

REGISTER_TYPED_TEST_SUITE_P(ComparisonUtilsIntegralTypedTest, CompareIntegrals);

using MyIntegralTypes = ::testing::Types<int8_t, int16_t, int32_t, int64_t,
                                         uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(Integrals, ComparisonUtilsIntegralTypedTest,
                               MyIntegralTypes);

template <typename T>
class ComparisonUtilsSequenceContainerTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ComparisonUtilsSequenceContainerTypedTest);

TYPED_TEST_P(ComparisonUtilsSequenceContainerTypedTest,
             Compare1DSequenceContainerIntegral) {
  EXPECT_THAT(
      ::xls::Compare("Container is equal", TypeParam{42, 1}, TypeParam{42, 1}),
      IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("Container", TypeParam{42, 1}, TypeParam{42, 1, 0}),
      HasSubstr("Size of element Container differ: expected (2), got (3)."));
  EXPECT_THAT(
      ::xls::Compare("Container", TypeParam{42, 1}, TypeParam{42, 42}),
      HasSubstr("Element Container[1] differ: expected (1), got (42)."));
}

REGISTER_TYPED_TEST_SUITE_P(ComparisonUtilsSequenceContainerTypedTest,
                            Compare1DSequenceContainerIntegral);

// Test a single integral type per container type, since the integral types were
// thoroughly tested in a previous tests.
using My1DSequenceContainerIntegralTypes =
    ::testing::Types<absl::Span<const int8_t>, std::deque<int8_t>,
                     std::list<int8_t>, std::vector<int8_t>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Integrals,
                               ComparisonUtilsSequenceContainerTypedTest,
                               My1DSequenceContainerIntegralTypes);

template <typename T>
class ComparisonUtilsSetContainerTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ComparisonUtilsSetContainerTypedTest);

TYPED_TEST_P(ComparisonUtilsSetContainerTypedTest,
             Compare1DSetContainerIntegral) {
  EXPECT_THAT(::xls::Compare("SetContainer is equal", TypeParam{42, 1},
                             TypeParam{42, 1}),
              IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("SetContainer", TypeParam{42, 1}, TypeParam{42, 1, 0}),
      HasSubstr("Size of element SetContainer differ: expected (2), got (3)."));
  EXPECT_THAT(
      ::xls::Compare("SetContainer", TypeParam{1, 42}, TypeParam{0, 42}),
      HasSubstr("differ: expected ("));
}

REGISTER_TYPED_TEST_SUITE_P(ComparisonUtilsSetContainerTypedTest,
                            Compare1DSetContainerIntegral);

// Test a single integral type per container type, since the integral types were
// thoroughly tested in a previous tests.
using My1DSetContainerIntegralTypes =
    ::testing::Types<std::multiset<int8_t>, std::set<int8_t>,
                     std::unordered_multiset<int8_t>,
                     std::unordered_set<int8_t>>;

INSTANTIATE_TYPED_TEST_SUITE_P(Integrals, ComparisonUtilsSetContainerTypedTest,
                               My1DSetContainerIntegralTypes);

// Test a single container type for a multi-dimensional containers, since the
// container types were thoroughly tested in previous tests.
TEST(ComparisonUtilsTest, Compare2DVectorIntegral) {
  EXPECT_THAT(
      ::xls::Compare("Vector is equal",
                     std::vector<std::vector<int64_t>>{{42, 42}, {42, 42}},
                     std::vector<std::vector<int64_t>>{{42, 42}, {42, 42}}),
      IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("Vector",
                     std::vector<std::vector<int64_t>>{{42, 42}, {42, 42}},
                     std::vector<std::vector<int64_t>>{{42, 42}, {42, 1}}),
      HasSubstr("Element Vector[1][1] differ: expected "
                "(42), got (1)."));
}

struct UserDefinedType {
  int64_t integer;
  bool boolean;
};
static bool operator==(const UserDefinedType& lhs, const UserDefinedType& rhs) {
  return lhs.integer == rhs.integer && lhs.boolean == rhs.boolean;
}
static bool operator!=(const UserDefinedType& lhs, const UserDefinedType& rhs) {
  return !(lhs == rhs);
}

std::string Compare(std::string_view element_name,
                    const UserDefinedType& expected,
                    const UserDefinedType& computed) {
  std::string comparison;
  std::string ref_element_name;
  if (!element_name.empty()) {
    ref_element_name = absl::StrCat(element_name, ".");
  }
  absl::StrAppend(
      &comparison,
      absl::StrFormat("%s",
                      ::xls::Compare(absl::StrCat(ref_element_name, "integer"),
                                     expected.integer, computed.integer)));
  absl::StrAppend(
      &comparison,
      absl::StrFormat("%s",
                      ::xls::Compare(absl::StrCat(ref_element_name, "boolean"),
                                     expected.boolean, computed.boolean)));
  return comparison;
}

TEST(ComparisonUtilsTest, CompareUserDefinedType) {
  EXPECT_THAT(Compare("UserDefinedType is equal", UserDefinedType{42, true},
                      UserDefinedType{42, true}),
              IsEmpty());
  EXPECT_THAT(Compare("UserDefinedType", UserDefinedType{42, true},
                      UserDefinedType{42, false}),
              HasSubstr("Element UserDefinedType.boolean differ: expected "
                        "(true), got (false)."));
  // Differ with no name reference.
  EXPECT_THAT(Compare("", UserDefinedType{42, true}, UserDefinedType{1, true}),
              HasSubstr("Element integer differ: expected (42), got (1)."));
}

TEST(ComparisonUtilsTest, CompareSpanUserDefinedType) {
  EXPECT_THAT(
      ::xls::Compare("Span is equal",
                     absl::Span<const UserDefinedType>{
                         UserDefinedType{42, true}, UserDefinedType{1, false}},
                     absl::Span<const UserDefinedType>{
                         UserDefinedType{42, true}, UserDefinedType{1, false}}),
      IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("Span",
                     absl::Span<const UserDefinedType>{
                         UserDefinedType{42, true}, UserDefinedType{42, false}},
                     absl::Span<const UserDefinedType>{
                         UserDefinedType{1, true}, UserDefinedType{42, false}}),
      HasSubstr("Element Span[0].integer differ: expected "
                "(42), got (1)."));
}

TEST(ToStringHelpersTest, ToStringStdPair) {
  EXPECT_THAT(::xls::Compare("UserDefinedType is equal",
                             std::pair<int8_t, int8_t>{42, 1},
                             std::pair<int8_t, int8_t>{42, 1}),
              IsEmpty());
  EXPECT_THAT(
      ::xls::Compare("UserDefinedType", std::pair<int8_t, int8_t>{42, 1},
                     std::pair<int8_t, int8_t>{42, 42}),
      HasSubstr(
          "Element UserDefinedType.second differ: expected (1), got (42)."));
  // Differ with no name reference.
  EXPECT_THAT(::xls::Compare("", std::pair<int8_t, int8_t>{42, 1},
                             std::pair<int8_t, int8_t>{42, 42}),
              HasSubstr("Element second differ: expected (1), got (42)."));
}

}  // namespace
}  // namespace xls
