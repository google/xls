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

#include "xls/common/type_traits_utils.h"

#include <array>
#include <cstdint>
#include <deque>
#include <forward_list>
#include <list>
#include <queue>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"

namespace xls {
namespace {

template <typename T>
class HasConstIteratorSupportedTypesTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(HasConstIteratorSupportedTypesTest);

TYPED_TEST_P(HasConstIteratorSupportedTypesTest,
             HasConstIteratorSupportedTypes) {
  EXPECT_TRUE(has_const_iterator_v<TypeParam>);
}

REGISTER_TYPED_TEST_SUITE_P(HasConstIteratorSupportedTypesTest,
                            HasConstIteratorSupportedTypes);

using MyHasConstIteratorSupportedTypes =
    ::testing::Types<absl::Span<int64_t>, std::array<int64_t, 1>,
                     std::deque<int64_t>, std::forward_list<int64_t>,
                     std::list<int64_t>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HasConstIteratorSupportedTypes,
                               HasConstIteratorSupportedTypesTest,
                               MyHasConstIteratorSupportedTypes);

template <typename T>
class HasConstIteratorNonSupportedTypesTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(HasConstIteratorNonSupportedTypesTest);

TYPED_TEST_P(HasConstIteratorNonSupportedTypesTest,
             HasConstIteratorNonSupportedTypes) {
  EXPECT_FALSE(has_const_iterator_v<TypeParam>);
}

REGISTER_TYPED_TEST_SUITE_P(HasConstIteratorNonSupportedTypesTest,
                            HasConstIteratorNonSupportedTypes);

using MyHasConstIteratorNonSupportedTypes =
    ::testing::Types<std::priority_queue<int64_t>, std::queue<int64_t>,
                     std::stack<int64_t>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HasConstIteratorNonSupportedTypes,
                               HasConstIteratorNonSupportedTypesTest,
                               MyHasConstIteratorNonSupportedTypes);

template <typename T>
class HasMemberSizeSupportedTypesTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(HasMemberSizeSupportedTypesTest);

TYPED_TEST_P(HasMemberSizeSupportedTypesTest, HasMemberSizeSupportedTypes) {
  EXPECT_TRUE(has_member_size_v<TypeParam>);
}

REGISTER_TYPED_TEST_SUITE_P(HasMemberSizeSupportedTypesTest,
                            HasMemberSizeSupportedTypes);

using MyHasMemberSizeSupportedTypes =
    ::testing::Types<absl::Span<int64_t>, std::array<int64_t, 1>,
                     std::deque<int64_t>, std::list<int64_t>,
                     std::priority_queue<int64_t>, std::queue<int64_t>,
                     std::stack<int64_t>, std::vector<int64_t>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HasMemberSizeSupportedTypes,
                               HasMemberSizeSupportedTypesTest,
                               MyHasMemberSizeSupportedTypes);

TEST(TypeTraitHelperTest, HasMemberSizeNonSupportedTypes) {
  EXPECT_FALSE(has_member_size_v<std::forward_list<int64_t>>);
}

struct MyStruct {};

struct MyStructWithToString {
  std::string ToString() { return "MyStructWithToString"; }
};

TEST(TypeTraitHelperTest, HasMemberToString) {
  EXPECT_FALSE(has_member_to_string_v<int64_t>);
  EXPECT_FALSE(has_member_to_string_v<std::vector<int64_t>>);
  EXPECT_FALSE(has_member_to_string_v<MyStruct>);
  EXPECT_TRUE(has_member_to_string_v<MyStructWithToString>);
}

TEST(TypeTraitHelperTest, IsOneOf) {
  bool result;
  EXPECT_FALSE(is_one_of<>::value);
  EXPECT_FALSE(is_one_of<int64_t>::value);
  result = is_one_of<int8_t, std::variant<int64_t>>::value;
  EXPECT_FALSE(result);
  result = is_one_of<int8_t, std::variant<int64_t, MyStruct>>::value;
  EXPECT_FALSE(result);
  result = is_one_of<int64_t, std::variant<int64_t>>::value;
  EXPECT_TRUE(result);
  result = is_one_of<MyStruct, std::variant<int64_t, MyStruct>>::value;
  EXPECT_TRUE(result);
}

}  // namespace
}  // namespace xls
