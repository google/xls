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

#include "xls/ir/ternary.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/iterator_range.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

TEST(Ternary, FromKnownBits) {
  // Basic test of functionality
  EXPECT_EQ(*StringToTernaryVector("0b1101X1X001"),
            ternary_ops::FromKnownBits(UBits(0b1111010111, 10),
                                       UBits(0b1101010001, 10)));
  // Empty bitstring should be handled correctly
  EXPECT_EQ(TernaryVector(), ternary_ops::FromKnownBits(Bits(), Bits()));
}

TEST(Ternary, Difference) {
  // Basic test of functionality.
  EXPECT_EQ(*StringToTernaryVector("0b11XXXXXXX1"),
            ternary_ops::Difference(*StringToTernaryVector("0b1101X1X001"),
                                    *StringToTernaryVector("0bXX01X1X00X")));
  // Test that conflict (in the last bit) leads to `std::nullopt`.
  EXPECT_EQ(std::nullopt,
            ternary_ops::Difference(*StringToTernaryVector("0b1101X1X001"),
                                    *StringToTernaryVector("0bXX01X1X000")));
  // It's okay if there are unknown bits in lhs that are known in rhs.
  // The point is just to determine what information was gained in lhs that
  // is not already in rhs.
  EXPECT_EQ(*StringToTernaryVector("0b11XXXXXXX1"),
            ternary_ops::Difference(*StringToTernaryVector("0b110XXXX001"),
                                    *StringToTernaryVector("0bXX01X1X00X")));
}

TEST(Ternary, NumberOfKnownBits) {
  // Basic test of functionality
  EXPECT_EQ(
      ternary_ops::NumberOfKnownBits(*StringToTernaryVector("0b1101X1X001")),
      8);
  // Empty ternary vector should be handled correctly
  EXPECT_EQ(ternary_ops::NumberOfKnownBits(TernaryVector()), 0);
}

TEST(Ternary, MinimumUnsignedBitCount) {
  // Basic test of functionality
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b1101X1X001")),
            10);
  // Empty ternary vector should be handled correctly
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(TernaryVector()), 0);
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b00X1X1X001")),
            8);
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b0000000000")),
            0);
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b1000000000")),
            10);
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b0100000000")),
            9);
  EXPECT_EQ(ternary_ops::MinimumUnsignedBitCount(
                *StringToTernaryVector("0b0X00000000")),
            9);
}
MATCHER_P(ToVector, m,
          testing::DescribeMatcher<std::vector<Bits>>(m, negation)) {
  return testing::ExplainMatchResult(
      m, std::vector<Bits>(arg.begin(), arg.end()), result_listener);
}
template <typename... Args>
auto IteratorElementsAre(Args... args) {
  return ToVector(testing::ElementsAre(args...));
}

MATCHER_P(IterIs, m,
          absl::StrFormat("element is %s",
                          testing::DescribeMatcher<Bits>(m, negation))) {
  return testing::ExplainMatchResult(m, *arg, result_listener);
}

TEST(TernaryIterator, Iterate) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary, StringToTernaryVector("0bX0X1"));
  auto range = ternary_ops::AllBitsValues(ternary);
  auto it = range.begin();
  EXPECT_THAT(it, IterIs(UBits(0b0001, 4)));
  EXPECT_THAT(++it, IterIs(UBits(0b0011, 4)));
  EXPECT_THAT(++it, IterIs(UBits(0b1001, 4)));
  EXPECT_THAT(++it, IterIs(UBits(0b1011, 4)));
  EXPECT_THAT(++it, range.end());
  EXPECT_THAT(++it, range.end());
  EXPECT_THAT(++it, range.end());
  EXPECT_THAT(++it, range.end());
}

// Instructions treat a 0-bit value as a thing that exists so ternary should as
// well.
TEST(TernaryIterator, IterateZeroLength) {
  auto range = ternary_ops::AllBitsValues({});
  EXPECT_THAT(range, IteratorElementsAre(UBits(0, 0)));
}

TEST(TernaryIterator, IteratePost) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary, StringToTernaryVector("0bX0X1"));
  auto range = ternary_ops::AllBitsValues(ternary);
  auto it = range.begin();
  EXPECT_THAT(it, IterIs(UBits(0b0001, 4)));
  EXPECT_THAT((it++, it), IterIs(UBits(0b0011, 4)));
  EXPECT_THAT((it++, it), IterIs(UBits(0b1001, 4)));
  EXPECT_THAT((it++, it), IterIs(UBits(0b1011, 4)));
  EXPECT_THAT((it++, it), range.end());
  EXPECT_THAT((it++, it), range.end());
  EXPECT_THAT((it++, it), range.end());
  EXPECT_THAT((it++, it), range.end());
}
TEST(TernaryIterator, Add) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary, StringToTernaryVector("0bX0X1"));
  auto range = ternary_ops::AllBitsValues(ternary);
  auto it = range.begin();
  EXPECT_THAT(it, IterIs(UBits(0b0001, 4)));
  EXPECT_THAT(it + 1, IterIs(UBits(0b0011, 4)));
  EXPECT_THAT(it + 2, IterIs(UBits(0b1001, 4)));
  EXPECT_THAT(it + 3, IterIs(UBits(0b1011, 4)));
  EXPECT_THAT(it + 4, range.end());
  EXPECT_THAT(it + 5, range.end());
  EXPECT_THAT(it + 6, range.end());
  EXPECT_THAT(it + 7, range.end());
}

TEST(TernaryIterator, SingleValue) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary, StringToTernaryVector("0b1010"));
  auto range = ternary_ops::AllBitsValues(ternary);
  EXPECT_THAT(range, IteratorElementsAre(UBits(0b1010, 4)));
}

TEST(TernaryIterator, MultipleValues) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary, StringToTernaryVector("0bX0X1"));
  auto range = ternary_ops::AllBitsValues(ternary);
  EXPECT_THAT(range, IteratorElementsAre(UBits(0b0001, 4), UBits(0b0011, 4),
                                         UBits(0b1001, 4), UBits(0b1011, 4)));
}

TEST(TernaryIterator, SomeValues) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary,
                           StringToTernaryVector("0bXXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"));
  ASSERT_THAT(ternary, testing::SizeIs(256));
  auto range = ternary_ops::AllBitsValues(ternary);
  xabsl::iterator_range<ternary_ops::RealizedTernaryIterator> sub_range(
      range.begin(), range.begin() + 10);
  EXPECT_THAT(sub_range,
              IteratorElementsAre(UBits(0, 256), UBits(1, 256), UBits(2, 256),
                                  UBits(3, 256), UBits(4, 256), UBits(5, 256),
                                  UBits(6, 256), UBits(7, 256), UBits(8, 256),
                                  UBits(9, 256)));
}

TEST(TernaryIterator, SomeHugeValues) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary,
                           StringToTernaryVector("0bXXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"
                                                 "_XXXX_XXXX_XXXX_XXXX"));
  ASSERT_THAT(ternary, testing::SizeIs(256));
  auto range = ternary_ops::AllBitsValues(ternary);
  auto offset = [](int64_t b) -> Bits {
    return bits_ops::Add(
        bits_ops::ZeroExtend(bits_ops::Concat({UBits(1, 1), UBits(0, 127)}),
                             256),
        UBits(b, 256));
  };
  xabsl::iterator_range<ternary_ops::RealizedTernaryIterator> sub_range(
      range.begin() + bits_ops::Concat({UBits(1, 1), UBits(0, 127)}),
      range.begin() + 10 + bits_ops::Concat({UBits(1, 1), UBits(0, 127)}));
  EXPECT_THAT(sub_range,
              IteratorElementsAre(offset(0), offset(1), offset(2), offset(3),
                                  offset(4), offset(5), offset(6), offset(7),
                                  offset(8), offset(9)));
}

TEST(TernaryIterator, HugeValues) {
  XLS_ASSERT_OK_AND_ASSIGN(auto ternary,
                           StringToTernaryVector("0bX000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_0000"
                                                 "_0000_0000_0000_000X"));
  ASSERT_THAT(ternary, testing::SizeIs(128));
  auto range = ternary_ops::AllBitsValues(ternary);
  EXPECT_THAT(range, IteratorElementsAre(
                         UBits(0, 128), UBits(1, 128),
                         bits_ops::Concat({UBits(1, 1), UBits(0, 127)}),
                         bits_ops::Concat({UBits(1, 1), UBits(1, 127)})));
}

class AllValuesTest : public ::testing::Test {
 protected:
  AllValuesTest() : package_("AllValuesTest") {}

  // Parse and return the given string as an XLS type.
  Type* AsType(std::string_view s) {
    return Parser::ParseType(s, &package_).value();
  }

  Value AsValue(std::string_view s, Type* t) {
    return *Parser::ParseValue(s, t);
  }

 private:
  Package package_;
};

TEST_F(AllValuesTest, AllValues) {
  Type* type = AsType("(bits[2], (), bits[3][4])");
  auto ternary_ltt = LeafTypeTree<TernaryVector>::CreateFromVector(
      type, {{*StringToTernaryVector("0b1X"), *StringToTernaryVector("0b000"),
              *StringToTernaryVector("0bXX1"), *StringToTernaryVector("0b010"),
              *StringToTernaryVector("0b011")}});
  EXPECT_THAT(ternary_ops::AllValues(ternary_ltt.AsView()),
              IsOkAndHolds(ElementsAre(
                  AsValue("(0b10, (), [0b000, 0b001, 0b010, 0b011])", type),
                  AsValue("(0b11, (), [0b000, 0b001, 0b010, 0b011])", type),
                  AsValue("(0b10, (), [0b000, 0b011, 0b010, 0b011])", type),
                  AsValue("(0b11, (), [0b000, 0b011, 0b010, 0b011])", type),
                  AsValue("(0b10, (), [0b000, 0b101, 0b010, 0b011])", type),
                  AsValue("(0b11, (), [0b000, 0b101, 0b010, 0b011])", type),
                  AsValue("(0b10, (), [0b000, 0b111, 0b010, 0b011])", type),
                  AsValue("(0b11, (), [0b000, 0b111, 0b010, 0b011])", type))));
}

TEST_F(AllValuesTest, AllValuesEmptyTypes) {
  auto empty_tuple_ltt =
      LeafTypeTree<TernaryVector>::CreateFromVector(AsType("()"), {});
  EXPECT_THAT(ternary_ops::AllValues(empty_tuple_ltt.AsView()),
              IsOkAndHolds(ElementsAre(Value::Tuple({}))));

  auto empty_array_ltt =
      LeafTypeTree<TernaryVector>::CreateFromVector(AsType("()[2]"), {});
  EXPECT_THAT(ternary_ops::AllValues(empty_array_ltt.AsView()),
              IsOkAndHolds(ElementsAre(
                  *Value::Array({Value::Tuple({}), Value::Tuple({})}))));

  Type* complex_empty_type = AsType("(()[2], (), ()[2], ((), (), ()[3][2]))");
  auto complex_empty_ltt =
      LeafTypeTree<TernaryVector>::CreateFromVector(complex_empty_type, {});
  EXPECT_THAT(
      ternary_ops::AllValues(complex_empty_ltt.AsView()),
      IsOkAndHolds(ElementsAre(AsValue("([(), ()],"
                                       " (),"
                                       " [(), ()],"
                                       " ((), (), [[(), (), ()], [(), (), ()]])"
                                       ")",
                                       complex_empty_type))));
}

}  // namespace
}  // namespace xls
