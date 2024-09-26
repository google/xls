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

#include "xls/data_structures/leaf_type_tree.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::xls::status_testing::StatusIs;

class LeafTypeTreeTest : public ::testing::Test {
 protected:
  LeafTypeTreeTest() : package_("LeafTypeTreeTest") {}

  // Returns a vector of strings constructed by calling the ToString method of
  // each of the given elements.
  template <typename T>
  std::vector<std::string> AsStrings(absl::Span<T* const> elements) {
    std::vector<std::string> res;
    for (const T* e : elements) {
      res.push_back(e->ToString());
    }
    return res;
  }

  // Parse and return the given string as an XLS type.
  Type* AsType(std::string_view s) {
    return Parser::ParseType(s, &package_).value();
  }

 private:
  Package package_;
};

TEST_F(LeafTypeTreeTest, BitsTypes) {
  LeafTypeTree<int64_t> tree(AsType("bits[42]"));

  EXPECT_EQ(tree.size(), 1);
  // Elements should be value-initialized to zero.
  EXPECT_EQ(tree.Get({}), 0);
  EXPECT_THAT(AsStrings(tree.leaf_types()), ElementsAre("bits[42]"));
  EXPECT_THAT(tree.elements(), ElementsAre(0));
  EXPECT_THAT(tree.AsView().elements(), ElementsAre(0));

  tree.Set({}, 42);
  EXPECT_EQ(tree.Get({}), 42);
  EXPECT_THAT(tree.elements(), ElementsAre(42));
  EXPECT_EQ(tree.ToString(), "42");
  EXPECT_EQ(tree.ToMultilineString(), "42");

  tree.Set({}, 123);
  EXPECT_EQ(tree.Get({}), 123);
  EXPECT_THAT(tree.elements(), ElementsAre(123));

  LeafTypeTree<int64_t> subtree = leaf_type_tree::Clone(tree.AsView());
  EXPECT_THAT(subtree.elements(), ElementsAre(123));

  LeafTypeTree<int64_t> tree_with_init(AsType("bits[42]"), 123456);
  EXPECT_EQ(tree_with_init.Get({}), 123456);
}

TEST_F(LeafTypeTreeTest, TupleType) {
  LeafTypeTree<Bits> tree(AsType("(bits[123], bits[2], bits[42])"));

  EXPECT_EQ(tree.size(), 3);
  EXPECT_EQ(tree.type()->ToString(), "(bits[123], bits[2], bits[42])");
  EXPECT_THAT(AsStrings(tree.leaf_types()),
              ElementsAre("bits[123]", "bits[2]", "bits[42]"));
  EXPECT_THAT(tree.elements(), ElementsAre(Bits(), Bits(), Bits()));

  tree.Set({1}, UBits(33, 44));
  EXPECT_EQ(tree.Get({1}), UBits(33, 44));
  EXPECT_THAT(tree.elements(), ElementsAre(Bits(), UBits(33, 44), Bits()));

  EXPECT_EQ(tree.ToString([](const Bits& b) { return BitsToString(b); }),
            "(0, 33, 0)");
  EXPECT_EQ(
      tree.ToMultilineString([](const Bits& b) { return BitsToString(b); }),
      R"((
  0,
  33,
  0
))");

  tree.elements()[0] = UBits(7, 8);
  EXPECT_EQ(tree.elements()[0], UBits(7, 8));
  EXPECT_THAT(tree.elements(), ElementsAre(UBits(7, 8), UBits(33, 44), Bits()));

  tree.elements()[2] = UBits(1, 1);
  EXPECT_THAT(tree.elements(),
              ElementsAre(UBits(7, 8), UBits(33, 44), UBits(1, 1)));

  EXPECT_THAT(tree.elements(),
              ElementsAre(UBits(7, 8), UBits(33, 44), UBits(1, 1)));
  EXPECT_THAT(tree.AsView({0}).elements(), ElementsAre(UBits(7, 8)));
  EXPECT_THAT(tree.AsView({1}).elements(), ElementsAre(UBits(33, 44)));
  EXPECT_THAT(tree.AsView({2}).elements(), ElementsAre(UBits(1, 1)));

  LeafTypeTree<Bits> subtree0 = leaf_type_tree::Clone(tree.AsView());
  EXPECT_EQ(subtree0.type()->ToString(), "(bits[123], bits[2], bits[42])");
  EXPECT_THAT(subtree0.elements(),
              ElementsAre(UBits(7, 8), UBits(33, 44), UBits(1, 1)));

  LeafTypeTree<Bits> subtree1 = leaf_type_tree::Clone(tree.AsView({1}));
  EXPECT_EQ(subtree1.type()->ToString(), "bits[2]");
  EXPECT_THAT(subtree1.elements(), ElementsAre(UBits(33, 44)));

  LeafTypeTree<Bits> tree_with_init(AsType("(bits[123], bits[2], bits[42])"),
                                    UBits(999, 99));
  EXPECT_THAT(tree_with_init.elements(),
              ElementsAre(UBits(999, 99), UBits(999, 99), UBits(999, 99)));
}

TEST_F(LeafTypeTreeTest, ArrayType) {
  LeafTypeTree<int64_t> tree(AsType("bits[1][5]"));

  EXPECT_EQ(tree.size(), 5);
  EXPECT_THAT(
      AsStrings(tree.leaf_types()),
      ElementsAre("bits[1]", "bits[1]", "bits[1]", "bits[1]", "bits[1]"));
  EXPECT_EQ(tree.type()->ToString(), "bits[1][5]");

  // Elements should be value initialized to zero.
  EXPECT_THAT(tree.elements(), ElementsAre(0, 0, 0, 0, 0));

  tree.Set({1}, 3);
  tree.Set({2}, 4);
  tree.Set({3}, 5);
  tree.Set({4}, 6);
  EXPECT_THAT(tree.elements(), ElementsAre(0, 3, 4, 5, 6));
  EXPECT_EQ(tree.Get({3}), 5);
  EXPECT_THAT(tree.AsView({3}).elements(), ElementsAre(5));

  EXPECT_EQ(tree.ToString(), "[0, 3, 4, 5, 6]");
  EXPECT_EQ(tree.ToMultilineString(),
            R"([
  0,
  3,
  4,
  5,
  6
])");

  LeafTypeTree<int64_t> clone = leaf_type_tree::Clone(tree.AsView());
  EXPECT_THAT(clone.elements(), ElementsAre(0, 3, 4, 5, 6));

  LeafTypeTree<int64_t> subclone = leaf_type_tree::Clone(tree.AsView({2}));
  EXPECT_THAT(subclone.elements(), ElementsAre(4));
}

TEST_F(LeafTypeTreeTest, NestedTupleType) {
  LeafTypeTree<int64_t> tree(
      AsType("(bits[37][3], (bits[22], (bits[1], bits[1][2]), bits[42]))"));

  EXPECT_EQ(tree.size(), 8);
  EXPECT_EQ(tree.type()->ToString(),
            "(bits[37][3], (bits[22], (bits[1], bits[1][2]), bits[42]))");
  EXPECT_THAT(tree.elements(), Each(Eq(0)));

  tree.Set({0, 2}, 3);
  EXPECT_EQ(tree.Get({0, 2}), 3);
  tree.Set({1, 1, 0}, 42);
  EXPECT_EQ(tree.Get({1, 1, 0}), 42);
  tree.Set({1, 2}, 77);
  EXPECT_EQ(tree.Get({1, 2}), 77);

  EXPECT_THAT(tree.elements(), ElementsAre(0, 0, 3, 0, 42, 0, 0, 77));

  EXPECT_THAT(tree.AsView({0}).elements(), ElementsAre(0, 0, 3));
  EXPECT_THAT(tree.AsView({1}).elements(), ElementsAre(0, 42, 0, 0, 77));
  EXPECT_THAT(tree.AsView({1, 1}).elements(), ElementsAre(42, 0, 0));

  EXPECT_EQ(tree.ToString(), "([0, 0, 3], (0, (42, [0, 0]), 77))");
  EXPECT_EQ(tree.ToMultilineString(),
            R"((
  [
    0,
    0,
    3
  ],
  (
    0,
    (
      42,
      [
        0,
        0
      ]
    ),
    77
  )
))");

  LeafTypeTree<int64_t> mapped =
      leaf_type_tree::MapIndex<int64_t, int64_t>(
          tree.AsView(),
          [](Type* leaf_type, int64_t x, absl::Span<const int64_t> index)
              -> absl::StatusOr<int64_t> { return x + 1; })
          .value();
  EXPECT_THAT(mapped.elements(), ElementsAre(1, 1, 4, 1, 43, 1, 1, 78));

  LeafTypeTree<int64_t> simple_mapped = leaf_type_tree::Map<int64_t, int64_t>(
      tree.AsView(), [](int64_t x) { return 2 * x; });
  EXPECT_THAT(simple_mapped.elements(), ElementsAre(0, 0, 6, 0, 84, 0, 0, 154));

  LeafTypeTree<int64_t> other_tree(tree.type());
  other_tree.Set({0, 1}, 5);
  other_tree.Set({1, 1, 1, 0}, 12);
  LeafTypeTree<int64_t> zipped =
      leaf_type_tree::ZipIndex<int64_t, int64_t>(
          {tree.AsView(), other_tree.AsView()},
          [](Type* leaf_type, absl::Span<const int64_t* const> elements,
             absl::Span<const int64_t> index) -> absl::StatusOr<int64_t> {
            return std::max(*elements[0], *elements[1]);
          })
          .value();
  EXPECT_THAT(zipped.elements(), ElementsAre(0, 5, 3, 0, 42, 12, 0, 77));

  LeafTypeTree<int64_t> simple_zipped = leaf_type_tree::Zip<int64_t, int64_t>(
      tree.AsView(), other_tree.AsView(),
      [](int64_t a, int64_t b) { return a - b; });
  EXPECT_THAT(simple_zipped.elements(),
              ElementsAre(0, -5, 3, 0, 42, -12, 0, 77));

  LeafTypeTree<int64_t> subtree = leaf_type_tree::Clone(tree.AsView({1, 1}));
  EXPECT_EQ(subtree.type()->ToString(), "(bits[1], bits[1][2])");
  EXPECT_THAT(subtree.elements(), ElementsAre(42, 0, 0));
}

TEST_F(LeafTypeTreeTest, NestedArrayType) {
  LeafTypeTree<int64_t> tree(AsType("(bits[42], bits[123])[3]"));

  EXPECT_EQ(tree.size(), 6);
  EXPECT_THAT(tree.elements(), Each(Eq(0)));
  EXPECT_THAT(AsStrings(tree.leaf_types()),
              ElementsAre("bits[42]", "bits[123]", "bits[42]", "bits[123]",
                          "bits[42]", "bits[123]"));

  tree.Set({0, 1}, 3);
  EXPECT_EQ(tree.Get({0, 1}), 3);
  tree.Set({2, 0}, 42);
  EXPECT_EQ(tree.Get({2, 0}), 42);

  EXPECT_THAT(tree.elements(), ElementsAre(0, 3, 0, 0, 42, 0));

  EXPECT_EQ(tree.ToString(), "[(0, 3), (0, 0), (42, 0)]");

  LeafTypeTree<int64_t> subtree = leaf_type_tree::Clone(tree.AsView({2}));
  EXPECT_EQ(subtree.type()->ToString(), "(bits[42], bits[123])");
  EXPECT_THAT(subtree.elements(), ElementsAre(42, 0));
}

TEST_F(LeafTypeTreeTest, EmptyTuple) {
  LeafTypeTree<int64_t> tree(AsType("()"));
  EXPECT_EQ(tree.size(), 0);
  EXPECT_TRUE(tree.elements().empty());
  EXPECT_TRUE(tree.leaf_types().empty());
  EXPECT_EQ(tree.ToString(), "()");
  EXPECT_EQ(tree.ToMultilineString(), "()");

  EXPECT_EQ(tree.AsView().ToString(), "()");
}

TEST_F(LeafTypeTreeTest, Token) {
  LeafTypeTree<int64_t> tree(AsType("token"));
  EXPECT_EQ(tree.size(), 1);
  // Elements should be value-initialized to zero.
  EXPECT_EQ(tree.Get({}), 0);
  EXPECT_THAT(AsStrings(tree.leaf_types()), ElementsAre("token"));
  EXPECT_THAT(tree.elements(), ElementsAre(0));
  EXPECT_EQ(tree.ToString(), "0");
  EXPECT_EQ(tree.ToMultilineString(), "0");

  EXPECT_EQ(tree.AsView().ToString(), "0");
}

TEST_F(LeafTypeTreeTest, ForEachTest) {
  std::string result;
  auto append_to_result = [&result](Type* type, int64_t data,
                                    absl::Span<const int64_t> index) {
    absl::StrAppendFormat(&result, "[%s, %d, {%s}]", type->ToString(), data,
                          absl::StrJoin(index, ", "));
    return absl::OkStatus();
  };
  {
    LeafTypeTree<int64_t> tree(AsType("()"), 42);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(result, "");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(((())),(),(()))"), 42);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(result, "");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("bits[32]"), 42);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(result, "[bits[32], 42, {}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(bits[32], bits[64])"));
    tree.Set({0}, 23);
    tree.Set({1}, 42);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(result, "[bits[32], 23, {0}][bits[64], 42, {1}]");

    result.clear();
    XLS_ASSERT_OK(leaf_type_tree::ForEachIndex(tree.AsView({0}),
                                               append_to_result,
                                               /*index_prefix=*/{0}));
    EXPECT_EQ(result, "[bits[32], 23, {0}]");

    result.clear();
    XLS_ASSERT_OK(leaf_type_tree::ForEachIndex(tree.AsView({1}),
                                               append_to_result,
                                               /*index_prefix=*/{1}));
    EXPECT_EQ(result, "[bits[64], 42, {1}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("bits[32][2]"));
    tree.Set({0}, 23);
    tree.Set({1}, 42);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(result, "[bits[32], 23, {0}][bits[32], 42, {1}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(bits[32][2], ((bits[32])))"));
    tree.Set({0, 0}, 1);
    tree.Set({0, 1}, 2);
    tree.Set({1, 0, 0}, 3);
    result.clear();
    XLS_ASSERT_OK(
        leaf_type_tree::ForEachIndex(tree.AsView(), append_to_result));
    EXPECT_EQ(
        result,
        "[bits[32], 1, {0, 0}][bits[32], 2, {0, 1}][bits[32], 3, {1, 0, 0}]");

    result.clear();
    XLS_ASSERT_OK(leaf_type_tree::ForEachIndex(tree.AsView({0}),
                                               append_to_result,
                                               /*index_prefix=*/{0}));
    EXPECT_EQ(result, "[bits[32], 1, {0, 0}][bits[32], 2, {0, 1}]");

    result.clear();
    XLS_ASSERT_OK(leaf_type_tree::ForEachIndex(tree.AsView({0, 1}),
                                               append_to_result,
                                               /*index_prefix=*/{0, 1}));
    EXPECT_EQ(result, "[bits[32], 2, {0, 1}]");
  }
}

TEST_F(LeafTypeTreeTest, ForEachSubArray) {
  LeafTypeTree<int64_t> tree(AsType("bits[32][1][2][3]"));
  // The data value of each leaf is the linear index in the leaf array.
  int64_t i = 0;
  for (int64_t& element : tree.elements()) {
    element = i++;
  }
  std::vector<std::string> result;
  auto append_element = [&](LeafTypeTreeView<int64_t> ltt,
                            absl::Span<const int64_t> index) {
    result.push_back(absl::StrFormat(
        "X[%s]: %s = [%s]", absl::StrJoin(index, ","), ltt.type()->ToString(),
        absl::StrJoin(ltt.elements(), ",")));
    return absl::OkStatus();
  };

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 0,
                                                         append_element));
  EXPECT_THAT(result, ElementsAre("X[]: bits[32][1][2][3] = [0,1,2,3,4,5]"));

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 1,
                                                         append_element));
  EXPECT_THAT(result, ElementsAre("X[0]: bits[32][1][2] = [0,1]",
                                  "X[1]: bits[32][1][2] = [2,3]",
                                  "X[2]: bits[32][1][2] = [4,5]"));

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 2,
                                                         append_element));
  EXPECT_THAT(
      result,
      ElementsAre("X[0,0]: bits[32][1] = [0]", "X[0,1]: bits[32][1] = [1]",
                  "X[1,0]: bits[32][1] = [2]", "X[1,1]: bits[32][1] = [3]",
                  "X[2,0]: bits[32][1] = [4]", "X[2,1]: bits[32][1] = [5]"));

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 3,
                                                         append_element));
  EXPECT_THAT(
      result,
      ElementsAre("X[0,0,0]: bits[32] = [0]", "X[0,1,0]: bits[32] = [1]",
                  "X[1,0,0]: bits[32] = [2]", "X[1,1,0]: bits[32] = [3]",
                  "X[2,0,0]: bits[32] = [4]", "X[2,1,0]: bits[32] = [5]"));
}

TEST_F(LeafTypeTreeTest, ForEachSubArray1D) {
  LeafTypeTree<int64_t> tree(AsType("bits[32][5]"));
  // The data value of each leaf is the linear index in the leaf array.
  int64_t i = 0;
  for (int64_t& element : tree.elements()) {
    element = i++;
  }

  std::vector<std::string> result;
  auto append_element = [&](LeafTypeTreeView<int64_t> ltt,
                            absl::Span<const int64_t> index) {
    result.push_back(absl::StrFormat(
        "X[%s]: %s = [%s]", absl::StrJoin(index, ","), ltt.type()->ToString(),
        absl::StrJoin(ltt.elements(), ",")));
    return absl::OkStatus();
  };

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 0,
                                                         append_element));
  EXPECT_THAT(result, ElementsAre("X[]: bits[32][5] = [0,1,2,3,4]"));

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 1,
                                                         append_element));
  EXPECT_THAT(result,
              ElementsAre("X[0]: bits[32] = [0]", "X[1]: bits[32] = [1]",
                          "X[2]: bits[32] = [2]", "X[3]: bits[32] = [3]",
                          "X[4]: bits[32] = [4]"));
}

TEST_F(LeafTypeTreeTest, ForEachSubArrayBitsType) {
  LeafTypeTree<int64_t> tree(AsType("bits[32]"));
  // The data value of each leaf is the linear index in the leaf array.
  int64_t i = 0;
  for (int64_t& element : tree.elements()) {
    element = i++;
  }

  std::vector<std::string> result;
  auto append_element = [&](LeafTypeTreeView<int64_t> ltt,
                            absl::Span<const int64_t> index) {
    result.push_back(absl::StrFormat(
        "X[%s]: %s = [%s]", absl::StrJoin(index, ","), ltt.type()->ToString(),
        absl::StrJoin(ltt.elements(), ",")));
    return absl::OkStatus();
  };

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 0,
                                                         append_element));
  EXPECT_THAT(result, ElementsAre("X[]: bits[32] = [0]"));
}

TEST_F(LeafTypeTreeTest, ForEachArrayWithEmptyTupleLeaf) {
  LeafTypeTree<int64_t> tree(AsType("()[2]"));

  std::vector<std::string> result;
  auto append_element = [&](LeafTypeTreeView<int64_t> ltt,
                            absl::Span<const int64_t> index) {
    result.push_back(absl::StrFormat(
        "X[%s]: %s = [%s]", absl::StrJoin(index, ","), ltt.type()->ToString(),
        absl::StrJoin(ltt.elements(), ",")));
    return absl::OkStatus();
  };

  result.clear();
  XLS_ASSERT_OK(leaf_type_tree::ForEachSubArray<int64_t>(tree.AsView(), 1,
                                                         append_element));
  EXPECT_THAT(result, ElementsAre("X[0]: () = []", "X[1]: () = []"));
}

TEST_F(LeafTypeTreeTest, ForEachArrayErrors) {
  auto f = [](LeafTypeTreeView<int64_t> ltt, absl::Span<const int64_t> index) {
    return absl::OkStatus();
  };

  LeafTypeTree<int64_t> empty_tuple_tree(AsType("()"));
  EXPECT_THAT(
      leaf_type_tree::ForEachSubArray<int64_t>(empty_tuple_tree.AsView(), 1, f),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Type has fewer than 1 array dimensions: ()")));

  LeafTypeTree<int64_t> three_d_tree(AsType("bits[32][1][2][3]"));
  EXPECT_THAT(
      leaf_type_tree::ForEachSubArray<int64_t>(three_d_tree.AsView(), 4, f),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Type has fewer than 4 array dimensions: bits[32][1][2][3]")));

  EXPECT_THAT(
      leaf_type_tree::ForEachSubArray<int64_t>(
          three_d_tree.AsView(), 2,
          [](LeafTypeTreeView<int64_t> ltt, absl::Span<const int64_t> index) {
            return absl::InternalError("Oh noes!");
          }),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Oh noes!")));
}

TEST_F(LeafTypeTreeTest, ArrayOfEmptyTuples) {
  LeafTypeTree<int64_t> tree(AsType("()[5]"));
  EXPECT_EQ(tree.ToString(), "[(), (), (), (), ()]");
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({})).ToString(),
            "[(), (), (), (), ()]");
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({0})).ToString(), "()");
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({4})).ToString(), "()");
}

TEST_F(LeafTypeTreeTest, OutOfBoundsTest) {
  LeafTypeTree<int64_t> tree(AsType("(bits[32], (), (()))"));
  EXPECT_THAT(tree.AsView({1}).elements(), ElementsAre());
  EXPECT_THAT(tree.AsView({2}).elements(), ElementsAre());
  EXPECT_THAT(tree.AsView({2, 0}).elements(), ElementsAre());
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({1})).ToString(), "()");
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({2})).ToString(), "(())");
  EXPECT_EQ(leaf_type_tree::Clone(tree.AsView({2, 0})).ToString(), "()");
}

TEST_F(LeafTypeTreeTest, IncrementArrayIndex0D) {
  std::vector<int64_t> bounds;
  std::vector<int64_t> index;
  EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
  EXPECT_TRUE(index.empty());
}

TEST_F(LeafTypeTreeTest, IncrementArrayIndex1D) {
  {
    std::vector<int64_t> bounds = {1};
    std::vector<int64_t> index = {0};
    EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0));
  }
  {
    std::vector<int64_t> bounds = {3};
    std::vector<int64_t> index = {0};
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(1));
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(2));
    EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0));
  }
}

TEST_F(LeafTypeTreeTest, IncrementArrayIndex2D) {
  {
    std::vector<int64_t> bounds = {1, 1};
    std::vector<int64_t> index = {0, 0};
    EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 0));
  }
  {
    std::vector<int64_t> bounds = {2, 3};
    std::vector<int64_t> index = {0, 0};
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 1));
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 2));
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(1, 0));
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(1, 1));
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(1, 2));
    EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 0));
  }
  {
    std::vector<int64_t> bounds = {1, 2};
    std::vector<int64_t> index = {0, 0};
    EXPECT_FALSE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 1));
    EXPECT_TRUE(leaf_type_tree_internal::IncrementArrayIndex(bounds, &index));
    EXPECT_THAT(index, ElementsAre(0, 0));
  }
}

TEST_F(LeafTypeTreeTest, GetSubArraySize) {
  XLS_ASSERT_OK_AND_ASSIGN(
      leaf_type_tree_internal::SubArraySize bits_size,
      leaf_type_tree_internal::GetSubArraySize(AsType("bits[32]"), 0));
  EXPECT_EQ(bits_size.type->ToString(), "bits[32]");
  EXPECT_TRUE(bits_size.bounds.empty());
  EXPECT_EQ(bits_size.element_count, 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      leaf_type_tree_internal::SubArraySize one_d_array_0,
      leaf_type_tree_internal::GetSubArraySize(AsType("bits[32][2]"), 0));
  EXPECT_EQ(one_d_array_0.type->ToString(), "bits[32][2]");
  EXPECT_THAT(one_d_array_0.bounds, ElementsAre());
  EXPECT_EQ(one_d_array_0.element_count, 2);

  XLS_ASSERT_OK_AND_ASSIGN(
      leaf_type_tree_internal::SubArraySize one_d_array_1,
      leaf_type_tree_internal::GetSubArraySize(AsType("bits[32][2]"), 1));
  EXPECT_EQ(one_d_array_1.type->ToString(), "bits[32]");
  EXPECT_THAT(one_d_array_1.bounds, ElementsAre(2));
  EXPECT_EQ(one_d_array_1.element_count, 1);

  XLS_ASSERT_OK_AND_ASSIGN(leaf_type_tree_internal::SubArraySize two_d_array_0,
                           leaf_type_tree_internal::GetSubArraySize(
                               AsType("(bits[1], bits[2])[7][3]"), 0));
  EXPECT_EQ(two_d_array_0.type->ToString(), "(bits[1], bits[2])[7][3]");
  EXPECT_THAT(two_d_array_0.bounds, ElementsAre());
  EXPECT_EQ(two_d_array_0.element_count, 42);

  XLS_ASSERT_OK_AND_ASSIGN(leaf_type_tree_internal::SubArraySize two_d_array_1,
                           leaf_type_tree_internal::GetSubArraySize(
                               AsType("(bits[1], bits[2])[7][3]"), 1));
  EXPECT_EQ(two_d_array_1.type->ToString(), "(bits[1], bits[2])[7]");
  EXPECT_THAT(two_d_array_1.bounds, ElementsAre(3));
  EXPECT_EQ(two_d_array_1.element_count, 14);

  XLS_ASSERT_OK_AND_ASSIGN(leaf_type_tree_internal::SubArraySize two_d_array_2,
                           leaf_type_tree_internal::GetSubArraySize(
                               AsType("(bits[1], bits[2])[7][3]"), 2));
  EXPECT_EQ(two_d_array_2.type->ToString(), "(bits[1], bits[2])");
  EXPECT_THAT(two_d_array_2.bounds, ElementsAre(3, 7));
  EXPECT_EQ(two_d_array_2.element_count, 2);
}

TEST_F(LeafTypeTreeTest, CreateTuple) {
  LeafTypeTree<std::string> u32_tree(AsType("bits[32]"),
                                     std::vector<std::string>({"foo"}));
  LeafTypeTree<std::string> subtuple_tree(
      AsType("(bits[11], bits[12])"), std::vector<std::string>({"baz", "qux"}));
  LeafTypeTree<std::string> empty_tuple_tree(AsType("()"));

  XLS_ASSERT_OK_AND_ASSIGN(
      LeafTypeTree<std::string> tree,
      leaf_type_tree::CreateTuple<std::string>(
          AsType("(bits[32], (bits[11], bits[12]), ())")->AsTupleOrDie(),
          {u32_tree.AsView(), subtuple_tree.AsView(),
           empty_tuple_tree.AsView()}));
  EXPECT_EQ(tree.ToString(), "(foo, (baz, qux), ())");

  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> empty_tree,
                           leaf_type_tree::CreateTuple<std::string>(
                               AsType("(), ())")->AsTupleOrDie(), {}));
  EXPECT_EQ(empty_tree.ToString(), "()");
}

TEST_F(LeafTypeTreeTest, CreateArray) {
  LeafTypeTree<std::string> foo(AsType("bits[32]"),
                                std::vector<std::string>({"foo"}));
  LeafTypeTree<std::string> bar(AsType("bits[32]"),
                                std::vector<std::string>({"bar"}));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> tree,
                           leaf_type_tree::CreateArray<std::string>(
                               AsType("bits[32][3]")->AsArrayOrDie(),
                               {foo.AsView(), bar.AsView(), foo.AsView()}));
  EXPECT_EQ(tree.ToString(), "[foo, bar, foo]");
}

TEST_F(LeafTypeTreeTest, ConcatArray) {
  LeafTypeTree<std::string> alpha(AsType("bits[32]"),
                                  std::vector<std::string>({"alpha"}));
  LeafTypeTree<std::string> bravo(AsType("bits[32]"),
                                  std::vector<std::string>({"bravo"}));
  LeafTypeTree<std::string> charlie(AsType("bits[32]"),
                                    std::vector<std::string>({"charlie"}));
  LeafTypeTree<std::string> delta(AsType("bits[32]"),
                                  std::vector<std::string>({"delta"}));
  LeafTypeTree<std::string> echo(AsType("bits[32]"),
                                 std::vector<std::string>({"echo"}));
  LeafTypeTree<std::string> foxtrot(AsType("bits[32]"),
                                    std::vector<std::string>({"foxtrot"}));
  LeafTypeTree<std::string> golf(AsType("bits[32]"),
                                 std::vector<std::string>({"golf"}));
  LeafTypeTree<std::string> hotel(AsType("bits[32]"),
                                  std::vector<std::string>({"hotel"}));
  LeafTypeTree<std::string> india(AsType("bits[32]"),
                                  std::vector<std::string>({"india"}));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> ab,
                           leaf_type_tree::CreateArray<std::string>(
                               AsType("bits[32][2]")->AsArrayOrDie(),
                               {alpha.AsView(), bravo.AsView()}));
  XLS_ASSERT_OK_AND_ASSIGN(
      LeafTypeTree<std::string> cde,
      leaf_type_tree::CreateArray<std::string>(
          AsType("bits[32][3]")->AsArrayOrDie(),
          {charlie.AsView(), delta.AsView(), echo.AsView()}));
  XLS_ASSERT_OK_AND_ASSIGN(
      LeafTypeTree<std::string> fghi,
      leaf_type_tree::CreateArray<std::string>(
          AsType("bits[32][4]")->AsArrayOrDie(),
          {foxtrot.AsView(), golf.AsView(), hotel.AsView(), india.AsView()}));
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> abcdefghi,
                           leaf_type_tree::ConcatArray<std::string>(
                               AsType("bits[32][9]")->AsArrayOrDie(),
                               {ab.AsView(), cde.AsView(), fghi.AsView()}));
  EXPECT_EQ(
      abcdefghi.ToString(),
      "[alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india]");
}

TEST_F(LeafTypeTreeTest, SliceArray) {
  LeafTypeTree<std::string> alpha(AsType("bits[32]"),
                                  std::vector<std::string>({"alpha"}));
  LeafTypeTree<std::string> bravo(AsType("bits[32]"),
                                  std::vector<std::string>({"bravo"}));
  LeafTypeTree<std::string> charlie(AsType("bits[32]"),
                                    std::vector<std::string>({"charlie"}));
  LeafTypeTree<std::string> delta(AsType("bits[32]"),
                                  std::vector<std::string>({"delta"}));
  LeafTypeTree<std::string> echo(AsType("bits[32]"),
                                 std::vector<std::string>({"echo"}));
  LeafTypeTree<std::string> foxtrot(AsType("bits[32]"),
                                    std::vector<std::string>({"foxtrot"}));
  LeafTypeTree<std::string> golf(AsType("bits[32]"),
                                 std::vector<std::string>({"golf"}));
  LeafTypeTree<std::string> hotel(AsType("bits[32]"),
                                  std::vector<std::string>({"hotel"}));
  LeafTypeTree<std::string> india(AsType("bits[32]"),
                                  std::vector<std::string>({"india"}));
  XLS_ASSERT_OK_AND_ASSIGN(
      LeafTypeTree<std::string> abcdefghi,
      leaf_type_tree::CreateArray<std::string>(
          AsType("bits[32][9]")->AsArrayOrDie(),
          {alpha.AsView(), bravo.AsView(), charlie.AsView(), delta.AsView(),
           echo.AsView(), foxtrot.AsView(), golf.AsView(), hotel.AsView(),
           india.AsView()}));

  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> ghiiii,
                           leaf_type_tree::SliceArray<std::string>(
                               AsType("bits[32][6]")->AsArrayOrDie(),
                               abcdefghi.AsView(), /*start=*/6));
  EXPECT_EQ(ghiiii.ToString(), "[golf, hotel, india, india, india, india]");
  XLS_ASSERT_OK_AND_ASSIGN(LeafTypeTree<std::string> bcd,
                           leaf_type_tree::SliceArray<std::string>(
                               AsType("bits[32][3]")->AsArrayOrDie(),
                               abcdefghi.AsView(), /*start=*/1));
  EXPECT_EQ(bcd.ToString(), "[bravo, charlie, delta]");
  // Don't be confused by overflow of start
  XLS_ASSERT_OK_AND_ASSIGN(
      LeafTypeTree<std::string> iiiiii,
      leaf_type_tree::SliceArray<std::string>(
          AsType("bits[32][6]")->AsArrayOrDie(), abcdefghi.AsView(),
          /*start=*/std::numeric_limits<int64_t>::max() - 2));
  EXPECT_EQ(iiiiii.ToString(), "[india, india, india, india, india, india]");
}

TEST_F(LeafTypeTreeTest, ReplaceElements) {
  LeafTypeTree<std::string> foo(AsType("bits[32]"),
                                std::vector<std::string>({"foo"}));
  LeafTypeTree<std::string> bar(AsType("bits[32]"),
                                std::vector<std::string>({"bar"}));
  EXPECT_EQ(foo.ToString(), "foo");
  XLS_ASSERT_OK(
      leaf_type_tree::ReplaceElements(foo.AsMutableView(), bar.AsView()));
  EXPECT_EQ(foo.ToString(), "bar");

  LeafTypeTree<std::string> tree(AsType("(bits[32], (bits[11], bits[12]), ())"),
                                 std::vector<std::string>({"a", "b", "c"}));
  LeafTypeTree<std::string> subtree(AsType("(bits[11], bits[12])"),
                                    std::vector<std::string>({"x", "y"}));

  EXPECT_EQ(tree.ToString(), "(a, (b, c), ())");
  XLS_ASSERT_OK(leaf_type_tree::ReplaceElements(tree.AsMutableView({1}),
                                                subtree.AsView()));
  EXPECT_EQ(tree.ToString(), "(a, (x, y), ())");
  XLS_ASSERT_OK(
      leaf_type_tree::ReplaceElements(tree.AsMutableView({0}), bar.AsView()));
  EXPECT_EQ(tree.ToString(), "(bar, (x, y), ())");
}

TEST_F(LeafTypeTreeTest, IteratorTest) {
  {
    LeafTypeTree<int64_t> tree(AsType("(bits[32][2], (), (())[3], bits[2])"));
    leaf_type_tree_internal::LeafTypeTreeIterator it(tree.type());
    EXPECT_EQ(it.ToString(),
              "root_type=(bits[32][2], (), (())[3], bits[2]), "
              "leaf_type=bits[32], type_index={0,0}, linear_index=0");
    it.Advance();
    EXPECT_EQ(it.ToString(),
              "root_type=(bits[32][2], (), (())[3], bits[2]), "
              "leaf_type=bits[32], type_index={0,1}, linear_index=1");
    it.Advance();
    EXPECT_EQ(it.ToString(),
              "root_type=(bits[32][2], (), (())[3], bits[2]), "
              "leaf_type=bits[2], type_index={3}, linear_index=2");
    it.Advance();
    EXPECT_EQ(it.ToString(),
              "root_type=(bits[32][2], (), (())[3], bits[2]), END");
    EXPECT_TRUE(it.AtEnd());
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(()[18], bits[14], bits[7])"));
    leaf_type_tree_internal::LeafTypeTreeIterator it(tree.type());
    EXPECT_EQ(it.ToString(),
              "root_type=(()[18], bits[14], bits[7]), leaf_type=bits[14], "
              "type_index={1}, linear_index=0");
    it.Advance();
    EXPECT_EQ(it.ToString(),
              "root_type=(()[18], bits[14], bits[7]), leaf_type=bits[7], "
              "type_index={2}, linear_index=1");
    it.Advance();
    EXPECT_EQ(it.ToString(), "root_type=(()[18], bits[14], bits[7]), END");
    EXPECT_TRUE(it.AtEnd());
  }
  {
    LeafTypeTree<int64_t> empty_tuple(AsType("()"));
    leaf_type_tree_internal::LeafTypeTreeIterator it(empty_tuple.type());
    EXPECT_EQ(it.ToString(), "root_type=(), END");
    EXPECT_TRUE(it.AtEnd());
  }
  {
    LeafTypeTree<int64_t> empty_tuple_tuple(AsType("(())"));
    leaf_type_tree_internal::LeafTypeTreeIterator it(empty_tuple_tuple.type());
    EXPECT_EQ(it.ToString(), "root_type=(()), END");
    EXPECT_TRUE(it.AtEnd());
  }
  {
    LeafTypeTree<int64_t> bits(AsType("bits[42]"));
    leaf_type_tree_internal::LeafTypeTreeIterator it(bits.type());
    EXPECT_EQ(it.ToString(),
              "root_type=bits[42], leaf_type=bits[42], type_index={}, "
              "linear_index=0");
    it.Advance();
    EXPECT_EQ(it.ToString(), "root_type=bits[42], END");
    EXPECT_TRUE(it.AtEnd());
  }
}

TEST_F(LeafTypeTreeTest, ZipMultiple) {
  Type* type = AsType("(bits[32][2], (), (())[3], bits[2])");
  LeafTypeTree<int64_t> input0(type, {10, 20, 30});
  LeafTypeTree<int64_t> input1(type, {11, 21, 31});
  LeafTypeTree<int64_t> input2(type, {12, 22, 32});
  LeafTypeTree<std::string> result =
      leaf_type_tree::ZipIndex<std::string, int64_t>(
          {input0.AsView(), input1.AsView(), input2.AsView()},
          [](Type* leaf_type, absl::Span<const int64_t* const> inputs,
             absl::Span<const int64_t> index) -> absl::StatusOr<std::string> {
            return absl::StrFormat(
                "%s:%s@{%s}", leaf_type->ToString(),
                absl::StrJoin(inputs, ",",
                              [](std::string* s, const int64_t* v) {
                                absl::StrAppend(s, *v);
                              }),
                absl::StrJoin(index, ","));
          })
          .value();
  EXPECT_EQ(result.ToMultilineString(),
            R"((
  [
    bits[32]:10,11,12@{0,0},
    bits[32]:20,21,22@{0,1}
  ],
  (),
  [
    (
      ()
    ),
    (
      ()
    ),
    (
      ()
    )
  ],
  bits[2]:30,31,32@{3}
))");
}

TEST_F(LeafTypeTreeTest, CreateFromFunction) {
  Type* type = AsType("(bits[32][2], bits[2])");
  LeafTypeTree<std::string> result =
      LeafTypeTree<std::string>::CreateFromFunction(
          type,
          [](Type* leaf_type,
             absl::Span<const int64_t> index) -> absl::StatusOr<std::string> {
            return absl::StrFormat("%s@{%s}", leaf_type->ToString(),
                                   absl::StrJoin(index, ","));
          })
          .value();
  EXPECT_EQ(result.ToString(),
            "([bits[32]@{0,0}, bits[32]@{0,1}], bits[2]@{1})");
}

TEST_F(LeafTypeTreeTest, CreateFromFunctionSimple) {
  Type* type = AsType("(bits[32][2], bits[2])");
  LeafTypeTree<std::string> result =
      LeafTypeTree<std::string>::CreateFromFunction(
          type,
          [](Type* leaf_type) -> absl::StatusOr<std::string> {
            return absl::StrCat(leaf_type->ToString());
          })
          .value();
  EXPECT_EQ(result.ToString(), "([bits[32], bits[32]], bits[2])");
}

}  // namespace
}  // namespace xls
