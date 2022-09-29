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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Eq;

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
  EXPECT_THAT(tree.GetSubelements({}), ElementsAre(0));

  tree.Set({}, 42);
  EXPECT_EQ(tree.Get({}), 42);
  EXPECT_THAT(tree.elements(), ElementsAre(42));
  EXPECT_EQ(tree.ToString(), "42");
  EXPECT_EQ(tree.ToMultilineString(), "42");

  tree.Set({}, 123);
  EXPECT_EQ(tree.Get({}), 123);
  EXPECT_THAT(tree.elements(), ElementsAre(123));

  LeafTypeTree<int64_t> subtree = tree.CopySubtree({});
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

  EXPECT_EQ(tree.ToString([](const Bits& b) { return b.ToString(); }),
            "(0, 33, 0)");
  EXPECT_EQ(tree.ToMultilineString([](const Bits& b) { return b.ToString(); }),
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

  EXPECT_THAT(tree.GetSubelements({}),
              ElementsAre(UBits(7, 8), UBits(33, 44), UBits(1, 1)));
  EXPECT_THAT(tree.GetSubelements({0}), ElementsAre(UBits(7, 8)));
  EXPECT_THAT(tree.GetSubelements({1}), ElementsAre(UBits(33, 44)));
  EXPECT_THAT(tree.GetSubelements({2}), ElementsAre(UBits(1, 1)));

  LeafTypeTree<Bits> subtree0 = tree.CopySubtree({});
  EXPECT_EQ(subtree0.type()->ToString(), "(bits[123], bits[2], bits[42])");
  EXPECT_THAT(subtree0.elements(),
              ElementsAre(UBits(7, 8), UBits(33, 44), UBits(1, 1)));

  LeafTypeTree<Bits> subtree1 = tree.CopySubtree({1});
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
  EXPECT_THAT(tree.GetSubelements({3}), ElementsAre(5));

  EXPECT_EQ(tree.ToString(), "[0, 3, 4, 5, 6]");
  EXPECT_EQ(tree.ToMultilineString(),
            R"([
  0,
  3,
  4,
  5,
  6
])");

  LeafTypeTree<int64_t> subtree = tree.CopySubtree({});
  EXPECT_THAT(subtree.elements(), ElementsAre(0, 3, 4, 5, 6));
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

  EXPECT_THAT(tree.GetSubelements({0}), ElementsAre(0, 0, 3));
  EXPECT_THAT(tree.GetSubelements({1}), ElementsAre(0, 42, 0, 0, 77));
  EXPECT_THAT(tree.GetSubelements({1, 1}), ElementsAre(42, 0, 0));

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
      tree.Map<int64_t>([](int64_t x) { return x + 1; });
  EXPECT_THAT(mapped.elements(), ElementsAre(1, 1, 4, 1, 43, 1, 1, 78));

  LeafTypeTree<int64_t> other_tree(tree.type());
  other_tree.Set({0, 1}, 5);
  other_tree.Set({1, 1, 1, 0}, 12);
  LeafTypeTree<int64_t> zipped = LeafTypeTree<int64_t>::Zip<int64_t, int64_t>(
      [](int64_t x, int64_t y) { return std::max(x, y); }, tree, other_tree);
  EXPECT_THAT(zipped.elements(), ElementsAre(0, 5, 3, 0, 42, 12, 0, 77));

  LeafTypeTree<int64_t> subtree = tree.CopySubtree({1, 1});
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

  LeafTypeTree<int64_t> subtree = tree.CopySubtree({2});
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
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(result, "");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(((())),(),(()))"), 42);
    result.clear();
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(result, "");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("bits[32]"), 42);
    result.clear();
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(result, "[bits[32], 42, {}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(bits[32], bits[64])"));
    tree.Set({0}, 23);
    tree.Set({1}, 42);
    result.clear();
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(result, "[bits[32], 23, {0}][bits[64], 42, {1}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("bits[32][2]"));
    tree.Set({0}, 23);
    tree.Set({1}, 42);
    result.clear();
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(result, "[bits[32], 23, {0}][bits[32], 42, {1}]");
  }
  {
    LeafTypeTree<int64_t> tree(AsType("(bits[32][2], ((bits[32])))"));
    tree.Set({0, 0}, 1);
    tree.Set({0, 1}, 2);
    tree.Set({1, 0, 0}, 3);
    result.clear();
    XLS_ASSERT_OK(tree.ForEach(append_to_result));
    EXPECT_EQ(
        result,
        "[bits[32], 1, {0, 0}][bits[32], 2, {0, 1}][bits[32], 3, {1, 0, 0}]");
  }
}

TEST_F(LeafTypeTreeTest, ArrayOfEmptyTuples) {
  LeafTypeTree<int64_t> tree(AsType("()[5]"));
  EXPECT_EQ(tree.ToString(), "[(), (), (), (), ()]");
  EXPECT_EQ(tree.CopySubtree({}).ToString(), "[(), (), (), (), ()]");
  EXPECT_EQ(tree.CopySubtree({0}).ToString(), "()");
  EXPECT_EQ(tree.CopySubtree({4}).ToString(), "()");
}

TEST_F(LeafTypeTreeTest, OutOfBoundsTest) {
  LeafTypeTree<int64_t> tree(AsType("(bits[32], (), (()))"));
  EXPECT_THAT(tree.GetSubelements({1}), ElementsAre());
  EXPECT_THAT(tree.GetSubelements({2}), ElementsAre());
  EXPECT_THAT(tree.GetSubelements({2, 0}), ElementsAre());
  EXPECT_EQ(tree.CopySubtree({1}).ToString(), "()");
  EXPECT_EQ(tree.CopySubtree({2}).ToString(), "(())");
  EXPECT_EQ(tree.CopySubtree({2, 0}).ToString(), "()");
}

}  // namespace
}  // namespace xls
