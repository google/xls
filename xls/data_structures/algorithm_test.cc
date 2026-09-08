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

#include "xls/data_structures/algorithm.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::ElementsAre;

TEST(NodeUtilTest, IndicesWhere) {
  EXPECT_THAT(IndicesWhere<int>({2, 5, 7, 4, -1}, [](int i) { return i > 3; }),
              ElementsAre(1, 2, 3));
  EXPECT_THAT(
      IndicesWhereNot<int>({2, 5, 7, 4, -1}, [](int i) { return i > 3; }),
      ElementsAre(0, 4));
  EXPECT_THAT(
      IndicesWhere<int>({2, 5, 7, 4, -1}, [](int i) { return i > 12345; }),
      ElementsAre());
  EXPECT_THAT(
      IndicesWhereNot<int>({2, 5, 7, 4, -1}, [](int i) { return i > 12345; }),
      ElementsAre(0, 1, 2, 3, 4));
}

TEST(NodeUtilTest, GatherFromSequence) {
  EXPECT_THAT(GatherFromSequence<char>({'a', 'b', 'c', 'd', 'e'}, {0, 1, 3}),
              ElementsAre('a', 'b', 'd'));
  EXPECT_THAT(GatherFromSequence<char>({'a', 'b', 'c', 'd', 'e'}, {3, 0, 1}),
              ElementsAre('d', 'a', 'b'));
  EXPECT_THAT(GatherFromSequence<char>({'a', 'b', 'c', 'd', 'e'}, {2, 2}),
              ElementsAre('c', 'c'));
  EXPECT_THAT(GatherFromSequence<char>({'a', 'b', 'c', 'd', 'e'}, {}),
              ElementsAre());
  EXPECT_THAT(GatherFromSequence<char>({}, {}), ElementsAre());
}

TEST(NodeUtilTest, SortByKeyTriviallyCopyable) {
  std::vector<int> v = {10, 5, 20, 15};
  int key_call_count = 0;
  SortByKey(
      v,
      [&](int x) {
        ++key_call_count;
        return -x;
      },
      std::greater<>{});

  EXPECT_EQ(key_call_count, 4);
  EXPECT_THAT(v, ElementsAre(5, 10, 15, 20));
}

struct HeavyObject {
  int id;
  int key;
  int data[1024];

  HeavyObject(int id, int key) : id(id), key(key) {}

  bool operator==(const HeavyObject& o) const {
    return id == o.id && key == o.key;
  }
};

TEST(NodeUtilTest, SortByKeyInPlaceLargeType) {
  std::vector<HeavyObject> vec;
  vec.push_back(HeavyObject(1, 30));
  vec.push_back(HeavyObject(2, 10));
  vec.push_back(HeavyObject(3, 20));

  int key_call_count = 0;
  SortByKey(vec, [&](const HeavyObject& item) {
    ++key_call_count;
    return item.key;
  });

  EXPECT_EQ(key_call_count, 3);
  EXPECT_EQ(vec[0].id, 2);
  EXPECT_EQ(vec[1].id, 3);
  EXPECT_EQ(vec[2].id, 1);
}

}  // namespace
}  // namespace xls
