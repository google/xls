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

#include "xls/data_structures/union_find_map.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"

namespace xls {
namespace {

TEST(UnionFindMapTest, BasicUsage) {
  UnionFindMap<char, int32_t> ufm;

  auto add = [](int32_t x, int32_t y) { return x + y; };

  ufm.Insert('a', 7);
  ufm.Insert('b', 5);
  ufm.Insert('c', 10);
  ufm.Insert('d', 10);
  ufm.Insert('a', 3);       // test overwriting
  ufm.Insert('a', 2, add);  // test merging

  EXPECT_TRUE(ufm.Union('a', 'b', add));
  EXPECT_TRUE(ufm.Union('c', 'd', add));
  EXPECT_FALSE(ufm.Union('a', 'x', add));
  EXPECT_FALSE(ufm.Union('x', 'a', add));
  EXPECT_FALSE(ufm.Union('x', 'y', add));

  EXPECT_TRUE(ufm.Contains('a'));
  EXPECT_TRUE(ufm.Contains('b'));
  EXPECT_TRUE(ufm.Contains('c'));
  EXPECT_TRUE(ufm.Contains('d'));
  EXPECT_FALSE(ufm.Contains('e'));
  EXPECT_FALSE(ufm.Contains('f'));

  EXPECT_EQ(ufm.Find('a'), ufm.Find('b'));
  EXPECT_EQ(ufm.Find('c'), ufm.Find('d'));
  EXPECT_EQ(ufm.Find('a')->second, 10);
  EXPECT_EQ(ufm.Find('c')->second, 20);
  EXPECT_FALSE(ufm.Find('e'));
  EXPECT_FALSE(ufm.Find('f'));

  {
    std::vector<char> keys{'a', 'b', 'c', 'd'};
    EXPECT_EQ(ufm.GetKeys(), keys);
  }

  {
    absl::flat_hash_set<char> reps{'a', 'c'};
    EXPECT_EQ(ufm.GetRepresentatives(), reps);
  }

  ufm.Insert('e', 10);
  ufm.Insert('f', 20);
  EXPECT_TRUE(ufm.Union('d', 'e', add));
  EXPECT_TRUE(ufm.Union('e', 'f', add));
  EXPECT_TRUE(ufm.Union('a', 'c', add));

  {
    std::vector<char> keys{'a', 'b', 'c', 'd', 'e', 'f'};
    std::optional<std::pair<char, int32_t>> expected({'c', 60});
    for (char key : keys) {
      std::optional<std::pair<char, int32_t>> actual(ufm.Find(key));
      EXPECT_EQ(actual, expected);
    }
  }
}

}  // namespace
}  // namespace xls
