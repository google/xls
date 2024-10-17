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

}  // namespace
}  // namespace xls
