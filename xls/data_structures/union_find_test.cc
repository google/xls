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

#include "xls/data_structures/union_find.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::AnyOf;

TEST(UnionFindTest, UnionFind) {
  UnionFind<char> uf;
  uf.Insert('a');
  uf.Insert('b');
  uf.Insert('c');
  uf.Insert('d');

  EXPECT_EQ(uf.Find('a'), 'a');
  EXPECT_EQ(uf.Find('b'), 'b');
  EXPECT_EQ(uf.Find('c'), 'c');
  EXPECT_EQ(uf.Find('d'), 'd');

  // Inserting an element a second time should have no effect.
  uf.Insert('a');
  EXPECT_EQ(uf.Find('a'), 'a');
  EXPECT_EQ(uf.Find('b'), 'b');
  EXPECT_EQ(uf.Find('c'), 'c');
  EXPECT_EQ(uf.Find('d'), 'd');

  // Unioning an element with itself should have no effect.
  uf.Union('a', 'a');
  EXPECT_EQ(uf.Find('a'), 'a');
  EXPECT_EQ(uf.Find('b'), 'b');
  EXPECT_EQ(uf.Find('c'), 'c');
  EXPECT_EQ(uf.Find('d'), 'd');

  uf.Union('a', 'b');
  EXPECT_EQ(uf.Find('a'), uf.Find('b'));
  EXPECT_THAT(uf.Find('a'), AnyOf('a', 'b'));
  EXPECT_EQ(uf.Find('c'), 'c');
  EXPECT_EQ(uf.Find('d'), 'd');

  uf.Union('b', 'c');
  EXPECT_EQ(uf.Find('a'), uf.Find('b'));
  EXPECT_EQ(uf.Find('a'), uf.Find('c'));
  EXPECT_THAT(uf.Find('a'), AnyOf('a', 'b', 'c'));
  EXPECT_EQ(uf.Find('d'), 'd');

  uf.Union('a', 'd');
  EXPECT_EQ(uf.Find('a'), uf.Find('b'));
  EXPECT_EQ(uf.Find('a'), uf.Find('c'));
  EXPECT_EQ(uf.Find('a'), uf.Find('d'));
  EXPECT_THAT(uf.Find('a'), AnyOf('a', 'b', 'c', 'd'));
}

}  // namespace
}  // namespace xls
