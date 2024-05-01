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

#include "xls/ir/name_uniquer.h"

#include "gtest/gtest.h"

namespace xls {
namespace {

TEST(NameUniquerTest, SimpleUniquer) {
  NameUniquer uniquer("__");

  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("foo__3", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("qux", uniquer.GetSanitizedUniqueName("qux"));

  EXPECT_EQ("baz__42", uniquer.GetSanitizedUniqueName("baz__42"));
  EXPECT_EQ("baz", uniquer.GetSanitizedUniqueName("baz"));
  EXPECT_EQ("baz__1", uniquer.GetSanitizedUniqueName("baz"));

  EXPECT_EQ("abc__2", uniquer.GetSanitizedUniqueName("abc__2"));
  EXPECT_EQ("abc__4", uniquer.GetSanitizedUniqueName("abc__4"));
  EXPECT_EQ("abc", uniquer.GetSanitizedUniqueName("abc"));
  EXPECT_EQ("abc__1", uniquer.GetSanitizedUniqueName("abc"));
  EXPECT_EQ("abc__3", uniquer.GetSanitizedUniqueName("abc__2"));
  EXPECT_EQ("abc__5", uniquer.GetSanitizedUniqueName("abc"));
}

TEST(NameUniquerTest, DifferentSeparator) {
  NameUniquer uniquer(".");

  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo.1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo.2", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("foo.3", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar.1", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("qux", uniquer.GetSanitizedUniqueName("qux"));
}

TEST(NameUniquerTest, NumericSuffixes) {
  NameUniquer uniquer("__");

  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__3", uniquer.GetSanitizedUniqueName("foo__3"));
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo__3"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo__3"));
  EXPECT_EQ("foo__4", uniquer.GetSanitizedUniqueName("foo__3"));
  EXPECT_EQ("foo__5", uniquer.GetSanitizedUniqueName("foo__1"));
  EXPECT_EQ("foo__6", uniquer.GetSanitizedUniqueName("foo"));

  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__3", uniquer.GetSanitizedUniqueName("bar__3"));
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__2", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__4", uniquer.GetSanitizedUniqueName("bar"));
}

TEST(NameUniquerTest, SanitizeNames) {
  NameUniquer uniquer("__", {"res1", "res2", "_res", "__res"});
  EXPECT_EQ("CamelCase", uniquer.GetSanitizedUniqueName("CamelCase"));
  EXPECT_EQ("snake_case", uniquer.GetSanitizedUniqueName("snake_case"));
  EXPECT_EQ("a1234", uniquer.GetSanitizedUniqueName("a1234"));
  EXPECT_EQ("_33", uniquer.GetSanitizedUniqueName("_33"));

  EXPECT_EQ("_", uniquer.GetSanitizedUniqueName(" "));
  EXPECT_EQ("___1", uniquer.GetSanitizedUniqueName("."));
  EXPECT_EQ("_a_b_c", uniquer.GetSanitizedUniqueName(" a b c"));
  EXPECT_EQ("_A5_3D", uniquer.GetSanitizedUniqueName(" A5%3D"));
  EXPECT_EQ("_A5_3D__1", uniquer.GetSanitizedUniqueName(" A5-3D"));

  EXPECT_EQ("_42", uniquer.GetSanitizedUniqueName("42"));
  EXPECT_EQ("_42__1", uniquer.GetSanitizedUniqueName("42"));
  EXPECT_EQ("_42__2", uniquer.GetSanitizedUniqueName("_42"));

  EXPECT_EQ("_res1", uniquer.GetSanitizedUniqueName("res1"));
  EXPECT_EQ("_res1__1", uniquer.GetSanitizedUniqueName("res1"));
  EXPECT_EQ("_res2", uniquer.GetSanitizedUniqueName("res2"));
  EXPECT_EQ("___res", uniquer.GetSanitizedUniqueName("_res"));
}

TEST(NameUniquerTest, CornerCases) {
  NameUniquer uniquer("__");
  EXPECT_EQ("name", uniquer.GetSanitizedUniqueName(""));
  EXPECT_EQ("name__1", uniquer.GetSanitizedUniqueName(""));
  EXPECT_EQ("__", uniquer.GetSanitizedUniqueName("__"));
  EXPECT_EQ("____1", uniquer.GetSanitizedUniqueName("__"));
  EXPECT_EQ("name__3", uniquer.GetSanitizedUniqueName("__3"));
  EXPECT_EQ("name__2", uniquer.GetSanitizedUniqueName(""));
  EXPECT_EQ("name__4", uniquer.GetSanitizedUniqueName(""));
  EXPECT_EQ("name__5", uniquer.GetSanitizedUniqueName("__2"));
}

TEST(NameUniquerTest, IsValidIdentifier) {
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("foo"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("foo_bar"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("foo_bar33"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("_baz"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("_"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("___"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("_42"));
  EXPECT_TRUE(NameUniquer::IsValidIdentifier("_42abc"));

  EXPECT_FALSE(NameUniquer::IsValidIdentifier(""));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier(" "));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("\n"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("1"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("42"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("42abcd"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("42_"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("foo-bar"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("foo bar"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("foo&bar"));
  EXPECT_FALSE(NameUniquer::IsValidIdentifier("foo+bar"));
}

}  // namespace
}  // namespace xls
