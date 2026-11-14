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

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace xls {
namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using testing::Not;
using testing::Optional;

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

TEST(NameUniquerTest, ReserveIdentifier) {
  NameUniquer uniquer("__");
  EXPECT_THAT(uniquer.ReserveIdentifier("foo"), IsOk());
  EXPECT_THAT(uniquer.ReserveIdentifier("foo"), Not(IsOk()));
  EXPECT_THAT(uniquer.ReserveIdentifier("baz__2"), IsOk());
  EXPECT_THAT(uniquer.ReserveIdentifier("baz__2"), Not(IsOk()));
  EXPECT_THAT(uniquer.ReserveIdentifier("BOO__2"), IsOk());
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("baz__1", uniquer.GetSanitizedUniqueName("baz__2"));
  EXPECT_EQ("baz", uniquer.GetSanitizedUniqueName("baz"));
  EXPECT_EQ("baz__3", uniquer.GetSanitizedUniqueName("baz"));
  EXPECT_EQ("BOO", uniquer.GetSanitizedUniqueName("BOO"));
  EXPECT_EQ("BOO__1", uniquer.GetSanitizedUniqueName("BOO"));
  EXPECT_EQ("BOO__3", uniquer.GetSanitizedUniqueName("BOO"));

  // A failed duplicate ReserveIdentifier call on a suffixed identifier must not
  // advance or consume the next available sequential ID in the generator.
  EXPECT_THAT(uniquer.ReserveIdentifier("qux__2"), IsOk());
  EXPECT_THAT(uniquer.ReserveIdentifier("qux__2"), Not(IsOk()));
  EXPECT_EQ("qux", uniquer.GetSanitizedUniqueName("qux"));
  EXPECT_EQ("qux__1", uniquer.GetSanitizedUniqueName("qux"));
  EXPECT_EQ("qux__3", uniquer.GetSanitizedUniqueName("qux"));
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

  // Non-canonical numeric suffixes (such as leading zeros) are treated as bare
  // prefixes rather than colliding with canonical numeric suffixes.
  EXPECT_EQ("lead__01", uniquer.GetSanitizedUniqueName("lead__01"));
  EXPECT_EQ("lead__1", uniquer.GetSanitizedUniqueName("lead__1"));
  EXPECT_THAT(uniquer.ReleaseIdentifier("lead__01"), IsOk());
  EXPECT_THAT(uniquer.ReleaseIdentifier("lead__1"), IsOk());
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

TEST(NameUniquerTest, ReleaseIdentifier) {
  NameUniquer uniquer("__");

  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar"));

  // Release a bare name.
  EXPECT_THAT(uniquer.ReleaseIdentifier("foo"), IsOk());
  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  // foo__1 and foo__2 are still used. Next should be foo__3
  EXPECT_EQ("foo__3", uniquer.GetSanitizedUniqueName("foo"));

  // Release a name with suffix.
  EXPECT_THAT(uniquer.ReleaseIdentifier("foo__1"), IsOk());
  // Requesting 'foo' again gets the next available sequential id.
  EXPECT_EQ("foo__4", uniquer.GetSanitizedUniqueName("foo"));
  // Requesting 'foo__1' re-uses the released identifier.
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo__1"));
  // But requesting 'foo__1' again will be uniquified.
  EXPECT_EQ("foo__5", uniquer.GetSanitizedUniqueName("foo__1"));

  // Release bar and bar__1
  EXPECT_THAT(uniquer.ReleaseIdentifier("bar"), IsOk());
  EXPECT_THAT(uniquer.ReleaseIdentifier("bar__1"), IsOk());
  EXPECT_EQ("bar", uniquer.GetSanitizedUniqueName("bar"));
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar__1"));
  EXPECT_EQ("bar__2", uniquer.GetSanitizedUniqueName("bar"));

  // Releasing something not generated should fail.
  // This fails because 'dne' is not a registered bare name and it doesn't have
  // a numeric suffix.
  EXPECT_THAT(uniquer.ReleaseIdentifier("dne"), Not(IsOk()));
  // This fails because 'dne' is not a registered root.
  EXPECT_THAT(uniquer.ReleaseIdentifier("dne__1"), Not(IsOk()));
  // This fails because foo__123 was never generated.
  EXPECT_THAT(uniquer.ReleaseIdentifier("foo__123"), Not(IsOk()));
}

TEST(NameUniquerTest, TrackOriginalNames) {
  NameUniquer uniquer("__", {"res1"});

  // Bare names that do not collide are not uniquified.
  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ(uniquer.GetOriginalName("foo"), std::nullopt);
  EXPECT_FALSE(uniquer.IsNameAvailable("foo"));
  EXPECT_TRUE(uniquer.IsNameAvailable("foo__1"));

  // Colliding bare names record "foo" as the original name.
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_THAT(uniquer.GetOriginalName("foo__1"), Optional(std::string("foo")));
  EXPECT_THAT(uniquer.GetOriginalName("foo__2"), Optional(std::string("foo")));

  // Directly requested suffixed names that are available are not uniquified.
  EXPECT_EQ("bar__3", uniquer.GetSanitizedUniqueName("bar__3"));
  EXPECT_EQ(uniquer.GetOriginalName("bar__3"), std::nullopt);
  EXPECT_TRUE(uniquer.IsNameAvailable("bar"));
  EXPECT_FALSE(uniquer.IsNameAvailable("bar__3"));

  // Colliding suffixed names record the requested suffixed name ("bar__3").
  EXPECT_EQ("bar__1", uniquer.GetSanitizedUniqueName("bar__3"));
  EXPECT_THAT(uniquer.GetOriginalName("bar__1"),
              Optional(std::string("bar__3")));

  // Sanitized and reserved names record the sanitized original name.
  EXPECT_EQ("_res1", uniquer.GetSanitizedUniqueName("res1"));
  EXPECT_EQ("_res1__1", uniquer.GetSanitizedUniqueName("res1"));
  EXPECT_EQ(uniquer.GetOriginalName("_res1"), std::nullopt);
  EXPECT_THAT(uniquer.GetOriginalName("_res1__1"),
              Optional(std::string("_res1")));

  // Empty and suffix-only names use "name" as root.
  EXPECT_EQ("name", uniquer.GetSanitizedUniqueName(""));
  EXPECT_EQ("name__1", uniquer.GetSanitizedUniqueName(""));
  EXPECT_THAT(uniquer.GetOriginalName("name__1"),
              Optional(std::string("name")));

  // Releasing a uniquified identifier clears its original name entry.
  EXPECT_THAT(uniquer.ReleaseIdentifier("foo__1"), IsOk());
  EXPECT_EQ(uniquer.GetOriginalName("foo__1"), std::nullopt);
  EXPECT_TRUE(uniquer.IsNameAvailable("foo__1"));

  // Reset clears all tracked state.
  uniquer.Reset();
  EXPECT_EQ(uniquer.GetOriginalName("foo__2"), std::nullopt);
  EXPECT_TRUE(uniquer.IsNameAvailable("foo"));
}

TEST(NameUniquerTest, TryReclaimUnqualifiedName) {
  NameUniquer uniquer("__");

  EXPECT_EQ("foo", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__1", uniquer.GetSanitizedUniqueName("foo"));
  EXPECT_EQ("foo__2", uniquer.GetSanitizedUniqueName("foo"));
  // Request "foo__1" while it is taken by the second "foo" -> gets "foo__3".
  EXPECT_EQ("foo__3", uniquer.GetSanitizedUniqueName("foo__1"));

  // While "foo" is still taken, neither "foo" nor "foo__1" can be reclaimed.
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo"),
              IsOkAndHolds(std::nullopt));
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__1"),
              IsOkAndHolds(std::nullopt));
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__3"),
              IsOkAndHolds(std::nullopt));

  // Release "foo". Now "foo__1" can reclaim "foo".
  EXPECT_THAT(uniquer.ReleaseIdentifier("foo"), IsOk());
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__1"),
              IsOkAndHolds(Optional(std::string("foo"))));
  EXPECT_FALSE(uniquer.IsNameAvailable("foo"));
  EXPECT_TRUE(uniquer.IsNameAvailable("foo__1"));
  EXPECT_EQ(uniquer.GetOriginalName("foo"), std::nullopt);

  // "foo" is now taken by the promoted node, so "foo__2" cannot reclaim it.
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__2"),
              IsOkAndHolds(std::nullopt));

  // Because "foo__1" was freed when it reclaimed "foo", "foo__3" (which
  // originally requested "foo__1") can now reclaim "foo__1"!
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__3"),
              IsOkAndHolds(Optional(std::string("foo__1"))));
  EXPECT_FALSE(uniquer.IsNameAvailable("foo__1"));
  EXPECT_TRUE(uniquer.IsNameAvailable("foo__3"));
  EXPECT_EQ(uniquer.GetOriginalName("foo__1"), std::nullopt);

  // Reclaiming an unregistered identifier returns an error.
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("foo__3"), Not(IsOk()));
  EXPECT_THAT(uniquer.TryReclaimUnqualifiedName("dne"), Not(IsOk()));
}

}  // namespace
}  // namespace xls
