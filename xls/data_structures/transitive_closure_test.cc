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

#include "xls/data_structures/transitive_closure.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

using V = std::string;

TEST(TransitiveClosureTest, Simple) {
  HashRelation<V> rel;
  rel["foo"].insert("bar");
  rel["bar"].insert("baz");
  rel["bar"].insert("qux");
  rel["baz"].insert("qux");
  rel["foo2"].insert("baz");
  HashRelation<V> tc = TransitiveClosure<V>(rel);
  EXPECT_THAT(tc.at("foo"), UnorderedElementsAre("bar", "baz", "qux"));
  EXPECT_THAT(tc.at("foo2"), UnorderedElementsAre("baz", "qux"));
  EXPECT_THAT(tc.at("bar"), UnorderedElementsAre("baz", "qux"));
  EXPECT_THAT(tc.at("baz"), UnorderedElementsAre("qux"));
  EXPECT_FALSE(tc.contains("qux"));
}

}  // namespace
}  // namespace xls
