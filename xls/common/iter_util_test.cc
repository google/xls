// Copyright 2024 The XLS Authors
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

#include "xls/common/iter_util.h"

#include <array>
#include <ostream>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"

namespace xls {
namespace {
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

struct Tup {
  int a;
  int b;
  int c;
};
bool operator==(const Tup& l, const Tup& r) {
  return l.a == r.a && l.b == r.b && l.c == r.c;
}
std::ostream& operator<<(std::ostream& os, Tup t) {
  return os << "{ " << t.a << ", " << t.b << ", " << t.c << " }";
}
TEST(IterUtilTest, Iterate) {
  std::vector<Tup> vs;
  EXPECT_FALSE(IteratorProduct<absl::Span<int const>>(
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, [&](auto v) -> bool {
        vs.push_back({*v[0], *v[1], *v[2]});
        return false;
      }));
  EXPECT_THAT(vs, ElementsAreArray(std::array<Tup, 27>{
                      Tup{1, 4, 7}, Tup{2, 4, 7}, Tup{3, 4, 7}, Tup{1, 5, 7},
                      Tup{2, 5, 7}, Tup{3, 5, 7}, Tup{1, 6, 7}, Tup{2, 6, 7},
                      Tup{3, 6, 7}, Tup{1, 4, 8}, Tup{2, 4, 8}, Tup{3, 4, 8},
                      Tup{1, 5, 8}, Tup{2, 5, 8}, Tup{3, 5, 8}, Tup{1, 6, 8},
                      Tup{2, 6, 8}, Tup{3, 6, 8}, Tup{1, 4, 9}, Tup{2, 4, 9},
                      Tup{3, 4, 9}, Tup{1, 5, 9}, Tup{2, 5, 9}, Tup{3, 5, 9},
                      Tup{1, 6, 9}, Tup{2, 6, 9}, Tup{3, 6, 9},
                  }));
}

TEST(IterUtilTest, IterateMixedRadix) {
  std::vector<Tup> vs;
  EXPECT_FALSE(IteratorProduct<absl::Span<int const>>(
      {{1, 2}, {3, 4, 5, 6}, {7}}, [&](auto v) -> bool {
        vs.push_back({*v[0], *v[1], *v[2]});
        return false;
      }));
  EXPECT_THAT(vs, ElementsAreArray(std::array<Tup, 8>{
                      Tup{1, 3, 7}, Tup{2, 3, 7}, Tup{1, 4, 7}, Tup{2, 4, 7},
                      Tup{1, 5, 7}, Tup{2, 5, 7}, Tup{1, 6, 7}, Tup{2, 6, 7}}));
}

TEST(IterUtilTest, IterateEmptyIter) {
  std::vector<Tup> vs;
  EXPECT_FALSE(IteratorProduct<absl::Span<int const>>(
      {{1, 2}, {3, 4, 5, 6}, {}, {7}}, [&](auto v) -> bool {
        vs.push_back({*v[0], *v[1], *v[2]});
        return false;
      }));
  EXPECT_THAT(vs, IsEmpty());
}

TEST(IterUtilTest, IterateNothing) {
  std::vector<Tup> vs;
  EXPECT_FALSE(IteratorProduct<absl::Span<int const>>({}, [&](auto v) -> bool {
    vs.push_back({*v[0], *v[1], *v[2]});
    return false;
  }));
  EXPECT_THAT(vs, IsEmpty());
}

}  // namespace
}  // namespace xls
