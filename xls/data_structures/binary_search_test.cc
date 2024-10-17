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

#include "xls/data_structures/binary_search.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(BinarySearchTest, MaxTrue) {
  const int64_t kMaxSize = 10;
  for (int start = 0; start < kMaxSize; ++start) {
    for (int end = start; end < kMaxSize; ++end) {
      for (int target = start; target <= end; ++target) {
        auto got = BinarySearchMaxTrue(start, end,
                                       [&](int64_t i) { return i <= target; });
        EXPECT_EQ(got, target);
      }
    }
  }
}

TEST(BinarySearchTest, MinTrue) {
  const int64_t kMaxSize = 10;
  for (int start = 0; start < kMaxSize; ++start) {
    for (int end = start; end < kMaxSize; ++end) {
      for (int target = start; target <= end; ++target) {
        auto got = BinarySearchMinTrue(start, end,
                                       [&](int64_t i) { return i >= target; });
        EXPECT_EQ(got, target);
      }
    }
  }
}

TEST(BinarySearchTest, MaxTrueWithStatus) {
  const int64_t kMaxSize = 10;
  for (int start = 0; start < kMaxSize; ++start) {
    for (int end = start; end < kMaxSize; ++end) {
      for (int target = start; target <= end; ++target) {
        auto got = BinarySearchMaxTrueWithStatus(
            start, end,
            [&](int64_t i) -> absl::StatusOr<bool> { return i <= target; });
        EXPECT_THAT(got, IsOkAndHolds(target));
      }
    }
  }
}

TEST(BinarySearchTest, MinTrueWithStatus) {
  const int64_t kMaxSize = 10;
  for (int start = 0; start < kMaxSize; ++start) {
    for (int end = start; end < kMaxSize; ++end) {
      for (int target = start; target <= end; ++target) {
        auto got = BinarySearchMinTrueWithStatus(
            start, end,
            [&](int64_t i) -> absl::StatusOr<bool> { return i >= target; });
        EXPECT_THAT(got, IsOkAndHolds(target));
      }
    }
  }
}

TEST(BinarySearchTest, NumTimesFunctionCalled) {
  int64_t f_called = 0;
  // Note: some compilers dislike the lambda living inside the macro, so we
  // hoist the lambdas.
  auto f1 = [&](int64_t i) {
    f_called++;
    return i <= 555555;
  };
  EXPECT_EQ(BinarySearchMaxTrue(0, 1024 * 1024, f1), 555555);
  EXPECT_LT(f_called, 25);

  f_called = 0;
  auto f2 = [&](int64_t i) {
    f_called++;
    return i >= 123456;
  };
  EXPECT_EQ(BinarySearchMinTrue(0, 1024 * 1024, f2), 123456);
  EXPECT_LT(f_called, 25);
}

TEST(BinarySearchTest, ErrorConditions) {
  // Note: some compilers dislike the lambda living inside the macro, so we
  // hoist the compared-to values.
  auto a = BinarySearchMaxTrueWithStatus(
      123, 42, [&](int64_t i) -> absl::StatusOr<bool> { return true; });
  EXPECT_THAT(a,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("start <= end")));
  auto b = BinarySearchMinTrueWithStatus(
      123, 42, [&](int64_t i) -> absl::StatusOr<bool> { return true; });
  EXPECT_THAT(b,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("start <= end")));

  auto c = BinarySearchMaxTrueWithStatus(
      1, 42, [&](int64_t i) -> absl::StatusOr<bool> { return false; });
  EXPECT_THAT(c, StatusIs(absl::StatusCode::kInvalidArgument,
                          HasSubstr("Lowest value in range fails condition")));
  EXPECT_THAT(
      BinarySearchMinTrueWithStatus(
          1, 42, [&](int64_t i) -> absl::StatusOr<bool> { return false; }),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Highest value in range fails condition")));

  auto d = BinarySearchMaxTrueWithStatus(
      1, 42, [&](int64_t i) -> absl::StatusOr<bool> {
        return absl::InvalidArgumentError("foobar");
      });
  EXPECT_THAT(
      d, StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("foobar")));

  auto e = BinarySearchMaxTrueWithStatus(
      1, 42, [&](int64_t i) -> absl::StatusOr<bool> {
        return absl::UnimplementedError("qux");
      });
  EXPECT_THAT(e, StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("qux")));
}

}  // namespace
}  // namespace xls
