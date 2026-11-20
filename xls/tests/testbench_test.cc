// Copyright 2026 The XLS Authors
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

#include "xls/tests/testbench.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

std::vector<uint64_t> MakeIota(uint64_t start, uint64_t end) {
  std::vector<uint64_t> expected(end - start);
  std::iota(expected.begin(), expected.end(), start);
  return expected;
}

absl::StatusOr<std::vector<uint64_t>> RunAndCollectIndices(
    uint64_t start, uint64_t end, int64_t num_threads) {
  absl::Mutex mutex;
  std::vector<uint64_t> visited;

  Testbench<uint64_t, uint64_t> testbench(
      start, end, num_threads, /*max_failures=*/1,
      [](uint64_t index) { return index; },
      [&](uint64_t input) {
        absl::MutexLock lock(mutex);
        visited.push_back(input);
        return input;
      },
      [](uint64_t input) { return input; },
      [](uint64_t expected, uint64_t actual) { return expected == actual; },
      [](int64_t /*index*/, uint64_t /*input*/, uint64_t /*expected*/,
         uint64_t /*actual*/) {});

  XLS_RETURN_IF_ERROR(testbench.Run());
  absl::c_sort(visited);
  return visited;
}

TEST(TestbenchTest, EvenlyDivisibleMultiThreadRange) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/100, /*num_threads=*/4),
              IsOkAndHolds(ElementsAreArray(MakeIota(0, 100))));
}

TEST(TestbenchTest, UnevenRemainderMultiThreadRange) {
  // 10 / 4 = 2 with remainder 2 (whereas 10 % 2 == 0).
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/10, /*num_threads=*/4),
              IsOkAndHolds(ElementsAreArray(MakeIota(0, 10))));
  // 11 / 4 = 2 with remainder 3 (whereas 11 % 2 == 1).
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/11, /*num_threads=*/4),
              IsOkAndHolds(ElementsAreArray(MakeIota(0, 11))));
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/17, /*num_threads=*/5),
              IsOkAndHolds(ElementsAreArray(MakeIota(0, 17))));
}

TEST(TestbenchTest, NonZeroStartMultiThreadRange) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/50, /*end=*/75, /*num_threads=*/4),
              IsOkAndHolds(ElementsAreArray(MakeIota(50, 75))));
}

TEST(TestbenchTest, FewerSamplesThanThreads) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/10, /*end=*/13, /*num_threads=*/8),
              IsOkAndHolds(ElementsAreArray(MakeIota(10, 13))));
}

TEST(TestbenchTest, SingleThreadExactRange) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/7, /*end=*/17, /*num_threads=*/1),
              IsOkAndHolds(ElementsAreArray(MakeIota(7, 17))));
}

TEST(TestbenchTest, EmptyRangeReturnsOk) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/5, /*end=*/5, /*num_threads=*/4),
              IsOkAndHolds(IsEmpty()));
}

TEST(TestbenchTest, InvalidRangeOrThreadCountReturnsError) {
  EXPECT_THAT(RunAndCollectIndices(/*start=*/10, /*end=*/5, /*num_threads=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/10, /*num_threads=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RunAndCollectIndices(/*start=*/0, /*end=*/10, /*num_threads=*/-2),
              StatusIs(absl::StatusCode::kInvalidArgument));

  Testbench<uint64_t, uint64_t> testbench(
      /*start=*/0, /*end=*/10, /*num_threads=*/2, /*max_failures=*/1,
      [](uint64_t index) { return index; },
      [](uint64_t input) { return input; },
      [](uint64_t input) { return input; },
      [](uint64_t expected, uint64_t actual) { return expected == actual; },
      [](int64_t /*index*/, uint64_t /*input*/, uint64_t /*expected*/,
         uint64_t /*actual*/) {});
  EXPECT_THAT(testbench.SetNumThreads(0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(testbench.SetNumThreads(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TestbenchTest, ShardDataMultiThreadRange) {
  absl::Mutex mutex;
  std::vector<uint64_t> visited;

  Testbench<uint64_t, uint64_t, int> testbench(
      /*start=*/10, /*end=*/23, /*num_threads=*/4, /*max_failures=*/1,
      [](uint64_t index) { return index; },
      []() { return std::make_unique<int>(7); },
      [&](int* shard_data, uint64_t input) {
        absl::MutexLock lock(mutex);
        visited.push_back(input);
        return input + static_cast<uint64_t>(*shard_data);
      },
      [](int* shard_data, uint64_t input) {
        return input + static_cast<uint64_t>(*shard_data);
      },
      [](uint64_t expected, uint64_t actual) { return expected == actual; },
      [](int64_t /*index*/, uint64_t /*input*/, uint64_t /*expected*/,
         uint64_t /*actual*/) {});

  EXPECT_THAT(testbench.Run(), absl_testing::IsOk());
  absl::c_sort(visited);
  EXPECT_THAT(visited, ElementsAreArray(MakeIota(10, 23)));
}

TEST(TestbenchTest, HighRangeNearUint64Max) {
  constexpr uint64_t kEnd = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kStart = kEnd - 10;
  EXPECT_THAT(RunAndCollectIndices(kStart, kEnd, /*num_threads=*/4),
              IsOkAndHolds(ElementsAreArray(MakeIota(kStart, kEnd))));
}

}  // namespace
}  // namespace xls
