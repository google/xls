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

#include "xls/common/logging/capture_stream.h"

#include <cstdio>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace testing {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::Eq;

TEST(CaptureStreamTest, CaptureStdoutData) {
  absl::StatusOr<std::string> data = CaptureStream(STDOUT_FILENO, [] {
    fprintf(stdout, "Hello!");
    fprintf(stderr, "This should be ignored.");
  });
  EXPECT_THAT(data, IsOkAndHolds(Eq("Hello!")));
}

TEST(CaptureStreamTest, CaptureStderrData) {
  absl::StatusOr<std::string> data = CaptureStream(STDERR_FILENO, [] {
    fprintf(stderr, "Some output.\nHello!");
    fprintf(stdout, "This should be ignored.");
  });
  EXPECT_THAT(data, IsOkAndHolds(Eq("Some output.\nHello!")));
}

}  // namespace
}  // namespace testing
}  // namespace xls
