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

#include "xls/common/status/error_code_to_status.h"

#include <errno.h>

#include <system_error>  // NOLINT(build/c++11)

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::StatusIs;

TEST(ErrorCodeToStatusTest, EmptyErrorCodeIsConvertedToOkStatus) {
  XLS_EXPECT_OK(ErrorCodeToStatus(std::error_code()));
}

TEST(ErrorCodeToStatusTest, NotFoundCodeIsConvertedToNotFoundStatus) {
  absl::Status status = ErrorCodeToStatus(
      std::make_error_code(std::errc::no_such_file_or_directory));

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
}

TEST(ErrorCodeToStatusTest, ErrnoToStatusConvertsEnoentToNotFoundStatus) {
  absl::Status status = ErrnoToStatus(ENOENT);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls
