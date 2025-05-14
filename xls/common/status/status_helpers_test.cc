// Copyright 2025 The XLS Authors
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

#include "xls/common/status/status_helpers.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {
namespace {

TEST(ValueOrDieTest, Success) {
  EXPECT_EQ(xabsl::ValueOrDie(absl::StatusOr<int>(10)), 10);
  EXPECT_EQ(xabsl::ValueOrDie(absl::StatusOr<std::string>("prefix")), "prefix");
}

TEST(ValueOrDieTest, Death) {
  EXPECT_DEATH(xabsl::ValueOrDie(
                   absl::StatusOr<int>(absl::UnimplementedError("error1"))),
               "Unexpected error: UNIMPLEMENTED: error1");
  EXPECT_DEATH(xabsl::ValueOrDie(absl::StatusOr<double>(
                   absl::FailedPreconditionError("error2"))),
               "Unexpected error: FAILED_PRECONDITION: error2");
}

}  // namespace
}  // namespace xls
