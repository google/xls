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

#include "xls/common/subprocess.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(SubprocessTest, EmptyArgvFails) {
  auto result = InvokeSubprocess({});

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SubprocessTest, FailingCommandFails) {
  auto result = InvokeSubprocess(
      {"/bin/sh", "-c", "/bin/echo hey && /bin/echo hello >&2 && /bin/false"});

  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hey"));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hello"));
}

TEST(SubprocessTest, WorkingCommandWorks) {
  auto result = InvokeSubprocess(
      {"/bin/sh", "-c", "/bin/echo hey && /bin/echo hello >&2"});

  XLS_ASSERT_OK(result);
  EXPECT_EQ(result->first, "hey\n");
  EXPECT_EQ(result->second, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStdoutFirstWorks) {
  auto result = InvokeSubprocess(
      {"/bin/sh", "-c", "/usr/bin/env seq 10000 && /bin/echo hello >&2"});

  XLS_ASSERT_OK(result);
  EXPECT_THAT(result->first, HasSubstr("\n10000\n"));
  EXPECT_EQ(result->second, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStderrFirstWorks) {
  auto result = InvokeSubprocess(
      {"/bin/sh", "-c", "/usr/bin/env seq 10000 >&2 && /bin/echo hello"});

  XLS_ASSERT_OK(result);
  EXPECT_EQ(result->first, "hello\n");
  EXPECT_THAT(result->second, HasSubstr("\n10000\n"));
}

}  // namespace
}  // namespace xls
