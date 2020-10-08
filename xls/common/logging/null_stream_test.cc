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

#include "xls/common/logging/null_stream.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace logging_internal {
namespace {

using ::testing::HasSubstr;

TEST(NullStreamTest, NullStreamFatalExitsWhenDestructed) {
    EXPECT_DEATH(
      {
        NullStreamFatal fatal;
        fprintf(stderr, "did not die yet\n");
      },
      HasSubstr("did not die yet"));
}

}  // namespace
}  // namespace logging_internal
}  // namespace xls
