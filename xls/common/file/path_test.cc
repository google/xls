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

#include "xls/common/file/path.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(PathTest, RelativizePathRelativizesPath) {
  EXPECT_THAT(RelativizePath("/a/d", "/a"), IsOkAndHolds("d"));
  EXPECT_THAT(RelativizePath("/a/d", "/a/b/c"), IsOkAndHolds("../../d"));
  EXPECT_THAT(RelativizePath("/a/b/c", "/a/d"), IsOkAndHolds("../b/c"));
}

}  // namespace
}  // namespace xls
