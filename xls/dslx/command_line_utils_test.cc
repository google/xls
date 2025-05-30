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

#include "xls/dslx/command_line_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"

namespace xls::dslx {
namespace {

using absl_testing::IsOkAndHolds;

TEST(CommandLineUtilsTest, PathToName) {
  EXPECT_THAT(PathToName("/tmp/foo/bar.x.y.z"), IsOkAndHolds("bar"));
  EXPECT_THAT(PathToName("bar.x.y.z"), IsOkAndHolds("bar"));
  EXPECT_THAT(PathToName("/tmp/foo/beyond-all-repair.x.y.z"),
              IsOkAndHolds("beyond__H0x2D__all__H0x2D__repair"));
  EXPECT_THAT(PathToName("/tmp/foo/to INFINITY!\n.x.y.z"),
              IsOkAndHolds("to__H0x20__INFINITY__H0x21____H0x0A__"));
}

TEST(CommandLineUtilsTest, NameNeedsCanonicalization) {
  EXPECT_FALSE(NameNeedsCanonicalization("/tmp/foo/bar.x.y.z"));
  EXPECT_FALSE(NameNeedsCanonicalization("bar.x.y.z"));
  EXPECT_TRUE(NameNeedsCanonicalization("/tmp/foo/beyond-all-repair.x.y.z"));
  EXPECT_TRUE(NameNeedsCanonicalization("/tmp/foo/to INFINITY!\n.x.y.z"));
}

}  // namespace
}  // namespace xls::dslx
