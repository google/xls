// Copyright 2021 The XLS Authors
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

#include "xls/common/logging/check_ops.h"

#include <sstream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/casts.h"

namespace xls::logging_internal {
namespace {

TEST(CheckOpsTest, MessageBuilder) {
  CheckOpMessageBuilder mb("foo");
  std::ostream* os = mb.ForVar1();
  auto* oss = down_cast<std::ostringstream*>(os);
  EXPECT_EQ(oss->str(), "foo (");

  EXPECT_EQ(mb.ForVar2(), os);
  EXPECT_EQ(oss->str(), "foo ( vs. ");

  std::unique_ptr<std::string> s(mb.NewString());
  EXPECT_EQ(*s, "foo ( vs. )");
}

}  // namespace
}  // namespace xls::logging_internal
