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

#include "xls/common/visitor.h"

#include <string>
#include <variant>

#include "gtest/gtest.h"
#include "absl/types/variant.h"

namespace xls {
namespace {

TEST(OverloadedTest, Test) {
  std::variant<int, std::string> v = 3;
  EXPECT_TRUE(absl::visit(Visitor{
                              [](int x) { return x == 3; },
                              [](const std::string&) { return false; },
                          },
                          v));

  v = "str";
  EXPECT_TRUE(absl::visit(Visitor{
                              [](int) { return false; },
                              [](const std::string& s) { return s == "str"; },
                          },
                          v));
}

}  // namespace
}  // namespace xls
