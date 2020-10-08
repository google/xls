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

#include "xls/dslx/parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

class ParserTest : public ::testing::Test {
 public:
  void RoundTrip(std::string program) {
    Scanner s("test.x", program);
    Parser p("test", &s);
    XLS_ASSERT_OK_AND_ASSIGN(auto module, p.ParseModule());
    EXPECT_EQ(module->ToString(), program);
  }
};

TEST_F(ParserTest, TestIdentityFunction) {
  RoundTrip(R"(fn f(x: u32) -> u32 {
  x
})");
}

TEST_F(ParserTest, TestIdentityFunctionWithLet) {
  RoundTrip(R"(fn f(x: u32) -> u32 {
  let y = x;
  y
})");
}

}  // namespace
}  // namespace xls::dslx
