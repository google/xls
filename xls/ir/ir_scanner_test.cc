// Copyright 2020 Google LLC
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

#include "xls/ir/ir_scanner.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::StatusIs;

TEST(IrScannerTest, TokenizeWhitespaceString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString("   \n\t"));
  EXPECT_EQ(0, tokens.size());
}

TEST(IrScannerTest, TokenizeEmptyString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, TokenizeString(""));
  EXPECT_EQ(0, tokens.size());
}

TEST(IrScannerTest, TokenizeEmptyStringWithMinComment) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, TokenizeString("//"));
  EXPECT_EQ(0, tokens.size());
}

TEST(IrScannerTest, TokenizeEmptyStringWithComment) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString("// comment"));
  EXPECT_EQ(0, tokens.size());
}

TEST(IrScannerTest, TokenizeStringWithComment) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"(fn n( // comment)"));
  EXPECT_EQ(3, tokens.size());
}

TEST(IrScannerTest, TokenizeInvalidCharacter) {
  {
    auto tokens_status = TokenizeString("$");
    EXPECT_FALSE(tokens_status.ok());
    EXPECT_THAT(
        tokens_status.status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 ::testing::HasSubstr("Invalid character in IR text \"$\"")));
  }
  {
    auto tokens_status = TokenizeString("\x07");
    EXPECT_FALSE(tokens_status.ok());
    EXPECT_THAT(tokens_status.status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         ::testing::HasSubstr(
                             "Invalid character in IR text \"\\x07\"")));
  }
}

}  // namespace
}  // namespace xls
