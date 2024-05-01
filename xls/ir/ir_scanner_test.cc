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

#include "xls/ir/ir_scanner.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;

// Returns the string values of the given tokens as a vector of strings.
std::vector<std::string> TokensToStrings(absl::Span<const Token> tokens) {
  std::vector<std::string> strs;
  for (const Token& token : tokens) {
    strs.push_back(token.value());
  }
  return strs;
}

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
    EXPECT_THAT(tokens_status.status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Invalid character in IR text \"$\"")));
  }
  {
    auto tokens_status = TokenizeString("\x07");
    EXPECT_FALSE(tokens_status.ok());
    EXPECT_THAT(tokens_status.status(),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Invalid character in IR text \"\\x07\"")));
  }
}

TEST(IrScannerTest, QuotedString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"("foo")"));
  EXPECT_EQ(tokens.size(), 1);
  const Token& foo = tokens.front();
  EXPECT_EQ(foo.type(), LexicalTokenType::kQuotedString);
  EXPECT_EQ(foo.value(), "foo");
  EXPECT_EQ(foo.pos().colno, 0);
  EXPECT_EQ(foo.pos().lineno, 0);
}

TEST(IrScannerTest, OffsetQuotedString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString("\n\n \"foo\""));
  EXPECT_EQ(tokens.size(), 1);
  const Token& foo = tokens.front();
  EXPECT_EQ(foo.type(), LexicalTokenType::kQuotedString);
  EXPECT_EQ(foo.value(), "foo");
  EXPECT_EQ(foo.pos().colno, 1);
  EXPECT_EQ(foo.pos().lineno, 2);
}

TEST(IrScannerTest, EmptyQuotedString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, TokenizeString(R"("")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre(""));
}

TEST(IrScannerTest, MultipleQuotedStrings) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"("foo""bar""baz""")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre("foo", "bar", "baz", ""));
}

TEST(IrScannerTest, TripleQuotedString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"("""asdf""")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre("asdf"));
}

TEST(IrScannerTest, MultilineQuotedStrings) {
  EXPECT_THAT(TokenizeString("\"foo bar\n\"").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unterminated quoted string")));
}

TEST(IrScannerTest, EmptyTripleQuotedString) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"("""""")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre(""));
}

TEST(IrScannerTest, EmptyTripleQuotedStrings) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"("""""""""""""""""")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre("", "", ""));
}

TEST(IrScannerTest, TripleQuotedStringWithSingleQuotes) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           TokenizeString(R"(""""dog" "cat" " "" "  """)"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre(R"("dog" "cat" " "" "  )"));
}

TEST(IrScannerTest, MultilineTripleQuotedStrings) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, TokenizeString(R"("""
something
somethingelse

foo

bar""")"));
  EXPECT_THAT(TokensToStrings(tokens), ElementsAre(R"(
something
somethingelse

foo

bar)"));
}

TEST(IrScannerTest, UnterminatedQuotedStrings) {
  EXPECT_THAT(
      TokenizeString(R"("unterminated)").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unterminated quoted string starting at 1:1")));
}

TEST(IrScannerTest, UnterminatedTripleQuotedStrings) {
  EXPECT_THAT(
      TokenizeString(R"("""does not terminate)").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unterminated quoted string starting at 1:1")));
}

}  // namespace
}  // namespace xls
