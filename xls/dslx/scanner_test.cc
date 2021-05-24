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

#include "xls/dslx/scanner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

TEST(ScannerTest, SimpleTokens) {
  std::string text = "+ - ++ << >>";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(5, tokens.size());
  EXPECT_EQ(tokens[0].kind(), TokenKind::kPlus);
  EXPECT_EQ(tokens[1].kind(), TokenKind::kMinus);
  EXPECT_EQ(tokens[2].kind(), TokenKind::kDoublePlus);
  EXPECT_EQ(tokens[3].kind(), TokenKind::kDoubleOAngle);
  EXPECT_EQ(tokens[4].kind(), TokenKind::kDoubleCAngle);
}

TEST(ScannerTest, HexNumbers) {
  std::string text = "0xf00 0xba5 0xA";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(3, tokens.size());
  EXPECT_TRUE(tokens[0].IsNumber("0xf00"));
  EXPECT_TRUE(tokens[1].IsNumber("0xba5"));
  EXPECT_TRUE(tokens[2].IsNumber("0xA"));
}

TEST(ScannerTest, BoolKeywords) {
  std::string text = "true false bool";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(3, tokens.size());
  EXPECT_TRUE(tokens[0].IsKeyword(Keyword::kTrue));
  EXPECT_TRUE(tokens[1].IsKeyword(Keyword::kFalse));
  EXPECT_TRUE(tokens[2].IsKeyword(Keyword::kBool));
  EXPECT_EQ(tokens[0].ToErrorString(), "keyword:true");
}

TEST(ScannerTest, IdentifierWithTick) {
  std::string text = "state state' state'' s'";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(4, tokens.size());
  EXPECT_TRUE(tokens[0].IsIdentifier("state"));
  EXPECT_TRUE(tokens[1].IsIdentifier("state'"));
  EXPECT_TRUE(tokens[2].IsIdentifier("state''"));
  EXPECT_TRUE(tokens[3].IsIdentifier("s'"));
}

TEST(ScannerTest, TickCannotStartAnIdentifier) {
  std::string text = "'state";
  Scanner s("fake_file.x", text);
  EXPECT_THAT(
      s.PopAll(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected closing single quote for character literal; got t")));
}

// Verifies that Scanner::ProcessNextStringChar() correctly handles all
// supported escape sequences.
TEST(ScannerTest, RecognizesEscapes) {
  std::string text = R"(\n\r\t\\\0\'\"\x6f\u{fEdCb}Hello"extrastuff)";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::string result, s.ScanUntilDoubleQuote());
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 10);    // Newline.
  EXPECT_EQ(static_cast<uint8_t>(result[1]), 13);    // Carriage return.
  EXPECT_EQ(static_cast<uint8_t>(result[2]), 9);     // Tab.
  EXPECT_EQ(static_cast<uint8_t>(result[3]), 92);    // Backslash.
  EXPECT_EQ(static_cast<uint8_t>(result[4]), 0);     // Null.
  EXPECT_EQ(static_cast<uint8_t>(result[5]), 39);    // Single quote.
  EXPECT_EQ(static_cast<uint8_t>(result[6]), 34);    // Double quote.
  EXPECT_EQ(static_cast<uint8_t>(result[7]), 111);   // Lowercase o.
  EXPECT_EQ(static_cast<uint8_t>(result[8]), 0xcb);  // Byte 3 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[9]), 0xed);  // Byte 2 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[10]), 0xf);  // Byte 1 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[11]), 'H');  // Final word.
  EXPECT_EQ(static_cast<uint8_t>(result[12]), 'e');
  EXPECT_EQ(static_cast<uint8_t>(result[13]), 'l');
  EXPECT_EQ(static_cast<uint8_t>(result[14]), 'l');
  EXPECT_EQ(static_cast<uint8_t>(result[15]), 'o');
  EXPECT_EQ(result.size(), 16);
}

}  // namespace
}  // namespace xls::dslx
