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

#include "xls/dslx/cpp_scanner.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls::dslx {
namespace {

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

}  // namespace
}  // namespace xls::dslx
