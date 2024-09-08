// Copyright 2023 The XLS Authors
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

#include "xls/dslx/frontend/token.h"

#include "gtest/gtest.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {
namespace {

TEST(TokenTest, KeywordGetValueIsKeywordText) {
  Token token(FakeSpan(), Keyword::kFn);
  EXPECT_TRUE(token.GetValue().has_value());
  EXPECT_EQ(token.GetValue().value(), "fn");
  EXPECT_EQ(token.ToErrorString(), "keyword:fn");
  EXPECT_EQ(token.ToString(), "fn");
  FileTable file_table;
  EXPECT_EQ(
      token.ToRepr(file_table),
      "Token(Span(Pos(\"<no-file>\", 0, 0), Pos(\"<no-file>\", 0, 0)), fn)");
}

TEST(TokenTest, SimpleSelfEquality) {
  Token token(FakeSpan(), Keyword::kFn);
  EXPECT_TRUE(token == token);
  EXPECT_FALSE(token != token);
}

TEST(TokenTest, IsTypeKeyword) {
  EXPECT_FALSE(Token(FakeSpan(), Keyword::kFn).IsTypeKeyword());
  EXPECT_TRUE(Token(FakeSpan(), Keyword::kUN).IsTypeKeyword());
  EXPECT_TRUE(Token(FakeSpan(), Keyword::kU32).IsTypeKeyword());
}

TEST(TokenTest, IdentifierTokenAccessors) {
  Token token(TokenKind::kIdentifier, FakeSpan(), "my_identifier");
  EXPECT_TRUE(token.GetValue().has_value());
  EXPECT_EQ(token.GetValue().value(), "my_identifier");
  EXPECT_EQ(token.ToErrorString(), "identifier");
  EXPECT_EQ(token.ToString(), "my_identifier");
  FileTable file_table;
  EXPECT_EQ(
      token.ToRepr(file_table),
      "Token(identifier, Span(Pos(\"<no-file>\", 0, 0), Pos(\"<no-file>\", 0, "
      "0)), \"my_identifier\")");

  EXPECT_TRUE(token.IsIdentifier("my_identifier"));
  EXPECT_FALSE(token.IsIdentifier("my_ident"));
  EXPECT_FALSE(token.IsNumber("my_identifier"));

  EXPECT_TRUE(token.IsKindIn({TokenKind::kIdentifier}));
  EXPECT_TRUE(token.IsKindIn({TokenKind::kIdentifier, Keyword::kFn}));
  EXPECT_FALSE(token.IsKindIn({Keyword::kFn}));
}

}  // namespace
}  // namespace xls::dslx
