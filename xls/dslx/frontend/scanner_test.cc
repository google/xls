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

#include "xls/dslx/frontend/scanner.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/error_test_utils.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

absl::StatusOr<std::vector<Token>> ToTokens(std::string text) {
  Scanner s("fake_file.x", std::move(text));
  return s.PopAll();
}

}  // namespace

TEST(ScannerTest, SimpleTokens) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("+ - ++ << >>"));
  ASSERT_EQ(5, tokens.size());
  EXPECT_EQ(tokens[0].kind(), TokenKind::kPlus);
  EXPECT_EQ(tokens[1].kind(), TokenKind::kMinus);
  EXPECT_EQ(tokens[2].kind(), TokenKind::kDoublePlus);
  EXPECT_EQ(tokens[3].kind(), TokenKind::kDoubleOAngle);
  EXPECT_EQ(tokens[4].kind(), TokenKind::kDoubleCAngle);
}

TEST(ScannerTest, HexNumbers) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           ToTokens("0xf00 0xba5 0xA"));
  ASSERT_EQ(3, tokens.size());
  EXPECT_TRUE(tokens[0].IsNumber("0xf00"));
  EXPECT_TRUE(tokens[1].IsNumber("0xba5"));
  EXPECT_TRUE(tokens[2].IsNumber("0xA"));
}

TEST(ScannerTest, BoolKeywords) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           ToTokens("true false bool"));
  ASSERT_EQ(3, tokens.size());
  EXPECT_TRUE(tokens[0].IsKeyword(Keyword::kTrue));
  EXPECT_TRUE(tokens[1].IsKeyword(Keyword::kFalse));
  EXPECT_TRUE(tokens[2].IsKeyword(Keyword::kBool));
  EXPECT_EQ(tokens[0].ToErrorString(), "keyword:true");
}

TEST(ScannerTest, IdentifierWithTick) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           ToTokens("state state' state'' s'"));
  ASSERT_EQ(4, tokens.size());
  EXPECT_TRUE(tokens[0].IsIdentifier("state"));
  EXPECT_TRUE(tokens[1].IsIdentifier("state'"));
  EXPECT_TRUE(tokens[2].IsIdentifier("state''"));
  EXPECT_TRUE(tokens[3].IsIdentifier("s'"));
}

TEST(ScannerTest, TickCannotStartAnIdentifier) {
  const char* kText = "'state";
  EXPECT_THAT(
      ToTokens(kText),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected closing single quote for character literal; got t")));
}

// Verifies that Scanner::ProcessNextStringChar() correctly handles all
// supported escape sequences.
TEST(ScannerTest, RecognizesEscapes) {
  std::string text = R"(\n\r\t\\\0\'\"\x6f\u{102DCB}Hello"extrastuff)";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::string result, s.ScanUntilDoubleQuote());
  EXPECT_EQ(static_cast<uint8_t>(result[0]), 10);   // Newline.
  EXPECT_EQ(static_cast<uint8_t>(result[1]), 13);   // Carriage return.
  EXPECT_EQ(static_cast<uint8_t>(result[2]), 9);    // Tab.
  EXPECT_EQ(static_cast<uint8_t>(result[3]), 92);   // Backslash.
  EXPECT_EQ(static_cast<uint8_t>(result[4]), 0);    // Null.
  EXPECT_EQ(static_cast<uint8_t>(result[5]), 39);   // Single quote.
  EXPECT_EQ(static_cast<uint8_t>(result[6]), 34);   // Double quote.
  EXPECT_EQ(static_cast<uint8_t>(result[7]), 111);  // Lowercase o.
  EXPECT_EQ(static_cast<uint8_t>(result[8]),
            0xF4);  // Byte 1 in UTF-8 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[9]),
            0x82);  // Byte 2 in UTF-8 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[10]),
            0xB7);  // Byte 3 in UTF-8 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[11]),
            0x8B);  // Byte 4 in UTF-8 of Unicode code.
  EXPECT_EQ(static_cast<uint8_t>(result[12]), 'H');  // Final word.
  EXPECT_EQ(static_cast<uint8_t>(result[13]), 'e');
  EXPECT_EQ(static_cast<uint8_t>(result[14]), 'l');
  EXPECT_EQ(static_cast<uint8_t>(result[15]), 'l');
  EXPECT_EQ(static_cast<uint8_t>(result[16]), 'o');
  EXPECT_EQ(result.size(), 17);
}

TEST(ScannerTest, ScanJustWhitespace) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens(" "));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].kind(), TokenKind::kEof);
}

TEST(ScannerTest, ScanKeyword) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("fn"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsKeyword(Keyword::kFn));
}

TEST(ScannerTest, FunctionDefinition) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens,
                           ToTokens("fn ident(x) { x }"));

  ASSERT_EQ(tokens.size(), 8);

  EXPECT_TRUE(tokens[0].IsKeyword(Keyword::kFn));
  EXPECT_EQ(tokens[0].ToString(), "fn");

  EXPECT_TRUE(tokens[1].IsIdentifier("ident"));
  EXPECT_EQ(tokens[1].ToString(), "ident");

  EXPECT_EQ(tokens[2].kind(), TokenKind::kOParen);
  EXPECT_EQ(tokens[2].ToString(), "(");

  EXPECT_TRUE(tokens[3].IsIdentifier("x"));
  EXPECT_EQ(tokens[3].ToString(), "x");

  EXPECT_EQ(tokens[4].kind(), TokenKind::kCParen);
  EXPECT_EQ(tokens[4].ToString(), ")");

  EXPECT_EQ(tokens[5].kind(), TokenKind::kOBrace);
  EXPECT_EQ(tokens[5].ToString(), "{");

  EXPECT_TRUE(tokens[6].IsIdentifier("x"));
  EXPECT_EQ(tokens[6].ToString(), "x");

  EXPECT_EQ(tokens[7].kind(), TokenKind::kCBrace);
  EXPECT_EQ(tokens[7].ToString(), "}");
}

TEST(ScannerTest, DoublePlus) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("x++y"));
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_TRUE(tokens[0].IsIdentifier("x"));
  EXPECT_EQ(tokens[1].ToString(), "++");
  EXPECT_TRUE(tokens[2].IsIdentifier("y"));
}

TEST(ScannerTest, NumberHex) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("0xf00"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("0xf00"));
}

TEST(ScannerTest, NegativeNumberHex) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("-0xf00"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("-0xf00"));
}

TEST(ScannerTest, NumberBin) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("0b10"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("0b10"));
}

TEST(ScannerTest, NumberBinInvalidDigit) {
  EXPECT_THAT(ToTokens("0b102"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid digit for binary number: '2'")));
}

TEST(ScannerTest, NegativeNumberBin) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("-0b10"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("-0b10"));
}

TEST(ScannerTest, NegativeNumber) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("-42"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("-42"));
}

TEST(ScannerTest, NumberWithUnderscores) {
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, ToTokens("0b11_1100"));
  ASSERT_EQ(tokens.size(), 1);
  EXPECT_TRUE(tokens[0].IsNumber("0b11_1100"));
}

TEST(ScannerTest, ScanIncompleteNumbers) {
  EXPECT_THAT(ToTokens("0x"), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Expected hex characters")));
  EXPECT_THAT(ToTokens("0b"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected binary characters")));
}

TEST(ScannerTest, BadlyFormedNumber) {
  EXPECT_THAT(ToTokens("u1:01"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid radix for number")));
}

TEST(ScannerTest, IncompleteCharacter) {
  EXPECT_THAT(ToTokens("'a"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected closing single quote")));
  EXPECT_THAT(ToTokens("'"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected character after single quote")));
}

TEST(ScannerTest, WhitespaceAndCommentsMode) {
  Scanner s("fake_file.x", R"(// Hello comment world.
  42
  // EOF)",
            /*include_whitespace_and_comments=*/true);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  EXPECT_EQ(tokens.size(), 5);
  EXPECT_EQ(tokens[0].kind(), TokenKind::kComment);
  EXPECT_EQ(tokens[1].kind(), TokenKind::kWhitespace);
  EXPECT_EQ(tokens[2].kind(), TokenKind::kNumber);
  EXPECT_EQ(tokens[3].kind(), TokenKind::kWhitespace);
  EXPECT_EQ(tokens[4].kind(), TokenKind::kComment);
}

TEST(ScannerTest, PopSeveral) {
  Scanner s("fake_file.x", "[!](-)");
  std::vector<TokenKind> expected = {
      TokenKind::kOBrack, TokenKind::kBang,  TokenKind::kCBrack,
      TokenKind::kOParen, TokenKind::kMinus, TokenKind::kCParen,
  };
  for (TokenKind tk : expected) {
    ASSERT_FALSE(s.AtEof());
    XLS_ASSERT_OK_AND_ASSIGN(Token t, s.Pop());
    EXPECT_EQ(t.kind(), tk);
  }
  EXPECT_TRUE(s.AtEof());
}

TEST(ScannerTest, ScanRandomLookingForCrashes) {
  absl::BitGen bitgen;
  for (int64_t i = 0; i < 256 * 1024; ++i) {
    int64_t length = absl::Uniform(bitgen, 0, 512);
    std::string text;
    for (int64_t charno = 0; charno < length; ++charno) {
      text.push_back(absl::Uniform(bitgen, 0, 256));
    }
    Scanner s("fake_file.x", text);
    absl::StatusOr<std::vector<Token>> tokens = s.PopAll();
    if (!tokens.ok()) {
      continue;
    }
    // Ensure any scanned tokens can be converted to strings.
    for (const Token& token : tokens.value()) {
      (void)token.ToString();
    }
  }
}

TEST(ScannerTest, TokenEqNeqTests) {
  Span span_a(Pos("test.x", 0, 0), Pos("test.x", 1, 1));
  Span span_b(Pos("test.x", 2, 2), Pos("test.x", 3, 3));

  Token test_token(TokenKind::kIdentifier, span_a, "payload");
  EXPECT_EQ(test_token, test_token);

  Token identically_constructed(TokenKind::kIdentifier, span_a, "payload");
  EXPECT_EQ(test_token, identically_constructed);

  Token test_payload_mismatch(TokenKind::kIdentifier, span_a, "bad_payload");
  EXPECT_NE(test_token, test_payload_mismatch);

  Token test_payload_missing(TokenKind::kIdentifier, span_a);
  EXPECT_NE(test_token, test_payload_missing);

  Token test_span_mismatch(TokenKind::kIdentifier, span_b, "payload");
  EXPECT_NE(test_token, test_span_mismatch);

  Token test_kind_mismatch(TokenKind::kNumber, span_a, "payload");
  EXPECT_NE(test_token, test_kind_mismatch);

  Token keyword_token(span_a, Keyword::kBool);
  EXPECT_EQ(keyword_token, keyword_token);

  Token keyword_identically_constructed(span_a, Keyword::kBool);
  EXPECT_EQ(keyword_token, keyword_identically_constructed);

  Token keyword_token_mismatch(span_a, Keyword::kFn);
  EXPECT_NE(keyword_token, keyword_token_mismatch);
}

TEST(ScannerTest, HexCharLiteralBadDigit) {
  std::string text = R"('\xjk')";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::vector<Token>> result = s.PopAll();
  EXPECT_THAT(
      result.status(),
      IsPosError("ScanError", HasSubstr("Only hex digits are allowed")));
}

TEST(ScannerTest, NoCloseQuoteOnString) {
  std::string text = R"("abc)";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::vector<Token>> result = s.PopAll();
  EXPECT_THAT(
      result.status(),
      IsPosError(
          "ScanError",
          HasSubstr(
              "Reached end of file without finding a closing double quote.")));
}

TEST(ScannerTest, StringCharUnicodeEscapeNonHexDigit) {
  std::string text = R"(\u{jk}")";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(
      result.status(),
      IsPosError(
          "ScanError",
          HasSubstr(
              "Only hex digits are allowed within a Unicode character code")));
}

TEST(ScannerTest, StringCharUnicodeEscapeEmpty) {
  std::string text = R"(\u{}")";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(
      result.status(),
      IsPosError(
          "ScanError",
          HasSubstr("Unicode escape must contain at least one character")));
}

TEST(ScannerTest, StringCharUnicodeInvalidSequence) {
  std::string text = R"(\u{d835}")";  // surrogate character
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(result.status(),
              IsPosError("ScanError",
                         HasSubstr("Invalid unicode sequence: '\\u{d835}'")));
}

TEST(ScannerTest, StringCharUnicodeMoreThanSixDigits) {
  std::string text = R"(\u{1234567}")";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(
      result.status(),
      IsPosError("ScanError",
                 HasSubstr("Unicode character code escape sequence must "
                           "terminate (after 6 digits at most)")));
}

TEST(ScannerTest, StringCharUnicodeBadTerminator) {
  std::string text = R"(\u{123456!")";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(
      result.status(),
      IsPosError("ScanError",
                 HasSubstr("Unicode character code escape sequence must "
                           "terminate (after 6 digits at most) with a '}'")));
}

TEST(ScannerTest, StringCharUnicodeBadStartChar) {
  std::string text = R"(\u!")";
  Scanner s("fake_file.x", text);
  absl::StatusOr<std::string> result = s.ScanUntilDoubleQuote();
  EXPECT_THAT(
      result.status(),
      IsPosError(
          "ScanError",
          HasSubstr("Unicode character code escape sequence start (\\u) must "
                    "be followed by a character code, such as \"{...}\"")));
}

TEST(ScannerTest, SimpleString) {
  std::string text = R"({"hello world!"})";
  Scanner s("fake_file.x", text);

  XLS_ASSERT_OK_AND_ASSIGN(Token ocurl, s.Pop());
  EXPECT_EQ(ocurl.kind(), TokenKind::kOBrace);

  XLS_ASSERT_OK_AND_ASSIGN(Token t, s.Pop());

  XLS_ASSERT_OK_AND_ASSIGN(Token ccurl, s.Pop());
  EXPECT_EQ(ccurl.kind(), TokenKind::kCBrace);

  ASSERT_TRUE(s.AtEof());
  EXPECT_EQ(t.kind(), TokenKind::kString);
  EXPECT_EQ(*t.GetValue(), "hello world!");
}

TEST(ScannerTest, SimpleCommentData) {
  std::string text = R"(// I haz comments!)";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());

  ASSERT_EQ(tokens.size(), 1);
  EXPECT_EQ(tokens[0].kind(), TokenKind::kEof);

  ASSERT_EQ(s.comments().size(), 1);
  const CommentData& comment = s.comments()[0];
  const Span want_span{Pos("fake_file.x", 0, 0), Pos("fake_file.x", 0, 18)};
  EXPECT_EQ(comment.span, want_span);
  EXPECT_EQ(comment.text, " I haz comments!");
}

TEST(ScannerTest, CommentTokenSandwich) {
  std::string text = R"(+ // I haz comments!
*)";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(tokens.size(), 2);
  EXPECT_EQ(tokens[0].kind(), TokenKind::kPlus);
  EXPECT_EQ(tokens[1].kind(), TokenKind::kStar);

  ASSERT_EQ(s.comments().size(), 1);
  const CommentData& comment = s.comments()[0];
  const Span want_span{Pos("fake_file.x", 0, 2), Pos("fake_file.x", 1, 0)};
  EXPECT_EQ(comment.span, want_span);
  EXPECT_EQ(comment.text, " I haz comments!\n");
}

TEST(ScannerTest, TwoInlineStyleComments) {
  std::string text = R"(foo // one thing
bar  // another thing
)";
  Scanner s("fake_file.x", text);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, s.PopAll());
  ASSERT_EQ(tokens.size(), 3);
  EXPECT_EQ(tokens[0].kind(), TokenKind::kIdentifier);
  EXPECT_EQ(tokens[1].kind(), TokenKind::kIdentifier);
  EXPECT_EQ(tokens[2].kind(), TokenKind::kEof);

  ASSERT_EQ(s.comments().size(), 2);

  {
    const CommentData& comment = s.comments()[0];
    const Span want_span{Pos("fake_file.x", 0, 4), Pos("fake_file.x", 1, 0)};
    EXPECT_EQ(comment.span, want_span);
    EXPECT_EQ(comment.text, " one thing\n");
  }

  {
    const CommentData& comment = s.comments()[1];
    const Span want_span{Pos("fake_file.x", 1, 5), Pos("fake_file.x", 2, 0)};
    EXPECT_EQ(comment.span, want_span);
    EXPECT_EQ(comment.text, " another thing\n");
  }
}

}  // namespace xls::dslx
