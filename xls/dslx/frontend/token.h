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

#ifndef XLS_DSLX_FRONTEND_TOKEN_H_
#define XLS_DSLX_FRONTEND_TOKEN_H_

#include <cstdint>
#include <cstdio>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner_keywords.inc"

namespace xls::dslx {

#define XLS_DSLX_TOKEN_KINDS(X)                                        \
  /* enum, pyname, str */                                              \
  X(kDot, DOT, ".")                                                    \
  X(kEof, EOF, "EOF")                                                  \
  X(kKeyword, KEYWORD, "keyword")                                      \
  X(kIdentifier, IDENTIFIER, "identifier")                             \
  X(kNumber, NUMBER, "number")                                         \
  X(kCharacter, CHARACTER, "character")                                \
  X(kOParen, OPAREN, "(")                                              \
  X(kCParen, CPAREN, ")")                                              \
  X(kOBrace, OBRACE, "{")                                              \
  X(kCBrace, CBRACE, "}")                                              \
  X(kPlus, PLUS, "+")                                                  \
  X(kMinus, MINUS, "-")                                                \
  X(kPlusColon, PLUS_COLON, "+:")                                      \
  X(kDoubleCAngle, DOUBLE_CANGLE, ">>")                                \
  X(kDoubleOAngle, DOUBLE_OANGLE, "<<")                                \
  X(kEquals, EQUALS, "=")                                              \
  X(kDoubleColon, DOUBLE_COLON, "::")                                  \
  X(kDoublePlus, DOUBLE_PLUS, "++")                                    \
  X(kDoubleEquals, DOUBLE_EQUALS, "==")                                \
  X(kCAngleEquals, CANGLE_EQUALS, ">=")                                \
  X(kOAngleEquals, OANGLE_EQUALS, "<=")                                \
  X(kBangEquals, BANG_EQUALS, "!=")                                    \
  X(kCAngle, CANGLE, ">")                                              \
  X(kOAngle, OANGLE, "<")                                              \
  X(kBang, BANG, "!")                                                  \
  X(kOBrack, OBRACK, "[")                                              \
  X(kCBrack, CBRACK, "]")                                              \
  X(kColon, COLON, ":")                                                \
  X(kComma, COMMA, ",")                                                \
  X(kDoubleQuote, DOUBLE_QUOTE, "\"")                                  \
  X(kStar, STAR, "*")                                                  \
  X(kSlash, SLASH, "/")                                                \
  X(kPercent, PERCENT, "%")                                            \
  X(kArrow, ARROW, "->")                                               \
  X(kSemi, SEMI, ";")                                                  \
  X(kAmpersand, AMPERSAND, "&")                                        \
  X(kDoubleAmpersand, DOUBLE_AMPERSAND, "&&")                          \
  X(kBar, BAR, "|")                                                    \
  X(kDoubleBar, DOUBLE_BAR, "||")                                      \
  X(kHat, HAT, "^")                                                    \
  X(kFatArrow, FAT_ARROW, "=>")                                        \
  X(kDoubleDot, DOUBLE_DOT, "..")                                      \
  X(kEllipsis, ELLIPSIS, "...")                                        \
  X(kHash, HASH, "#")                                                  \
  X(kString, STRING, "string")                                         \
  /* When in whitespace/comment mode; e.g. for syntax highlighting. */ \
  X(kWhitespace, WHITESPACE, "whitespace")                             \
  X(kComment, COMMENT, "comment")

#define XLS_FIRST_COMMA(A, ...) A,

enum class TokenKind : uint8_t { XLS_DSLX_TOKEN_KINDS(XLS_FIRST_COMMA) };

std::string TokenKindToString(TokenKind kind);

inline std::ostream& operator<<(std::ostream& os, TokenKind kind) {
  os << TokenKindToString(kind);
  return os;
}

enum class Keyword : uint8_t { XLS_DSLX_KEYWORDS(XLS_FIRST_COMMA) };

std::string KeywordToString(Keyword keyword);

std::optional<Keyword> KeywordFromString(std::string_view s);

// Returns a singleton set of type keywords.
const absl::flat_hash_set<Keyword>& GetTypeKeywords();

// Token yielded by the Scanner below.
class Token {
 public:
  Token(TokenKind kind, Span span,
        std::optional<std::string> value = std::nullopt)
      : kind_(kind), span_(std::move(span)), payload_(value) {}

  Token(Span span, Keyword keyword)
      : kind_(TokenKind::kKeyword), span_(std::move(span)), payload_(keyword) {}

  TokenKind kind() const { return kind_; }
  const Span& span() const { return span_; }

  std::optional<std::string> GetValue() const {
    if (std::holds_alternative<Keyword>(payload_)) {
      return KeywordToString(GetKeyword());
    }
    return std::get<std::optional<std::string>>(payload_);
  }

  // Note: assumes that the payload is not a keyword.
  const std::string& GetStringValue() const {
    return *std::get<std::optional<std::string>>(payload_);
  }

  absl::StatusOr<int64_t> GetValueAsInt64() const;

  bool IsKeywordIn(const absl::flat_hash_set<Keyword>& targets) const {
    return kind_ == TokenKind::kKeyword &&
           (targets.find(GetKeyword()) != targets.end());
  }

  bool IsTypeKeyword() const {
    return kind_ == TokenKind::kKeyword &&
           GetTypeKeywords().contains(GetKeyword());
  }

  Keyword GetKeyword() const { return std::get<Keyword>(payload_); }

  bool IsKeyword(Keyword target) const {
    return kind_ == TokenKind::kKeyword && GetKeyword() == target;
  }
  bool IsIdentifier(std::string_view target) const {
    return kind_ == TokenKind::kIdentifier && *GetValue() == target;
  }
  bool IsNumber(std::string_view target) const {
    return kind_ == TokenKind::kNumber && *GetValue() == target;
  }

  bool IsKindIn(
      absl::Span<std::variant<TokenKind, Keyword> const> targets) const {
    for (auto target : targets) {
      if (std::holds_alternative<TokenKind>(target)) {
        if (kind() == std::get<TokenKind>(target)) {
          return true;
        }
      } else {
        if (IsKeyword(std::get<Keyword>(target))) {
          return true;
        }
      }
    }
    return false;
  }

  bool operator==(const Token& other) const {
    return kind_ == other.kind_ && span_ == other.span_ &&
           payload_ == other.payload_;
  }
  bool operator!=(const Token& other) const { return !(*this == other); }

  // Returns a string that represents this token suitable for use in displaying
  // this token for user error reporting; e.g. "keyword:true".
  std::string ToErrorString() const;

  std::string ToString() const;

  std::string ToRepr(const FileTable& file_table) const;

 private:
  TokenKind kind_;
  Span span_;
  std::variant<std::optional<std::string>, Keyword> payload_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_TOKEN_H_
