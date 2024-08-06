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

#include <cctype>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/number_parser.h"

namespace xls {

std::string LexicalTokenTypeToString(LexicalTokenType token_type) {
  switch (token_type) {
    case LexicalTokenType::kAdd:
      return "+";
    case LexicalTokenType::kBracketClose:
      return "]";
    case LexicalTokenType::kBracketOpen:
      return "[";
    case LexicalTokenType::kColon:
      return ":";
    case LexicalTokenType::kComma:
      return ",";
    case LexicalTokenType::kCurlClose:
      return "}";
    case LexicalTokenType::kCurlOpen:
      return "{";
    case LexicalTokenType::kDot:
      return ".";
    case LexicalTokenType::kEquals:
      return "=";
    case LexicalTokenType::kGt:
      return ">";
    case LexicalTokenType::kIdent:
      return "ident";
    case LexicalTokenType::kKeyword:
      return "keyword";
    case LexicalTokenType::kLiteral:
      return "literal";
    case LexicalTokenType::kLt:
      return "<";
    case LexicalTokenType::kMinus:
      return "-";
    case LexicalTokenType::kParenClose:
      return ")";
    case LexicalTokenType::kParenOpen:
      return "(";
    case LexicalTokenType::kQuotedString:
      return "quoted string";
    case LexicalTokenType::kRightArrow:
      return "->";
    case LexicalTokenType::kHash:
      return "#";
  }
  return absl::StrCat("LexicalTokenType(", static_cast<int>(token_type), ")");
}

std::string TokenPos::ToHumanString() const {
  return absl::StrFormat("%d:%d", lineno + 1, colno + 1);
}

absl::StatusOr<bool> Token::IsNegative() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError("Can only get sign for literal tokens.");
  }
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(value()));
  return pair.first;
}

absl::StatusOr<Bits> Token::GetValueBits() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError(
        "Can only get value as integer for literal tokens.");
  }
  return ParseNumber(value());
}

absl::StatusOr<int64_t> Token::GetValueInt64() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError(
        "Can only get value as integer for literal tokens.");
  }
  return ParseNumberAsInt64(value());
}

absl::StatusOr<bool> Token::GetValueBool() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError(
        "Can only get value as integer for literal tokens.");
  }
  return ParseNumberAsBool(value());
}

std::string Token::ToString() const {
  return absl::StrFormat("Token(\"%s\", value=\"%s\") @ %s",
                         LexicalTokenTypeToString(type_), value_,
                         pos_.ToHumanString());
}

namespace {

// Helper class for tokenizing a string.
// TODO(meheff): This could be combined into Scanner class and made lazy at the
// same time.
class Tokenizer {
 public:
  // Tokenizes the given string and returns the vector of Tokens.
  static absl::StatusOr<std::vector<Token>> TokenizeString(
      std::string_view str) {
    Tokenizer tokenizer(str);
    return tokenizer.Tokenize();
  }

 private:
  // Drops all whitespace starting at current index. Returns true if any
  // whitespace was dropped.
  bool DropWhiteSpace() {
    int64_t old_index = index();
    while (!EndOfString() && absl::ascii_isspace(current())) {
      Advance();
    }
    return old_index != index();
  }

  // Tries to drop an end of line comment starting with "//" at the current
  // index up to the newline. Returns true an end of line comment was found.
  bool DropEndOfLineComment() {
    if (MatchSubstring("//")) {
      Advance(2);
      while (!EndOfString() && current() != '\n') {
        Advance(1);
      }
      return true;
    }
    return false;
  }

  // Returns true if the given string matches the substring starting at the
  // current index in the tokenized string.
  bool MatchSubstring(std::string_view substr) {
    return index_ + substr.size() <= str_.size() &&
           substr == std::string_view(str_.data() + index_, substr.size());
  }

  // Tries to match a quoted string with the given quote character sequence
  // (e.g., """). Returns the contents of the quoted string or nullopt if no
  // quoted string was matched. allow_multine indicates whether a newline
  // character is allowed in the quoted string.
  absl::StatusOr<std::optional<std::string_view>> MatchQuotedString(
      std::string_view quote, bool allow_multiline) {
    if (!MatchSubstring(quote)) {
      return std::nullopt;
    }
    int64_t start_colno = colno();
    int64_t start_lineno = lineno();
    Advance(quote.size());
    int64_t content_start = index();
    while (!EndOfString()) {
      if (MatchSubstring(quote)) {
        std::string_view content = std::string_view(str_.data() + content_start,
                                                    index() - content_start);
        Advance(quote.size());
        return content;
      }
      if (!allow_multiline && current() == '\n') {
        break;
      }
      Advance();
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Unterminated quoted string starting at %s",
                        TokenPos{start_lineno, start_colno}.ToHumanString()));
  }

  // Advances the current index into the tokenized string by the given
  // amount. Updates column and line numbers.
  int64_t Advance(int64_t amount = 1) {
    CHECK_LE(index_ + amount, str_.size());
    for (int64_t i = 0; i < amount; ++i) {
      if (current() == '\t') {
        colno_ += 2;
      } else if (current() == '\n') {
        colno_ = 0;
        ++lineno_;
      } else {
        ++colno_;
      }
      ++index_;
    }
    return index_;
  }

  // Returns whether the current index is at the end of the string.
  bool EndOfString() const { return index_ >= str_.size(); }

  // Returns the sequence of all characters which satisfy the given test
  // starting at the current index. Current index is updated to one past the
  // last matching character. min_chars is the minimum number of characters
  // which are unconditionally captured.
  std::string_view CaptureWhile(std::function<bool(char)> test_f,
                                int64_t min_chars = 0) {
    int64_t start = index();
    while (!EndOfString() &&
           ((index() < min_chars + start) || test_f(current()))) {
      Advance();
    }
    return std::string_view(str_.data() + start, index_ - start);
  }

  // Tokenizes the internal string.
  absl::StatusOr<std::vector<Token>> Tokenize() {
    std::vector<Token> tokens;
    while (!EndOfString()) {
      if (DropWhiteSpace() || DropEndOfLineComment()) {
        continue;
      }

      const int64_t start_lineno = lineno();
      const int64_t start_colno = colno();

      // Literal numbers can decimal, binary (eg, 0b0101) or hexadecimal (eg,
      // 0xbeef) so capture all alphanumeric characters after the initial
      // digit. Literal numbers can also contain '_'s after the first
      // character which are used to improve readability (example:
      // '0xabcd_ef00').
      if ((isdigit(current()) != 0) ||
          (current() == '-' && next().has_value() && (isdigit(*next()) != 0))) {
        std::string_view value = CaptureWhile(
            [](char c) { return absl::ascii_isalnum(c) || c == '_'; },
            /*min_chars=*/1);
        tokens.push_back(Token(LexicalTokenType::kLiteral, value, start_lineno,
                               start_colno));
        continue;
      }

      if (isalpha(current()) != 0 || current() == '_') {
        std::string_view value = CaptureWhile([](char c) {
          return isalpha(c) != 0 || c == '_' || c == '.' || isdigit(c) != 0;
        });
        tokens.push_back(
            Token::MakeIdentOrKeyword(value, start_lineno, start_colno));
        continue;
      }

      // Look for multi-character tokens.
      if (MatchSubstring("->")) {
        tokens.push_back(Token(LexicalTokenType::kRightArrow, "->",
                               start_lineno, start_colno));
        Advance(2);
        continue;
      }

      // Match quoted strings. Double-quoted strings (e.g., "foo") and
      // triple-double-quoted strings (e.g., """foo""") are allowed. Only
      // triple-double-quoted strings can contain new lines.
      std::optional<std::string_view> content;
      XLS_ASSIGN_OR_RETURN(
          content, MatchQuotedString("\"\"\"", /*allow_multiline=*/true));
      if (content.has_value()) {
        tokens.push_back(Token(LexicalTokenType::kQuotedString, content.value(),
                               start_lineno, start_colno));
        continue;
      }
      XLS_ASSIGN_OR_RETURN(content,
                           MatchQuotedString("\"", /*allow_multiline=*/false));
      if (content.has_value()) {
        tokens.push_back(Token(LexicalTokenType::kQuotedString, content.value(),
                               start_lineno, start_colno));
        continue;
      }

      // Handle single-character tokens.
      LexicalTokenType token_type;

      switch (current()) {
        case '-':
          token_type = LexicalTokenType::kMinus;
          break;
        case '+':
          token_type = LexicalTokenType::kAdd;
          break;
        case '.':
          token_type = LexicalTokenType::kDot;
          break;
        case ':':
          token_type = LexicalTokenType::kColon;
          break;
        case ',':
          token_type = LexicalTokenType::kComma;
          break;
        case '=':
          token_type = LexicalTokenType::kEquals;
          break;
        case '[':
          token_type = LexicalTokenType::kBracketOpen;
          break;
        case ']':
          token_type = LexicalTokenType::kBracketClose;
          break;
        case '{':
          token_type = LexicalTokenType::kCurlOpen;
          break;
        case '}':
          token_type = LexicalTokenType::kCurlClose;
          break;
        case '(':
          token_type = LexicalTokenType::kParenOpen;
          break;
        case ')':
          token_type = LexicalTokenType::kParenClose;
          break;
        case '>':
          token_type = LexicalTokenType::kGt;
          break;
        case '<':
          token_type = LexicalTokenType::kLt;
          break;
        case '#':
          token_type = LexicalTokenType::kHash;
          break;
        default:
          std::string char_str = absl::ascii_iscntrl(current())
                                     ? absl::StrFormat("\\x%02x", current())
                                     : std::string(1, current());
          LOG(ERROR) << "IR text with error: " << str_;
          return absl::InvalidArgumentError(absl::StrFormat(
              "Invalid character in IR text \"%s\" @ %s", char_str,
              TokenPos{lineno(), colno()}.ToHumanString()));
      }
      tokens.push_back(Token(token_type, lineno(), colno()));
      Advance();
    }
    return tokens;
  }

  // Returns the character at the current index.
  char current() const { return str_.at(index_); }

  // Returns the character at the current index + 1, or nullopt if current index
  // + 1 is beyond the end of the string.
  std::optional<char> next() const {
    if (index_ + 1 < str_.size()) {
      return str_[index_ + 1];
    }
    return std::nullopt;
  }

  // Returns the current index in the string.
  int64_t index() const { return index_; }

  // Returns the current line/column number.
  int64_t lineno() const { return lineno_; }
  int64_t colno() const { return colno_; }

 private:
  explicit Tokenizer(std::string_view str) : str_(str) {}

  // The string being tokenized.
  std::string_view str_;

  // Current index.
  int64_t index_ = 0;

  // Line/column number based on the current index.
  int64_t lineno_ = 0;
  int64_t colno_ = 0;
};

}  // namespace

absl::StatusOr<std::vector<Token>> TokenizeString(std::string_view str) {
  return Tokenizer::TokenizeString(str);
}

absl::StatusOr<Scanner> Scanner::Create(std::string_view text) {
  XLS_ASSIGN_OR_RETURN(auto tokens, TokenizeString(text));
  return Scanner(std::move(tokens));
}

absl::StatusOr<Token> Scanner::PeekToken() const {
  if (AtEof()) {
    return absl::InvalidArgumentError("Expected token, but found EOF.");
  }
  return tokens_[token_idx_];
}

absl::StatusOr<Token> Scanner::PopTokenOrError(std::string_view context) {
  if (AtEof()) {
    std::string context_str =
        context.empty() ? std::string("") : absl::StrCat(" in ", context);
    return absl::InvalidArgumentError("Expected token" + context_str +
                                      ", but found EOF.");
  }
  return PopToken();
}

bool Scanner::TryDropToken(LexicalTokenType target) {
  if (PeekTokenIs(target)) {
    PopToken();
    return true;
  }
  return false;
}

absl::Status Scanner::DropTokenOrError(LexicalTokenType target,
                                       std::string_view context) {
  if (AtEof()) {
    std::string context_str =
        context.empty() ? std::string("") : absl::StrCat(" in ", context);
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected token of type %s%s; found EOF.",
                        LexicalTokenTypeToString(target), context_str));
  }
  XLS_ASSIGN_OR_RETURN(Token dropped, PopTokenOrError(target, context));
  (void)dropped;
  return absl::OkStatus();
}

absl::StatusOr<Token> Scanner::PopTokenOrError(LexicalTokenType target,
                                               std::string_view context) {
  XLS_ASSIGN_OR_RETURN(Token token, PopTokenOrError());
  if (token.type() != target) {
    std::string context_str =
        context.empty() ? std::string("") : absl::StrCat(" in ", context);
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected token of type \"%s\"%s @ %s, but found: %s",
                        LexicalTokenTypeToString(target), context_str,
                        token.pos().ToHumanString(), token.ToString()));
  }
  return token;
}

absl::StatusOr<Token> Scanner::PopKeywordOrIdentToken(
    std::string_view context) {
  XLS_ASSIGN_OR_RETURN(Token token, PopTokenOrError());
  if (token.type() != LexicalTokenType::kIdent &&
      token.type() != LexicalTokenType::kKeyword) {
    std::string context_str =
        context.empty() ? std::string("") : absl::StrCat(" in ", context);
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected keyword or identifier token %s @ %s, but found: %s",
        context_str, token.pos().ToHumanString(), token.ToString()));
  }
  return token;
}

absl::StatusOr<Token> Scanner::PopLiteralOrStringToken(
    std::string_view context) {
  XLS_ASSIGN_OR_RETURN(Token token, PopTokenOrError());
  if (token.type() != LexicalTokenType::kLiteral &&
      token.type() != LexicalTokenType::kQuotedString) {
    std::string context_str =
        context.empty() ? std::string("") : absl::StrCat(" in ", context);
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected literal or quoted string token %s @ %s, but found: %s",
        context_str, token.pos().ToHumanString(), token.ToString()));
  }
  return token;
}

absl::Status Scanner::DropKeywordOrError(std::string_view keyword) {
  absl::StatusOr<Token> popped_status = PopTokenOrError();
  if (!popped_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected keyword '%s': %s", keyword,
                        popped_status.status().message()));
  }
  XLS_ASSIGN_OR_RETURN(Token popped, popped_status);
  if (popped.type() == LexicalTokenType::kKeyword &&
      keyword == popped.value()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Expected '%s' keyword; got: %s @ %s", keyword,
                      popped.ToString(), popped.pos().ToHumanString()));
}

}  // namespace xls
