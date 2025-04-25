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

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner_keywords.inc"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {

absl::Status ScanErrorStatus(const Span& span, std::string_view message,
                             const FileTable& file_table) {
  return absl::InvalidArgumentError(
      absl::StrFormat("ScanError: %s %s", span.ToString(file_table), message));
}

char Scanner::PopChar() {
  CHECK(!AtEof()) << "Cannot pop character when at EOF.";
  char c = PeekChar();
  index_ += 1;
  if (c == '\n') {
    lineno_ += 1;
    colno_ = 0;
  } else {
    colno_ += 1;
  }
  return c;
}

void Scanner::DropChar(int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    (void)PopChar();
  }
}

bool Scanner::TryDropChar(char target) {
  if (!AtCharEof() && PeekChar() == target) {
    DropChar();
    return true;
  }
  return false;
}

Token Scanner::PopComment(const Pos& start_pos, bool allow_multiline) {
  std::string chars;
  Pos end_pos = GetPos();
  while (!AtCharEof()) {
    char c = PopChar();
    chars.append(1, c);
    end_pos = GetPos();
    if (c == '\n') {
      // If we've collected a comment, conditionally look for a continuation of
      // it on the next line at the same colno.
      if (allow_multiline && !AtCharEof() && PeekChar() != '\n' &&
          chars.size() > 1) {
        DropLeadingWhitespace();
        if (!AtCharEof() && PeekChar() == '/' && PeekChar2OrNull() == '/' &&
            GetPos().colno() == start_pos.colno()) {
          DropChar(2);
          continue;
        }
      }
      break;
    }
  }
  return Token(TokenKind::kComment, Span(start_pos, end_pos), chars);
}

absl::StatusOr<Token> Scanner::PopWhitespace(const Pos& start_pos) {
  CHECK(AtWhitespace());
  std::string chars;
  while (!AtCharEof() && AtWhitespace()) {
    chars.append(1, PopChar());
  }
  return Token(TokenKind::kWhitespace, Span(start_pos, GetPos()), chars);
}

// This is too simple to need to return absl::Status. Just never call it
// with a non-hex character.
static uint8_t HexCharToU8(char hex_char) {
  if (std::isdigit(hex_char) != 0) {
    return hex_char - '0';
  }
  if ('a' <= hex_char && hex_char <= 'f') {
    return hex_char - 'a' + 10;
  }
  if ('A' <= hex_char && hex_char <= 'F') {
    return hex_char - 'A' + 10;
  }
  LOG(FATAL) << "Non-hex character received: " << hex_char;
}

// Returns the next character literal.
absl::StatusOr<char> Scanner::ScanCharLiteral() {
  const Pos start_pos = GetPos();
  char current = PopChar();
  if (current != '\\' || AtCharEof()) {
    return current;
  }
  // All codes given in hex for consistency.
  char next = PeekChar();
  if (next == 'n') {
    DropChar();
    return '\x0a';  // Newline.
  }
  if (next == 'r') {
    DropChar();
    return '\x0d';  // Carriage return.
  }
  if (next == 't') {
    DropChar();
    return '\x09';  // Tab.
  }
  if (next == '\\') {
    DropChar();
    return '\x5c';  // Backslash.
  }
  if (next == '0') {
    DropChar();
    return '\x00';  // Null.
  }
  if (next == '\'') {
    DropChar();
    return '\x27';  // Single quote/apostrophe.
  }
  if (next == '"') {
    DropChar();
    return '\x22';  // Double quote/apostrophe.
  }
  if (next == 'x') {
    // Hex character code. Now read [exactly] two more digits.
    DropChar();

    uint8_t code = 0;
    for (int i = 0; i < 2; i++) {
      if (AtEof()) {
        return ScanErrorStatus(
            Span(start_pos, GetPos()),
            "Unexpected EOF in escaped hexadecimal character.");
      }
      next = PeekChar();
      if (!absl::ascii_isxdigit(next)) {
        return ScanErrorStatus(
            Span(start_pos, GetPos()),
            "Only hex digits are allowed within a 7-bit character code.");
      }
      code = static_cast<uint8_t>((code << 4) | HexCharToU8(next));
      DropChar();
    }

    return code & 255;
  }
  return ScanErrorStatus(Span(start_pos, GetPos()),
                         absl::StrCat("Unrecognized escape sequence: `\\",
                                      std::string(1, next), "`"));
}

// Returns a string with the next "character" in the string. A string is
// returned instead of a "char", since multi-byte Unicode characters are valid
// constituents of a string.
absl::StatusOr<std::string> Scanner::ProcessNextStringChar() {
  const Pos start_pos = GetPos();
  if (PeekChar() == '\\' && PeekChar2OrNull() == 'u') {
    // Unicode character code.
    DropChar();
    DropChar();
    if (AtEof()) {
      return ScanErrorStatus(Span(start_pos, GetPos()),
                             "Unexpected EOF in escaped unicode character.");
    }
    if (PeekChar() != '{') {
      return ScanErrorStatus(
          Span(start_pos, GetPos()),
          "Unicode character code escape sequence start (\\u) "
          "must be followed by a character code, such as \"{...}\".");
    }
    DropChar();

    // At most 6 hex digits allowed.
    std::string unicode;
    for (int i = 0; i < 6 && !AtEof(); i++) {
      char next = PeekChar();
      if (absl::ascii_isxdigit(next)) {
        absl::StrAppend(&unicode, std::string(1, next));
        DropChar();
      } else if (next == '}') {
        break;
      } else {
        return ScanErrorStatus(
            Span(start_pos, GetPos()),
            "Only hex digits are allowed within a Unicode character code.");
      }
    }
    if (AtEof() || PeekChar() != '}') {
      return ScanErrorStatus(
          Span(start_pos, GetPos()),
          "Unicode character code escape sequence must terminate "
          "(after 6 digits at most) with a '}'");
    }
    DropChar();

    if (unicode.empty()) {
      return ScanErrorStatus(
          Span(start_pos, GetPos()),
          "Unicode escape must contain at least one character.");
    }

    // Add padding and unicode escape characters to conform with
    // absl::CUnescape.
    std::string to_unescape;
    if (unicode.size() <= 4) {
      to_unescape = "\\u";
      to_unescape.insert(to_unescape.size(), 4 - unicode.size(), '0');
      to_unescape += unicode;
    } else {
      XLS_RET_CHECK(unicode.size() <= 8);
      to_unescape = "\\U";
      to_unescape.insert(to_unescape.size(), 8 - unicode.size(), '0');
      to_unescape += unicode;
    }

    std::string utf8;
    if (!absl::CUnescape(to_unescape, &utf8)) {
      // Note: we report the error using the characters the user provided
      // instead of what we created to feed CUnescape.
      return ScanErrorStatus(
          Span(start_pos, GetPos()),
          absl::StrFormat("Invalid unicode sequence: '%s'.",
                          absl::StrCat(R"(\u{)", unicode, "}")));
    }
    return utf8;
  }
  XLS_ASSIGN_OR_RETURN(char c, ScanCharLiteral());
  return std::string(1, c);
}

absl::StatusOr<std::string> Scanner::ScanUntilDoubleQuote() {
  const Pos start_pos = GetPos();

  std::string result;
  while (!AtCharEof() && PeekChar() != '\"') {
    XLS_ASSIGN_OR_RETURN(std::string next, ProcessNextStringChar());
    absl::StrAppend(&result, next);
  }

  if (AtEof()) {
    return ScanErrorStatus(
        Span(start_pos, GetPos()),
        "Reached end of file without finding a closing double quote.");
  }

  return result;
}

/* static */ std::optional<Keyword> Scanner::GetKeyword(std::string_view s) {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, Keyword>>
      mapping({
#define MAKE_ITEM(__enum, unused, __str, ...) {__str, Keyword::__enum},
          XLS_DSLX_KEYWORDS(MAKE_ITEM)
#undef MAKE_ITEM
      });
  auto it = mapping->find(s);
  if (it == mapping->end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::StatusOr<Token> Scanner::ScanIdentifierOrKeyword(char startc,
                                                       const Pos& start_pos) {
  // The leading character is `startc` so we scan out trailing identifiers.
  auto is_trailing_identifier_char = [](char c) {
    return std::isalpha(c) != 0 || std::isdigit(c) != 0 || c == '_' ||
           c == '!' || c == '\'';
  };
  std::string s = ScanWhile(startc, is_trailing_identifier_char);
  Span span(start_pos, GetPos());
  if (std::optional<Keyword> keyword = GetKeyword(s)) {
    return Token(span, *keyword);
  }
  return Token(TokenKind::kIdentifier, span, std::move(s));
}

std::optional<CommentData> Scanner::TryPopComment(bool allow_multiline) {
  const Pos start_pos = GetPos();
  if (!AtEof() && PeekChar() == '/' && PeekChar2OrNull() == '/') {
    DropChar(2);
    Token token = PopComment(start_pos, allow_multiline);
    CHECK(token.GetValue().has_value());
    return CommentData{.span = token.span(), .text = token.GetValue().value()};
  }
  return std::nullopt;
}

absl::StatusOr<std::optional<Token>> Scanner::TryPopWhitespaceOrComment() {
  const Pos start_pos = GetPos();
  if (AtCharEof()) {
    return Token(TokenKind::kEof, Span(start_pos, start_pos));
  }
  if (AtWhitespace()) {
    XLS_ASSIGN_OR_RETURN(Token token, PopWhitespace(start_pos));
    return token;
  }
  if (PeekChar() == '/' && PeekChar2OrNull() == '/') {
    DropChar(2);
    Token token = PopComment(start_pos, /* allow_multiline= */ false);
    return token;
  }
  return std::nullopt;
}

absl::StatusOr<Token> Scanner::ScanString(const Pos& start_pos) {
  DropChar();
  XLS_ASSIGN_OR_RETURN(std::string value, ScanUntilDoubleQuote());
  if (!TryDropChar('"')) {
    return ScanErrorStatus(
        Span(start_pos, GetPos()),
        "Expected close quote character to terminate open quote character.");
  }
  return Token(TokenKind::kString, Span(start_pos, GetPos()), value);
}

absl::StatusOr<Token> Scanner::ScanNumber(char startc, const Pos& start_pos) {
  bool negative = startc == '-';
  if (negative) {
    startc = PopChar();
  }

  std::string s;
  if (startc == '0' && TryDropChar('x')) {  // Hex radix.
    s = ScanWhile("0x", [](char c) {
      return ('0' <= c && c <= '9') || ('a' <= c && c <= 'f') ||
             ('A' <= c && c <= 'F') || c == '_';
    });
    if (s == "0x") {
      return ScanErrorStatus(Span(GetPos(), GetPos()),
                             "Expected hex characters following 0x prefix.");
    }
  } else if (startc == '0' && TryDropChar('b')) {  // Bin prefix.
    s = ScanWhile("0b",
                  [](char c) { return ('0' <= c && c <= '1') || c == '_'; });
    if (s == "0b") {
      return ScanErrorStatus(Span(GetPos(), GetPos()),
                             "Expected binary characters following 0b prefix");
    }
    if (!AtEof() && '0' <= PeekChar() && PeekChar() <= '9') {
      return ScanErrorStatus(
          Span(GetPos(), GetPos()),
          absl::StrFormat("Invalid digit for binary number: '%c'", PeekChar()));
    }
  } else {
    s = ScanWhile(startc, [](char c) { return std::isdigit(c) != 0; });
    if (absl::StartsWith(s, "0") && s.size() != 1) {
      return ScanErrorStatus(
          Span(GetPos(), GetPos()),
          "Invalid radix for number, expect 0b or 0x because of leading 0.");
    }
    CHECK(!s.empty())
        << "Must have seen numerical digits to attempt to scan a number.";
  }
  if (negative) {
    s = "-" + s;
  }
  return Token(TokenKind::kNumber, Span(start_pos, GetPos()), s);
}

bool Scanner::AtWhitespace() const {
  switch (PeekChar()) {
    case ' ':
    case '\r':
    case '\n':
    case '\t':
    case '\xa0':
      return true;
    default:
      return false;
  }
}

void Scanner::DropLeadingWhitespace() {
  while (!AtCharEof()) {
    if (AtWhitespace()) {
      DropChar();
    } else {
      break;
    }
  }
}

absl::StatusOr<Token> Scanner::ScanChar(const Pos& start_pos) {
  const char open_quote = PopChar();
  CHECK_EQ(open_quote, '\'');
  if (AtCharEof()) {
    return ScanErrorStatus(
        Span(GetPos(), GetPos()),
        "Expected character after single quote, saw end of file.");
  }
  if (PeekChar() == '\n') {
    return ScanErrorStatus(
        Span(start_pos, GetPos()),
        "Newline found in character literal. Newlines are not allowed in "
        "character literals.");
  }
  XLS_ASSIGN_OR_RETURN(char c, ScanCharLiteral());
  if (AtCharEof() || !TryDropChar('\'')) {
    std::string msg = absl::StrFormat(
        "Expected closing single quote for character literal; got %s",
        AtCharEof() ? std::string("end of file") : std::string(1, PeekChar()));
    return ScanErrorStatus(Span(GetPos(), GetPos()), msg);
  }
  return Token(TokenKind::kCharacter, Span(start_pos, GetPos()),
               std::string(1, c));
}

absl::StatusOr<Token> Scanner::Pop() {
  if (include_whitespace_and_comments_) {
    XLS_ASSIGN_OR_RETURN(std::optional<Token> tok, TryPopWhitespaceOrComment());
    if (tok) {
      return *tok;
    }
  } else {
    while (true) {
      // Allow inline comments to be multi-line.
      bool allow_multiline =
          !AtCharEof() && !(GetPos().colno() == 0 || PeekChar() == '\n');
      DropLeadingWhitespace();

      if (std::optional<CommentData> comment = TryPopComment(allow_multiline)) {
        comments_.push_back(std::move(comment).value());
      } else {
        // Dropped whitespace and not seeing a comment, good to go scan a token.
        break;
      }
    }
  }

  // Record the position the token starts at.
  const Pos start_pos = GetPos();
  auto mk_span = [&]() -> Span { return Span(start_pos, GetPos()); };

  // After dropping whitespace this may be EOF.
  if (AtCharEof()) {
    return Token(TokenKind::kEof, mk_span());
  }

  // Peek at one character for prefix scanning.
  const char startc = PeekChar();
  std::optional<Token> result;
  switch (startc) {
    case '"': {
      XLS_ASSIGN_OR_RETURN(result, ScanString(start_pos));
      break;
    }
    case '\'': {
      XLS_ASSIGN_OR_RETURN(result, ScanChar(start_pos));
      break;
    }
    case '#':
      DropChar();
      result = Token(TokenKind::kHash, mk_span());
      break;
    case '!':
      DropChar();
      if (TryDropChar('=')) {
        result = Token(TokenKind::kBangEquals, mk_span());
      } else {
        result = Token(TokenKind::kBang, mk_span());
      }
      break;
    case '=':
      DropChar();
      if (TryDropChar('=')) {
        result = Token(TokenKind::kDoubleEquals, mk_span());
      } else if (TryDropChar('>')) {
        result = Token(TokenKind::kFatArrow, mk_span());
      } else {
        result = Token(TokenKind::kEquals, mk_span());
      }
      break;
    case '+':
      DropChar();
      if (TryDropChar('+')) {
        result = Token(TokenKind::kDoublePlus, mk_span());
      } else if (TryDropChar(':')) {
        result = Token(TokenKind::kPlusColon, mk_span());
      } else {
        result = Token(TokenKind::kPlus, mk_span());
      }
      break;
    case '<':
      DropChar();
      if (TryDropChar('<')) {
        result = Token(TokenKind::kDoubleOAngle, mk_span());
      } else if (TryDropChar('=')) {
        result = Token(TokenKind::kOAngleEquals, mk_span());
      } else {
        result = Token(TokenKind::kOAngle, mk_span());
      }
      break;
    case '>':
      DropChar();
      if (double_c_angle_enabled_ && TryDropChar('>')) {
        result = Token(TokenKind::kDoubleCAngle, mk_span());
      } else if (TryDropChar('=')) {
        result = Token(TokenKind::kCAngleEquals, mk_span());
      } else {
        result = Token(TokenKind::kCAngle, mk_span());
      }
      break;
    case '.':
      DropChar();
      if (TryDropChar('.')) {
        if (TryDropChar('.')) {
          result = Token(TokenKind::kEllipsis, mk_span());
        } else if (TryDropChar('=')) {
          result = Token(TokenKind::kDoubleDotEquals, mk_span());
        } else {
          result = Token(TokenKind::kDoubleDot, mk_span());
        }
      } else {
        result = Token(TokenKind::kDot, mk_span());
      }
      break;
    case ':':
      DropChar();
      if (TryDropChar(':')) {
        result = Token(TokenKind::kDoubleColon, mk_span());
      } else {
        result = Token(TokenKind::kColon, mk_span());
      }
      break;
    case '|':
      DropChar();
      if (TryDropChar('|')) {
        result = Token(TokenKind::kDoubleBar, mk_span());
      } else {
        result = Token(TokenKind::kBar, mk_span());
      }
      break;
    case '&':
      DropChar();
      if (TryDropChar('&')) {
        result = Token(TokenKind::kDoubleAmpersand, mk_span());
      } else {
        result = Token(TokenKind::kAmpersand, mk_span());
      }
      break;
      // clang-format off
    case '(': DropChar(); result = Token(TokenKind::kOParen, mk_span()); break;  // NOLINT
    case ')': DropChar(); result = Token(TokenKind::kCParen, mk_span()); break;  // NOLINT
    case '[': DropChar(); result = Token(TokenKind::kOBrack, mk_span()); break;  // NOLINT
    case ']': DropChar(); result = Token(TokenKind::kCBrack, mk_span()); break;  // NOLINT
    case '{': DropChar(); result = Token(TokenKind::kOBrace, mk_span()); break;  // NOLINT
    case '}': DropChar(); result = Token(TokenKind::kCBrace, mk_span()); break;  // NOLINT
    case ',': DropChar(); result = Token(TokenKind::kComma, mk_span()); break;  // NOLINT
    case ';': DropChar(); result = Token(TokenKind::kSemi, mk_span()); break;  // NOLINT
    case '*': DropChar(); result = Token(TokenKind::kStar, mk_span()); break;  // NOLINT
    case '%': DropChar(); result = Token(TokenKind::kPercent, mk_span()); break;  // NOLINT
    case '^': DropChar(); result = Token(TokenKind::kHat, mk_span()); break;  // NOLINT
    case '/': DropChar(); result = Token(TokenKind::kSlash, mk_span()); break;  // NOLINT
    // clang-format on
    default:
      if (std::isalpha(startc) != 0 || startc == '_') {
        XLS_ASSIGN_OR_RETURN(result,
                             ScanIdentifierOrKeyword(PopChar(), start_pos));
      } else if (std::isdigit(startc) != 0 ||
                 (startc == '-' && std::isdigit((PeekChar2OrNull())) != 0)) {
        XLS_ASSIGN_OR_RETURN(result, ScanNumber(PopChar(), start_pos));
      } else if (startc == '-') {  // Minus handling is after the "number" path.
        DropChar();
        if (TryDropChar('>')) {
          result = Token(TokenKind::kArrow, mk_span());
        } else {
          result = Token(TokenKind::kMinus, mk_span());
        }
      } else {
        return ScanErrorStatus(
            Span(GetPos(), GetPos()),
            absl::StrFormat("Unrecognized character: '%c' (%#x)", startc,
                            startc));
      }
  }

  CHECK(result.has_value());
  return std::move(result).value();
}

}  // namespace xls::dslx
