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

#include "xls/netlist/netlist_parser.h"

#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace netlist {
namespace rtl {

std::string Pos::ToHumanString() const {
  return absl::StrFormat("%d:%d", lineno + 1, colno + 1);
}

std::string TokenKindToString(TokenKind kind) {
  switch (kind) {
    case TokenKind::kStartParams:
      return "start-params";
    case TokenKind::kOpenParen:
      return "open-paren";
    case TokenKind::kCloseParen:
      return "close-paren";
    case TokenKind::kOpenBracket:
      return "open-bracket";
    case TokenKind::kCloseBracket:
      return "close-bracket";
    case TokenKind::kOpenBrace:
      return "open-brace";
    case TokenKind::kCloseBrace:
      return "close-brace";
    case TokenKind::kDot:
      return "dot";
    case TokenKind::kComma:
      return "comma";
    case TokenKind::kSemicolon:
      return "semicolon";
    case TokenKind::kColon:
      return "colon";
    case TokenKind::kEquals:
      return "equals";
    case TokenKind::kQuote:
      return "quote";
    case TokenKind::kName:
      return "name";
    case TokenKind::kNumber:
      return "number";
  }
  return absl::StrCat("<invalid-kind-%d>", static_cast<int>(kind));
}

std::string Token::ToString() const {
  if (kind == TokenKind::kName) {
    return absl::StrFormat("Token{kName, @%s, \"%s\"}", pos.ToHumanString(),
                           value);
  }
  return absl::StrFormat("Token{%s, @%s}", TokenKindToString(kind),
                         pos.ToHumanString());
}

char Scanner::PeekCharOrDie() const {
  CHECK(!AtEofInternal());
  return text_[index_];
}

char Scanner::PeekChar2OrDie() const {
  CHECK_GT(text_.size(), index_ + 1);
  return text_[index_ + 1];
}

char Scanner::PopCharOrDie() {
  CHECK(!AtEofInternal());
  char c = text_[index_++];
  if (c == '\n') {
    lineno_++;
    colno_ = 0;
  } else {
    colno_++;
  }
  return c;
}

void Scanner::DropIgnoredChars() {
  auto drop_to_eol_or_eof = [this] {
    while (!AtEofInternal() && PeekCharOrDie() != '\n') {
      DropCharOrDie();
    }
  };
  auto drop_to_block_comment_end_or_eof = [this] {
    char previous_char = '0';  // arbitrary char that is not '*'
    while (!AtEofInternal()) {
      if (PeekCharOrDie() == '/' && previous_char == '*') {
        DropCharOrDie();
        break;
      }
      previous_char = PeekCharOrDie();
      DropCharOrDie();
    }
  };
  auto drop_to_attr_end_or_eof = [this] {
    char previous_char = '0';  // arbitrary char that is not '*'
    while (!AtEofInternal()) {
      if (PeekCharOrDie() == ')' && previous_char == '*') {
        DropCharOrDie();
        break;
      }
      previous_char = PeekCharOrDie();
      DropCharOrDie();
    }
  };
  while (!AtEofInternal()) {
    switch (PeekCharOrDie()) {
      case '/': {
        if (PeekChar2OrDie() == '/') {
          DropCharOrDie();
          DropCharOrDie();
          drop_to_eol_or_eof();
          continue;
        }
        if (PeekChar2OrDie() == '*') {
          DropCharOrDie();
          DropCharOrDie();
          drop_to_block_comment_end_or_eof();
          continue;
        }
        return;
      }
      case '(': {
        if (PeekChar2OrDie() == '*') {
          DropCharOrDie();
          DropCharOrDie();
          drop_to_attr_end_or_eof();
          continue;
        }
        return;
      }
      case ' ':
      case '\n':
      case '\t':
        DropCharOrDie();
        break;
      default:
        return;
    }
  }
}

absl::StatusOr<Token> Scanner::Peek() {
  if (lookahead_.has_value()) {
    return lookahead_.value();
  }
  XLS_ASSIGN_OR_RETURN(Token token, PeekInternal());
  lookahead_.emplace(token);
  return lookahead_.value();
}

absl::StatusOr<Token> Scanner::Pop() {
  XLS_ASSIGN_OR_RETURN(Token result, Peek());
  lookahead_.reset();
  VLOG(3) << "Popping token: " << result.ToString();
  return result;
}

absl::StatusOr<Token> Scanner::ScanNumber(char startc, Pos pos) {
  std::string chars;
  chars.push_back(startc);
  bool seen_separator = false;
  auto is_hex_char = [](char c) {
    return absl::ascii_isxdigit(absl::ascii_toupper(c));
  };

  // This isn't quite right - if there's an apostrophe (i.e., if this is a sized
  // number), then the size (the first component) must be decimal-only.
  // It's probably fine to ignore that restriction, though.
  //
  // This also can't handle reals (no decimal or sign support)...but we don't
  // expect them to show up in netlists.
  while (!AtEofInternal()) {
    char c = PeekCharOrDie();
    if (is_hex_char(c)) {
      chars.push_back(PopCharOrDie());
    } else if (c == '\'' && !seen_separator) {
      // If we see a base separator, pop it, then the optional signedness
      // indicator (s|S), then the base indicator (d|b|o|h|D|B|O|H).
      chars.push_back(PopCharOrDie());
      XLS_RET_CHECK(!AtEofInternal()) << "Saw EOF while scanning number base!";
      chars.push_back(PopCharOrDie());
      if (chars.back() == 's' || chars.back() == 'S') {
        XLS_RET_CHECK(!AtEofInternal())
            << "Saw EOF while scanning number base (post-signedness)!";
        chars.push_back(PopCharOrDie());
      }

      c = chars.back();
      XLS_RET_CHECK(c == 'd' || c == 'b' || c == 'o' || c == 'h' || c == 'D' ||
                    c == 'B' || c == 'O' || c == 'H')
          << "Expected [dbohDBOH], saw '" << c << "'";

      seen_separator = true;
    } else {
      break;
    }
  }

  return Token{TokenKind::kNumber, pos, chars};
}

absl::StatusOr<Token> Scanner::ScanName(char startc, Pos pos, bool is_escaped) {
  std::string chars;
  chars.push_back(startc);
  while (!AtEofInternal()) {
    char c = PeekCharOrDie();
    bool is_whitespace = c == ' ' || c == '\t' || c == '\n';
    if ((is_escaped && !is_whitespace) || isalpha(c) || isdigit(c) ||
        c == '_') {
      chars.push_back(PopCharOrDie());
    } else {
      break;
    }
  }
  return Token{TokenKind::kName, pos, chars};
}

absl::StatusOr<Token> Scanner::PeekInternal() {
  DropIgnoredChars();
  if (index_ >= text_.size()) {
    return absl::FailedPreconditionError("Scan has reached EOF.");
  }
  auto pos = GetPos();
  char c = PopCharOrDie();
  switch (c) {
    case '(':
      return Token{TokenKind::kOpenParen, pos};
    case ')':
      return Token{TokenKind::kCloseParen, pos};
    case '[':
      return Token{TokenKind::kOpenBracket, pos};
    case ']':
      return Token{TokenKind::kCloseBracket, pos};
    case '{':
      return Token{TokenKind::kOpenBrace, pos};
    case '}':
      return Token{TokenKind::kCloseBrace, pos};
    case '.':
      return Token{TokenKind::kDot, pos};
    case ',':
      return Token{TokenKind::kComma, pos};
    case ';':
      return Token{TokenKind::kSemicolon, pos};
    case ':':
      return Token{TokenKind::kColon, pos};
    case '=':
      return Token{TokenKind::kEquals, pos};
    case '"':
      return Token{TokenKind::kQuote, pos};
    case '#':
      if (index_ < text_.size() && text_[index_] == '(') {
        DropCharOrDie();
        return Token{TokenKind::kStartParams, GetPos()};
      }
      [[fallthrough]];
    default:
      if (isdigit(c)) {
        return ScanNumber(c, pos);
      }
      if (isalpha(c) || c == '\\' || c == '_') {
        return ScanName(c, pos, c == '\\');
      }
      return absl::UnimplementedError(absl::StrFormat(
          "Unsupported character: '%c' (%#x) @ %s", c, c, pos.ToHumanString()));
  }
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls
