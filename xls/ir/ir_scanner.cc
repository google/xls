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

#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/number_parser.h"

namespace xls {

std::string LexicalTokenTypeToString(LexicalTokenType token_type) {
  switch (token_type) {
    case LexicalTokenType::kIdent:
      return "ident";
    case LexicalTokenType::kKeyword:
      return "keyword";
    case LexicalTokenType::kLiteral:
      return "literal";
    case LexicalTokenType::kMinus:
      return "-";
    case LexicalTokenType::kAdd:
      return "+";
    case LexicalTokenType::kColon:
      return ":";
    case LexicalTokenType::kGt:
      return ">";
    case LexicalTokenType::kLt:
      return "<";
    case LexicalTokenType::kDot:
      return ".";
    case LexicalTokenType::kComma:
      return ",";
    case LexicalTokenType::kEquals:
      return "=";
    case LexicalTokenType::kCurlOpen:
      return "{";
    case LexicalTokenType::kCurlClose:
      return "}";
    case LexicalTokenType::kBracketOpen:
      return "[";
    case LexicalTokenType::kBracketClose:
      return "]";
    case LexicalTokenType::kParenOpen:
      return "(";
    case LexicalTokenType::kParenClose:
      return ")";
    case LexicalTokenType::kRightArrow:
      return "->";
  }
  return absl::StrCat("LexicalTokenType(", static_cast<int>(token_type), ")");
}

std::string TokenPos::ToHumanString() const {
  return absl::StrFormat("%d:%d", lineno + 1, colno + 1);
}

xabsl::StatusOr<bool> Token::IsNegative() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError("Can only get sign for literal tokens.");
  }
  std::pair<bool, Bits> pair;
  XLS_ASSIGN_OR_RETURN(pair, GetSignAndMagnitude(value()));
  return pair.first;
}

xabsl::StatusOr<Bits> Token::GetValueBits() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError(
        "Can only get value as integer for literal tokens.");
  }
  return ParseNumber(value());
}

xabsl::StatusOr<int64> Token::GetValueInt64() const {
  if (type() != LexicalTokenType::kLiteral) {
    return absl::InternalError(
        "Can only get value as integer for literal tokens.");
  }
  return ParseNumberAsInt64(value());
}

xabsl::StatusOr<bool> Token::GetValueBool() const {
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

xabsl::StatusOr<std::vector<Token>> TokenizeString(absl::string_view str) {
  int lineno = 0;
  int colno = 0;

  auto in_bounds = [&str](int64 index) -> bool { return index < str.size(); };

  // Returns the first index greater than 'index' in 'str' with a non-whitespace
  // character. Updates source location information.
  auto eat_white_space = [&](absl::string_view str, int64 index) -> int64 {
    while (in_bounds(index) && absl::ascii_isspace(str[index])) {
      char c = str[index];
      if (c == ' ') {
        ++colno;
      }
      if (c == '\t') {
        colno += 4;
      }
      if (c == '\n') {
        colno = 1;
        ++lineno;
      }
      index++;
    }
    return index;
  };

  // In case of end-of-line comments, return the first index after
  // the comment-terminating end-of-line character. Otherwise return
  // the originally-given index.
  auto eat_end_of_line_comment = [&](absl::string_view str,
                                     int64 index) -> int64 {
    if (in_bounds(index + 1) && str[index] == '/' && str[index + 1] == '/') {
      index += 2;
      while (in_bounds(index) && str[index] != '\n') {
        index++;
      }
    }
    return index;
  };

  auto drop_whitespace_and_comments = [&](absl::string_view str,
                                          int64 index) -> int64 {
    int64 old_index = -1;
    while (old_index != index) {
      old_index = index;
      index = eat_white_space(str, index);
      index = eat_end_of_line_comment(str, index);
    }
    return index;
  };

  std::vector<Token> tokens;
  int64 index = 0;  // Running index into the input string.
  while (in_bounds(index)) {
    index = drop_whitespace_and_comments(str, index);
    if (!in_bounds(index)) {
      break;
    }
    int64 token_start_index = index;

    // Literal numbers can decimal, binary (eg, 0b0101) or hexadecimal (eg,
    // 0xbeef) so capture all alphanumeric characters after the initial digit.
    // Literal numbers can also contain '_'s after the first character which are
    // used to improve readability (example: '0xabcd_ef00').
    if (isdigit(str[index]) || (str[index] == '-' && in_bounds(index + 1) &&
                                isdigit(str[index + 1]))) {
      while ((token_start_index == index && str[index] == '-') ||
             (in_bounds(index) &&
              (absl::ascii_isalnum(str[index]) || str[index] == '_'))) {
        index++;
      }
      const int64 token_len = index - token_start_index;
      absl::string_view value = str.substr(token_start_index, token_len);
      tokens.push_back(Token(LexicalTokenType::kLiteral, value, lineno, colno));
      colno += token_len;
      continue;
    }
    if (isalpha(str[index]) || str[index] == '_') {
      std::string res = "";
      while (isalpha(str[index]) || str[index] == '_' || str[index] == '.' ||
             isdigit(str[index])) {
        res.append(1, str[index++]);
      }
      tokens.push_back(Token::MakeIdentOrKeyword(res, lineno, colno));
      colno += (index - token_start_index);
      continue;
    }

    // Look for multi-character tokens.
    if (str[index] == '-' && in_bounds(index + 1) && str[index + 1] == '>') {
      tokens.push_back(
          Token(LexicalTokenType::kRightArrow, "->", lineno, colno));
      index += 2;
      colno += 2;
      continue;
    }

    // Handle single-character tokens.
    LexicalTokenType token_type;
    const char c = str[index];
    switch (c) {
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
      default:
        std::string char_str = absl::ascii_iscntrl(c)
                                   ? absl::StrFormat("\\x%02x", c)
                                   : std::string(1, c);
        XLS_LOG(ERROR) << "IR text with error: " << str;
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid character in IR text \"%s\" @ %s",
                            char_str, TokenPos{lineno, colno}.ToHumanString()));
    }
    tokens.push_back(Token(token_type, lineno, colno));
    index++;
    colno++;
  }
  return tokens;
}

xabsl::StatusOr<Scanner> Scanner::Create(absl::string_view text) {
  XLS_ASSIGN_OR_RETURN(auto tokens, TokenizeString(text));
  return Scanner(std::move(tokens));
}

xabsl::StatusOr<Token> Scanner::PeekToken() const {
  if (AtEof()) {
    return absl::InvalidArgumentError("Expected token, but found EOF.");
  }
  return tokens_[token_idx_];
}

xabsl::StatusOr<Token> Scanner::PopTokenOrError(absl::string_view context) {
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
                                       absl::string_view context) {
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

xabsl::StatusOr<Token> Scanner::PopTokenOrError(LexicalTokenType target,
                                                absl::string_view context) {
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

absl::Status Scanner::DropKeywordOrError(absl::string_view keyword) {
  xabsl::StatusOr<Token> popped_status = PopTokenOrError();
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
