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

#include "xls/netlist/netlist_parser.h"

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "re2/re2.h"

namespace xls {
namespace netlist {
namespace rtl {

std::string Pos::ToHumanString() const {
  return absl::StrFormat("%d:%d", lineno + 1, colno + 1);
}

std::string TokenKindToString(TokenKind kind) {
  switch (kind) {
    case TokenKind::kOpenParen:
      return "open-paren";
    case TokenKind::kCloseParen:
      return "close-paren";
    case TokenKind::kOpenBracket:
      return "open-bracket";
    case TokenKind::kCloseBracket:
      return "close-bracket";
    case TokenKind::kDot:
      return "dot";
    case TokenKind::kComma:
      return "comma";
    case TokenKind::kSemicolon:
      return "semicolon";
    case TokenKind::kColon:
      return "colon";
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
  XLS_CHECK(!AtEofInternal());
  return text_[index_];
}

char Scanner::PeekChar2OrDie() const {
  XLS_CHECK_GT(text_.size(), index_ + 1);
  return text_[index_ + 1];
}

char Scanner::PopCharOrDie() {
  XLS_CHECK(!AtEofInternal());
  char c = text_[index_++];
  if (c == '\n') {
    lineno_++;
    colno_ = 0;
  } else {
    colno_++;
  }
  return c;
}

void Scanner::DropCommentsAndWhitespace() {
  auto drop_to_eol_or_eof = [this] {
    while (!AtEofInternal() && PeekCharOrDie() != '\n') {
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

xabsl::StatusOr<Token> Scanner::Peek() {
  if (lookahead_.has_value()) {
    return lookahead_.value();
  }
  XLS_ASSIGN_OR_RETURN(Token token, PeekInternal());
  lookahead_.emplace(token);
  return lookahead_.value();
}

xabsl::StatusOr<Token> Scanner::Pop() {
  XLS_ASSIGN_OR_RETURN(Token result, Peek());
  lookahead_.reset();
  XLS_VLOG(3) << "Popping token: " << result.ToString();
  return result;
}

xabsl::StatusOr<Token> Scanner::ScanNumber(char startc, Pos pos) {
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

xabsl::StatusOr<Token> Scanner::ScanName(char startc, Pos pos,
                                         bool is_escaped) {
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

xabsl::StatusOr<Token> Scanner::PeekInternal() {
  DropCommentsAndWhitespace();
  if (index_ >= text_.size()) {
    return absl::FailedPreconditionError("Scan has reached EOF.");
  }
  auto pos = GetPos();
  char c = text_[index_++];
  switch (c) {
    case '(':
      return Token{TokenKind::kOpenParen, pos};
    case ')':
      return Token{TokenKind::kCloseParen, pos};
    case '[':
      return Token{TokenKind::kOpenBracket, pos};
    case ']':
      return Token{TokenKind::kCloseBracket, pos};
    case '.':
      return Token{TokenKind::kDot, pos};
    case ',':
      return Token{TokenKind::kComma, pos};
    case ';':
      return Token{TokenKind::kSemicolon, pos};
    case ':':
      return Token{TokenKind::kColon, pos};
    default:
      if (isdigit(c)) {
        return ScanNumber(c, pos);
      }
      if (isalpha(c) || c == '\\') {
        return ScanName(c, pos, c == '\\');
      }
      return absl::UnimplementedError(absl::StrFormat(
          "Unsupported character: '%c' (%#x) @ %s", c, c, pos.ToHumanString()));
  }
}

xabsl::StatusOr<std::string> Parser::PopNameOrError() {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName) {
    return token.value;
  }
  return absl::InvalidArgumentError("Expected name token; got: " +
                                    token.ToString());
}

xabsl::StatusOr<int64> Parser::PopNumberOrError() {
  // We're assuming we won't see > 64b values. Fine for now, at least.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kNumber) {
    // Check for the big version first.
    std::string width_string, signed_string, base_string, value_string;
    if (RE2::FullMatch(
            token.value, R"(([0-9]+)'([Ss]?)([bodhBODH])([0-9a-f]+))",
            &width_string, &signed_string, &base_string, &value_string)) {
      int64 width;
      XLS_RET_CHECK(absl::numbers_internal::safe_strto64_base(
          width_string, reinterpret_cast<int64_t*>(&width), 10))
          << "Unable to parse number width: " << width_string;
      int base;
      if (base_string == "b" || base_string == "B") {
        base = 2;
      } else if (base_string == "o" || base_string == "O") {
        base = 8;
      } else if (base_string == "d" || base_string == "D") {
        base = 10;
      } else if (base_string == "h" || base_string == "H") {
        base = 16;
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Invalid numeric base: ", base_string));
      }

      uint64_t temp;
      XLS_RET_CHECK(
          absl::numbers_internal::safe_strtou64_base(value_string, &temp, base))
          << "Unable to parse number value: " << value_string;
      if (signed_string.empty()) {
        return static_cast<int64>(temp);
      }

      // If the number is actually signed, then throw it into a Bits for sign
      // conversion.
      XLS_ASSIGN_OR_RETURN(Bits bits, UBitsWithStatus(temp, width));
      return bits.ToInt64();
    }

    int64 result;
    if (!absl::SimpleAtoi(token.value, &result)) {
      return absl::InternalError(
          "Number token's value cannot be parsed as an int64: " + token.value);
    }
    return result;
  }
  return absl::InvalidArgumentError("Expected number token; got: " +
                                    token.ToString());
}

xabsl::StatusOr<absl::variant<std::string, int64>>
Parser::PopNameOrNumberOrError() {
  TokenKind kind = scanner_->Peek()->kind;
  if (kind == TokenKind::kName) {
    XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
    return token.value;
  } else if (kind == TokenKind::kNumber) {
    return PopNumberOrError();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Expected name or number token; got: ", static_cast<int>(kind)));
}

absl::Status Parser::DropTokenOrError(TokenKind target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == target) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Want token %s; got %s.", TokenKindToString(target), token.ToString()));
}

xabsl::StatusOr<std::vector<std::string>> Parser::PopParenNameList() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  std::vector<std::string> results;
  bool must_end = false;
  while (true) {
    if (TryDropToken(TokenKind::kCloseParen)) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
      break;
    }
    XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
    results.push_back(name);
    must_end = !TryDropToken(TokenKind::kComma);
  }
  return results;
}

absl::Status Parser::DropKeywordOrError(absl::string_view target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName && token.value == target) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Want keyword '%s', got: %s", target, token.ToString()));
}

xabsl::StatusOr<const CellLibraryEntry*> Parser::ParseCellModule(
    const Netlist& netlist) {
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  const Module* module = netlist.GetModule(name);
  if (module != nullptr) {
    return module->AsCellLibraryEntry();
  }
  return cell_library_->GetEntry(name);
}

xabsl::StatusOr<NetRef> Parser::ParseNetRef(Module* module) {
  using TokenT = absl::variant<std::string, int64>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (absl::holds_alternative<int64>(token)) {
    int64 value = absl::get<int64>(token);
    return module->AddOrResolveNumber(value);
  }

  std::string name = absl::get<std::string>(token);
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64 index, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    absl::StrAppend(&name, "[", index, "]");
  }
  return module->ResolveNet(name);
}

absl::Status Parser::ParseInstance(Module* module, const Netlist& netlist) {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_->Peek());
  const Pos pos = peek.pos;

  XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* cle, ParseCellModule(netlist));
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  // LRM 23.3.2 Calls these "named parameter assignments".
  absl::flat_hash_map<std::string, NetRef> named_parameter_assignments;
  while (true) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDot));
    XLS_ASSIGN_OR_RETURN(std::string pin_name, PopNameOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
    XLS_ASSIGN_OR_RETURN(NetRef net, ParseNetRef(module));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    XLS_VLOG(3) << "Adding named parameter assignment: " << pin_name;
    bool is_new = named_parameter_assignments.insert({pin_name, net}).second;
    if (!is_new) {
      return absl::InvalidArgumentError("Duplicate port seen: " + pin_name);
    }
    if (!TryDropToken(TokenKind::kComma)) {
      break;
    }
  }
  absl::optional<NetRef> clock;
  if (cle->clock_name().has_value()) {
    auto it = named_parameter_assignments.find(cle->clock_name().value());
    if (it == named_parameter_assignments.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cell %s named %s requires a clock connection %s but none was found.",
          cle->name(), name, cle->clock_name().value()));
    }
    clock = it->second;
    named_parameter_assignments.erase(it);
  }
  XLS_ASSIGN_OR_RETURN(Cell cell,
                   Cell::Create(cle, name, named_parameter_assignments, clock),
                   _ << " @ " << pos.ToHumanString());
  XLS_ASSIGN_OR_RETURN(Cell * cell_ptr, module->AddCell(std::move(cell)));
  for (auto& item : named_parameter_assignments) {
    item.second->NoteConnectedCell(cell_ptr);
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));
  return absl::OkStatus();
}

bool Parser::TryDropToken(TokenKind target) {
  if (scanner_->AtEof()) {
    return false;
  }
  if (scanner_->Peek().value().kind == target) {
    XLS_CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

bool Parser::TryDropKeyword(absl::string_view target) {
  if (scanner_->AtEof()) {
    return false;
  }
  Token peek = scanner_->Peek().value();
  if (peek.kind == TokenKind::kName && peek.value == target) {
    XLS_CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

absl::Status Parser::ParseNetDecl(Module* module, NetDeclKind kind) {
  absl::optional<std::pair<int64, int64>> range;
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64 high, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kColon));
    XLS_ASSIGN_OR_RETURN(int64 low, PopNumberOrError());
    if (high < low) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected net range to be [high:low] with low <= "
                          "high, got low: %d; high: %d",
                          low, high));
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    range = {high, low};
  }

  std::vector<std::string> names;
  do {
    XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
    names.push_back(name);
  } while (TryDropToken(TokenKind::kComma));

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  if (names.size() > 1 && range.has_value()) {
    // Note: we could support this but not sure if the netlist ever contains
    // such a construct.
    return absl::UnimplementedError(
        "Multiple declarations for a ranged net is not yet supported.");
  }

  for (const std::string& name : names) {
    if (range.has_value()) {
      for (int64 i = range->second; i <= range->first; ++i) {
        XLS_RETURN_IF_ERROR(
            module->AddNetDecl(kind, absl::StrFormat("%s[%d]", name, i)));
      }
    } else {
      XLS_RETURN_IF_ERROR(module->AddNetDecl(kind, name));
    }
  }
  return absl::OkStatus();
}

absl::Status Parser::ParseModuleStatement(Module* module,
                                          const Netlist& netlist) {
  if (TryDropKeyword("input")) {
    return ParseNetDecl(module, NetDeclKind::kInput);
  }
  if (TryDropKeyword("output")) {
    return ParseNetDecl(module, NetDeclKind::kOutput);
  }
  if (TryDropKeyword("wire")) {
    return ParseNetDecl(module, NetDeclKind::kWire);
  }
  return ParseInstance(module, netlist);
}

xabsl::StatusOr<std::unique_ptr<Module>> Parser::ParseModule(
    const Netlist& netlist) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError("module"));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PopNameOrError());
  auto module = std::make_unique<Module>(module_name);
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> ports, PopParenNameList());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  while (true) {
    if (TryDropKeyword("endmodule")) {
      break;
    }
    XLS_RETURN_IF_ERROR(ParseModuleStatement(module.get(), netlist));
  }
  return module;
}

/* static */ xabsl::StatusOr<std::unique_ptr<Module>> Parser::ParseModule(
    CellLibrary* cell_library, Scanner* scanner) {
  Parser p(cell_library, scanner);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module, p.ParseModule());
  if (!scanner->AtEof()) {
    return absl::InvalidArgumentError("Unexpected characters at end of file.");
  }
  return module;
}

xabsl::StatusOr<Netlist> Parser::ParseNetlist(CellLibrary* cell_library,
                                              Scanner* scanner) {
  Netlist netlist;
  Parser p(cell_library, scanner);
  while (!scanner->AtEof()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                         p.ParseModule(netlist));
    netlist.AddModule(std::move(module));
  }
  return std::move(netlist);
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls
