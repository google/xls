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

#include <tuple>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/string_to_int.h"
#include "xls/ir/bits.h"
#include "xls/netlist/netlist.h"
#include "re2/re2.h"

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
  XLS_VLOG(3) << "Popping token: " << result.ToString();
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

absl::StatusOr<std::string> Parser::PopNameOrError() {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName) {
    return token.value;
  }
  return absl::InvalidArgumentError("Expected name token; got: " +
                                    token.ToString());
}

absl::StatusOr<int64_t> Parser::PopNumberOrError() {
  // We're assuming we won't see > 64b values. Fine for now, at least.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kNumber) {
    // Check for the big version first.
    std::string width_string, signed_string, base_string, value_string;
    if (RE2::FullMatch(
            token.value, R"(([0-9]+)'([Ss]?)([bodhBODH])([0-9a-f]+))",
            &width_string, &signed_string, &base_string, &value_string)) {
      int64_t width;
      XLS_RET_CHECK(
          absl::SimpleAtoi(width_string, reinterpret_cast<int64_t*>(&width)))
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

      XLS_ASSIGN_OR_RETURN(uint64_t temp, StrTo64Base(value_string, base));
      if (signed_string.empty()) {
        return static_cast<int64_t>(temp);
      }

      // If the number is actually signed, then throw it into a Bits for sign
      // conversion.
      XLS_ASSIGN_OR_RETURN(Bits bits, UBitsWithStatus(temp, width));
      return bits.ToInt64();
    }

    int64_t result;
    if (!absl::SimpleAtoi(token.value, &result)) {
      return absl::InternalError(
          "Number token's value cannot be parsed as an int64_t: " +
          token.value);
    }
    return result;
  }
  return absl::InvalidArgumentError("Expected number token; got: " +
                                    token.ToString());
}

absl::StatusOr<absl::variant<std::string, int64_t>>
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

absl::StatusOr<std::vector<std::string>> Parser::PopParenNameList() {
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

absl::StatusOr<const CellLibraryEntry*> Parser::ParseCellModule(
    Netlist& netlist) {
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  auto status_or_module = netlist.GetModule(name);
  if (status_or_module.ok()) {
    return status_or_module.value()->AsCellLibraryEntry();
  }
  if (name == "SB_LUT4") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kStartParams));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDot));
    XLS_ASSIGN_OR_RETURN(std::string param_name, PopNameOrError());
    if (param_name != "LUT_INIT") {
      return absl::InvalidArgumentError(
          "Expected a single .LUT_INIT named parameter, got: " + param_name);
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
    XLS_ASSIGN_OR_RETURN(int64_t lut_mask, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    return netlist.GetOrCreateLut4CellEntry(lut_mask);
  }
  return cell_library_->GetEntry(name);
}

absl::StatusOr<NetRef> Parser::ParseNetRef(Module* module) {
  using TokenT = absl::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (absl::holds_alternative<int64_t>(token)) {
    int64_t value = absl::get<int64_t>(token);
    return module->AddOrResolveNumber(value);
  }

  std::string name = absl::get<std::string>(token);
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64_t index, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    absl::StrAppend(&name, "[", index, "]");
  }
  return module->ResolveNet(name);
}

absl::Status Parser::ParseInstance(Module* module, Netlist& netlist) {
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
                       Cell::Create(cle, name, named_parameter_assignments,
                                    clock, module->GetDummyRef()),
                       _ << " @ " << pos.ToHumanString());
  XLS_ASSIGN_OR_RETURN(Cell * cell_ptr, module->AddCell(std::move(cell)));
  absl::flat_hash_set<NetRef> connected_wires;
  for (auto& item : named_parameter_assignments) {
    if (connected_wires.contains(item.second)) {
      continue;
    }
    item.second->NoteConnectedCell(cell_ptr);
    connected_wires.insert(item.second);
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

absl::StatusOr<absl::optional<Parser::Range>> Parser::ParseOptionalRange(
    bool strict) {
  absl::optional<Range> range;
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64_t high, PopNumberOrError());
    int64_t low = high;
    if (TryDropToken(TokenKind::kColon)) {
      XLS_ASSIGN_OR_RETURN(low, PopNumberOrError());
      if (high < low) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected net range to be [high:low] with low <= "
                            "high, got low: %d; high: %d",
                            low, high));
      }
    } else if (strict) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expecting net range, got a subscript instead"));
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    range = {high, low};
  }
  return range;
}

absl::Status Parser::ParseNetDecl(Module* module, NetDeclKind kind) {
  XLS_ASSIGN_OR_RETURN(auto range, ParseOptionalRange());
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
    if (kind == NetDeclKind::kInput || kind == NetDeclKind::kOutput) {
      int64_t width = 1;
      if (range.has_value()) {
        width = range->high - range->low + 1;
      }
      XLS_RETURN_IF_ERROR(
          module->DeclarePort(name, width, kind == NetDeclKind::kOutput));
    }
    if (range.has_value()) {
      for (int64_t i = range->low; i <= range->high; ++i) {
        XLS_RETURN_IF_ERROR(
            module->AddNetDecl(kind, absl::StrFormat("%s[%d]", name, i)));
      }
    } else {
      XLS_RETURN_IF_ERROR(module->AddNetDecl(kind, name));
    }
  }
  return absl::OkStatus();
}

absl::Status Parser::ParseOneAssignment(Module* module,
                                        absl::string_view lhs_name,
                                        absl::optional<Range> lhs_range) {
  // Extract the range from the lhs wire.  The high and low ends are identical
  // because the optional range might be an index dereference.
  int64_t lhs_high = 0, lhs_low = 0;
  if (lhs_range.has_value()) {
    lhs_high = lhs_range.value().high;
    lhs_low = lhs_range.value().low;
  }
  XLS_RET_CHECK(lhs_high >= lhs_low);

  using TokenT = absl::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (absl::holds_alternative<int64_t>(token)) {
    int64_t rhs_value = absl::get<int64_t>(token);
    // We'll be right-shifting below, make sure that sign extensions do not
    // trip us up.
    if (rhs_value < 0) {
      return absl::UnimplementedError(
          "Negative number literals are not supported in assign statements.");
    }

    // Start converting the value to the input wires zero_ or one_, and
    // assign each input bit to the corresponding NetDecl, starting with the
    // low end of the range.  If we run out of wires while converting the
    // number, error out.

    while (lhs_low <= lhs_high) {
      bool bit = rhs_value & 1;
      if (lhs_range.has_value()) {
        XLS_RETURN_IF_ERROR(module->AddAssignDecl(
            absl::StrFormat("%s[%d]", lhs_name, lhs_low), bit));
      } else {
        // The loop will execute only once.
        XLS_RETURN_IF_ERROR(module->AddAssignDecl(lhs_name, bit));
      }
      lhs_low++;
      rhs_value >>= 1;
    }
    if (rhs_value != 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number literal is too wide for %s.", lhs_name));
    }

  } else {
    std::string rhs_name = absl::get<std::string>(token);
    XLS_ASSIGN_OR_RETURN(auto rhs_range, ParseOptionalRange(false));

    // Extract the range from the rhs wire.
    int64_t rhs_high = 0, rhs_low = 0;
    if (rhs_range.has_value()) {
      rhs_high = rhs_range.value().high;
      rhs_low = rhs_range.value().low;
    }
    XLS_CHECK(rhs_high >= rhs_low);

    // The two ranges must be the same width.
    if (rhs_high - rhs_low != lhs_high - lhs_low) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Mismatched bit widths: left-hand side is %lld, "
                          "right-hand side is %lld.",
                          lhs_high - lhs_low + 1, rhs_high - rhs_low + 1));
    }

    // Start mapping the rhs wires to the lhs ones.
    while (lhs_low <= lhs_high) {
      std::string lhs_wire_name;
      if (lhs_range.has_value()) {
        lhs_wire_name = absl::StrFormat("%s[%d]", lhs_name, lhs_low);
      } else {
        lhs_wire_name = lhs_name;
      }
      std::string rhs_wire_name;
      if (rhs_range.has_value()) {
        rhs_wire_name = absl::StrFormat("%s[%d]", rhs_name, rhs_low);
      } else {
        rhs_wire_name = rhs_name;
      }
      XLS_RETURN_IF_ERROR(module->AddAssignDecl(lhs_wire_name, rhs_wire_name));
      lhs_low++;
      rhs_low++;
    }
    XLS_CHECK(rhs_low >= rhs_high);
  }

  return absl::OkStatus();
}

absl::Status Parser::ParseAssignDecl(Module* module) {
  // Parse assign statements of the following format:
  //
  // assign idA = idB;
  // assign { idA0, idA1, ... } = { idB0, idB1, ... }
  //
  // Each identifier can be a literal, a single wire, or a wire with a
  // subscript, or a wire with a subscript range, e.g. "8'h00", "a", or "a[0]",
  // or "a[7:0]".
  //
  // The identifiers on the LHS and the RHS must be the same width, e.g.
  //
  // assign a = 1'b0
  // assign a[7:0] = 8'hff
  // assign a = b
  // assign { a, b[1], c[7:0] }  = { d, e[5], f[15:8] }
  //
  // Note: we do not handle all possible kinds of assign syntax.  For example,
  // the line "assign {a,b} = 2'h0;" is legal.  We error out in this case rather
  // than doing the wrong thing.  Support can be added in the future, if needed.

  if (TryDropToken(TokenKind::kOpenBrace)) {
    std::vector<std::pair<std::string, absl::optional<Range>>> lhs;
    // Parse the left-hand side.
    do {
      XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
      XLS_ASSIGN_OR_RETURN(auto range, ParseOptionalRange(false));
      lhs.push_back({name, range});
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));

    // Parse the right-hand side.  While parsing, iterate over the lhs elements
    // we collected, verify that widths match, then break them up into
    // individual NetDecl instances, and save the associationss.  The right-hand
    // side could map an integer to a wire range, in which case we break up the
    // integer bitwise and assign the values to the lhs wires.

    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenBrace));
    auto left = lhs.begin();
    do {
      absl::string_view lhs_name = left->first;
      absl::optional<Range> lhs_range = left->second;
      XLS_RETURN_IF_ERROR(ParseOneAssignment(module, lhs_name, lhs_range));
      left++;
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
  } else {
    // Parse the left-hand side.
    XLS_ASSIGN_OR_RETURN(std::string lhs_name, PopNameOrError());
    XLS_ASSIGN_OR_RETURN(auto lhs_range, ParseOptionalRange(false));
    // Parse the right-hand side.
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
    XLS_RETURN_IF_ERROR(ParseOneAssignment(module, lhs_name, lhs_range));
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  return absl::OkStatus();
}

absl::Status Parser::ParseModuleStatement(Module* module, Netlist& netlist) {
  if (TryDropKeyword("input")) {
    return ParseNetDecl(module, NetDeclKind::kInput);
  }
  if (TryDropKeyword("output")) {
    return ParseNetDecl(module, NetDeclKind::kOutput);
  }
  if (TryDropKeyword("wire")) {
    return ParseNetDecl(module, NetDeclKind::kWire);
  }
  if (TryDropKeyword("assign")) {
    return ParseAssignDecl(module);
  }
  return ParseInstance(module, netlist);
}

absl::StatusOr<std::unique_ptr<Module>> Parser::ParseModule(Netlist& netlist) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError("module"));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PopNameOrError());
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> module_ports,
                       PopParenNameList());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  auto module = std::make_unique<Module>(module_name);
  module->DeclarePortsOrder(module_ports);

  while (true) {
    if (TryDropKeyword("endmodule")) {
      break;
    }
    XLS_RETURN_IF_ERROR(ParseModuleStatement(module.get(), netlist));
  }
  return module;
}

absl::StatusOr<std::unique_ptr<Netlist>> Parser::ParseNetlist(
    CellLibrary* cell_library, Scanner* scanner) {
  auto netlist = std::make_unique<Netlist>();
  Parser p(cell_library, scanner);
  while (!scanner->AtEof()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                         p.ParseModule(*netlist));
    netlist->AddModule(std::move(module));
  }
  return std::move(netlist);
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls
