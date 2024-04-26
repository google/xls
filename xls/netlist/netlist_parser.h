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

#ifndef XLS_NETLIST_NETLIST_PARSER_H_
#define XLS_NETLIST_NETLIST_PARSER_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/string_to_int.h"
#include "xls/ir/bits.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/netlist.h"
#include "re2/re2.h"

namespace xls {
namespace netlist {
namespace rtl {

// Kinds of tokens the scanner emits.
enum class TokenKind {
  kStartParams,   // #(
  kOpenParen,     // (
  kCloseParen,    // )
  kOpenBracket,   // [
  kCloseBracket,  // ]
  kOpenBrace,     // {
  kCloseBrace,    // }
  kDot,
  kComma,
  kColon,
  kSemicolon,
  kEquals,
  kQuote,
  kName,
  kNumber,
};

// Returns a string representation of "kind" suitable for debugging.
std::string TokenKindToString(TokenKind kind);

// Represents a position in input text.
struct Pos {
  int64_t lineno;
  int64_t colno;

  std::string ToHumanString() const;
};

// Represents a scanned token (that comes from scanning a character stream).
struct Token {
  TokenKind kind;
  Pos pos;
  std::string value;

  std::string ToString() const;
};

// Token scanner for netlist files.
class Scanner {
 public:
  explicit Scanner(std::string_view text) : text_(text) {}

  absl::StatusOr<Token> Peek();

  absl::StatusOr<Token> Pop();

  bool AtEof() {
    DropIgnoredChars();
    return index_ >= text_.size();
  }

 private:
  absl::StatusOr<Token> ScanName(char startc, Pos pos, bool is_escaped);
  absl::StatusOr<Token> ScanNumber(char startc, Pos pos);
  absl::StatusOr<Token> PeekInternal();

  // Drops any characters that should not be converted to Tokens, including
  // whitespace, comments, and attributes.
  // Note that we may eventually want to expose attributes to the
  // AbstractParser, but until then it's much simpler to treat attributes like
  // block comments and ignore everything inside of them. This also means that
  // the Scanner will accept attributes that are in invalid positions.
  void DropIgnoredChars();

  char PeekCharOrDie() const;
  char PeekChar2OrDie() const;
  char PopCharOrDie();
  void DropCharOrDie() { (void)PopCharOrDie(); }
  Pos GetPos() const { return Pos{lineno_, colno_}; }

  // Internal version of EOF checking that doesn't attempt to discard the
  // comments/whitespace as the public AtEof() does above -- this simply checks
  // whether the character stream index has reached the end of the text.
  bool AtEofInternal() const { return index_ >= text_.size(); }

  std::string_view text_;
  int64_t index_ = 0;
  int64_t lineno_ = 0;
  int64_t colno_ = 0;
  std::optional<Token> lookahead_;
};

template <typename EvalT = bool>
class AbstractParser {
 public:
  // Parses a netlist with the given cell library and token scanner.
  // Returns a status on parse error.
  static absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>> ParseNetlist(
      AbstractCellLibrary<EvalT>* cell_library, Scanner* scanner, EvalT zero,
      EvalT one);
  template <typename = std::is_constructible<EvalT, bool>>
  static absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>> ParseNetlist(
      AbstractCellLibrary<EvalT>* cell_library, Scanner* scanner) {
    return ParseNetlist(cell_library, scanner, EvalT{false}, EvalT{true});
  }

 private:
  explicit AbstractParser(AbstractCellLibrary<EvalT>* cell_library,
                          Scanner* scanner, EvalT zero, EvalT one)
      : cell_library_(cell_library),
        scanner_(scanner),
        zero_(zero),
        one_(one) {}

  // Parses a cell instantiation (e.g. in module scope).
  absl::Status ParseInstance(AbstractModule<EvalT>* module,
                             AbstractNetlist<EvalT>& netlist);

  // Parses a cell module name out of the token stream and returns the
  // corresponding AbstractCellLibraryEntry for that module name.
  absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*> ParseCellModule(
      AbstractNetlist<EvalT>& netlist);

  // Parses a wire declaration at the module scope.
  absl::Status ParseNetDecl(AbstractModule<EvalT>* module, NetDeclKind kind);

  // Parses an assign declaration at the module scope.  When encountering an
  // assign declaration, we break the LHS and RHS into individual wires, from
  // MSB to LSB.  Then, starting with the LSB, we start assigning the RHS to the
  // LHS.  This permits the loose mapping of any combinations of wires and
  // number-literal values.
  //
  // Given a list of wires representing the LHS and RHS of the assignment, each
  // complex wire is broken into its individual bits; single wires are
  // represented by their name only. (Number literals and input ports are not
  // allowed to be present on the LHS.)
  //
  // For example, consider in the following port (equivalently, wire)
  // declarations:
  //
  //   input a;
  //   input [2:0] b;
  //   output [3:0] c;
  //   output d;
  //
  // And the following assignments:
  //
  //   assign c = { a, b }
  //   assign d = a;
  //   assign { c, d } = { a, b[2:1], 1'b10 };
  //
  // The LHS for the assigns would each become a vector of strings with values
  // as follows:
  //
  //   "c[3]", "c[2]", "c[1]", and "c[0]".
  //   "d"
  //   "c[3]", "c[2]", "c[1]", "c[0]", and "d".
  //
  // For the RHS:
  //   "a", "b[2]", "b[1]", "b[0]"
  //   "a"
  //   "a", "b[2]", "b[1]", "1", "0"
  //
  // These would be matched up wire by wire from left to right.  Note that the
  // LHS and RHS need not be of the same size.  If there are more terms in the
  // LHS, then the extraneous most-significant ones will be unassigned.  If
  // there are more terms on the RHS, then the most-significant values would not
  // be assigned to anything.
  absl::Status ParseAssignDecl(AbstractModule<EvalT>* module);

  // Given a wire or port identifier or a number literal next in the token
  // stream, and also given whether we expect that token to be on the LHS or RHS
  // of an assign statement, this method will convert the input as the following
  // examples illustrate:
  //
  // Given port or wire "x" that was declared without width, it will emit "x"
  // Given port or wire "x" that was declared with width [7:2], it will emit
  // "x[7]", "x[6]", "x[5]", .., "x[2]".
  // Given a number "4'b1010" and is_lhs == false, it will emit "1", "0", "1",
  // "0".
  absl::Status ParseOneEntryOfAssignDecl(AbstractModule<EvalT>* module,
                                         std::vector<std::string>& side,
                                         bool is_lhs);

  // Attempts to parse a range of the kind [high:low].  It also handles
  // indexing by setting parameter strict to false, by representing the range as
  // [high:high].  For example:
  //   "a" --> no range
  //   "a[1] --> [1:1] (strict == false)
  //   "a[1:0] --> [1:0]
  absl::StatusOr<std::optional<Range>> ParseOptionalRange(bool strict = true);

  // Parses a module-level statement (e.g. wire decl or cell instantiation).
  absl::Status ParseModuleStatement(AbstractModule<EvalT>* module,
                                    AbstractNetlist<EvalT>& netlist);

  // Parses a module definition (e.g. at the top of the file).
  absl::StatusOr<std::unique_ptr<AbstractModule<EvalT>>> ParseModule(
      AbstractNetlist<EvalT>& netlist);

  // Parses a reference to an already- declared net.
  absl::StatusOr<AbstractNetRef<EvalT>> ParseNetRef(
      AbstractModule<EvalT>* module);

  // Pops a name token and returns its contents or gives an error status if a
  // name token is not immediately present in the stream.
  absl::StatusOr<std::string> PopNameOrError();

  // Pops a name token and returns its value or gives an error status if a
  // number token is not immediately present in the stream.  The overload
  // accepting the width parameters sets it to the bit width of the parsed
  // number.
  absl::StatusOr<int64_t> PopNumberOrError();
  absl::StatusOr<int64_t> PopNumberOrError(size_t& width);

  // Pops either a name or number token or returns an error.  The overload
  // accepting a width parameter sets that parameter to the bit width of the
  // parsed number, if a number was parsed; otherwise, width is not modified.
  absl::StatusOr<std::variant<std::string, int64_t>> PopNameOrNumberOrError();
  absl::StatusOr<std::variant<std::string, int64_t>> PopNameOrNumberOrError(
      size_t& width);

  // Drops a token of kind target from the head of the stream or gives an error
  // status.
  absl::Status DropTokenOrError(TokenKind target);

  // Drops a keyword token from the head of the stream or gives an error status.
  absl::Status DropKeywordOrError(std::string_view target);

  // Attempts to drop a token of the target kind, or returns false if that
  // target token kind is not at the head of the token stream.
  bool TryDropToken(TokenKind target);

  // Attempts to drop a keyword token with the value "target" from the head of
  // the token stream, or returns false if it cannot.
  bool TryDropKeyword(std::string_view target);

  // Pops a parenthesized name list from the token stream and returns it as a
  // vector of those names.
  absl::StatusOr<std::vector<std::string>> PopParenNameList();

  // Cell library definitions are resolved against.
  AbstractCellLibrary<EvalT>* cell_library_;

  // Set of (already-parsed) Modules that may be present in the AbstractModule
  // currently being processed as AbstractCell-type references.
  absl::flat_hash_map<std::string, AbstractModule<EvalT>> modules_;

  // Scanner used for scanning out tokens (in a stream sequence).
  Scanner* scanner_;

  // Values representing zero/false and one/true in the EvalT type.
  EvalT zero_;
  EvalT one_;
};

using Parser = AbstractParser<>;

template <typename EvalT>
absl::StatusOr<std::string> AbstractParser<EvalT>::PopNameOrError() {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName) {
    return token.value;
  }
  return absl::InvalidArgumentError("Expected name token; got: " +
                                    token.ToString());
}

template <typename EvalT>
absl::StatusOr<int64_t> AbstractParser<EvalT>::PopNumberOrError(size_t& width) {
  // We're assuming we won't see > 64b values. Fine for now, at least.
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kNumber) {
    // Check for the big version first.
    std::string width_string, signed_string, base_string, value_string;
    // Precompute the regex matcher for Verilog number literals.
    static LazyRE2 number_re_ = {R"(([0-9]+)'([Ss]?)([bodhBODH])([0-9a-f]+))"};
    if (RE2::FullMatch(token.value, *number_re_, &width_string, &signed_string,
                       &base_string, &value_string)) {
      XLS_RET_CHECK(
          absl::SimpleAtoi(width_string, reinterpret_cast<size_t*>(&width)))
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
    // Size field defaults to 32 when not explicitly specified.
    width = 32;
    return result;
  }
  return absl::InvalidArgumentError("Expected number token; got: " +
                                    token.ToString());
}

template <typename EvalT>
absl::StatusOr<int64_t> AbstractParser<EvalT>::PopNumberOrError() {
  size_t width;
  return PopNumberOrError(width);
}

template <typename EvalT>
absl::StatusOr<std::variant<std::string, int64_t>>
AbstractParser<EvalT>::PopNameOrNumberOrError(size_t& width) {
  const TokenKind kind = scanner_->Peek()->kind;
  switch (kind) {
    case TokenKind::kName: {
      XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
      return token.value;
    }
    case TokenKind::kNumber:
      return PopNumberOrError(width);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Expected name or number token; got: ", static_cast<int>(kind)));
  }
}

template <typename EvalT>
absl::StatusOr<std::variant<std::string, int64_t>>
AbstractParser<EvalT>::PopNameOrNumberOrError() {
  size_t width;
  return PopNameOrNumberOrError(width);
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::DropTokenOrError(TokenKind target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == target) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(absl::StrFormat(
      "Want token %s; got %s.", TokenKindToString(target), token.ToString()));
}

template <typename EvalT>
absl::StatusOr<std::vector<std::string>>
AbstractParser<EvalT>::PopParenNameList() {
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

template <typename EvalT>
absl::Status AbstractParser<EvalT>::DropKeywordOrError(
    std::string_view target) {
  XLS_ASSIGN_OR_RETURN(Token token, scanner_->Pop());
  if (token.kind == TokenKind::kName && token.value == target) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Want keyword '%s', got: %s", target, token.ToString()));
}

template <typename EvalT>
absl::StatusOr<const AbstractCellLibraryEntry<EvalT>*>
AbstractParser<EvalT>::ParseCellModule(AbstractNetlist<EvalT>& netlist) {
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  auto maybe_module = netlist.MaybeGetModule(name);
  if (maybe_module.has_value()) {
    return maybe_module.value()->AsCellLibraryEntry();
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
    return netlist.GetOrCreateLut4CellEntry(lut_mask, zero_, one_);
  }
  return cell_library_->GetEntry(name);
}

template <typename EvalT>
absl::StatusOr<AbstractNetRef<EvalT>> AbstractParser<EvalT>::ParseNetRef(
    AbstractModule<EvalT>* module) {
  using TokenT = std::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError());
  if (std::holds_alternative<int64_t>(token)) {
    int64_t value = std::get<int64_t>(token);
    return module->AddOrResolveNumber(value);
  }

  std::string name = std::get<std::string>(token);
  if (TryDropToken(TokenKind::kOpenBracket)) {
    XLS_ASSIGN_OR_RETURN(int64_t index, PopNumberOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBracket));
    absl::StrAppend(&name, "[", index, "]");
  }
  return module->ResolveNet(name);
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseInstance(
    AbstractModule<EvalT>* module, AbstractNetlist<EvalT>& netlist) {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_->Peek());
  const Pos pos = peek.pos;

  XLS_ASSIGN_OR_RETURN(const AbstractCellLibraryEntry<EvalT>* cle,
                       ParseCellModule(netlist));
  XLS_ASSIGN_OR_RETURN(std::string name, PopNameOrError());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  // LRM 23.3.2 Calls these "named parameter assignments".
  absl::flat_hash_map<std::string, AbstractNetRef<EvalT>>
      named_parameter_assignments;
  while (true) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDot));
    XLS_ASSIGN_OR_RETURN(std::string pin_name, PopNameOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
    XLS_ASSIGN_OR_RETURN(AbstractNetRef<EvalT> net, ParseNetRef(module));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseParen));
    VLOG(3) << "Adding named parameter assignment: " << pin_name;
    bool is_new = named_parameter_assignments.insert({pin_name, net}).second;
    if (!is_new) {
      return absl::InvalidArgumentError("Duplicate port seen: " + pin_name);
    }
    if (!TryDropToken(TokenKind::kComma)) {
      break;
    }
  }
  std::optional<AbstractNetRef<EvalT>> clock;
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
  XLS_ASSIGN_OR_RETURN(
      AbstractCell<EvalT> cell,
      AbstractCell<EvalT>::Create(cle, name, named_parameter_assignments, clock,
                                  module->GetDummyRef()),
      _ << " @ " << pos.ToHumanString());
  XLS_ASSIGN_OR_RETURN(AbstractCell<EvalT> * cell_ptr,
                       module->AddCell(std::move(cell)));
  absl::flat_hash_set<AbstractNetRef<EvalT>> connected_wires;
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

template <typename EvalT>
bool AbstractParser<EvalT>::TryDropToken(TokenKind target) {
  if (scanner_->AtEof()) {
    return false;
  }
  if (scanner_->Peek().value().kind == target) {
    CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

template <typename EvalT>
bool AbstractParser<EvalT>::TryDropKeyword(std::string_view target) {
  if (scanner_->AtEof()) {
    return false;
  }
  Token peek = scanner_->Peek().value();
  if (peek.kind == TokenKind::kName && peek.value == target) {
    CHECK_OK(scanner_->Pop().status());
    return true;
  }
  return false;
}

template <typename EvalT>
absl::StatusOr<std::optional<Range>> AbstractParser<EvalT>::ParseOptionalRange(
    bool strict) {
  std::optional<Range> range;
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

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseNetDecl(AbstractModule<EvalT>* module,
                                                 NetDeclKind kind) {
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
    switch (kind) {
      case NetDeclKind::kInput:
      case NetDeclKind::kOutput:
        XLS_RETURN_IF_ERROR(
            module->DeclarePort(name, range, kind == NetDeclKind::kOutput));
        break;
      case NetDeclKind::kWire:
        XLS_RETURN_IF_ERROR(module->DeclareWire(name, range));
        break;
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

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseOneEntryOfAssignDecl(
    AbstractModule<EvalT>* module, std::vector<std::string>& side,
    bool is_lhs) {
  size_t number_bit_width;
  using TokenT = std::variant<std::string, int64_t>;
  XLS_ASSIGN_OR_RETURN(TokenT token, PopNameOrNumberOrError(number_bit_width));
  std::string name;
  std::optional<Range> range = std::nullopt;
  if (std::holds_alternative<std::string>(token)) {
    name = std::get<std::string>(token);
    XLS_ASSIGN_OR_RETURN(range, ParseOptionalRange(false));
  } else {
    // If we parsed a number, but we're expecting an lvalue, throw an error.
    if (is_lhs == true) {
      return absl::InvalidArgumentError(
          "Parsed a number when expecting lvalue in assign statement.");
    }
    // We got a number, and we're parsing the RHS. Break up the number into
    // bits, and splay its values as "0"s and "1"s in the string, starting with
    // the MSB.
    auto bitmap = InlineBitmap::FromWord(std::get<int64_t>(token),
                                         number_bit_width, /*fill=*/false);
    CHECK_GT(number_bit_width, 0);
    for (; number_bit_width; number_bit_width--) {
      side.push_back(bitmap.Get(number_bit_width - 1) ? "1" : "0");
    }
    return absl::OkStatus();
  }
  if (range.has_value()) {
    auto high = range->high;
    auto low = range->low;
    while (high >= low) {
      side.push_back(absl::Substitute("$0[$1]", name, high));
      high--;
    }
  } else {
    // No subscript was used; check to see the width of the wire.  Note that
    // because wires and output ports are tracked separately, if the port
    // lookup fails, we'll have to check the wires.
    auto has_opt_range = module->GetPortRange(name, /*is_assignable=*/is_lhs);
    if (!has_opt_range.has_value()) {
      has_opt_range = module->GetWireRange(name);
      CHECK(has_opt_range.has_value());
    }
    // Get the optional range from the result. If the optional range exists,
    // then the wire is ranged, e.g. it was declared as "output [7:0] out".
    // In this case, we insert "out[7]", "out[6]", ..., "out[0]" to the LHS
    // array.  If there is no range, e.g. if it was declared as "output
    // out", then we only insert "out".
    auto opt_range = has_opt_range.value();
    if (opt_range.has_value()) {
      int64_t high = opt_range->high;
      int64_t low = opt_range->low;
      while (high >= low) {
        side.push_back(absl::Substitute("$0[$1]", name, high));
        high--;
      }
    } else {
      side.push_back(name);
    }
  }
  return absl::OkStatus();
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseAssignDecl(
    AbstractModule<EvalT>* module) {
  // Parse the left-hand side.
  std::vector<std::string> lhs;
  if (TryDropToken(TokenKind::kOpenBrace)) {
    do {
      XLS_RETURN_IF_ERROR(
          ParseOneEntryOfAssignDecl(module, lhs, /*is_lhs=*/true));
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
  } else {
    XLS_RETURN_IF_ERROR(
        ParseOneEntryOfAssignDecl(module, lhs, /*is_lhs=*/true));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));

  // Parse the right-hand side.
  std::vector<std::string> rhs;
  if (TryDropToken(TokenKind::kOpenBrace)) {
    do {
      XLS_RETURN_IF_ERROR(
          ParseOneEntryOfAssignDecl(module, rhs, /*is_lhs=*/false));
    } while (TryDropToken(TokenKind::kComma));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCloseBrace));
  } else {
    XLS_RETURN_IF_ERROR(
        ParseOneEntryOfAssignDecl(module, rhs, /*is_lhs=*/false));
  }

  // Starting with the least-significant bits (from the right), match the LHS
  // and the RHS.  Note that the two sides need not be of equal length.  If the
  // LHS is longer than the RHS, then the remaning wires will have undefined
  // values.
  auto left = lhs.end();
  auto right = rhs.cend();
  while (left-- != lhs.begin() && right-- != rhs.cbegin()) {
    if (*right == "0" || *right == "1") {
      bool bit = *right == "1";
      XLS_RETURN_IF_ERROR(module->AddAssignDecl(*left, bit));
    } else {
      XLS_RETURN_IF_ERROR(module->AddAssignDecl(*left, *right));
    }
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));
  return absl::OkStatus();
}

template <typename EvalT>
absl::Status AbstractParser<EvalT>::ParseModuleStatement(
    AbstractModule<EvalT>* module, AbstractNetlist<EvalT>& netlist) {
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

template <typename EvalT>
absl::StatusOr<std::unique_ptr<AbstractModule<EvalT>>>
AbstractParser<EvalT>::ParseModule(AbstractNetlist<EvalT>& netlist) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError("module"));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PopNameOrError());
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> module_ports,
                       PopParenNameList());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemicolon));

  auto module = std::make_unique<AbstractModule<EvalT>>(module_name);
  module->DeclarePortsOrder(module_ports);

  while (true) {
    if (TryDropKeyword("endmodule")) {
      break;
    }
    XLS_RETURN_IF_ERROR(ParseModuleStatement(module.get(), netlist));
  }
  return module;
}

template <typename EvalT>
absl::StatusOr<std::unique_ptr<AbstractNetlist<EvalT>>>
AbstractParser<EvalT>::ParseNetlist(AbstractCellLibrary<EvalT>* cell_library,
                                    Scanner* scanner, EvalT zero, EvalT one) {
  auto netlist = std::make_unique<AbstractNetlist<EvalT>>();
  AbstractParser<EvalT> p(cell_library, scanner, zero, one);
  while (!scanner->AtEof()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<AbstractModule<EvalT>> module,
                         p.ParseModule(*netlist));
    netlist->AddModule(std::move(module));
  }
  return std::move(netlist);
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_NETLIST_PARSER_H_
