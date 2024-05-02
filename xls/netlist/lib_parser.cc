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

#include "xls/netlist/lib_parser.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace netlist {
namespace cell_lib {

/* static */ absl::StatusOr<CharStream> CharStream::FromPath(
    std::string_view path) {
  std::ifstream file_stream{std::string(path)};
  if (file_stream.is_open()) {
    return CharStream(std::move(file_stream));
  }
  return absl::NotFoundError(
      absl::StrCat("Could not open file at path: ", path));
}

/* static */ absl::StatusOr<CharStream> CharStream::FromText(std::string text) {
  return CharStream(std::move(text));
}

std::string TokenKindToString(TokenKind kind) {
  switch (kind) {
    case TokenKind::kIdentifier:
      return "identifier";
    case TokenKind::kOpenParen:
      return "open-paren";
    case TokenKind::kCloseParen:
      return "close-paren";
    case TokenKind::kOpenCurl:
      return "open-curl";
    case TokenKind::kCloseCurl:
      return "close-curl";
    case TokenKind::kSemi:
      return "semi";
    case TokenKind::kColon:
      return "colon";
    case TokenKind::kQuotedString:
      return "quoted-string";
    case TokenKind::kNumber:
      return "number";
    case TokenKind::kComma:
      return "comma";
  }
  return absl::StrFormat("<invalid TokenKind(%d)>", static_cast<int64_t>(kind));
}

absl::StatusOr<Token> Scanner::ScanIdentifier() {
  const Pos start_pos = cs_->GetPos();
  CHECK(IsIdentifierStart(cs_->PeekCharOrDie()));
  absl::InlinedVector<char, 16> chars;
  while (!cs_->AtEof() && IsIdentifierRest(cs_->PeekCharOrDie())) {
    chars.push_back(cs_->PopCharOrDie());
  }
  return Token::Identifier(
      start_pos, std::string(std::string_view(chars.data(), chars.size())));
}

// Scans a number token.
absl::StatusOr<Token> Scanner::ScanNumber() {
  const Pos start_pos = cs_->GetPos();
  CHECK_NE(std::isdigit(cs_->PeekCharOrDie()), 0);
  absl::InlinedVector<char, 16> chars;
  while (!cs_->AtEof()) {
    if (IsNumberRest(cs_->PeekCharOrDie())) {
      chars.push_back(cs_->PopCharOrDie());
    } else if (cs_->TryDropChars('e', '-')) {
      chars.push_back('e');
      chars.push_back('-');
    } else {
      break;
    }
  }
  return Token::Number(
      start_pos, std::string(std::string_view(chars.data(), chars.size())));
}

// Scans a string token.
absl::StatusOr<Token> Scanner::ScanQuotedString() {
  const Pos start_pos = cs_->GetPos();
  CHECK(cs_->TryDropChar('"'));
  absl::InlinedVector<char, 16> chars;
  while (true) {
    if (cs_->AtEof()) {
      return absl::InvalidArgumentError(
          "Unexpected end-of-file in string token starting @ " +
          start_pos.ToHumanString());
    }
    char c = cs_->PopCharOrDie();
    if (c == '"') {
      break;
    }
    chars.push_back(c);
  }
  return Token::QuotedString(
      start_pos, std::string(std::string_view(chars.data(), chars.size())));
}

absl::Status Scanner::PeekInternal() {
  DCHECK(!lookahead_.has_value());
  DCHECK(!cs_->AtEof());
  if (IsIdentifierStart(cs_->PeekCharOrDie())) {
    XLS_ASSIGN_OR_RETURN(lookahead_, ScanIdentifier());
    DropWhitespaceAndComments();
    return absl::OkStatus();
  }
  if (isdigit(cs_->PeekCharOrDie())) {
    XLS_ASSIGN_OR_RETURN(lookahead_, ScanNumber());
    DropWhitespaceAndComments();
    return absl::OkStatus();
  }
  const Pos start_pos = cs_->GetPos();
  switch (char c = cs_->PeekCharOrDie()) {
    case '"': {
      XLS_ASSIGN_OR_RETURN(lookahead_, ScanQuotedString());
      break;
    }
    case ':':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kColon);
      break;
    case ';':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kSemi);
      break;
    case '{':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kOpenCurl);
      break;
    case '}':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kCloseCurl);
      break;
    case '(':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kOpenParen);
      break;
    case ')':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kCloseParen);
      break;
    case ',':
      cs_->DropCharOrDie();
      lookahead_ = Token::Simple(start_pos, TokenKind::kComma);
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unhandled character for scanning @ %s: '%c'=%#x",
                          start_pos.ToHumanString(), c, c));
  }
  DropWhitespaceAndComments();
  return absl::OkStatus();
}

/* static */ std::string Block::EntryToString(const BlockEntry& entry) {
  if (const KVEntry* kv = absl::get_if<KVEntry>(&entry)) {
    return absl::StrFormat("(%s \"%s\")", kv->key, kv->value);
  }
  const auto& block = std::get<std::unique_ptr<Block>>(entry);
  return block->ToString();
}

std::string Block::ToString() const {
  std::string result =
      absl::StrCat("(block ", kind, " (", absl::StrJoin(args, " "), ")", " (");
  absl::StrAppend(&result,
                  absl::StrJoin(entries, " ",
                                [](std::string* out, const BlockEntry& entry) {
                                  absl::StrAppend(out, EntryToString(entry));
                                }));
  absl::StrAppend(&result, "))");
  return result;
}

std::vector<const Block*> Block::GetSubBlocks(
    std::optional<std::string_view> target_kind) const {
  std::vector<const Block*> results;
  for (const BlockEntry& item : entries) {
    if (const auto* block = absl::get_if<std::unique_ptr<Block>>(&item)) {
      if (!target_kind.has_value() || target_kind.value() == (*block)->kind) {
        results.push_back(block->get());
      }
    }
  }
  return results;
}

const std::string& Block::GetKVOrDie(std::string_view target_key) const {
  for (const BlockEntry& item : entries) {
    if (const KVEntry* kv_entry = absl::get_if<KVEntry>(&item);
        kv_entry->key == target_key) {
      return kv_entry->value;
    }
  }
  LOG(FATAL) << "Target key is not present in " << kind
             << " block: " << target_key;
}

int64_t Block::CountEntries(std::string_view target) const {
  int64_t count = 0;
  for (const BlockEntry& entry : entries) {
    if (const KVEntry* kv = absl::get_if<KVEntry>(&entry)) {
      count += static_cast<int64_t>(kv->key == target);
    } else {
      const Block* block = std::get<std::unique_ptr<Block>>(entry).get();
      count += static_cast<int64_t>(block->kind == target);
    }
  }
  return count;
}

absl::StatusOr<bool> Parser::TryDropToken(TokenKind target, Pos* pos) {
  XLS_ASSIGN_OR_RETURN(const Token* peek, scanner_->Peek());
  if (peek->kind() == target) {
    if (pos != nullptr) {
      *pos = peek->pos();
    }
    CHECK(scanner_->Pop().ok());
    return true;
  }
  return false;
}
absl::Status Parser::DropTokenOrError(TokenKind kind) {
  XLS_ASSIGN_OR_RETURN(bool dropped, TryDropToken(kind));
  if (!dropped) {
    return absl::InvalidArgumentError(
        "Could not pop token with kind: " + TokenKindToString(kind) + " @ " +
        scanner_->GetPos().ToHumanString());
  }
  return absl::OkStatus();
}
absl::Status Parser::DropIdentifierOrError(std::string_view target) {
  XLS_ASSIGN_OR_RETURN(std::string identifier, PopIdentifierOrError());
  if (identifier != target) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected identifier '%s'; got '%s'", target, identifier));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> Parser::PopIdentifierOrError() {
  XLS_ASSIGN_OR_RETURN(Token t, scanner_->Pop());
  if (t.kind() != TokenKind::kIdentifier) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected an identifier; got %s @ %s",
                        TokenKindToString(t.kind()), t.pos().ToHumanString()));
  }
  return t.PopPayload();
}

absl::StatusOr<std::string> Parser::PopValueOrError(Pos* last_pos) {
  XLS_ASSIGN_OR_RETURN(Token t, scanner_->Pop());
  if (last_pos != nullptr) {
    *last_pos = t.pos();
  }
  switch (t.kind()) {
    case TokenKind::kNumber:
    case TokenKind::kQuotedString:
    case TokenKind::kIdentifier:
      return std::string(t.payload());
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected a value; got %s @ %s", TokenKindToString(t.kind()),
          t.pos().ToHumanString()));
  }
}

absl::StatusOr<std::vector<BlockEntry>> Parser::ParseEntries() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenCurl));
  std::vector<BlockEntry> result;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_curl,
                         TryDropToken(TokenKind::kCloseCurl));
    if (dropped_curl) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(std::string identifier, PopIdentifierOrError());
    XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
    if (dropped_colon) {
      Pos last_pos;
      XLS_ASSIGN_OR_RETURN(std::string value, PopValueOrError(&last_pos));

      // Could be a colon-ref-type value, e.g., Foo:Bar.
      XLS_ASSIGN_OR_RETURN(bool dropped_another_colon,
                           TryDropToken(TokenKind::kColon));
      if (dropped_another_colon) {
        XLS_ASSIGN_OR_RETURN(std::string sub_value, PopValueOrError());
        absl::StrAppend(&value, ":", sub_value);
      }
      result.push_back(KVEntry{identifier, value});
      XLS_ASSIGN_OR_RETURN(bool dropped_semi, TryDropToken(TokenKind::kSemi));
      if (!dropped_semi) {
        if (scanner_->GetPos().lineno == last_pos.lineno) {
          return absl::InvalidArgumentError(
              "Expected semicolon or newline after entry @ " +
              last_pos.ToHumanString());
        }
      }
    } else {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Block> block,
                           ParseBlock(identifier));
      result.push_back(std::move(block));
    }
  }
  return result;
}

absl::StatusOr<absl::InlinedVector<std::string, 4>> Parser::ParseValues(
    Pos* end_pos) {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOpenParen));
  absl::InlinedVector<std::string, 4> result;
  while (true) {
    Pos pos;
    XLS_ASSIGN_OR_RETURN(bool dropped_close_paren,
                         TryDropToken(TokenKind::kCloseParen, &pos));
    if (dropped_close_paren) {
      if (end_pos != nullptr) {
        *end_pos = pos;
      }
      break;
    }
    if (!result.empty()) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kComma));
    }
    XLS_ASSIGN_OR_RETURN(std::string value, PopValueOrError());
    result.push_back(value);
  }
  return result;
}

absl::StatusOr<std::unique_ptr<Block>> Parser::ParseBlock(
    std::string identifier) {
  auto block = std::make_unique<Block>();
  block->kind = std::move(identifier);

  // Once we've seen the block kind we know whether it's in the allowlist or
  // not.
  bool kind_allowed =
      !kind_allowlist_.has_value() || kind_allowlist_->contains(block->kind);

  Pos last_pos;
  XLS_ASSIGN_OR_RETURN(block->args, ParseValues(&last_pos));
  XLS_ASSIGN_OR_RETURN(bool dropped_semi, TryDropToken(TokenKind::kSemi));
  if (dropped_semi) {
    return block;
  }

  // We do this to hack around missing semicolons...
  //
  // Normally an identifier following the parens would be a syntax error, but
  // we allow it to terminate the block because it happens at least once we've
  // seen.
  XLS_ASSIGN_OR_RETURN(const Token* peek_next, scanner_->Peek());
  if (peek_next->kind() == TokenKind::kIdentifier) {
    if (peek_next->pos().lineno > last_pos.lineno) {
      return block;
    }
  }

  XLS_ASSIGN_OR_RETURN(block->entries, ParseEntries());
  if (!kind_allowed) {
    // Save memory on disallowed blocks by clearing out its entries.
    block->entries.clear();
  }
  return block;
}

}  // namespace cell_lib
}  // namespace netlist
}  // namespace xls
