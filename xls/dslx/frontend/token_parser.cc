// Copyright 2021 The XLS Authors
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

#include "xls/dslx/frontend/token_parser.h"

#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/token.h"

namespace xls::dslx {

absl::StatusOr<bool> TokenParser::PeekTokenIs(TokenKind target) {
  XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
  return tok->kind() == target;
}
absl::StatusOr<bool> TokenParser::PeekTokenIs(Keyword target) {
  XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
  return tok->IsKeyword(target);
}
absl::StatusOr<bool> TokenParser::PeekTokenIn(
    absl::Span<TokenKind const> targets) {
  XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
  for (TokenKind target : targets) {
    if (target == tok->kind()) {
      return true;
    }
  }
  return false;
}
absl::StatusOr<bool> TokenParser::PeekKeywordIn(
    absl::Span<Keyword const> targets) {
  XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
  if (tok->kind() != TokenKind::kKeyword) {
    return false;
  }
  for (Keyword target : targets) {
    if (target == tok->GetKeyword()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<Token> TokenParser::PopTokenOrError(TokenKind target,
                                                   const Token* start,
                                                   std::string_view context,
                                                   Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(const Token* tok, PeekToken());
  if (limit_pos != nullptr) {
    *limit_pos = tok->span().limit();
  }
  if (tok->kind() == target) {
    return PopToken();
  }

  std::string msg;
  if (start == nullptr) {
    msg = absl::StrFormat("Expected '%s', got '%s'", TokenKindToString(target),
                          tok->ToErrorString());
    if (tok->IsKeyword(Keyword::kIf)) {
      msg +=
          " :: note that conditional syntax is `if test_expr { then_expr } "
          "else { else_expr }`";
    }
  } else {
    msg = absl::StrFormat(
        "Expected '%s' for construct starting with '%s' @ %s, got '%s'",
        TokenKindToString(target), start->ToErrorString(),
        start->span().ToString(), tok->ToErrorString());
  }
  if (!context.empty()) {
    absl::StrAppend(&msg, ": ", context);
  }
  return ParseErrorStatus(tok->span(), msg);
}

absl::Status TokenParser::DropTokenOrError(TokenKind target, const Token* start,
                                           std::string_view context,
                                           Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(Token token,
                       PopTokenOrError(target, start, context, limit_pos));
  return absl::OkStatus();
}

absl::StatusOr<Token> TokenParser::PopKeywordOrError(Keyword keyword,
                                                     std::string_view context,
                                                     Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(Token tok, PopToken());
  if (tok.IsKeyword(keyword)) {
    return std::move(tok);
  }
  std::string msg =
      absl::StrFormat("Expected keyword '%s', got %s'",
                      KeywordToString(keyword), tok.ToErrorString());
  if (!context.empty()) {
    absl::StrAppend(&msg, ": ", context);
  }
  return ParseErrorStatus(tok.span(), msg);
}

absl::Status TokenParser::DropKeywordOrError(Keyword target, Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(Token token, PopKeywordOrError(target, /*context=*/"",
                                                      /*limit_pos=*/limit_pos));
  return absl::OkStatus();
}

}  // namespace xls::dslx
