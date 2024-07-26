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

#include "xls/dslx/frontend/token.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/frontend/scanner_keywords.inc"

namespace xls::dslx {

const absl::flat_hash_set<Keyword>& GetTypeKeywords() {
  static const absl::flat_hash_set<Keyword>* singleton = ([] {
    auto* s = new absl::flat_hash_set<Keyword>;
#define ADD_TO_SET(__enum, ...) s->insert(Keyword::__enum);
    XLS_DSLX_TYPE_KEYWORDS(ADD_TO_SET)
#undef ADD_TO_SET
    return s;
  })();
  return *singleton;
}

std::string KeywordToString(Keyword keyword) {
  switch (keyword) {
#define MAKE_CASE(__enum, unused, __str, ...) \
  case Keyword::__enum:                       \
    return __str;
    XLS_DSLX_KEYWORDS(MAKE_CASE)
#undef MAKE_CASE
  }
  return absl::StrFormat("<invalid Keyword(%d)>", static_cast<int>(keyword));
}

std::optional<Keyword> KeywordFromString(std::string_view s) {
#define MAKE_CASE(__enum, unused, __str, ...) \
  if (s == __str) {                           \
    return Keyword::__enum;                   \
  }
  XLS_DSLX_KEYWORDS(MAKE_CASE)
#undef MAKE_CASE
  return std::nullopt;
}

std::string TokenKindToString(TokenKind kind) {
  switch (kind) {
#define MAKE_CASE(__enum, unused, __str, ...) \
  case TokenKind::__enum:                     \
    return __str;
    XLS_DSLX_TOKEN_KINDS(MAKE_CASE)
#undef MAKE_CASE
  }
  return absl::StrFormat("<invalid TokenKind(%d)>", static_cast<int>(kind));
}

absl::StatusOr<TokenKind> TokenKindFromString(std::string_view s) {
#define MAKE_CASE(__enum, unused, __str, ...) \
  if (s == __str) {                           \
    return TokenKind::__enum;                 \
  }
  XLS_DSLX_TOKEN_KINDS(MAKE_CASE)
#undef MAKE_CASE
  return absl::InvalidArgumentError(
      absl::StrFormat("Not a token kind: \"%s\"", s));
}

absl::StatusOr<int64_t> Token::GetValueAsInt64() const {
  std::optional<std::string> value = GetValue();
  if (!value) {
    return absl::InvalidArgumentError(
        "Token does not have a (string) value; cannot convert to int64_t.");
  }
  int64_t result;
  if (absl::SimpleAtoi(*value, &result)) {
    return result;
  }
  return absl::InvalidArgumentError("Could not convert value to int64_t: " +
                                    *value);
}

std::string Token::ToErrorString() const {
  if (kind_ == TokenKind::kKeyword) {
    return absl::StrFormat("keyword:%s", KeywordToString(GetKeyword()));
  }
  return TokenKindToString(kind_);
}

std::string Token::ToString() const {
  if (kind() == TokenKind::kKeyword) {
    return KeywordToString(GetKeyword());
  }
  if (kind() == TokenKind::kComment) {
    return absl::StrCat("//", GetValue().value());
  }
  if (kind() == TokenKind::kCharacter) {
    return absl::StrCat("'", absl::Utf8SafeCHexEscape(GetValue().value()), "'");
  }
  if (kind() == TokenKind::kString) {
    return absl::StrCat("\"", absl::Utf8SafeCHexEscape(GetValue().value()),
                        "\"");
  }
  if (GetValue().has_value()) {
    return GetValue().value();
  }
  return TokenKindToString(kind_);
}

std::string Token::ToRepr() const {
  if (kind_ == TokenKind::kKeyword) {
    return absl::StrFormat("Token(%s, %s)", span_.ToRepr(),
                           KeywordToString(GetKeyword()));
  }
  if (GetValue().has_value()) {
    return absl::StrFormat("Token(%s, %s, \"%s\")", TokenKindToString(kind_),
                           span_.ToRepr(), GetValue().value());
  }
  return absl::StrFormat("Token(%s, %s)", TokenKindToString(kind_),
                         span_.ToRepr());
}

}  // namespace xls::dslx
