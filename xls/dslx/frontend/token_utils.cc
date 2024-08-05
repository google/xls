// Copyright 2022 The XLS Authors
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

#include "xls/dslx/frontend/token_utils.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/dslx/frontend/scanner_keywords.inc"

namespace xls::dslx {

const absl::flat_hash_map<std::string, SizedTypeData>&
GetSizedTypeKeywordsMetadata() {
  static const auto* m = ([] {
    auto result =
        std::make_unique<absl::flat_hash_map<std::string, SizedTypeData>>();
#define ADD_SIZED_TYPE_KEYWORD(__enum, __caps, __str)                   \
  {                                                                     \
    bool is_signed = absl::StartsWith(__str, "s");                      \
    uint32_t width;                                                     \
    CHECK(absl::SimpleAtoi(std::string_view(__str).substr(1), &width)); \
    result->emplace(__str, SizedTypeData{is_signed, width});            \
  }
    XLS_DSLX_SIZED_TYPE_KEYWORDS(ADD_SIZED_TYPE_KEYWORD);
#undef ADD_SIZED_TYPE_KEYWORD
    return result.release();
  })();
  return *m;
}

std::string Escape(std::string_view original) {
  // Simple literal delimiter doesn't work because strlen will report an empty
  // string when constructing the string_view delimiter in absl::StrSplit.
  constexpr std::string_view kNullDelim("\0", 1);
  std::vector<std::string_view> segments = absl::StrSplit(original, kNullDelim);
  std::vector<std::string> escaped_segments;
  escaped_segments.reserve(segments.size());
  for (const auto& segment : segments) {
    escaped_segments.push_back(absl::CHexEscape(segment));
  }
  return absl::StrJoin(escaped_segments, "\\0");
}

bool IsScreamingSnakeCase(std::string_view identifier) {
  for (char c : identifier) {
    bool acceptable = ('0' <= c && c <= '9') || ('A' <= c && c <= 'Z') ||
                      c == '_' || c == '\'' || c == '!';
    if (!acceptable) {
      return false;
    }
  }
  return true;
}

bool IsAcceptablySnakeCase(std::string_view identifier) {
  // Implementation note: this is not filtering to say "these are actually
  // reasonable identifiers", it is just a hard filter for "does it conform to
  // the snake case predicate". `__!!'!'9B__` is accepted by this, but it's a
  // horrible identifier (and has no lowercase chars). This is just trying to
  // flag where people type things in a clearly-different style; e.g. CamelCase
  // for purposes of providing a helpful warning. We don't want to clamp down on
  // style to the extent somebody would do in a review in automated warnings,
  // just provide rough guidance.

  for (size_t i = 0; i < identifier.size(); ++i) {
    char c = identifier[i];
    bool acceptable = ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') ||
                      c == '_' || c == '\'' || c == '!';
    if (!acceptable) {
      // We make an exception for "B" following a number as an indicator of
      // bytes. We don't want to encourage people to change it to 'b' as that
      // would indicate bits.
      if (c == 'B' && i > 0 && std::isdigit(identifier[i - 1])) {
        continue;
      }

      return false;
    }
  }
  return true;
}

}  // namespace xls::dslx
