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

#include "xls/dslx/token_utils.h"

#include "absl/strings/match.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/scanner_keywords.inc"

namespace xls::dslx {

const absl::flat_hash_map<std::string, SizedTypeData>&
GetSizedTypeKeywordsMetadata() {
  static const auto* m = ([] {
    auto result =
        std::make_unique<absl::flat_hash_map<std::string, SizedTypeData>>();
#define ADD_SIZED_TYPE_KEYWORD(__enum, __caps, __str)                       \
  {                                                                         \
    bool is_signed = absl::StartsWith(__str, "s");                          \
    uint32_t width;                                                         \
    XLS_CHECK(absl::SimpleAtoi(std::string_view(__str).substr(1), &width)); \
    result->emplace(__str, SizedTypeData{is_signed, width});                \
  }
    XLS_DSLX_SIZED_TYPE_KEYWORDS(ADD_SIZED_TYPE_KEYWORD);
#undef ADD_SIZED_TYPE_KEYWORD
    return result.release();
  })();
  return *m;
}

}  // namespace xls::dslx
