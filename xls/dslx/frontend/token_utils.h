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
//
// Metadata and utilities around token contents.

#ifndef XLS_DSLX_FRONTEND_TOKEN_UTILS_H_
#define XLS_DSLX_FRONTEND_TOKEN_UTILS_H_

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"

namespace xls::dslx {

// For a sized (type; e.g. `s7`) represents {.is_signed=true, .width=7}.
struct SizedTypeData {
  bool is_signed;
  uint32_t width;
};

// Returns a singleton mapping the type keyword token e.g. "s7" to its
// SizedTypeData.
const absl::flat_hash_map<std::string, SizedTypeData>&
GetSizedTypeKeywordsMetadata();

// Performs unescaping a la absl::CHexEscape but prefers the non-hex nul
// character '\0' over the hex one that CHexEscape uses.
std::string Escape(std::string_view original);

// Returns whether the given identifier is in `SCREAMING_SNAKE_CASE` style.
bool IsScreamingSnakeCase(std::string_view identifier);

// Returns whether the given identifier is in `snake_case` style.
//
// Note that we make an exception for when a number is followed by "B" for
// bytes. This is a very specific exception, we may want to think about how to
// generalize it properly.
bool IsAcceptablySnakeCase(std::string_view identifier);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_TOKEN_UTILS_H_
