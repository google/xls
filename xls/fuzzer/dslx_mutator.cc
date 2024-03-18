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

#include "xls/fuzzer/dslx_mutator.h"

#include <string>
#include <string_view>
#include <vector>

#include "absl/random/random.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/scanner.h"

using absl::InvalidArgumentError;
using absl::StatusOr;

namespace xls::dslx {
namespace {
std::string TokensToString(absl::Span<const Token> tokens) {
  std::string result;
  for (const Token& t : tokens) {
    result += t.ToString();
  }
  return result;
}
}  // namespace

StatusOr<std::string> RemoveDslxToken(std::string_view dslx,
                                      absl::BitGenRef gen) {
  if (dslx.empty()) {
    return InvalidArgumentError("dslx is empty");
  }
  Scanner scan("x.x", std::string(dslx),
               /*include_whitespace_and_comments=*/true);
  XLS_ASSIGN_OR_RETURN(std::vector<Token> tokens, scan.PopAll());
  int64_t index_to_remove =
      absl::Uniform(gen, 0, static_cast<int64_t>(tokens.size()));
  tokens.erase(tokens.begin() + index_to_remove);
  return TokensToString(tokens);
}

}  // namespace xls::dslx
