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
#include "xls/common/case_converters.h"

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace xls {

std::string Camelize(std::string_view input) {
  std::vector<std::string> pieces =
      absl::StrSplit(input, absl::ByAnyChar("-_"));
  for (std::string& piece : pieces) {
    piece[0] = toupper(piece[0]);
  }
  return absl::StrJoin(pieces, "");
}

}  // namespace xls
