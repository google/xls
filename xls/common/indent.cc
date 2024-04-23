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

#include "xls/common/indent.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

namespace xls {

std::string Indent(std::string_view text, int64_t spaces) {
  const std::string indent(spaces, ' ');
  std::vector<std::string_view> lines = absl::StrSplit(text, '\n');
  std::string result;
  // Indent lines. Don't indent empty lines to avoid creating trailing white
  // space.
  for (auto& line : lines) {
    if (!result.empty()) {
      absl::StrAppend(&result, "\n");
    }
    if (line.empty()) {
      absl::StrAppend(&result);
    } else {
      absl::StrAppend(&result, indent, line);
    }
  }
  return result;
}

}  // namespace xls
