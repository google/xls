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

#include "xls/common/string_interpolation.h"

#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

absl::StatusOr<std::string> InterpolateArgs(absl::string_view s,
                                            InterpolationCallback print_arg) {
  std::string result;
  result.reserve(s.size());
  int64_t i = 0;
  int64_t arg_idx = 0;
  while (i < s.size()) {
    if (s[i] == '{') {
      // escaped lcurly - skip over it and continue
      if (i + 1 < s.size() && s[i + 1] == '{') {
        result += '{';
        i += 2;
        continue;
      }
      // not escaped: try and find matching rcurly
      int64_t j = i + 1;
      while (j < s.size() && s[j] != '}') {
        ++j;
      }
      // couldn't find a matching rcurly
      if (j == s.size()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("unmatched '{' at index %d", i, s));
      }

      // found a matching rcurly, format the argument and bump i passed the "}"
      XLS_ASSIGN_OR_RETURN(std::string formatted_arg,
                           print_arg(s.substr(i + 1, j - i - 1), arg_idx++));
      result += formatted_arg;
      i = j + 1;
    } else if (s[i] == '}') {
      // escaped rcurly - skip over it and continue
      if (i + 1 < s.size() && s[i + 1] == '}') {
        result += '}';
        i += 2;
        continue;
      }
      // otherwise this is an error, because matched "}"s would be found after
      // encountering its corresponding "{" above.
      return absl::InvalidArgumentError(
          absl::StrFormat("unmatched '}' at index %d", i, s));
    } else {
      result += s[i++];
    }
  }

  return result;
}

}  // namespace xls
