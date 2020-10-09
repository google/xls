// Copyright 2020 Google LLC
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

#include "xls/synthesis/yosys/yosys_util.h"

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"

namespace xls {
namespace synthesis {

absl::StatusOr<int64> ParseNextpnrOutput(absl::string_view nextpnr_output) {
  bool found = false;
  double max_mhz;
  // We're looking for lines of the form:
  //
  //   Info: Max frequency for clock 'foo': 125.28 MHz (PASS at 100.00 MHz)
  //
  // And we want to extract 125.28.
  // TODO(meheff): Use regular expressions for this. Unfortunately using RE2
  // causes multiple definition link errors when building yosys_server_test.
  for (auto line : absl::StrSplit(nextpnr_output, '\n')) {
    if (absl::StartsWith(line, "Info: Max frequency for clock") &&
        absl::StrContains(line, " MHz ")) {
      std::vector<absl::string_view> tokens = absl::StrSplit(line, ' ');
      for (int64 i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "MHz") {
          if (absl::SimpleAtod(tokens[i - 1], &max_mhz)) {
            found = true;
            break;
          }
        }
      }
    }
  }
  if (!found) {
    return absl::NotFoundError(
        "Could not find maximum frequency in nextpnr output.");
  }

  return static_cast<int64>(max_mhz * 1e6);
}

}  // namespace synthesis
}  // namespace xls
