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

#include "re2/re2.h"

namespace xls {
namespace synthesis {

absl::StatusOr<int64> ParseNextpnrOutput(absl::string_view nextpnr_output) {
  bool found = false;
  double max_mhz;
  while (RE2::FindAndConsume(
      &nextpnr_output, R"(Info: Max frequency for clock '\S+': ([0-9.]+) MHz)",
      &max_mhz)) {
    found = true;
  }
  if (!found) {
    return absl::NotFoundError(
        "Could not find maximum frequency in nextpnr output.");
  }

  return static_cast<int64>(max_mhz * 1e6);
}

}  // namespace synthesis
}  // namespace xls
