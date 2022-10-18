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

#ifndef XLS_SYNTHESIS_YOSYS_UTIL_H_
#define XLS_SYNTHESIS_YOSYS_UTIL_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {
namespace synthesis {

// Parses the given string output of nextpnr and returns the maximum frequency
// in Hz.
// TODO(meheff): Extract more information from the nextpnr output.
absl::StatusOr<int64_t> ParseNextpnrOutput(std::string_view nextpnr_output);

// Parses the given string output of yosys and returns information about the
// synthesis results.
struct YosysSynthesisStatistics {
  // Could add other fields for things like wires / memory...
  absl::flat_hash_map<std::string, int64_t> cell_histogram;
};
absl::StatusOr<YosysSynthesisStatistics> ParseYosysOutput(
    std::string_view yosys_output);

}  // namespace synthesis
}  // namespace xls

#endif  // XLS_SYNTHESIS_YOSYS_UTIL_H_
