// Copyright 2024 The XLS Authors
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

#include "xls/codegen/concurrent_stage_groups.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "absl/strings/str_format.h"

namespace xls {

std::string ConcurrentStageGroups::ToString() const {
  std::ostringstream oss;
  oss << "Mutually Exclusive Stages:\n";
  for (int64_t i = 0; i < stage_count(); ++i) {
    oss << absl::StreamFormat("\tStage %d: [", i);
    for (int64_t j = 0; j < stage_count(); ++j) {
      if (i == j) {
        continue;
      }
      if (IsMutuallyExclusive(i, j)) {
        oss << j << ", ";
      }
    }
    oss << "]\n";
  }
  return oss.str();
}
}  // namespace xls
