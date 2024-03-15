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

#include "xls/data_structures/path_cut.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/strong_int.h"

namespace xls {

std::string PathCutToString(const PathCut& cut) {
  std::vector<std::string> pieces;
  for (const auto& piece : cut) {
    pieces.push_back(absl::StrFormat(
        "[%s]",
        absl::StrJoin(piece, ", ", [](std::string* out, PathNodeId node) {
          absl::StrAppend(out, node.value());
        })));
  }
  return absl::StrJoin(pieces, ", ");
}

}  // namespace xls
