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

#include "xls/passes/pass_base.h"

#include <string>

#include "xls/common/math_util.h"

namespace xls {

std::string_view RamKindToString(RamKind kind) {
  switch (kind) {
    case RamKind::kAbstract:
      return "abstract";
    case RamKind::k1RW:
      return "1rw";
    case RamKind::k1R1W:
      return "1r1w";
    case RamKind::k2RW:
      return "2rw";
  }
}

int64_t RamConfig::addr_width() const {
  XLS_CHECK_GE(depth, 0);
  return CeilOfLog2(static_cast<uint64_t>(depth));
}

int64_t RamConfig::mask_width() const {
  if (!word_partition_size.has_value()) {
    return 0;
  }
  return (width + word_partition_size.value() - 1) /
         word_partition_size.value();
}
}  // namespace xls
