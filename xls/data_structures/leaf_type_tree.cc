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

#include "xls/data_structures/leaf_type_tree.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/type.h"

namespace xls {
namespace internal {

bool IncrementArrayIndex(absl::Span<const int64_t> bounds,
                         std::vector<int64_t>* array_index) {
  XLS_CHECK_EQ(bounds.size(), array_index->size());
  for (int64_t i = array_index->size() - 1; i >= 0; --i) {
    ++(*array_index)[i];
    if ((*array_index)[i] < bounds[i]) {
      return false;
    }
    (*array_index)[i] = 0;
  }
  return true;
}

absl::StatusOr<SubArraySize> GetSubArraySize(Type* type, int64_t index_depth) {
  std::vector<int64_t> bounds;
  Type* subtype = type;
  for (int64_t i = 0; i < index_depth; ++i) {
    if (!subtype->IsArray()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Type has fewer than %d array dimensions: %s",
                          index_depth, type->ToString()));
    }
    int64_t bound = subtype->AsArrayOrDie()->size();
    bounds.push_back(bound);
    subtype = subtype->AsArrayOrDie()->element_type();
  }

  return SubArraySize{.type = subtype,
                      .bounds = std::move(bounds),
                      .element_count = subtype->leaf_count()};
}

}  // namespace internal
}  // namespace xls
