// Copyright 2022 The XLS Authors
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

#ifndef XLS_DATA_STRUCTURES_UNION_FIND_H_
#define XLS_DATA_STRUCTURES_UNION_FIND_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/variant.h"
#include "absl/types/variant.h"
#include "absl/types/variant.h"
#include "absl/types/variant.h"
#include "absl/types/variant.h"
#include "absl/types/variant.h"
#include "xls/data_structures/union_find_map.h"

namespace xls {

// A simple union-find data structure based on UnionFindMap.
template <typename T>
class UnionFind {
 public:
  UnionFind() = default;

  // Insert the element. Has no effect if the element has been inserted
  // before. Otherwise, the element is inserted in its own equivalence class.
  void Insert(const T& element) {
    union_find_map_.Insert(element, absl::monostate());
  }

  // Union together the equivalence classes of two elements.
  void Union(const T& x, const T& y) {
    CHECK(union_find_map_.Union(x, y, Merge))
        << "Both elements passed to Union have not been added.";
  }

  // Returns the representative element in the given element's equivalence
  // class.
  T Find(const T& element) {
    std::optional<std::pair<T, std::monostate&>> result =
        union_find_map_.Find(element);
    CHECK(result.has_value()) << "Element passed Find has not been inserted.";
    return result->first;
  }

  absl::flat_hash_set<T> GetRepresentatives() {
    return union_find_map_.GetRepresentatives();
  }

  // Returns every element ever inserted, with unspecified ordering.
  const std::vector<T>& GetElements() { return union_find_map_.GetKeys(); }

  // Returns the number of elements in the data structure.
  int64_t size() const { return union_find_map_.GetKeys().size(); }

 private:
  static absl::monostate Merge(absl::monostate x, absl::monostate y) {
    return absl::monostate();
  }

  UnionFindMap<T, absl::monostate> union_find_map_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_UNION_FIND_H_
