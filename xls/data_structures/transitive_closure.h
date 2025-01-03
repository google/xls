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

#ifndef XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
#define XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace xls {

template <typename V>
using HashRelation = absl::flat_hash_map<V, absl::flat_hash_set<V>>;

// Compute the transitive closure of a relation.
template <typename V>
HashRelation<V> TransitiveClosure(HashRelation<V> relation) {
  // Warshall's algorithm; https://cs.winona.edu/lin/cs440/ch08-2.pdf
  // modified in the typical way to avoid unnecessary copies of the expanded
  // relation. It's safe to update the relation as we go, since at each stage k,
  // i relates to k via nodes < k iff i relates to k via nodes <= k, and
  // similarly for k relating to j.
  for (const auto& [k, from_k] : relation) {
    for (auto& [i, from_i] : relation) {
      if (i == k) {
        // Updating would be a no-op, so skip it.
        continue;
      }
      if (from_i.contains(k)) {
        // i relates to k (via nodes < k), so:
        //   for any j where k relates to j (via nodes < k),
        //     i relates to j (via nodes <= k).
        from_i.insert(from_k.begin(), from_k.end());
      }
    }
  }
  return relation;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
