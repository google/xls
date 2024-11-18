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

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"

namespace xls {

template <typename V>
using HashRelation = absl::flat_hash_map<V, absl::flat_hash_set<V>>;

// Compute the transitive closure of a relation.
template <typename V>
HashRelation<V> TransitiveClosure(const HashRelation<V>& relation) {
  using Rel = HashRelation<V>;

  if (relation.empty()) {
    return Rel();
  }

  absl::flat_hash_set<V> unordered_nodes;
  for (const auto& [node, children] : relation) {
    unordered_nodes.insert(node);
    unordered_nodes.insert(children.begin(), children.end());
  }

  std::vector<V> ordered_nodes(unordered_nodes.begin(), unordered_nodes.end());
  std::sort(ordered_nodes.begin(), ordered_nodes.end());

  const int64_t n = ordered_nodes.size();

  absl::flat_hash_map<V, int64_t> node_to_index;
  node_to_index.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    CHECK(node_to_index.insert({ordered_nodes[i], i}).second);
  }

  // Warshall's algorithm; https://cs.winona.edu/lin/cs440/ch08-2.pdf

  auto get = [&](const HashRelation<int64_t>& rel, int64_t i,
                 int64_t j) -> bool {
    return rel.contains(i) && rel.at(i).contains(j);
  };

  HashRelation<int64_t> lag;
  lag.reserve(relation.size());
  for (const auto& [node, children] : relation) {
    auto [it, inserted] = lag.insert({node_to_index.at(node), {}});
    DCHECK(inserted);
    auto& children_indices = it->second;
    children_indices.reserve(children.size());
    absl::c_transform(children,
                      std::inserter(children_indices, children_indices.end()),
                      [&](const V& child) { return node_to_index.at(child); });
  }
  HashRelation<int64_t> closure = lag;
  for (int64_t k = 0; k < n; ++k) {
    for (int64_t i = 0; i < n; ++i) {
      if (!get(lag, i, k)) {
        // i doesn't relate to k (via nodes < k).
        continue;
      }
      for (int64_t j = 0; j < n; ++j) {
        if (get(lag, k, j)) {
          // i relates to k (via nodes < k), and k relates to j (via nodes < k),
          // so i relates to j (via nodes <= k).
          DCHECK(closure.contains(i));  // since i relates to k
          closure.at(i).insert(j);
        }
      }
    }
    lag = closure;
  }

  Rel result;
  result.reserve(closure.size());
  for (const auto& [node_index, children_indices] : closure) {
    auto [it, inserted] = result.insert({ordered_nodes[node_index], {}});
    DCHECK(inserted);
    auto& children = it->second;
    children.reserve(children_indices.size());
    absl::c_transform(
        children_indices, std::inserter(children, children.end()),
        [&](int64_t child_index) { return ordered_nodes[child_index]; });
  }

  return result;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
