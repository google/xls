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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

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
    for (const auto& child : children) {
      unordered_nodes.insert(child);
    }
  }

  std::vector<V> ordered_nodes(unordered_nodes.begin(), unordered_nodes.end());
  std::sort(ordered_nodes.begin(), ordered_nodes.end());

  const int64_t n = ordered_nodes.size();

  absl::flat_hash_map<V, int64_t> node_to_index;
  for (int64_t i = 0; i < n; ++i) {
    node_to_index[ordered_nodes[i]] = i;
  }

  // Warshall's algorithm; https://cs.winona.edu/lin/cs440/ch08-2.pdf

  auto get = [&](const HashRelation<int64_t>& rel, int64_t i,
                 int64_t j) -> bool {
    return rel.contains(i) && rel.at(i).contains(j);
  };

  auto set = [&](HashRelation<int64_t>* rel, int64_t i, int64_t j, bool value) {
    if (value) {
      (*rel)[i].insert(j);
    } else {
      (*rel)[i].erase(j);
    }
  };

  HashRelation<int64_t> lag;
  for (const auto& [node, children] : relation) {
    for (const auto& child : children) {
      set(&lag, node_to_index.at(node), node_to_index.at(child), true);
    }
  }
  HashRelation<int64_t> closure = lag;
  for (int64_t k = 0; k < n; ++k) {
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        set(&closure, i, j,
            get(lag, i, j) || (get(lag, i, k) && get(lag, k, j)));
      }
    }
    lag = closure;
  }

  Rel result;
  for (const auto& [node_index, children_indices] : closure) {
    for (const auto& child_index : children_indices) {
      result[ordered_nodes[node_index]].insert(ordered_nodes[child_index]);
    }
  }

  return result;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
