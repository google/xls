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

#ifndef XLS_DATA_STRUCTURES_STRONGLY_CONNECTED_COMPONENTS_H_
#define XLS_DATA_STRUCTURES_STRONGLY_CONNECTED_COMPONENTS_H_

#include <stack>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"

namespace xls {

// Computes the strongly connected components of a graph using Tarjan's strongly
// connected components algorithm.
//
// The parameter `graph` is an arbitrary adjacency matrix represented as a
// map from nodes to the set of out-neighbors of that node. Self-edges are
// permitted.
//
// A description of the Tarjan SCC algorithm exists on Wikipedia:
// https://w.wiki/5h9U
// This implementation is directly based off of that pseudocode, so please read
// that to make sense of this code.
template <typename V>
std::vector<absl::btree_set<V>> StronglyConnectedComponents(
    const absl::btree_map<V, absl::btree_set<V>>& graph) {
  absl::btree_set<V> vertices;
  for (const auto& [source, targets] : graph) {
    for (const V& target : targets) {
      vertices.insert(source);
      vertices.insert(target);
    }
  }

  int64_t index = 0;
  std::stack<V, std::vector<V>> stack;
  std::vector<absl::btree_set<V>> result;
  absl::btree_map<V, int64_t> indexes;
  absl::btree_map<V, int64_t> low_links;
  absl::btree_set<V> on_stack;

  std::function<void(const V&)> strong_connect = [&](const V& vertex) {
    indexes[vertex] = index;
    low_links[vertex] = index;
    ++index;
    stack.push(vertex);
    on_stack.insert(vertex);

    if (graph.contains(vertex)) {
      for (const V& neighbor : graph.at(vertex)) {
        if (!indexes.contains(neighbor)) {
          strong_connect(neighbor);
          low_links.at(vertex) =
              std::min(low_links.at(vertex), low_links.at(neighbor));
        } else if (on_stack.contains(neighbor)) {
          low_links.at(vertex) =
              std::min(low_links.at(vertex), indexes.at(neighbor));
        }
      }
    }

    if (low_links.at(vertex) == indexes.at(vertex)) {
      absl::btree_set<V> scc;
      V v;
      do {
        v = stack.top();
        stack.pop();
        on_stack.erase(v);
        scc.insert(v);
      } while (v != vertex);
      result.push_back(scc);
    }
  };

  for (const V& vertex : vertices) {
    if (!indexes.contains(vertex)) {
      strong_connect(vertex);
    }
  }

  return result;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_STRONGLY_CONNECTED_COMPONENTS_H_
