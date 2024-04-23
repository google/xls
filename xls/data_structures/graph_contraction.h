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

#ifndef XLS_DATA_STRUCTURES_GRAPH_CONTRACTION_H_
#define XLS_DATA_STRUCTURES_GRAPH_CONTRACTION_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xls/data_structures/union_find_map.h"

namespace xls {

template <typename K, typename V>
static absl::flat_hash_set<K> Keys(const absl::flat_hash_map<K, V>& map) {
  absl::flat_hash_set<K> result;
  for (const auto& pair : map) {
    result.insert(pair.first);
  }
  return result;
}

// A graph data structure that supports vertex identification.
//
// Parameter `V` is the type of vertices, which is expected to be cheap to copy.
// Parameter `VW` is the type of vertex weights.
// Parameter `EW` is the type of edge weights.
template <typename V, typename VW, typename EW>
class GraphContraction {
 public:
  // Adds a vertex to the graph with the given `weight` associated to it.
  void AddVertex(const V& vertex, const VW& weight) {
    vertex_weights_.Insert(vertex, weight);
    if (!out_edges_.Contains(vertex)) {
      out_edges_.Insert(vertex, {});
    }
    if (!in_edges_.Contains(vertex)) {
      in_edges_.Insert(vertex, {});
    }
  }

  // Adds an edge from the given `source` to the given `target` with the given
  // `weight` associated to it. If an edge with that source and target already
  // exists, the given `weight` replaces the previous weight. Returns false if
  // either the source or target is not a previously inserted vertex, and
  // returns true otherwise. The graph is unchanged if false is returned.
  bool AddEdge(const V& source, const V& target, const EW& weight) {
    if (!(vertex_weights_.Contains(source) &&
          vertex_weights_.Contains(target))) {
      return false;
    }
    auto merge_map = [](absl::flat_hash_map<V, EW> a,
                        absl::flat_hash_map<V, EW> b) {
      absl::flat_hash_map<V, EW> result;
      result.merge(a);
      result.merge(b);
      return result;
    };
    out_edges_.Insert(source, {{target, weight}}, merge_map);
    in_edges_.Insert(target, {{source, weight}}, merge_map);
    return true;
  }

  // Identifies two vertices (a generalization of edge contraction). The vertex
  // weights of the identified vertices are combined with `vw_merge`. If vertex
  // identification results in multiple edges (a multigraph), the edge weights
  // are combined using `ew_merge`. Returns false (and leaves the graph in an
  // unmodified state) iff one of the given vertices is nonexistent.
  //
  // `vw_merge` should have a type compatible with
  // `std::function<VW(const VW&, const VW&)>`.
  //
  // `ew_merge` should have a type compatible with
  // `std::function<EW(const EW&, const EW&)>`.
  template <typename FV, typename FE>
  bool IdentifyVertices(const V& x, const V& y, FV vw_merge, FE ew_merge) {
    if (!(vertex_weights_.Contains(x) && vertex_weights_.Contains(y))) {
      return false;
    }
    if (vertex_weights_.Find(x)->first == vertex_weights_.Find(y)->first) {
      return true;
    }
    vertex_weights_.Union(x, y, vw_merge);
    V rep = vertex_weights_.Find(x).value().first;  // y would give same result
    auto merge_map = [&](const absl::flat_hash_map<V, EW>& a,
                         const absl::flat_hash_map<V, EW>& b) {
      absl::flat_hash_map<V, EW> result;
      for (const auto& [vertex, edge_weight] : a) {
        V key = ((vertex == x) || (vertex == y)) ? rep : vertex;
        result.insert_or_assign(key, result.contains(key)
                                         ? ew_merge(result.at(key), edge_weight)
                                         : edge_weight);
      }
      for (const auto& [vertex, edge_weight] : b) {
        V key = ((vertex == x) || (vertex == y)) ? rep : vertex;
        result.insert_or_assign(key, result.contains(key)
                                         ? ew_merge(result.at(key), edge_weight)
                                         : edge_weight);
      }
      return result;
    };

    absl::flat_hash_set<V> edges_out_of_x = Keys(out_edges_.Find(x)->second);
    absl::flat_hash_set<V> edges_out_of_y = Keys(out_edges_.Find(y)->second);
    absl::flat_hash_set<V> edges_into_x = Keys(in_edges_.Find(x)->second);
    absl::flat_hash_set<V> edges_into_y = Keys(in_edges_.Find(y)->second);
    out_edges_.Union(x, y, merge_map);
    in_edges_.Union(x, y, merge_map);

    // Given a vertex with at least one edge pointing into/out of `x` or `y`,
    // patch those out edges up so that they point to/from `rep`.
    auto patch_up_edges =
        [&](const V& vertex, UnionFindMap<V, absl::flat_hash_map<V, EW>>* map) {
          CHECK(map->Find(vertex).has_value())
              << "Inconsistency between in_edges_ and out_edges_ detected";
          absl::flat_hash_map<V, EW>& edges = map->Find(vertex)->second;
          if (edges.contains(x) && edges.contains(y)) {
            EW weight = ew_merge(edges.at(x), edges.at(y));
            edges.erase(x);
            edges.erase(y);
            edges.insert_or_assign(rep, weight);
          } else if (edges.contains(x)) {
            EW weight = edges.at(x);
            edges.erase(x);
            edges.insert_or_assign(rep, weight);
          } else if (edges.contains(y)) {
            EW weight = edges.at(y);
            edges.erase(y);
            edges.insert_or_assign(rep, weight);
          } else {
            CHECK(false)
                << "Inconsistency between in_edges_ and out_edges_ detected";
          }
        };

    for (const V& source : edges_into_x) {
      patch_up_edges(source, &out_edges_);
    }

    for (const V& source : edges_into_y) {
      patch_up_edges(source, &out_edges_);
    }

    for (const V& target : edges_out_of_x) {
      patch_up_edges(target, &in_edges_);
    }

    for (const V& target : edges_out_of_y) {
      patch_up_edges(target, &in_edges_);
    }

    return true;
  }

  // Returns true iff the given vertex has previously been added to the graph
  // using `AddVertex`.
  bool Contains(const V& vertex) { return vertex_weights_.Contains(vertex); }

  // Returns the set of vertices in the graph. If some set of vertices have been
  // identified, an arbitrary element of that set will be present in this list.
  absl::flat_hash_set<V> Vertices() {
    return vertex_weights_.GetRepresentatives();
  }

  // Returns the representative of the equivalence class of identified vertices
  // to which the given vertex belongs.
  std::optional<V> RepresentativeOf(const V& vertex) {
    if (auto pair = vertex_weights_.Find(vertex)) {
      return pair->first;
    }
    return std::nullopt;
  }

  // Returns the edges that point out of the given vertex, and their weights.
  absl::flat_hash_map<V, EW> EdgesOutOf(const V& vertex) {
    if (auto pair = out_edges_.Find(vertex)) {
      return pair->second;
    }
    return {};
  }

  // Returns the edges that point into the given vertex, and their weights.
  absl::flat_hash_map<V, EW> EdgesInto(const V& vertex) {
    if (auto pair = in_edges_.Find(vertex)) {
      return pair->second;
    }
    return {};
  }

  // Returns the weight of the given vertex.
  std::optional<VW> WeightOf(const V& vertex) {
    if (auto v = vertex_weights_.Find(vertex)) {
      return v.value().second;
    }
    return std::nullopt;
  }

  // Returns the weight of the given edge.
  std::optional<EW> WeightOf(const V& source, const V& target) {
    absl::flat_hash_map<V, EW> edges = EdgesOutOf(source);
    if (edges.contains(target)) {
      return edges.at(target);
    }
    return std::nullopt;
  }

  // Returns a topological sort of the nodes in the graph if the graph is
  // acyclic, otherwise returns std::nullopt.
  std::optional<std::vector<V>> TopologicalSort() {
    std::vector<V> result;

    // Kahn's algorithm

    std::vector<V> active;
    absl::flat_hash_map<V, int64_t> edge_count;
    for (const V& vertex : Vertices()) {
      edge_count[vertex] = EdgesInto(vertex).size();
      if (edge_count.at(vertex) == 0) {
        active.push_back(vertex);
      }
    }

    while (!active.empty()) {
      V source = active.back();
      active.pop_back();
      result.push_back(source);
      for (const auto& [target, weight] : EdgesOutOf(source)) {
        edge_count.at(target)--;
        if (edge_count.at(target) == 0) {
          active.push_back(target);
        }
      }
    }

    if (result.size() != Vertices().size()) {
      return std::nullopt;
    }

    return result;
  }

  // All-pairs longest paths in an acyclic graph.
  // The length of a path is measured by the total vertex weight encountered
  // along that path, using `operator+` and `std::max` on `VW`.
  // The outer map key is the source of the path and the inner map key is the
  // sink of the path. Those keys only exist if a path exists from that source
  // to that sink.
  // Returns std::nullopt if the graph contains a cycle.
  std::optional<absl::flat_hash_map<V, absl::flat_hash_map<V, VW>>>
  LongestNodePaths() {
    absl::flat_hash_map<V, absl::flat_hash_map<V, VW>> result;

    for (V vertex : Vertices()) {
      // The graph must be acyclic, so the longest path from any vertex to
      // itself has weight equal to the weight of that vertex.
      result[vertex] = {{vertex, WeightOf(vertex).value()}};
    }

    if (std::optional<std::vector<V>> topo = TopologicalSort()) {
      for (const V& vertex : *topo) {
        for (auto& [source, targets] : result) {
          for (const auto& [pred, edge_weight] : EdgesInto(vertex)) {
            if (targets.contains(pred)) {
              VW new_weight = targets[pred] + WeightOf(vertex).value();
              targets[vertex] = targets.contains(vertex)
                                    ? std::max(targets.at(vertex), new_weight)
                                    : new_weight;
            }
          }
        }
      }
    } else {
      return std::nullopt;
    }

    return result;
  }

 private:
  UnionFindMap<V, VW> vertex_weights_;
  UnionFindMap<V, absl::flat_hash_map<V, EW>> out_edges_;
  UnionFindMap<V, absl::flat_hash_map<V, EW>> in_edges_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_GRAPH_CONTRACTION_H_
