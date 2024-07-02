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

#ifndef XLS_DATA_STRUCTURES_GRAPH_COLORING_H_
#define XLS_DATA_STRUCTURES_GRAPH_COLORING_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "external/z3/src/api/c++/z3++.h"

namespace xls {

// This finds a maximal independent set in the given graph, using heuristics
// appropriate for the purpose of implementing the Recursive Largest First
// graph coloring algorithm.
//
// `vertices` is the set of vertices of the graph, which must not be empty.
// `neighborhood` is a function that, given a vertex in the graph, returns a set
// containing its neighbors. This is agnostic to graph representation.
//
// This returns a set of nodes that comprises an independent set of the
// given graph.
//
// This algorithm is explained on page 60 of "Guide to Graph Colouring" second
// edition by R. M. R. Lewis. https://doi.org/10.1007%2F978-3-030-81054-2
template <typename V>
absl::flat_hash_set<V> FindMaximalIndependentSet(
    const absl::flat_hash_set<V>& vertices,
    std::function<absl::flat_hash_set<V>(const V&)> neighborhood) {
  static_assert(!std::is_pointer<V>::value,
                "To avoid nondetermistic behavior V cannot be a pointer type");

  CHECK(!vertices.empty());

  absl::flat_hash_set<V> result;  // named S in the book
  absl::btree_set<V> available(vertices.begin(), vertices.end());  // named X
  absl::flat_hash_set<V> neighboring_result;                       // named Y

  auto add_to_result = [&](const V& vertex) {
    result.insert(vertex);
    for (const V& neighbor : neighborhood(vertex)) {
      neighboring_result.insert(neighbor);
    }
    available.erase(vertex);
  };

  // Initialize result to contain only the vertex with highest degree.
  {
    int64_t largest_neighborhood = 0;
    std::optional<V> vertex_with_most_neighbors;
    for (const V& vertex : available) {
      int64_t neighborhood_size = neighborhood(vertex).size();
      if (neighborhood_size >= largest_neighborhood) {
        largest_neighborhood = neighborhood_size;
        vertex_with_most_neighbors = vertex;
      }
    }
    add_to_result(vertex_with_most_neighbors.value());
  }

  while (!available.empty()) {
    // We want to choose the best vertex V as measured by a tuple (a, b) where
    // a is the number of neighbors in Y and b is the negated number of
    // neighbors in X, and these tuples are ordered lexicographically
    // (i.e.: sort by a, and then resolve ties with b).
    //
    // The first element in a valid tuple is always positive, so setting this to
    // (-1, -1) will result in it getting replaced in the first iteration of the
    // `for` loop below.
    std::pair<int64_t, int64_t> measure = {-1, -1};
    std::optional<V> best;
    for (const V& vertex : available) {
      if (neighboring_result.contains(vertex)) {
        continue;
      }
      std::pair<int64_t, int64_t> vertex_measure{0, 0};
      for (const V& neighbor : neighborhood(vertex)) {
        if (neighboring_result.contains(neighbor)) {
          ++vertex_measure.first;
        }
        if (available.contains(neighbor)) {
          --vertex_measure.second;
        }
      }
      if (vertex_measure > measure) {
        best = vertex;
        measure = vertex_measure;
      }
    }
    if (!best.has_value()) {
      break;
    }
    add_to_result(best.value());
  }

  return result;
}

// Color the given graph using the Recursive Largest First (RLF) algorithm.
//
// `vertices` is the set of vertices of the graph.
// `neighborhood` is a function that, given a vertex in the graph, returns a set
// containing its neighbors. This is agnostic to graph representation.
//
// This returns a vector of sets of nodes, each of which represents a color
// in the colored graph.
//
// This algorithm is explained on page 60 of "Guide to Graph Colouring" second
// edition by R. M. R. Lewis. https://doi.org/10.1007%2F978-3-030-81054-2
template <typename V>
std::vector<absl::flat_hash_set<V>> RecursiveLargestFirstColoring(
    const absl::flat_hash_set<V>& vertices,
    std::function<absl::flat_hash_set<V>(const V&)> neighborhood) {
  std::vector<absl::flat_hash_set<V>> result;
  absl::flat_hash_set<V> available = vertices;
  while (!available.empty()) {
    // Find the maximal independent set in the subgraph induced by `available`.
    absl::flat_hash_set<V> chosen = FindMaximalIndependentSet<V>(
        available, [&](const V& v) -> absl::flat_hash_set<V> {
          absl::flat_hash_set<V> result;
          for (const V& neighbor : neighborhood(v)) {
            if (available.contains(neighbor)) {
              result.insert(neighbor);
            }
          }
          return result;
        });
    for (const V& vertex : chosen) {
      available.erase(vertex);
    }
    result.push_back(chosen);
  }
  return result;
}

inline std::optional<int64_t> LookupIntegerInZ3Model(z3::model model,
                                                     std::string_view name) {
  for (int32_t i = 0; i < model.size(); i++) {
    if (model[i].name().str() == name) {
      std::optional<int64_t> result;
      const z3::func_decl& decl = model[i];
      int64_t temp = -1;
      if (model.get_const_interp(decl).is_numeral_i64(temp)) {
        result = temp;
      }
      return result;
    }
  }
  return std::nullopt;
}

template <typename V>
std::vector<absl::flat_hash_set<V>> Z3Coloring(
    const absl::flat_hash_set<V>& vertices,
    std::function<absl::flat_hash_set<V>(const V&)> neighborhood) {
  z3::context c;
  z3::optimize s(c);
  z3::expr k = c.int_const("k");

  absl::flat_hash_map<V, int64_t> vertex_index;
  std::vector<z3::expr> vertex_vars;

  {
    int64_t i = 0;
    for (V vertex : vertices) {
      vertex_index[vertex] = i;
      std::string name = absl::StrFormat("v_%d", i);
      vertex_vars.push_back(c.int_const(name.c_str()));
      ++i;
    }
  }

  for (V v : vertices) {
    s.add(vertex_vars.at(vertex_index.at(v)) >= 0);
    s.add(vertex_vars.at(vertex_index.at(v)) < k);
  }

  for (V a : vertices) {
    for (V b : neighborhood(a)) {
      if (vertex_index.at(a) < vertex_index.at(b)) {
        s.add(vertex_vars.at(vertex_index.at(a)) !=
              vertex_vars.at(vertex_index.at(b)));
      }
    }
  }

  s.minimize(k);

  CHECK(s.check() == z3::sat);

  z3::model model = s.get_model();

  int64_t chromatic_number = LookupIntegerInZ3Model(model, "k").value();

  std::vector<absl::flat_hash_set<V>> result;
  result.resize(chromatic_number);

  for (const auto& [vertex, i] : vertex_index) {
    int64_t color =
        LookupIntegerInZ3Model(model, absl::StrFormat("v_%d", i)).value();
    result[color].insert(vertex);
  }

  return result;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_GRAPH_COLORING_H_
