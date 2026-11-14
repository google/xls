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

#ifndef XLS_DATA_STRUCTURES_Z3_GRAPH_COLORING_H_
#define XLS_DATA_STRUCTURES_Z3_GRAPH_COLORING_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "z3/src/api/c++/z3++.h"

namespace xls {

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

// Color the given graph optimally using Z3's optimizing SMT solver.
//
// `vertices` is the set of vertices of the graph.
// `neighborhood` is a function that, given a vertex in the graph, returns a set
// containing its neighbors. This is agnostic to graph representation.
//
// Returns a vector of sets of nodes, each of which represents a color class in
// a minimum coloring of the graph.
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

#endif  // XLS_DATA_STRUCTURES_Z3_GRAPH_COLORING_H_
