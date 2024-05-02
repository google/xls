// Copyright 2020 The XLS Authors
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

#ifndef XLS_NETLIST_FIND_LOGIC_CLOUDS_H_
#define XLS_NETLIST_FIND_LOGIC_CLOUDS_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {
namespace rtl {

// Represents a cluster of cells.
class Cluster {
 public:
  // Adds a cell to this cluster.
  void Add(Cell* cell);

  // Helper for sorting cells after a cluster is built (by name) so a stable
  // order can be observed in callers.
  void SortCells();

  absl::Span<const Cell* const> terminating_flops() const {
    return terminating_flops_;
  }
  absl::Span<const Cell* const> other_cells() const { return other_cells_; }

 private:
  std::vector<Cell*> terminating_flops_;
  std::vector<Cell*> other_cells_;
};

// Finds the connected clusters of logic cells between flops in the given module
// and returns them. As noted in the definition of "Cluster", flops are
// associated with their "input side" subgraphs; any logic following a final
// flop stage has no associated "terminating flop".
//
// include_vacuous indicates whether a terminating flop with no connected logic
// (e.g. a layer of flops that flop input to the module) should be considered to
// be a cluster, or just discarded.
std::vector<Cluster> FindLogicClouds(const Module& module,
                                     bool include_vacuous = false);

// Converts the clusters to a string suitable for debugging/testing.
std::string ClustersToString(absl::Span<const Cluster> clusters);

}  // namespace rtl
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_FIND_LOGIC_CLOUDS_H_
