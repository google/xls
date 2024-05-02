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

#include "xls/netlist/find_logic_clouds.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/data_structures/union_find.h"

namespace xls {
namespace netlist {
namespace rtl {

void Cluster::Add(Cell* cell) {
  if (cell->kind() == CellKind::kFlop) {
    terminating_flops_.push_back(cell);
  } else {
    other_cells_.push_back(cell);
  }
}

void Cluster::SortCells() {
  auto cell_name_lt = [](const Cell* a, const Cell* b) {
    return a->name() < b->name();
  };
  std::sort(terminating_flops_.begin(), terminating_flops_.end(), cell_name_lt);
  std::sort(other_cells_.begin(), other_cells_.end(), cell_name_lt);
}

std::vector<Cluster> FindLogicClouds(const Module& module,
                                     bool include_vacuous) {
  UnionFind<Cell*> cell_to_uf;

  // Gets-or-adds the given cell to the UnionFind.
  auto get_uf = [&cell_to_uf](Cell* c) -> Cell* {
    cell_to_uf.Insert(c);
    return cell_to_uf.Find(c);
  };

  // Helper for debugging that counts the number of equivalence classes in
  // cell_to_uf.
  auto count_equivalence_classes = [&cell_to_uf]() -> int64_t {
    return cell_to_uf.GetRepresentatives().size();
  };

  for (auto& item : module.cells()) {
    Cell* cell = item.get();
    VLOG(4) << "Considering cell: " << cell->name();

    // Flop output connectivity is excluded from the equivalence class, so we
    // get partitions along flop (output) boundaries.
    if (cell->kind() == CellKind::kFlop) {
      // For flops we just make sure an equivalence class is present in the
      // map, in case it doesn't have any cells on the input side.
      get_uf(cell);
      VLOG(4) << "--- Now " << count_equivalence_classes()
              << " equivalence classes for " << cell_to_uf.size() << " cells.";
      continue;
    }

    for (auto& input : cell->inputs()) {
      VLOG(4) << "- Considering input net: " << input.netref->name();
      for (Cell* connected : input.netref->connected_cells()) {
        if (cell == connected) {
          continue;
        }
        if (connected->kind() == CellKind::kFlop) {
          VLOG(4) << absl::StreamFormat(
              "-- Cell is connected to flop cell %s on an input pin; not "
              "merging equivalence classes.",
              connected->name());
          continue;
        }
        VLOG(4) << absl::StreamFormat("-- Cell %s is connected to cell %s",
                                      cell->name(), connected->name());
        cell_to_uf.Union(get_uf(cell), get_uf(connected));
        VLOG(4) << "--- Now " << count_equivalence_classes()
                << " equivalence classes for " << cell_to_uf.size()
                << " cells.";
      }
    }

    for (const auto& iter : cell->outputs()) {
      NetRef output = iter.netref;
      VLOG(4) << "- Considering output net: " << output->name();
      for (Cell* connected : output->connected_cells()) {
        if (cell == connected) {
          continue;
        }
        VLOG(4) << absl::StreamFormat("-- Cell %s is connected to cell %s",
                                      cell->name(), connected->name());
        cell_to_uf.Union(get_uf(cell), get_uf(connected));
        VLOG(4) << "--- Now " << count_equivalence_classes()
                << " equivalence classes for " << cell_to_uf.size()
                << " cells.";
      }
    }
  }

  // Run through the cells and put them into clusters according to their
  // equivalence classes.
  absl::flat_hash_map<Cell*, Cluster> equivalence_set_to_cluster;
  for (auto& item : module.cells()) {
    Cell* cell = item.get();
    equivalence_set_to_cluster[get_uf(cell)].Add(cell);
  }

  // Put them into a vector and sort each cluster's internal cells for
  // determinism.
  std::vector<Cluster> clusters;
  for (auto& item : equivalence_set_to_cluster) {
    Cluster& cluster = item.second;
    if (!include_vacuous && (cluster.terminating_flops().size() == 1 &&
                             cluster.other_cells().empty())) {
      // Vacuous 'just a flop' cluster.
      continue;
    }
    cluster.SortCells();
    clusters.push_back(std::move(cluster));
  }

  // For convenience (for now) we convert the cell names to a string and rely on
  // string comparison for deterministic order.
  auto cells_to_str = [](absl::Span<const Cell* const> cells) {
    return absl::StrJoin(cells, ", ", [](std::string* out, const Cell* cell) {
      absl::StrAppend(out, cell->name());
    });
  };
  auto cluster_sort_lt = [cells_to_str](const Cluster& a, const Cluster& b) {
    return std::make_pair(cells_to_str(a.terminating_flops()),
                          cells_to_str(a.other_cells())) <
           std::make_pair(cells_to_str(b.terminating_flops()),
                          cells_to_str(b.other_cells()));
  };
  std::sort(clusters.begin(), clusters.end(), cluster_sort_lt);
  return clusters;
}

std::string ClustersToString(absl::Span<const Cluster> clusters) {
  std::string s;
  for (const Cluster& cluster : clusters) {
    absl::StrAppend(&s, "cluster {\n");
    for (const Cell* f : cluster.terminating_flops()) {
      absl::StrAppend(&s, "  terminating_flop: ", f->name(), "\n");
    }
    for (const Cell* f : cluster.other_cells()) {
      absl::StrAppend(&s, "  other_cell: ", f->name(), "\n");
    }
    absl::StrAppend(&s, "}\n");
  }
  return s;
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls
