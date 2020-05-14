// Copyright 2020 Google LLC
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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
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
  // We need stable pointers for the UnionFind nodes because they hold raw
  // pointers to each other.
  absl::flat_hash_map<Cell*, std::unique_ptr<UnionFind<absl::monostate>>>
      cell_to_uf;

  // Gets-or-creates a UnionFind node for cell c.
  auto get_uf = [&cell_to_uf](Cell* c) -> UnionFind<absl::monostate>* {
    auto it = cell_to_uf.find(c);
    if (it == cell_to_uf.end()) {
      auto value = absl::make_unique<UnionFind<absl::monostate>>();
      auto* ptr = value.get();
      cell_to_uf[c] = std::move(value);
      return ptr;
    }
    return it->second.get();
  };

  // Helper for debugging that counts the number of equivalence classes in
  // cell_to_uf.
  auto count_equivalence_classes = [&cell_to_uf]() -> int64 {
    absl::flat_hash_set<UnionFind<absl::monostate>*> classes;
    for (auto& item : cell_to_uf) {
      classes.insert(item.second->FindRoot());
    }
    return classes.size();
  };

  for (auto& item : module.cells()) {
    Cell* cell = item.get();
    XLS_VLOG(4) << "Considering cell: " << cell->name();

    // Flop output connectivity is excluded from the equivalence class, so we
    // get partitions along flop (output) boundaries.
    if (cell->kind() == CellKind::kFlop) {
      // For flops we just make sure an equivalence class is present in the
      // map, in case it doesn't have any cells on the input side.
      (void)get_uf(cell);
      XLS_VLOG(4) << "--- Now " << count_equivalence_classes()
                  << " equivalence classes for " << cell_to_uf.size()
                  << " cells.";
      continue;
    }

    for (auto& input : cell->inputs()) {
      XLS_VLOG(4) << "- Considering input net: " << input.netref->name();
      for (Cell* connected : input.netref->connected_cells()) {
        if (cell == connected) {
          continue;
        }
        if (connected->kind() == CellKind::kFlop) {
          XLS_VLOG(4) << absl::StreamFormat(
              "-- Cell is connected to flop cell %s on an input pin; not "
              "merging equivalence classes.",
              connected->name());
          continue;
        }
        XLS_VLOG(4) << absl::StreamFormat("-- Cell %s is connected to cell %s",
                                          cell->name(), connected->name());
        get_uf(cell)->Merge(get_uf(connected));
        XLS_VLOG(4) << "--- Now " << count_equivalence_classes()
                    << " equivalence classes for " << cell_to_uf.size()
                    << " cells.";
      }
    }

    for (const auto& iter : cell->outputs()) {
      NetRef output = iter.netref;
      XLS_VLOG(4) << "- Considering output net: " << output->name();
      for (Cell* connected : output->connected_cells()) {
        if (cell == connected) {
          continue;
        }
        XLS_VLOG(4) << absl::StreamFormat("-- Cell %s is connected to cell %s",
                                          cell->name(), connected->name());
        get_uf(cell)->Merge(get_uf(connected));
        XLS_VLOG(4) << "--- Now " << count_equivalence_classes()
                    << " equivalence classes for " << cell_to_uf.size()
                    << " cells.";
      }
    }
  }

  // Run through the cells and put them into clusters according to their
  // equivalence classes.
  absl::flat_hash_map<UnionFind<absl::monostate>*, Cluster>
      equivalence_set_to_cluster;
  for (auto& item : module.cells()) {
    Cell* cell = item.get();
    UnionFind<absl::monostate>* equivalence_class = get_uf(cell)->FindRoot();
    equivalence_set_to_cluster[equivalence_class].Add(cell);
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
