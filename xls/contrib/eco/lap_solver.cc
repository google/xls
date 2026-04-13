// Copyright 2025 The XLS Authors
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

// Shortest augmenting path algorithm for the rectangular linear sum
// assignment problem (LSAP).  Based on:
//   D.F. Crouse, "On implementing 2D rectangular assignment algorithms,"
//   IEEE Trans. Aerospace and Electronic Systems, 52(4):1679-1696, 2016.

#include "xls/contrib/eco/lap_solver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "absl/algorithm/container.h"

std::pair<std::vector<int>, std::vector<int>> linear_sum_assignment(
    const std::vector<std::vector<double>>& cost_matrix) {
  if (cost_matrix.empty() || cost_matrix[0].empty()) return {{}, {}};

  int nr = static_cast<int>(cost_matrix.size());
  int nc = static_cast<int>(cost_matrix[0].size());

  // Transpose so nr <= nc (every row gets assigned).
  const bool transposed = nc < nr;
  if (transposed) std::swap(nr, nc);

  std::vector<double> cost(nr * nc);
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nc; ++j)
      cost[i * nc + j] = transposed ? cost_matrix[j][i] : cost_matrix[i][j];

  std::vector<double> u(nr, 0), v(nc, 0), shortest(nc);
  std::vector<int> path(nc, -1), col4row(nr, -1), row4col(nc, -1);
  std::vector<int> remaining(nc);
  std::vector<int> sr_stamp(nr, -1), sc_stamp(nc, -1);
  std::vector<int> sr_list;
  std::vector<int> sc_list;
  sr_list.reserve(nr);
  sc_list.reserve(nc);

  for (int cur_row = 0; cur_row < nr; ++cur_row) {
    // Reverse-fill so constant-cost matrices yield identity assignment.
    for (int it = 0; it < nc; ++it) remaining[it] = nc - it - 1;
    int num_remaining = nc;

    sr_list.clear();
    sc_list.clear();
    std::fill(shortest.begin(), shortest.end(), INFINITY);

    double min_val = 0;
    int sink = -1;
    int i = cur_row;

    while (sink == -1) {
      if (sr_stamp[i] != cur_row) {
        sr_stamp[i] = cur_row;
        sr_list.push_back(i);
      }

      const double row_bias = min_val - u[i];
      const double* cost_row = &cost[i * nc];
      double lowest = INFINITY;
      int idx = -1;

      for (int it = 0; it < num_remaining; ++it) {
        const int j = remaining[it];
        const double rc = row_bias + cost_row[j] - v[j];
        if (rc < shortest[j]) {
          path[j] = i;
          shortest[j] = rc;
        }
        if (shortest[j] < lowest ||
            (shortest[j] == lowest && row4col[j] == -1)) {
          lowest = shortest[j];
          idx = it;
        }
      }

      min_val = lowest;
      if (min_val == INFINITY) return {{}, {}};

      int j = remaining[idx];
      if (row4col[j] == -1) {
        sink = j;
      } else {
        i = row4col[j];
      }
      if (sc_stamp[j] != cur_row) {
        sc_stamp[j] = cur_row;
        sc_list.push_back(j);
      }
      remaining[idx] = remaining[--num_remaining];
    }

    // Dual update.
    u[cur_row] += min_val;
    for (int r : sr_list) {
      if (r != cur_row) u[r] += min_val - shortest[col4row[r]];
    }
    for (int c : sc_list) {
      v[c] -= min_val - shortest[c];
    }

    // Augment.
    int j = sink;
    while (true) {
      int r = path[j];
      row4col[j] = r;
      std::swap(col4row[r], j);
      if (r == cur_row) break;
    }
  }

  // Build output.
  std::vector<int> row_ind(nr), col_ind(nr);
  if (transposed) {
    std::vector<int> order(nr);
    std::iota(order.begin(), order.end(), 0);
    absl::c_sort(order, [&](int a, int b) { return col4row[a] < col4row[b]; });
    for (int k = 0; k < nr; ++k) {
      row_ind[k] = col4row[order[k]];
      col_ind[k] = order[k];
    }
  } else {
    for (int r = 0; r < nr; ++r) {
      row_ind[r] = r;
      col_ind[r] = col4row[r];
    }
  }

  return {row_ind, col_ind};
}
