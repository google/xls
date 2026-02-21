#include "xls/eco/lap_solver.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "scipy_lap/rectangular_lsap.h"

std::pair<std::vector<int>, std::vector<int>> linear_sum_assignment(
    const std::vector<std::vector<double>> &cost_matrix) {
  if (cost_matrix.empty() || cost_matrix[0].empty()) {
    return {{}, {}};
  }

  int m = static_cast<int>(cost_matrix.size());
  int n = static_cast<int>(cost_matrix[0].size());

  // Create contiguous cost matrix (row-major, no padding needed!)
  std::vector<double> assign_cost(m * n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      assign_cost[i * n + j] = cost_matrix[i][j];
    }
  }

  // Allocate solution arrays
  std::vector<int64_t> row_indices(std::min(m, n));
  std::vector<int64_t> col_indices(std::min(m, n));

  // Call scipy's rectangular LAP solver
  int result = solve_rectangular_linear_sum_assignment(
      m, n, assign_cost.data(), false, row_indices.data(), col_indices.data());

  if (result != 0) {
    // Handle error

    return {{}, {}};
  }

  // Convert to int vectors (scipy returns int64_t)
  std::vector<int> row_ind(row_indices.begin(), row_indices.end());
  std::vector<int> col_ind(col_indices.begin(), col_indices.end());

  return {row_ind, col_ind};
}
