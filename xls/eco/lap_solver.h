#pragma once
#include <vector>

// Optimized LAP solver using AVX2
std::pair<std::vector<int>, std::vector<int>>
linear_sum_assignment(const std::vector<std::vector<double>> &cost_matrix);
