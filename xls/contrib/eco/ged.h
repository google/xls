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

#ifndef XLS_ECO_GED_H_
#define XLS_ECO_GED_H_

#include <climits>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "xls/contrib/eco/graph.h"

namespace ged {

// Dense cost matrix stored in row-major order.
struct RawCostMatrix {
  std::vector<int> data;
  int n_rows = 0;
  int n_cols = 0;

  static constexpr int INF = std::numeric_limits<int>::max();

  RawCostMatrix() = default;
  RawCostMatrix(int rows, int cols)
      : data(static_cast<size_t>(rows) * cols, INF),  // Initialize with INF
        n_rows(rows),
        n_cols(cols) {}

  int Get(int i, int j) const { return data[i * n_cols + j]; }
  void Set(int i, int j, int c) { data[i * n_cols + j] = c; }
};

struct CostMatrix {
  RawCostMatrix C;
  int ls = 0;
  std::vector<int> lsa_row_ind;
  std::vector<int> lsa_col_ind;
};

CostMatrix MakeCostMatrix(const RawCostMatrix& C, int m, int n,
                          std::vector<std::vector<double>>& dense_buffer);

struct MatchEdgesResult {
  std::vector<std::pair<int, int>> ij;
  CostMatrix localCe;
};

MatchEdgesResult MatchEdges(int u, int v, const std::vector<int>& pending_g,
                            const std::vector<int>& pending_h,
                            const CostMatrix& Ce, const XLSGraph& G1,
                            const XLSGraph& G2,
                            const std::vector<std::pair<int, int>>& matched_uv,
                            std::vector<std::vector<double>>& dense_buffer);

CostMatrix ReduceCe(const CostMatrix& Ce,
                    const std::vector<std::pair<int, int>>& ij, int m, int n,
                    std::vector<std::vector<double>>& dense_buffer);

struct EditOp {
  std::pair<int, int> ij;
  CostMatrix Cv_ij;
  std::vector<std::pair<int, int>> xy;
  CostMatrix Ce_xy;
  int edit_cost;
};

void GetEditOps(const std::vector<std::pair<int, int>>& matched_uv,
                const std::vector<int>& pending_u,
                const std::vector<int>& pending_v, const CostMatrix& Cv,
                const std::vector<int>& pending_g,
                const std::vector<int>& pending_h, const CostMatrix& Ce,
                const XLSGraph& G1, const XLSGraph& G2, int matched_cost,
                std::function<bool(int64_t)> prune,
                std::vector<std::vector<double>>& dense_Cv_buffer,
                std::vector<std::vector<double>>& dense_Ce_buffer,
                const std::function<void(EditOp&)>& op_callback);

struct AssignmentResult {
  int cost = 0;
  std::vector<std::pair<int, int>> subs;
  std::vector<int> dels;
  std::vector<int> ins;
};

AssignmentResult SolveLSAP(const RawCostMatrix& M, int m, int n,
                           std::vector<std::vector<double>>& dense_buffer);

struct NodeCostFunctions {
  absl::AnyInvocable<int(const XLSNode&, const XLSNode&) const> subst;
  absl::AnyInvocable<int(const XLSNode&) const> ins;
  absl::AnyInvocable<int(const XLSNode&) const> del;
};

struct EdgeCostFunctions {
  absl::AnyInvocable<int(const XLSEdge&, const XLSEdge&) const> subst;
  absl::AnyInvocable<int(const XLSEdge&) const> ins;
  absl::AnyInvocable<int(const XLSEdge&) const> del;
};

struct GEDOptions {
  NodeCostFunctions nodeCosts;
  EdgeCostFunctions edgeCosts;

  double timeout;
  bool optimal;
  int upper_bound = INT_MAX;
  bool strictly_decreasing = true;
};

struct GEDResult {
  std::vector<std::pair<int, int>> node_substitutions;
  std::vector<int> node_insertions;
  std::vector<int> node_deletions;

  std::vector<std::pair<int, int>> edge_substitutions;
  std::vector<int> edge_insertions;
  std::vector<int> edge_deletions;

  int node_cost = INT_MAX;
  int edge_cost = INT_MAX;
  int total_cost = INT_MAX;
};

GEDResult SolveGED(const XLSGraph& graph1, const XLSGraph& graph2,
                   const GEDOptions& options);

}  // namespace ged

#endif  // XLS_ECO_GED_H_
