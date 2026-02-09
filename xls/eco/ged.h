#ifndef XLS_ECO_GED_H_
#define XLS_ECO_GED_H_

#include <climits>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>
#include <chrono>
#include <unordered_map>


#include "xls/eco/graph.h"

namespace ged
{
struct SparseCostMatrix {
  std::vector<int> row_ind;
  std::vector<int> col_ind;
  std::vector<int> cost;
  int n_rows = 0;
  int n_cols = 0;

  std::vector<int> row_map;
  std::vector<int> col_map;

  static constexpr int INF = std::numeric_limits<int>::max();

  explicit SparseCostMatrix(size_t reserve_nnz = 0) {
    if (reserve_nnz) {
      row_ind.reserve(reserve_nnz);
      col_ind.reserve(reserve_nnz);
      cost.reserve(reserve_nnz);
    }
  }

  inline void add(int i, int j, int c) {
    row_ind.push_back(i);
    col_ind.push_back(j);
    cost.push_back(c);
  }

  inline int get(int i, int j) const {
    for (size_t k = 0; k < row_ind.size(); ++k)
      if (row_ind[k] == i && col_ind[k] == j) return cost[k];
    return INF;
  }

  inline void set(int i, int j, int c) {
    for (size_t k = 0; k < row_ind.size(); ++k) {
      if (row_ind[k] == i && col_ind[k] == j) {
        cost[k] = c;
        return;
      }
    }
    add(i, j, c);
  }

  inline size_t nnz() const { return cost.size(); }
};

struct CostMatrix {
  SparseCostMatrix C;
  int ls;
  std::vector<int> lsa_row_ind;
  std::vector<int> lsa_col_ind;
};

CostMatrix MakeCostMatrix(const SparseCostMatrix& C, int m, int n,
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
                std::function<bool(long long)> prune,
                std::vector<std::vector<double>>& dense_Cv_buffer,
                std::vector<std::vector<double>>& dense_Ce_buffer,
                const std::function<void(EditOp&)>& op_callback);

struct AssignmentResult {
  int cost = 0;
  std::vector<std::pair<int, int>> subs;
  std::vector<int> dels;
  std::vector<int> ins;
};

AssignmentResult SolveLSAP(const SparseCostMatrix& M, int m, int n,
                           std::vector<std::vector<double>>& dense_buffer);

struct NodeCostFunctions {
  int (*subst)(const XLSNode&, const XLSNode&);
  int (*ins)(const XLSNode&);
  int (*del)(const XLSNode&);
};

struct EdgeCostFunctions {
  int (*subst)(const XLSEdge&, const XLSEdge&);
  int (*ins)(const XLSEdge&);
  int (*del)(const XLSEdge&);
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

struct SearchState;

void GetEditPaths(SearchState& state, CostMatrix& Cv, CostMatrix& Ce,
                  const XLSGraph& G1, const XLSGraph& G2,
                  std::function<bool(long long)> prune, int& maxcost_value,
                  int& best_cost, GEDResult& best_result, int& expansion_count,
                  std::vector<std::vector<double>>& dense_Cv_buffer,
                  std::vector<std::vector<double>>& dense_Ce_buffer);
inline std::function<bool(long long)>
MakePruneFunction(const GEDOptions& options,
                  const std::chrono::steady_clock::time_point& start_time,
  bool& timed_out, int maxcost_value, int& best_cost) {
  return [&](long long cost) -> bool {
    if (options.timeout == -1 && !options.optimal &&
        best_cost < SparseCostMatrix::INF) {
      // Best cost already set: exit immediately when timeout == -1.
      return true;
    }
    if (options.timeout > 0 && !options.optimal) {
      double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - start_time).count();
      if (elapsed > options.timeout) {
        timed_out = true;
        return true;
      }
    }
    return (cost > maxcost_value) ||
           (options.upper_bound != INT_MAX && cost > options.upper_bound) ||
           (options.strictly_decreasing && cost >= maxcost_value);
  };
}

GEDResult SolveGED(const XLSGraph& graph1, const XLSGraph& graph2,
                   const GEDOptions& options);
}  // namespace ged
#endif  // XLS_ECO_GED_H_
