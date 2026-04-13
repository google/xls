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

#include "xls/contrib/eco/ged.h"

#include <algorithm>
#include <chrono>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "xls/contrib/eco/lap_solver.h"

/* Algorithm based on:
Zeina Abu-Aisheh, Romain Raveaux, Jean-Yves Ramel, Patrick Martineau.
"An Exact Graph Edit Distance Algorithm for Solving Pattern Recognition
Problems." ICPRAM 2015. https://hal.archives-ouvertes.fr/hal-01168816 */

namespace ged {

RawCostMatrix BuildNodeCostMatrix(const XLSGraph& G1, const XLSGraph& G2,
                                  const NodeCostFunctions& costs) {
  VLOG(1) << "BuildNodeCostMatrix start: G1 nodes=" << G1.nodes.size()
          << " G2 nodes=" << G2.nodes.size();
  const int m = G1.nodes.size();
  const int n = G2.nodes.size();
  RawCostMatrix M(m + n, m + n);

  // Substitution block (m × n).
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      M.Set(i, j, costs.subst(G1.nodes[i], G2.nodes[j]));
    }
  }

  // Deletion block (m × m): diagonal = del cost, off-diagonal stays INF.
  for (int i = 0; i < m; ++i) {
    M.Set(i, n + i, costs.del(G1.nodes[i]));
  }

  // Insertion block (n × n): diagonal = ins cost, off-diagonal stays INF.
  for (int j = 0; j < n; ++j) {
    M.Set(m + j, j, costs.ins(G2.nodes[j]));
  }

  // Dummy block (n × m): all zeros.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      M.Set(m + i, n + j, 0);
    }
  }

  VLOG(1) << "BuildNodeCostMatrix done (rows=" << M.n_rows
          << ", cols=" << M.n_cols << ")";
  return M;
}

RawCostMatrix BuildEdgeCostMatrix(const XLSGraph& G1, const XLSGraph& G2,
                                  const EdgeCostFunctions& costs) {
  VLOG(1) << "BuildEdgeCostMatrix start: G1 edges=" << G1.edges.size()
          << " G2 edges=" << G2.edges.size();
  const int m = G1.edges.size();
  const int n = G2.edges.size();
  RawCostMatrix M(m + n, m + n);

  // Substitution block.
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      M.Set(i, j, costs.subst(G1.edges[i], G2.edges[j]));
    }
  }

  // Deletion block: diagonal only.
  for (int i = 0; i < m; ++i) {
    M.Set(i, n + i, costs.del(G1.edges[i]));
  }

  // Insertion block: diagonal only.
  for (int j = 0; j < n; ++j) {
    M.Set(m + j, j, costs.ins(G2.edges[j]));
  }

  // Dummy block: all zeros.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      M.Set(m + i, n + j, 0);
    }
  }

  VLOG(1) << "BuildEdgeCostMatrix done (rows=" << M.n_rows
          << ", cols=" << M.n_cols << ")";
  return M;
}

CostMatrix MakeCostMatrix(const RawCostMatrix& C, int m, int n,
                          std::vector<std::vector<double>>& dense) {
  CostMatrix result;
  result.C = C;

  const int size = std::max(C.n_rows, C.n_cols);
  if (dense.size() != static_cast<size_t>(size)) {
    dense.assign(size, std::vector<double>(size, (double)RawCostMatrix::INF));
  } else {
    for (auto& row : dense) {
      std::fill(row.begin(), row.end(), (double)RawCostMatrix::INF);
    }
  }

  // Direct copy from dense int grid to dense double buffer.
  for (int i = 0; i < C.n_rows; ++i) {
    for (int j = 0; j < C.n_cols; ++j) {
      dense[i][j] = static_cast<double>(C.Get(i, j));
    }
  }

  auto [row_ind, col_ind] = linear_sum_assignment(dense);

  result.lsa_row_ind.clear();
  result.lsa_col_ind.clear();
  result.lsa_row_ind.reserve(row_ind.size());
  result.lsa_col_ind.reserve(col_ind.size());

  int64_t ls_sum = 0;
  for (size_t p = 0; p < row_ind.size(); ++p) {
    int i = row_ind[p];
    int j = col_ind[p];
    result.lsa_row_ind.push_back(i);
    result.lsa_col_ind.push_back(j);
    double c = dense[i][j];
    if (c >= (double)RawCostMatrix::INF) {
      ls_sum = (int64_t)RawCostMatrix::INF;
      break;
    }
    ls_sum += (int64_t)c;
  }

  std::vector<int> subst_rows, subst_cols, dummy_idx;
  for (size_t k = 0; k < result.lsa_row_ind.size(); ++k) {
    int i = result.lsa_row_ind[k];
    int j = result.lsa_col_ind[k];
    if (i < m && j < n) {
      subst_rows.push_back(i);
      subst_cols.push_back(j);
    } else if (i >= m && j >= n) {
      dummy_idx.push_back(k);
    }
  }

  const int n_pairs = std::min((int)dummy_idx.size(), (int)subst_rows.size());
  for (int p = 0; p < n_pairs; ++p) {
    int idx = dummy_idx[p];
    result.lsa_row_ind[idx] = subst_cols[p] + m;
    result.lsa_col_ind[idx] = subst_rows[p] + n;
  }

  result.ls = (ls_sum > RawCostMatrix::INF) ? RawCostMatrix::INF : (int)ls_sum;
  VLOG(2) << "MakeCostMatrix: rows=" << result.C.n_rows
          << " cols=" << result.C.n_cols << " m=" << m << " n=" << n
          << " ls=" << result.ls
          << " assignments=" << result.lsa_row_ind.size();
  return result;
}

RawCostMatrix ExtractC(const RawCostMatrix& C, const std::vector<int>& i_vec,
                       const std::vector<int>& j_vec, int m, int n) {
  const int size = m + n;
  absl::flat_hash_set<int> i_set(i_vec.begin(), i_vec.end());
  absl::flat_hash_set<int> j_set(j_vec.begin(), j_vec.end());

  std::vector<bool> row_mask(size, false), col_mask(size, false);
  for (int k = 0; k < size; ++k) {
    bool in_i = i_set.contains(k);
    bool in_j = j_set.contains(k);
    bool in_i_shift = (k >= n) && i_set.contains(k - n);
    bool in_j_shift = (k >= m) && j_set.contains(k - m);
    row_mask[k] = in_i || in_j_shift;
    col_mask[k] = in_j || in_i_shift;
  }

  int new_rows = absl::c_count(row_mask, true);
  int new_cols = absl::c_count(col_mask, true);
  RawCostMatrix result(new_rows, new_cols);

  std::vector<int> row_inv(size, -1), col_inv(size, -1);
  int ridx = 0, cidx = 0;
  for (int k = 0; k < size; ++k) {
    if (row_mask[k]) {
      row_inv[k] = ridx++;
    }
  }
  for (int k = 0; k < size; ++k) {
    if (col_mask[k]) {
      col_inv[k] = cidx++;
    }
  }

  for (int r = 0; r < size; ++r) {
    if (!row_mask[r]) {
      continue;
    }
    for (int c = 0; c < size; ++c) {
      if (!col_mask[c]) {
        continue;
      }
      result.Set(row_inv[r], col_inv[c], C.Get(r, c));
    }
  }

  VLOG(3) << "ExtractC: m=" << m << " n=" << n << " keep_rows=" << result.n_rows
          << " keep_cols=" << result.n_cols;
  return result;
}

RawCostMatrix ReduceC(const RawCostMatrix& C, const std::vector<int>& i_vec,
                      const std::vector<int>& j_vec, int m, int n) {
  const int size = m + n;
  absl::flat_hash_set<int> i_set(i_vec.begin(), i_vec.end());
  absl::flat_hash_set<int> j_set(j_vec.begin(), j_vec.end());

  std::vector<bool> row_mask(size, false), col_mask(size, false);
  for (int k = 0; k < size; ++k) {
    bool in_i = i_set.contains(k);
    bool in_j = j_set.contains(k);
    bool in_i_shift = (k >= n) && i_set.contains(k - n);
    bool in_j_shift = (k >= m) && j_set.contains(k - m);
    row_mask[k] = !in_i && !in_j_shift;
    col_mask[k] = !in_j && !in_i_shift;
  }

  int new_rows = absl::c_count(row_mask, true);
  int new_cols = absl::c_count(col_mask, true);
  RawCostMatrix result(new_rows, new_cols);

  std::vector<int> row_inv(size, -1), col_inv(size, -1);
  int ridx = 0, cidx = 0;
  for (int k = 0; k < size; ++k) {
    if (row_mask[k]) {
      row_inv[k] = ridx++;
    }
  }
  for (int k = 0; k < size; ++k) {
    if (col_mask[k]) {
      col_inv[k] = cidx++;
    }
  }

  for (int r = 0; r < size; ++r) {
    if (!row_mask[r]) {
      continue;
    }
    for (int c = 0; c < size; ++c) {
      if (!col_mask[c]) {
        continue;
      }
      result.Set(row_inv[r], col_inv[c], C.Get(r, c));
    }
  }

  VLOG(3) << "ReduceC: m=" << m << " n=" << n << " keep_rows=" << result.n_rows
          << " keep_cols=" << result.n_cols;
  return result;
}

std::vector<int> ReduceInd(const std::vector<int>& ind,
                           const std::vector<int>& i) {
  if (i.empty()) {
    return ind;
  }

  absl::flat_hash_set<int> i_set(i.begin(), i.end());
  std::vector<int> rind;
  rind.reserve(ind.size());
  for (int k : ind) {
    if (!i_set.count(k)) {
      rind.push_back(k);
    }
  }

  std::vector<int> i_sorted(i.begin(), i.end());
  absl::c_sort(i_sorted);
  i_sorted.erase(std::unique(i_sorted.begin(), i_sorted.end()), i_sorted.end());

  for (int k : i_sorted) {
    for (int& idx : rind) {
      if (idx >= k) {
        idx--;
      }
    }
  }

  VLOG(3) << "ReduceInd: in=" << ind.size() << " remove=" << i.size()
          << " out=" << rind.size();
  return rind;
}

MatchEdgesResult MatchEdges(int u, int v, const std::vector<int>& pending_g,
                            const std::vector<int>& pending_h,
                            const CostMatrix& Ce, const XLSGraph& G1,
                            const XLSGraph& G2,
                            const std::vector<std::pair<int, int>>& matched_uv,
                            std::vector<std::vector<double>>& dense_buffer) {
  MatchEdgesResult result;

  const int M = (int)pending_g.size();
  const int N = (int)pending_h.size();

  std::vector<int> g_ind, h_ind;
  if (!matched_uv.empty()) {
    for (int i = 0; i < M; ++i) {
      const auto& e = G1.edges[pending_g[i]];
      int src = e.endpoints.first;
      int dst = e.endpoints.second;

      bool include = (u != -1 && src == u && dst == u);
      if (!include && u != -1) {
        for (const auto& pq : matched_uv) {
          int p = pq.first;
          if ((src == p && dst == u) || (src == u && dst == p) ||
              (src == p && dst == p)) {
            include = true;
            break;
          }
        }
      }
      if (include) g_ind.push_back(i);
    }

    for (int j = 0; j < N; ++j) {
      const auto& e = G2.edges[pending_h[j]];
      int src = e.endpoints.first;
      int dst = e.endpoints.second;

      bool include = (v != -1 && src == v && dst == v);
      if (!include && v != -1) {
        for (const auto& pq : matched_uv) {
          int q = pq.second;
          if ((src == q && dst == v) || (src == v && dst == q) ||
              (src == q && dst == q)) {
            include = true;
            break;
          }
        }
      }
      if (include) h_ind.push_back(j);
    }
  }

  const int m = (int)g_ind.size();
  const int n = (int)h_ind.size();

  if (m == 0 && n == 0) {
    result.localCe.C = RawCostMatrix();
    result.localCe.ls = 0;
    return result;
  }

  RawCostMatrix C = ExtractC(Ce.C, g_ind, h_ind, M, N);

  const bool directed = true;

  for (int k = 0; k < m; ++k) {
    const auto& eg = G1.edges[pending_g[g_ind[k]]];
    int g_src = eg.endpoints.first;
    int g_dst = eg.endpoints.second;

    for (int l = 0; l < n; ++l) {
      const auto& eh = G2.edges[pending_h[h_ind[l]]];
      int h_src = eh.endpoints.first;
      int h_dst = eh.endpoints.second;

      bool allowed = false;
      if (directed) {
        for (const auto& pq : matched_uv) {
          int p = pq.first, q = pq.second;
          if ((g_src == p && g_dst == u && h_src == q && h_dst == v) ||
              (g_src == u && g_dst == p && h_src == v && h_dst == q)) {
            allowed = true;
            break;
          }
        }
      } else {
        for (const auto& pq : matched_uv) {
          int p = pq.first, q = pq.second;
          bool g_ok =
              ((g_src == p && g_dst == u) || (g_src == u && g_dst == p));
          bool h_ok =
              ((h_src == q && h_dst == v) || (h_src == v && h_dst == q));
          if (g_ok && h_ok) {
            allowed = true;
            break;
          }
        }
      }
      if (allowed) continue;

      if ((u != -1 && g_src == u && g_dst == u)) continue;
      {
        bool g_self_matched = false;
        for (const auto& pq : matched_uv) {
          int p = pq.first;
          if (g_src == p && g_dst == p) {
            g_self_matched = true;
            break;
          }
        }
        if (g_self_matched) continue;
      }

      if ((v != -1 && h_src == v && h_dst == v)) continue;
      {
        bool h_self_matched = false;
        for (const auto& pq : matched_uv) {
          int q = pq.second;
          if (h_src == q && h_dst == q) {
            h_self_matched = true;
            break;
          }
        }
        if (h_self_matched) continue;
      }

      C.Set(k, l, RawCostMatrix::INF);
    }
  }

  CostMatrix localCe = MakeCostMatrix(C, m, n, dense_buffer);

  std::vector<std::pair<int, int>> ij;
  ij.reserve((size_t)std::max(m, n));

  for (size_t t = 0; t < localCe.lsa_row_ind.size(); ++t) {
    int k = localCe.lsa_row_ind[t];
    int l = localCe.lsa_col_ind[t];
    if (!(k < m || l < n)) continue;

    int left = (k < m) ? g_ind[k] : (M + h_ind[l]);
    int right = (l < n) ? h_ind[l] : (N + g_ind[k]);
    ij.emplace_back(left, right);
  }

  result.ij = std::move(ij);
  result.localCe = std::move(localCe);

  VLOG(3) << "MatchEdges: u=" << u << " v=" << v << " M=" << M << " N=" << N
          << " m=" << m << " n=" << n << " localCe.ls=" << result.localCe.ls
          << " pairs=" << result.ij.size();
  return result;
}

CostMatrix ReduceCe(const CostMatrix& Ce,
                    const std::vector<std::pair<int, int>>& ij, int m, int n,
                    std::vector<std::vector<double>>& dense_buffer) {
  if (ij.empty()) {
    return Ce;
  }
  std::vector<int> i_list, j_list;
  i_list.reserve(ij.size());
  j_list.reserve(ij.size());
  for (const auto& [i, j] : ij) {
    i_list.push_back(i);
    j_list.push_back(j);
  }
  absl::flat_hash_set<int> i_unique, j_unique;
  for (int i : i_list) {
    if (i < m) {
      i_unique.insert(i);
    }
  }
  for (int j : j_list) {
    if (j < n) {
      j_unique.insert(j);
    }
  }
  int m_i = m - static_cast<int>(i_unique.size());
  int n_j = n - static_cast<int>(j_unique.size());
  RawCostMatrix reduced = ReduceC(Ce.C, i_list, j_list, m, n);

  VLOG(3) << "ReduceCe: m=" << m << " n=" << n << " |ij|=" << ij.size()
          << " -> m_i=" << m_i << " n_j=" << n_j
          << " (unique_i=" << i_unique.size() << " unique_j=" << j_unique.size()
          << ")";
  return MakeCostMatrix(reduced, m_i, n_j, dense_buffer);
}

void GetEditOps(const std::vector<std::pair<int, int>>& matched_uv,
                const std::vector<int>& pending_u,
                const std::vector<int>& pending_v, const CostMatrix& Cv,
                const std::vector<int>& pending_g,
                const std::vector<int>& pending_h, const CostMatrix& Ce,
                const XLSGraph& G1, const XLSGraph& G2, int matched_cost,
                std::function<bool(int64_t)> prune,
                std::vector<std::vector<double>>& dense_Cv_buffer,
                std::vector<std::vector<double>>& dense_Ce_buffer,
                const std::function<void(EditOp&)>& op_callback) {
  const int m = pending_u.size();
  const int n = pending_v.size();
  int min_i = -1, min_j = -1;
  bool has_min = false;
  for (size_t idx = 0; idx < Cv.lsa_row_ind.size(); idx++) {
    int k = Cv.lsa_row_ind[idx];
    int l = Cv.lsa_col_ind[idx];
    if (!(k < m || l < n)) {
      continue;
    }
    if (!has_min ||
        std::pair<int, int>(k, l) < std::pair<int, int>(min_i, min_j)) {
      min_i = k;
      min_j = l;
      has_min = true;
    }
  }
  if (!has_min) {
    return;
  }

  std::vector<std::pair<int, int>> candidates;
  candidates.emplace_back(min_i, min_j);
  if (m <= n) {
    for (int t = 0; t < m + n; t++) {
      if (t != min_i && (t < m || t == m + min_j)) {
        candidates.emplace_back(t, min_j);
      }
    }
  } else {
    for (int t = 0; t < m + n; t++) {
      if (t != min_j && (t < n || t == n + min_i)) {
        candidates.emplace_back(min_i, t);
      }
    }
  }

  const int lsap_node_cost = Cv.C.Get(min_i, min_j);
  const int u_lsap = (min_i < m) ? pending_u[min_i] : -1;
  const int v_lsap = (min_j < n) ? pending_v[min_j] : -1;
  auto lsap_edge_res = MatchEdges(u_lsap, v_lsap, pending_g, pending_h, Ce, G1,
                                  G2, matched_uv, dense_Ce_buffer);
  CostMatrix Ce_xy_lsap = ReduceCe(Ce, lsap_edge_res.ij, pending_g.size(),
                                   pending_h.size(), dense_Ce_buffer);
  int lsap_bound =
      matched_cost + Cv.ls + lsap_edge_res.localCe.ls + Ce_xy_lsap.ls;

  if (!prune(lsap_bound)) {
    EditOp lsap_op;
    lsap_op.ij = {min_i, min_j};
    lsap_op.Cv_ij.C = ReduceC(Cv.C, {min_i}, {min_j}, m, n);
    lsap_op.Cv_ij.lsa_row_ind = ReduceInd(Cv.lsa_row_ind, {min_i, m + min_j});
    lsap_op.Cv_ij.lsa_col_ind = ReduceInd(Cv.lsa_col_ind, {min_j, n + min_i});
    lsap_op.Cv_ij.ls = Cv.ls - lsap_node_cost;
    lsap_op.xy = lsap_edge_res.ij;
    lsap_op.Ce_xy = Ce_xy_lsap;
    lsap_op.edit_cost = lsap_node_cost + lsap_edge_res.localCe.ls;
    op_callback(lsap_op);
  }

  std::vector<EditOp> other_ops;
  for (const auto& candidate : candidates) {
    auto [i, j] = candidate;
    if (i == min_i && j == min_j) {
      continue;
    }

    int u = (i < m) ? pending_u[i] : -1;
    int v = (j < n) ? pending_v[j] : -1;
    int node_cost = Cv.C.Get(i, j);
    if (prune(matched_cost + node_cost + Ce.ls)) {
      continue;
    }

    const int new_m = (i < m) ? m - 1 : m;
    const int new_n = (j < n) ? n - 1 : n;
    RawCostMatrix reduced_C = ReduceC(Cv.C, {i}, {j}, m, n);
    CostMatrix Cv_ij = MakeCostMatrix(reduced_C, new_m, new_n, dense_Cv_buffer);
    if (prune(matched_cost + node_cost + Cv_ij.ls + Ce.ls)) {
      continue;
    }

    auto edge_res = MatchEdges(u, v, pending_g, pending_h, Ce, G1, G2,
                               matched_uv, dense_Ce_buffer);
    if (prune(matched_cost + node_cost + Cv_ij.ls + edge_res.localCe.ls)) {
      continue;
    }

    CostMatrix Ce_xy = ReduceCe(Ce, edge_res.ij, pending_g.size(),
                                pending_h.size(), dense_Ce_buffer);
    if (prune(matched_cost + node_cost + Cv_ij.ls + edge_res.localCe.ls +
              Ce_xy.ls)) {
      continue;
    }

    VLOG(3) << "GetEditOps: i=" << i << " j=" << j << " node_cost=" << node_cost
            << " Cv_ij.ls=" << Cv_ij.ls << " localCe.ls=" << edge_res.localCe.ls
            << " Ce_xy.ls=" << Ce_xy.ls;

    EditOp op;
    op.ij = {i, j};
    op.Cv_ij = Cv_ij;
    op.xy = edge_res.ij;
    op.Ce_xy = Ce_xy;
    op.edit_cost = node_cost + edge_res.localCe.ls;
    other_ops.push_back(op);
  }

  absl::c_sort(other_ops, [](const EditOp& a, const EditOp& b) {
    int64_t bound_a = (int64_t)a.edit_cost + a.Cv_ij.ls + a.Ce_xy.ls;
    int64_t bound_b = (int64_t)b.edit_cost + b.Cv_ij.ls + b.Ce_xy.ls;
    return bound_a < bound_b;
  });

  for (auto& op : other_ops) {
    op_callback(op);
  }
}

struct SearchState {
  std::vector<std::pair<int, int>> matched_uv;
  std::vector<int> pending_u;
  std::vector<int> pending_v;
  std::vector<std::pair<int, int>> matched_gh;
  std::vector<int> pending_g;
  std::vector<int> pending_h;
  int matched_cost;
};

void GetEditPaths(SearchState& state, CostMatrix& Cv, CostMatrix& Ce,
                  const XLSGraph& G1, const XLSGraph& G2,
                  std::function<bool(int64_t)> prune, int& maxcost_value,
                  int& best_cost, GEDResult& best_result, int& expansion_count,
                  std::vector<std::vector<double>>& dense_Cv_buffer,
                  std::vector<std::vector<double>>& dense_Ce_buffer) {
  ++expansion_count;
  if (expansion_count <= 20 || expansion_count % 1000 == 0) {
    VLOG(3) << "Expansion " << expansion_count
            << " | pending_u=" << state.pending_u.size()
            << " pending_v=" << state.pending_v.size()
            << " pending_g=" << state.pending_g.size()
            << " pending_h=" << state.pending_h.size()
            << " matched_cost=" << state.matched_cost;
  }
  if (std::max(state.pending_u.size(), state.pending_v.size()) == 0) {
    maxcost_value = std::min(maxcost_value, state.matched_cost);
    if (state.matched_cost < best_cost) {
      best_cost = state.matched_cost;
      best_result.node_substitutions.clear();
      best_result.node_deletions.clear();
      best_result.node_insertions.clear();
      best_result.edge_substitutions.clear();
      best_result.edge_deletions.clear();
      best_result.edge_insertions.clear();
      for (const auto& [u, v] : state.matched_uv) {
        if (u != -1 && v != -1) {
          best_result.node_substitutions.emplace_back(u, v);
        } else if (u != -1) {
          best_result.node_deletions.push_back(u);
        } else if (v != -1) {
          best_result.node_insertions.push_back(v);
        }
      }
      for (const auto& [g, h] : state.matched_gh) {
        if (g != -1 && h != -1) {
          best_result.edge_substitutions.emplace_back(g, h);
        } else if (g != -1) {
          best_result.edge_deletions.push_back(g);
        } else if (h != -1) {
          best_result.edge_insertions.push_back(h);
        }
      }
      best_result.node_cost = 0;
      best_result.edge_cost = 0;
      best_result.total_cost = state.matched_cost;
      VLOG(1) << "Updated best solution: cost=" << best_cost
              << " node_subs=" << best_result.node_substitutions.size()
              << " node_dels=" << best_result.node_deletions.size()
              << " node_ins=" << best_result.node_insertions.size()
              << " edge_subs=" << best_result.edge_substitutions.size()
              << " edge_dels=" << best_result.edge_deletions.size()
              << " edge_ins=" << best_result.edge_insertions.size();
    }

    return;
  }
  auto edit_op_handler = [&](EditOp& op) {
    int64_t new_cost = (int64_t)state.matched_cost + (int64_t)op.edit_cost;
    int64_t branch_bound =
        new_cost + (int64_t)op.Cv_ij.ls + (int64_t)op.Ce_xy.ls;
    bool should_prune = prune(branch_bound);
    if (should_prune) {
      VLOG(2) << "Pruned branch: bound=" << branch_bound
              << " matched_cost=" << state.matched_cost
              << " edit_cost=" << op.edit_cost;
      return;
    }
    auto [i, j] = op.ij;
    int u = -1, v = -1;
    if (i < (int)state.pending_u.size()) {
      u = state.pending_u[i];
      state.pending_u.erase(state.pending_u.begin() + i);
    }
    if (j < (int)state.pending_v.size()) {
      v = state.pending_v[j];
      state.pending_v.erase(state.pending_v.begin() + j);
    }
    state.matched_uv.emplace_back(u, v);
    std::vector<std::pair<int, int>> xy_backup = op.xy;
    std::vector<int> sortedx, sortedy;
    std::vector<int> removed_g, removed_h;
    for (const auto& [x, y] : xy_backup) {
      int len_g = state.pending_g.size();
      int len_h = state.pending_h.size();
      int edge_g = (x < len_g) ? state.pending_g[x] : -1;
      int edge_h = (y < len_h) ? state.pending_h[y] : -1;
      state.matched_gh.emplace_back(edge_g, edge_h);
      sortedx.push_back(x);
      sortedy.push_back(y);
    }
    absl::c_sort(sortedx);
    absl::c_sort(sortedy);
    for (auto it = sortedx.rbegin(); it != sortedx.rend(); ++it) {
      int x = *it;
      if (x < (int)state.pending_g.size()) {
        removed_g.push_back(state.pending_g[x]);
        state.pending_g.erase(state.pending_g.begin() + x);
      } else {
        removed_g.push_back(-1);
      }
    }
    for (auto it = sortedy.rbegin(); it != sortedy.rend(); ++it) {
      int y = *it;
      if (y < (int)state.pending_h.size()) {
        removed_h.push_back(state.pending_h[y]);
        state.pending_h.erase(state.pending_h.begin() + y);
      } else {
        removed_h.push_back(-1);
      }
    }
    int old_cost = state.matched_cost;
    state.matched_cost =
        (int)std::min<int64_t>(new_cost, std::numeric_limits<int>::max());
    GetEditPaths(state, op.Cv_ij, op.Ce_xy, G1, G2, prune, maxcost_value,
                 best_cost, best_result, expansion_count, dense_Cv_buffer,
                 dense_Ce_buffer);
    state.matched_cost = old_cost;
    if (u != -1) {
      state.pending_u.insert(state.pending_u.begin() + i, u);
    }
    if (v != -1) {
      state.pending_v.insert(state.pending_v.begin() + j, v);
    }
    state.matched_uv.pop_back();
    std::reverse(removed_g.begin(), removed_g.end());
    std::reverse(removed_h.begin(), removed_h.end());
    for (size_t idx = 0; idx < sortedx.size(); idx++) {
      int x = sortedx[idx];
      int g = removed_g[idx];
      if (g != -1) {
        state.pending_g.insert(state.pending_g.begin() + x, g);
      }
    }
    for (size_t idx = 0; idx < sortedy.size(); idx++) {
      int y = sortedy[idx];
      int h = removed_h[idx];
      if (h != -1) {
        state.pending_h.insert(state.pending_h.begin() + y, h);
      }
    }
    for (size_t idx = 0; idx < xy_backup.size(); idx++) {
      state.matched_gh.pop_back();
    }
  };

  GetEditOps(state.matched_uv, state.pending_u, state.pending_v, Cv,
             state.pending_g, state.pending_h, Ce, G1, G2, state.matched_cost,
             prune, dense_Cv_buffer, dense_Ce_buffer, edit_op_handler);
}

AssignmentResult SolveLSAP(const RawCostMatrix& M, int m, int n,
                           std::vector<std::vector<double>>& dense) {
  AssignmentResult result;
  VLOG(2) << "SolveLSAP start: m=" << m << " n=" << n << " rows=" << M.n_rows
          << " cols=" << M.n_cols;

  const int size = std::max(M.n_rows, M.n_cols);
  if (dense.size() != static_cast<size_t>(size)) {
    dense.assign(size, std::vector<double>(size, (double)RawCostMatrix::INF));
  } else {
    for (auto& row : dense) {
      std::fill(row.begin(), row.end(), (double)RawCostMatrix::INF);
    }
  }
  for (int i = 0; i < M.n_rows; ++i) {
    for (int j = 0; j < M.n_cols; ++j) {
      dense[i][j] = static_cast<double>(M.Get(i, j));
    }
  }

  auto [row_ind, col_ind] = linear_sum_assignment(dense);

  int64_t total_cost = 0;
  for (size_t p = 0; p < row_ind.size(); ++p) {
    int i = row_ind[p];
    int j = col_ind[p];
    double c = dense[i][j];
    if (c >= (double)RawCostMatrix::INF) {
      result.cost = RawCostMatrix::INF;
      VLOG(1) << "SolveLSAP infeasible assignment detected";
      return result;
    }
    total_cost += (int64_t)c;

    if (i < m && j < n) {
      result.subs.emplace_back(i, j);
    } else if (i < m && j >= n) {
      result.dels.push_back(i);
    } else if (i >= m && j < n) {
      result.ins.push_back(j);
    }
  }

  result.cost = (total_cost > RawCostMatrix::INF)
                    ? RawCostMatrix::INF
                    : static_cast<int>(total_cost);
  VLOG(2) << "SolveLSAP done: cost=" << result.cost
          << " subs=" << result.subs.size() << " dels=" << result.dels.size()
          << " ins=" << result.ins.size();
  return result;
}

// Returns a pruning function that checks cost, timeout, and upper bound.
static std::function<bool(int64_t)> MakePruneFunction(
    const GEDOptions& options,
    const std::chrono::steady_clock::time_point& start_time, bool& timed_out,
    int maxcost_value, int& best_cost) {
  const double timeout = options.timeout;
  const bool optimal = options.optimal;
  const int upper_bound = options.upper_bound;
  const bool strictly_decreasing = options.strictly_decreasing;
  return [start_time, timeout, optimal, upper_bound, strictly_decreasing,
          maxcost_value, &timed_out, &best_cost](int64_t cost) -> bool {
    if (timeout == -1 && !optimal &&
        best_cost < RawCostMatrix::INF) {
      return true;
    }
    if (timeout > 0 && !optimal) {
      double elapsed = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start_time)
                           .count();
      if (elapsed > timeout) {
        timed_out = true;
        return true;
      }
    }
    return (cost > maxcost_value) ||
           (upper_bound != INT_MAX && cost > upper_bound) ||
           (strictly_decreasing && cost >= maxcost_value);
  };
}

GEDResult SolveGED(const XLSGraph& G1, const XLSGraph& G2,
                   const GEDOptions& options) {
  VLOG(1) << "SolveGED start: G1 nodes=" << G1.nodes.size()
          << " edges=" << G1.edges.size() << " | G2 nodes=" << G2.nodes.size()
          << " edges=" << G2.edges.size() << " timeout=" << options.timeout
          << " upper_bound=" << options.upper_bound
          << " strictly_decreasing=" << options.strictly_decreasing;
  GEDResult result;
  SearchState state;

  // Initialize pending node/edge lists.
  for (size_t i = 0; i < G1.nodes.size(); ++i) {
    state.pending_u.push_back((int)i);
  }
  for (size_t j = 0; j < G2.nodes.size(); ++j) {
    state.pending_v.push_back((int)j);
  }
  for (size_t i = 0; i < G1.edges.size(); ++i) {
    state.pending_g.push_back((int)i);
  }
  for (size_t j = 0; j < G2.edges.size(); ++j) {
    state.pending_h.push_back((int)j);
  }

  std::vector<std::vector<double>> dense_Cv_buffer;
  std::vector<std::vector<double>> dense_Ce_buffer;

  // Build cost matrices.
  RawCostMatrix M_node = BuildNodeCostMatrix(G1, G2, options.nodeCosts);
  CostMatrix Cv = MakeCostMatrix(M_node, (int)state.pending_u.size(),
                                 (int)state.pending_v.size(), dense_Cv_buffer);
  VLOG(3) << "MakeCostMatrix(kind=nodes): m=" << state.pending_u.size()
          << " n=" << state.pending_v.size() << " ls=" << Cv.ls
          << " pairs=" << Cv.lsa_row_ind.size();

  RawCostMatrix M_edge = BuildEdgeCostMatrix(G1, G2, options.edgeCosts);
  CostMatrix Ce = MakeCostMatrix(M_edge, (int)state.pending_g.size(),
                                 (int)state.pending_h.size(), dense_Ce_buffer);
  VLOG(3) << "MakeCostMatrix(kind=edges): m=" << state.pending_g.size()
          << " n=" << state.pending_h.size() << " ls=" << Ce.ls
          << " pairs=" << Ce.lsa_row_ind.size();
  VLOG(1) << "Initial LSAPs ready: pending_u=" << state.pending_u.size()
          << " pending_v=" << state.pending_v.size()
          << " pending_g=" << state.pending_g.size()
          << " pending_h=" << state.pending_h.size();

  state.matched_cost = 0;

  // Compute max cost limit.
  auto dense_sum = [](const auto& costs, const auto& elems1,
                      const auto& elems2) -> int64_t {
    int64_t subst_sum = 0, del_sum = 0, ins_sum = 0;
    for (const auto& a : elems1) {
      for (const auto& b : elems2) {
        subst_sum += costs.subst(a, b);
      }
    }
    for (const auto& a : elems1) {
      del_sum += costs.del(a);
    }
    for (const auto& b : elems2) {
      ins_sum += costs.ins(b);
    }
    return subst_sum + del_sum + ins_sum;
  };

  int64_t maxcost_ll = dense_sum(options.nodeCosts, G1.nodes, G2.nodes) +
                       dense_sum(options.edgeCosts, G1.edges, G2.edges) + 1;
  int maxcost_value =
      (maxcost_ll > RawCostMatrix::INF) ? RawCostMatrix::INF : (int)maxcost_ll;

  int best_cost = RawCostMatrix::INF;
  const auto start_time = std::chrono::steady_clock::now();
  bool timed_out = false;

  auto prune = MakePruneFunction(options, start_time, timed_out, maxcost_value,
                                 best_cost);
  int expansion_count = 0;

  GetEditPaths(state, Cv, Ce, G1, G2, prune, maxcost_value, best_cost, result,
               expansion_count, dense_Cv_buffer, dense_Ce_buffer);
  VLOG(1) << "Search completed after " << expansion_count
          << " expansions; best_cost=" << best_cost
          << " timed_out=" << (timed_out ? 1 : 0);

  if (best_cost == RawCostMatrix::INF) {
    if (timed_out) {
      VLOG(1) << "GED search timed out before finding a valid edit path.";
    } else {
      VLOG(1) << "GED search produced no valid result.";
    }
    return {};
  }

  // Accumulate final costs.
  auto sum_costs = [](int& total, const auto& pairs, const auto& elems1,
                      const auto& elems2, const auto& cost_fn) {
    for (const auto& [a, b] : pairs) {
      total += cost_fn.subst(elems1[a], elems2[b]);
    }
  };

  int node_cost = 0, edge_cost = 0;
  sum_costs(node_cost, result.node_substitutions, G1.nodes, G2.nodes,
            options.nodeCosts);
  for (int u : result.node_deletions) {
    node_cost += options.nodeCosts.del(G1.nodes[u]);
  }
  for (int v : result.node_insertions) {
    node_cost += options.nodeCosts.ins(G2.nodes[v]);
  }

  sum_costs(edge_cost, result.edge_substitutions, G1.edges, G2.edges,
            options.edgeCosts);
  for (int g : result.edge_deletions) {
    edge_cost += options.edgeCosts.del(G1.edges[g]);
  }
  for (int h : result.edge_insertions) {
    edge_cost += options.edgeCosts.ins(G2.edges[h]);
  }

  result.node_cost = node_cost;
  result.edge_cost = edge_cost;
  result.total_cost = node_cost + edge_cost;
  VLOG(1) << "Final GED cost total=" << result.total_cost
          << " node=" << result.node_cost << " edge=" << result.edge_cost
          << " node_subs=" << result.node_substitutions.size()
          << " edge_subs=" << result.edge_substitutions.size();
  return result;
}

}  // namespace ged
