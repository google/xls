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

#include "xls/eco/mcs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"

/* Implementation of the Maximum Common Subgraph (MCS) algorithm based on the
work: Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks Lakshmanan, and Reynold Cheng.
"Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking
Approach." Proc. ACM Manag. Data 3, 3, Article 160 (2025).
https://doi.org/10.1145/3725404 */

namespace mcs {
namespace {

struct SearchContext {
  bool stop = false;
  int best_size = 0;
  int mcs_cutoff = -1;
  int total_nodes = 0;
  State best_mapping;
};

inline std::uint64_t EncodePair(int u, int v) {
  return (static_cast<std::uint64_t>(static_cast<unsigned int>(u)) << 32) |
         static_cast<std::uint64_t>(static_cast<unsigned int>(v));
}

inline bool IsForbidden(int u, int v, const ForbiddenSet& forbidden) {
  return forbidden.find(EncodePair(u, v)) != forbidden.end();
}

inline int PickBranchVertex(const CandidateSet& candidates) {
  if (candidates.empty()) {
    return -1;
  }

  int chosen_u = -1;
  size_t best_size = std::numeric_limits<size_t>::max();

  for (const auto& [u, cand_list] : candidates) {
    if (!cand_list.empty() && cand_list.size() < best_size) {
      best_size = cand_list.size();
      chosen_u = u;
    }
  }

  if (chosen_u == -1) {
    chosen_u = candidates.begin()->first;
  }
  return chosen_u;
}

inline std::string Indent(int depth) { return std::string(depth * 2, ' '); }

bool MaybeUpdateBest(const State& current, const XLSGraph& query,
                     SearchContext& ctx, int depth) {
  const int current_size = static_cast<int>(current.size());
  if (current_size <= ctx.best_size) {
    return false;
  }

  ctx.best_mapping = current;
  ctx.best_size = current_size;

  VLOG(2) << Indent(depth) << "[best] improved best_size=" << ctx.best_size;

  if (ctx.mcs_cutoff >= 0) {
    const int remaining = ctx.total_nodes - ctx.best_size;
    if (remaining <= ctx.mcs_cutoff) {
      ctx.stop = true;
      VLOG(0) << "[cutoff] MCS cutoff reached: remaining nodes (" << remaining
              << ") <= cutoff (" << ctx.mcs_cutoff << "), stopping search";
      return true;
    }
  }

  if (ctx.best_size == static_cast<int>(query.nodes.size())) {
    ctx.stop = true;
    VLOG(2) << Indent(depth)
            << "[optimal] found complete mapping, stop flag set";
  }

  return true;
}

CandidateSet RefineCandidates(const State& partial_mapping,
                              const CandidateSet& candidates, int matched_u,
                              int matched_v, const XLSGraph& query,
                              const XLSGraph& target) {
  CandidateSet refined;

  for (const auto& [x, cand_list] : candidates) {
    if (x == matched_u) {
      continue;
    }

    std::vector<int> new_candidates;
    new_candidates.reserve(cand_list.size());

    for (int y : cand_list) {
      if (y == matched_v) {
        continue;
      }

      bool consistent = true;

      // Check edges between matched_u and x vs matched_v and y.
      auto q_ux = query.get_edges_between(matched_u, x);
      auto g_vy = target.get_edges_between(matched_v, y);
      std::sort(q_ux.begin(), q_ux.end());
      std::sort(g_vy.begin(), g_vy.end());
      if (q_ux != g_vy) {
        consistent = false;
      }

      // Check edges between x and matched_u vs y and matched_v.
      if (consistent) {
        auto q_xu = query.get_edges_between(x, matched_u);
        auto g_yv = target.get_edges_between(y, matched_v);
        std::sort(q_xu.begin(), q_xu.end());
        std::sort(g_yv.begin(), g_yv.end());
        if (q_xu != g_yv) {
          consistent = false;
        }
      }

      if (consistent) {
        for (const auto& [u2, v2] : partial_mapping) {
          auto q_u2x = query.get_edges_between(u2, x);
          auto g_v2y = target.get_edges_between(v2, y);
          std::sort(q_u2x.begin(), q_u2x.end());
          std::sort(g_v2y.begin(), g_v2y.end());
          if (q_u2x != g_v2y) {
            consistent = false;
            break;
          }

          auto q_xu2 = query.get_edges_between(x, u2);
          auto g_yv2 = target.get_edges_between(y, v2);
          std::sort(q_xu2.begin(), q_xu2.end());
          std::sort(g_yv2.begin(), g_yv2.end());
          if (q_xu2 != g_yv2) {
            consistent = false;
            break;
          }
        }
      }

      if (consistent) {
        new_candidates.push_back(y);
      }
    }

    if (!new_candidates.empty()) {
      refined[x] = std::move(new_candidates);
    }
  }

  return refined;
}

CandidateSet BuildInitialCandidates(const XLSGraph& query,
                                    const XLSGraph& target) {
  CandidateSet initial_candidates;

  for (int u = 0; u < static_cast<int>(query.nodes.size()); ++u) {
    const auto& query_node = query.nodes[u];
    std::vector<int> candidates;

    for (int v = 0; v < static_cast<int>(target.nodes.size()); ++v) {
      const auto& target_node = target.nodes[v];

      if (query_node.label != target_node.label) {
        continue;
      }
      if (query_node.signature != target_node.signature) {
        continue;
      }

      if (query.get_incoming_neighbors(u).size() !=
              target.get_incoming_neighbors(v).size() ||
          query.get_outgoing_neighbors(u).size() !=
              target.get_outgoing_neighbors(v).size()) {
        continue;
      }

      candidates.push_back(v);
    }

    if (!candidates.empty()) {
      initial_candidates[u] = std::move(candidates);
    }
  }

  return initial_candidates;
}

int UpperBound(const State& partial_mapping, const CandidateSet& candidates,
               const XLSGraph& query, const XLSGraph& target) {
  int upper_bound = static_cast<int>(partial_mapping.size());

  for (const auto& [u, cand_list] : candidates) {
    int consistent_count = 0;

    for (int v : cand_list) {
      bool compatible = true;

      for (const auto& [u2, v2] : partial_mapping) {
        const int q_edges = query.count_edges(u, u2);
        const int g_edges = target.count_edges(v, v2);
        if (q_edges != g_edges) {
          compatible = false;
          break;
        }
      }

      if (compatible) {
        ++consistent_count;
      }
    }

    upper_bound +=
        std::min(consistent_count, static_cast<int>(cand_list.size()));
  }

  return upper_bound;
}

bool HasEmptyCandidateList(const CandidateSet& candidates) {
  for (const auto& [u, cand_list] : candidates) {
    if (cand_list.empty()) {
      return true;
    }
  }
  return false;
}

void PopulateMatchedEdges(const XLSGraph& graph1, const XLSGraph& graph2,
                          const State& mapping, MCSResult& result) {
  std::vector<int> g1_to_g2(graph1.nodes.size(), -1);
  for (const auto& [u, v] : mapping) {
    if (u >= 0 && u < static_cast<int>(graph1.nodes.size()) && v >= 0 &&
        v < static_cast<int>(graph2.nodes.size())) {
      g1_to_g2[u] = v;
    }
  }

  std::vector<char> used_g2_edges(graph2.edges.size(), 0);
  result.edge_mapping.clear();
  result.edge_mapping.reserve(
      std::min(graph1.edges.size(), graph2.edges.size()));

  for (int e1_idx = 0; e1_idx < static_cast<int>(graph1.edges.size());
       ++e1_idx) {
    const auto& e1 = graph1.edges[e1_idx];
    const int src_g1 = e1.endpoints.first;
    const int dst_g1 = e1.endpoints.second;

    if (src_g1 < 0 || src_g1 >= static_cast<int>(g1_to_g2.size()) ||
        dst_g1 < 0 || dst_g1 >= static_cast<int>(g1_to_g2.size())) {
      continue;
    }

    const int src_g2 = g1_to_g2[src_g1];
    const int dst_g2 = g1_to_g2[dst_g1];
    if (src_g2 < 0 || dst_g2 < 0) {
      continue;
    }

    auto it = graph2.node_edges.find(src_g2);
    if (it == graph2.node_edges.end()) {
      continue;
    }

    for (int e2_idx : it->second) {
      if (e2_idx < 0 || e2_idx >= static_cast<int>(graph2.edges.size()) ||
          used_g2_edges[e2_idx]) {
        continue;
      }

      const auto& e2 = graph2.edges[e2_idx];
      if (e2.endpoints.first == src_g2 && e2.endpoints.second == dst_g2 &&
          e2.index == e1.index) {
        used_g2_edges[e2_idx] = 1;
        result.edge_mapping.emplace_back(e1_idx, e2_idx);
        break;
      }
    }
  }

  result.edge_size = static_cast<int>(result.edge_mapping.size());
}

void RRSplitRec(State& partial_mapping, const CandidateSet& candidates,
                const ForbiddenSet& forbidden, const XLSGraph& query,
                const XLSGraph& target, SearchContext& ctx, int depth) {
  if (ctx.stop) {
    return;
  }

  VLOG(3) << Indent(depth) << "[enter] depth=" << depth
          << ", partial_size=" << partial_mapping.size()
          << ", best_size=" << ctx.best_size
          << ", #vars_with_cands=" << candidates.size();

  MaybeUpdateBest(partial_mapping, query, ctx, depth);
  if (ctx.stop) {
    return;
  }

  if (candidates.empty()) {
    VLOG(3) << Indent(depth) << "[leaf] no candidate variables remain";
    return;
  }

  const int ub = UpperBound(partial_mapping, candidates, query, target);
  if (ub <= ctx.best_size) {
    VLOG(3) << Indent(depth) << "[prune] upper_bound=" << ub
            << " <= best_size=" << ctx.best_size;
    return;
  }

  const int u = PickBranchVertex(candidates);
  if (u == -1) {
    VLOG(3) << Indent(depth) << "[leaf] no variable to branch on";
    return;
  }

  size_t u_cands = 0;
  if (auto it = candidates.find(u); it != candidates.end()) {
    u_cands = it->second.size();
  }

  VLOG(3) << Indent(depth) << "[branch] u=" << u << " (" << query.nodes[u].name
          << ") with |C(u)|=" << u_cands;

  // Maximality-based reduction.
  if (u_cands > 0) {
    for (int v_try : candidates.at(u)) {
      if (query.get_incoming_neighbors(u).size() !=
              target.get_incoming_neighbors(v_try).size() ||
          query.get_outgoing_neighbors(u).size() !=
              target.get_outgoing_neighbors(v_try).size()) {
        continue;
      }

      bool maximal_ok = true;

      for (const auto& [u2, v2] : partial_mapping) {
        auto q_fwd = query.get_edges_between(u, u2);
        auto g_fwd = target.get_edges_between(v_try, v2);
        std::sort(q_fwd.begin(), q_fwd.end());
        std::sort(g_fwd.begin(), g_fwd.end());

        auto q_bwd = query.get_edges_between(u2, u);
        auto g_bwd = target.get_edges_between(v2, v_try);
        std::sort(q_bwd.begin(), q_bwd.end());
        std::sort(g_bwd.begin(), g_bwd.end());

        if (q_fwd != g_fwd || q_bwd != g_bwd) {
          maximal_ok = false;
          break;
        }
      }

      if (maximal_ok) {
        CandidateSet refined = RefineCandidates(partial_mapping, candidates, u,
                                                v_try, query, target);
        if (HasEmptyCandidateList(refined)) {
          continue;
        }

        ForbiddenSet next_forbidden = forbidden;
        for (int other_v : candidates.at(u)) {
          if (other_v != v_try) {
            next_forbidden.insert(EncodePair(u, other_v));
          }
        }

        State next_mapping = partial_mapping;
        next_mapping.emplace_back(u, v_try);
        RRSplitRec(next_mapping, refined, next_forbidden, query, target, ctx,
                   depth + 1);

        VLOG(3) << Indent(depth) << "[max-red] pruned other candidates for u";
        return;
      }
    }
  }

  const auto& cand_list = candidates.at(u);

  for (int v : cand_list) {
    if (ctx.stop) {
      return;
    }

    if (IsForbidden(u, v, forbidden)) {
      VLOG(3) << Indent(depth) << "  - skip v=" << v << " ("
              << target.nodes[v].name << ") [forbidden]";
      continue;
    }

    VLOG(3) << Indent(depth) << "  - try v=" << v << " ("
            << target.nodes[v].name << ")";

    CandidateSet refined =
        RefineCandidates(partial_mapping, candidates, u, v, query, target);

    VLOG(3) << Indent(depth) << "    refined: |C'|=" << refined.size()
            << ", removed u from candidate set";

    if (HasEmptyCandidateList(refined)) {
      continue;
    }

    ForbiddenSet next_forbidden = forbidden;

    partial_mapping.push_back({u, v});
    RRSplitRec(partial_mapping, refined, next_forbidden, query, target, ctx,
               depth + 1);
    partial_mapping.pop_back();

    VLOG(3) << Indent(depth) << "  - backtrack v=" << v;
  }

  VLOG(3) << Indent(depth) << "[exit] depth=" << depth;
}

}  // namespace

MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff) {
  const auto start = std::chrono::steady_clock::now();
  VLOG(0) << "MCS start: G1 nodes=" << graph1.nodes.size()
          << " edges=" << graph1.edges.size()
          << " | G2 nodes=" << graph2.nodes.size()
          << " edges=" << graph2.edges.size() << " | cutoff=" << mcs_cutoff;

  const int total_nodes = static_cast<int>(graph1.nodes.size());

  auto initial_candidates = BuildInitialCandidates(graph1, graph2);
  VLOG(1) << "Initial candidate variables=" << initial_candidates.size();

  if (mcs_cutoff >= 0 && total_nodes <= mcs_cutoff) {
    VLOG(0) << "MCS cutoff: total nodes (" << total_nodes << ") <= cutoff ("
            << mcs_cutoff << "), skipping MCS (will use GED for all nodes)";

    MCSResult empty_result;
    empty_result.size = 0;

    for (int i = 0; i < total_nodes; ++i) {
      empty_result.unmatched_g1.push_back(i);
    }
    for (int j = 0; j < static_cast<int>(graph2.nodes.size()); ++j) {
      empty_result.unmatched_g2.push_back(j);
    }
    return empty_result;
  }

  SearchContext ctx{
      .stop = false,
      .best_size = 0,
      .mcs_cutoff = mcs_cutoff,
      .total_nodes = total_nodes,
      .best_mapping = {},
  };

  State partial_mapping;
  RRSplitRec(partial_mapping, initial_candidates, ForbiddenSet{}, graph1,
             graph2, ctx, 0);

  MCSResult result;
  result.mapping = ctx.best_mapping;
  result.size = static_cast<int>(ctx.best_mapping.size());
  PopulateMatchedEdges(graph1, graph2, ctx.best_mapping, result);

  const int remaining_nodes = total_nodes - result.size;
  if (mcs_cutoff >= 0 && remaining_nodes <= mcs_cutoff) {
    VLOG(0) << "MCS cutoff reached: remaining unmatched nodes ("
            << remaining_nodes << ") <= cutoff (" << mcs_cutoff
            << "), stopping MCS early";
  }

  VLOG(1) << "MCS found with " << result.size << " matched nodes";
  VLOG(1) << "MCS matched edges=" << result.edge_size;

  absl::flat_hash_set<int> matched_g1;
  absl::flat_hash_set<int> matched_g2;
  for (const auto& [u, v] : ctx.best_mapping) {
    matched_g1.insert(u);
    matched_g2.insert(v);
    VLOG(2) << "  " << graph1.nodes[u].name << " -> " << graph2.nodes[v].name;
  }

  for (int i = 0; i < static_cast<int>(graph1.nodes.size()); ++i) {
    if (!matched_g1.contains(i)) {
      result.unmatched_g1.push_back(i);
      VLOG(3) << "  " << graph1.nodes[i].name << " -> unmatched";
    }
  }

  for (int j = 0; j < static_cast<int>(graph2.nodes.size()); ++j) {
    if (!matched_g2.contains(j)) {
      result.unmatched_g2.push_back(j);
      VLOG(3) << "  " << graph2.nodes[j].name << " -> unmatched";
    }
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();

  VLOG(0) << "MCS done: size=" << result.size
          << " unmatched_g1=" << result.unmatched_g1.size()
          << " unmatched_g2=" << result.unmatched_g2.size()
          << " time_ms=" << elapsed;

  return result;
}

absl::flat_hash_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                               const XLSGraph& graph1,
                                               const XLSGraph& graph2) {
  absl::flat_hash_map<int, int> boundary_nodes;
  if (mcs.mapping.empty()) {
    return boundary_nodes;
  }

  absl::flat_hash_set<int> mcs_g1;
  absl::flat_hash_set<int> mcs_g2;
  for (const auto& [u, v] : mcs.mapping) {
    mcs_g1.insert(u);
    mcs_g2.insert(v);
  }

  for (const auto& [u, v] : mcs.mapping) {
    bool is_boundary = false;

    for (int n1 : graph1.get_neighbors(u)) {
      if (!mcs_g1.contains(n1)) {
        is_boundary = true;
        break;
      }
    }

    if (!is_boundary) {
      for (int n2 : graph2.get_neighbors(v)) {
        if (!mcs_g2.contains(n2)) {
          is_boundary = true;
          break;
        }
      }
    }

    if (is_boundary) {
      boundary_nodes[u] = v;
    }
  }

  VLOG(1) << "Identified " << boundary_nodes.size()
          << " boundary nodes (MCS nodes with ≥1 non-MCS neighbor)";
  return boundary_nodes;
}

}  // namespace mcs