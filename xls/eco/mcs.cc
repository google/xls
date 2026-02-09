#include "xls/eco/mcs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>

#include "absl/log/log.h"

/* Algorithm inspired by:
Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks Lakshmanan, and Reynold Cheng.
"Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking
Approach." Proc. ACM Manag. Data 3, 3, Article 160 (2025).
https://doi.org/10.1145/3725404 */

namespace mcs {

static bool g_stop = false;
static int g_best_size = 0;
static int g_mcs_cutoff = -1;
static int g_total_nodes = 0;

namespace {
inline std::uint64_t EncodePair(int u, int v) {
  return (static_cast<std::uint64_t>(static_cast<unsigned int>(u)) << 32) |
         static_cast<std::uint64_t>(static_cast<unsigned int>(v));
}

inline bool IsForbidden(int u, int v, const ForbiddenSet& D) {
  return D.find(EncodePair(u, v)) != D.end();
}

inline int PickBranchVertex(const CandidateSet& C) {
  if (C.empty()) return -1;
  int u_pick = -1;
  size_t best = std::numeric_limits<size_t>::max();
  for (const auto& kv : C) {
    if (!kv.second.empty() && kv.second.size() < best) {
      best = kv.second.size();
      u_pick = kv.first;
    }
  }
  if (u_pick == -1) u_pick = C.begin()->first;  // fallback
  return u_pick;
}

CandidateSet RefineCandidates(const State& S, const CandidateSet& C, int u,
                              int v, const XLSGraph& Q, const XLSGraph& G) {
  CandidateSet R;

  auto u_out = Q.get_outgoing_neighbors(u);
  auto u_in = Q.get_incoming_neighbors(u);
  auto v_out = G.get_outgoing_neighbors(v);
  auto v_in = G.get_incoming_neighbors(v);

  std::unordered_set<int> u_neighbors(u_out.begin(), u_out.end());
  u_neighbors.insert(u_in.begin(), u_in.end());

  std::unordered_set<int> v_neighbors(v_out.begin(), v_out.end());
  v_neighbors.insert(v_in.begin(), v_in.end());

  for (const auto& [x, candidates] : C) {
    if (x == u) continue;
    std::vector<int> new_candidates;

    for (int y : candidates) {
      if (y == v) continue;

      bool consistent = true;

      // Check edges between u and x vs v and y (Forward u->x vs v->y)
      auto q_ux = Q.get_edges_between(u, x);
      auto g_vy = G.get_edges_between(v, y);
      std::sort(q_ux.begin(), q_ux.end());
      std::sort(g_vy.begin(), g_vy.end());
      if (q_ux != g_vy) consistent = false;

      // Check edges between x and u vs y and v (Backward x->u vs y->v)
      if (consistent) {
        auto q_xu = Q.get_edges_between(x, u);
        auto g_yv = G.get_edges_between(y, v);
        std::sort(q_xu.begin(), q_xu.end());
        std::sort(g_yv.begin(), g_yv.end());
        if (q_xu != g_yv) consistent = false;
      }

      if (consistent) {
        for (auto [u2, v2] : S) {
          // Check edges between u2 and x vs v2 and y
          auto q_u2x = Q.get_edges_between(u2, x);
          auto g_v2y = G.get_edges_between(v2, y);
          std::sort(q_u2x.begin(), q_u2x.end());
          std::sort(g_v2y.begin(), g_v2y.end());
          if (q_u2x != g_v2y) {
            consistent = false;
            break;
          }

          // Check edges between x and u2 vs y and v2
          auto q_xu2 = Q.get_edges_between(x, u2);
          auto g_yv2 = G.get_edges_between(y, v2);
          std::sort(q_xu2.begin(), q_xu2.end());
          std::sort(g_yv2.begin(), g_yv2.end());
          if (q_xu2 != g_yv2) {
            consistent = false;
            break;
          }
        }
      }

      if (consistent) new_candidates.push_back(y);
    }

    if (!new_candidates.empty()) R[x] = std::move(new_candidates);
  }

  return R;
}

CandidateSet BuildInitialCandidates(const XLSGraph& Q, const XLSGraph& G) {
  CandidateSet Psi;

  for (int u = 0; u < static_cast<int>(Q.nodes.size()); ++u) {
    const auto& nu = Q.nodes[u];
    std::vector<int> candidates;

    for (int v = 0; v < static_cast<int>(G.nodes.size()); ++v) {
      const auto& nv = G.nodes[v];

      if (nu.label != nv.label) continue;

      if (nu.signature != nv.signature) continue;

      if (Q.get_incoming_neighbors(u).size() !=
              G.get_incoming_neighbors(v).size() ||
          Q.get_outgoing_neighbors(u).size() !=
              G.get_outgoing_neighbors(v).size())
        continue;

      candidates.push_back(v);
    }

    if (!candidates.empty()) Psi[u] = std::move(candidates);
  }

  return Psi;
}

int UpperBound(const State& S, const CandidateSet& C, const XLSGraph& Q,
               const XLSGraph& G) {
  int ub = static_cast<int>(S.size());

  for (const auto& [u, cand_list] : C) {
    int consistent = 0;

    for (int v : cand_list) {
      bool compatible = true;

      // Check edge multiplicity consistency with all matched pairs
      for (const auto& [u2, v2] : S) {
        int q_edges = Q.count_edges(u, u2);
        int g_edges = G.count_edges(v, v2);

        if (q_edges != g_edges) {
          compatible = false;
          break;
        }
      }

      if (compatible) consistent++;
    }

    ub += std::min(consistent, (int)cand_list.size());
  }

  return ub;
}

inline std::string indent(int depth) { return std::string(depth * 2, ' '); }

void RRSplitRec(State& S, const CandidateSet& C, const ForbiddenSet& D,
                const XLSGraph& Q, const XLSGraph& G, State& S_best,
                int depth) {
  if (g_stop) return;

  VLOG(3) << indent(depth) << "[enter] depth=" << depth
          << ", partial_size=" << S.size() << ", best_size=" << g_best_size
          << ", #vars_with_cands=" << C.size();

  // Update best solution if current is better
  if ((int)S.size() > g_best_size) {
    S_best = S;
    g_best_size = (int)S.size();
    VLOG(2) << indent(depth) << "[best] improved best_size=" << g_best_size;
    
    // Check cutoff: stop if remaining unmatched nodes <= cutoff threshold
    if (g_mcs_cutoff >= 0) {
      int remaining = g_total_nodes - g_best_size;
      if (remaining <= g_mcs_cutoff) {
        g_stop = true;
        VLOG(0) << "[cutoff] MCS cutoff reached: remaining nodes (" << remaining
                << ") <= cutoff (" << g_mcs_cutoff << "), stopping search";
        return;
      }
    }
    
    if (g_best_size == (int)Q.nodes.size()) {
      g_stop = true;
      VLOG(2) << indent(depth)
              << "[optimal] found complete mapping, stop flag set";
    }
  }

  if (C.empty()) {
    VLOG(3) << indent(depth) << "[leaf] no candidate variables remain";
    return;
  }

  // Upper bound pruning (Lemma 5)
  int ub = UpperBound(S, C, Q, G);
  if (ub <= g_best_size) {
    VLOG(3) << indent(depth) << "[prune] upper_bound=" << ub
            << " <= best_size=" << g_best_size;
    return;
  }

  // Choose branching variable (fail-first heuristic)
  int u = PickBranchVertex(C);
  if (u == -1) {
    VLOG(3) << indent(depth) << "[leaf] no variable to branch on";
    return;
  }

  size_t u_cands = 0;
  if (auto it = C.find(u); it != C.end()) u_cands = it->second.size();

  VLOG(3) << indent(depth) << "[branch] u=" << u << " (" << Q.nodes[u].name
          << ") with |C(u)|=" << u_cands;

  // Maximality-based reduction (Equation 13 from paper)
  if (u_cands > 0) {
    for (int v_try : C.at(u)) {
      if (Q.get_incoming_neighbors(u).size() !=
              G.get_incoming_neighbors(v_try).size() ||
          Q.get_outgoing_neighbors(u).size() !=
              G.get_outgoing_neighbors(v_try).size())
        continue;

      bool maximal_ok = true;

      for (const auto& [u2, v2] : S) {
        auto q_fwd = Q.get_edges_between(u, u2);
        auto g_fwd = G.get_edges_between(v_try, v2);
        std::sort(q_fwd.begin(), q_fwd.end());
        std::sort(g_fwd.begin(), g_fwd.end());

        auto q_bwd = Q.get_edges_between(u2, u);
        auto g_bwd = G.get_edges_between(v2, v_try);
        std::sort(q_bwd.begin(), q_bwd.end());
        std::sort(g_bwd.begin(), g_bwd.end());

        if (q_fwd != g_fwd || q_bwd != g_bwd) {
          maximal_ok = false;
          break;
        }
      }

      if (maximal_ok) {
        CandidateSet C2 = RefineCandidates(S, C, u, v_try, Q, G);

        bool empty = false;
        for (const auto& kv : C2)
          if (kv.second.empty()) {
            empty = true;
            break;
          }
        if (empty) continue;

        ForbiddenSet D2 = D;
        for (int vy : C.at(u))
          if (vy != v_try) D2.insert(EncodePair(u, vy));

        State S2 = S;
        S2.emplace_back(u, v_try);
        RRSplitRec(S2, C2, D2, Q, G, S_best, depth + 1);
        VLOG(3) << indent(depth) << "[max-red] pruned other candidates for u";
        return;  // Early exit after maximal match
      }
    }
  }

  const auto& cand = C.at(u);

  for (int v : cand) {
    if (IsForbidden(u, v, D)) {
      VLOG(3) << indent(depth) << "  - skip v=" << v << " (" << G.nodes[v].name
              << ") [forbidden]";
      continue;
    }

    VLOG(3) << indent(depth) << "  - try v=" << v << " (" << G.nodes[v].name
            << ")";
    auto C_refined = RefineCandidates(S, C, u, v, Q, G);
    VLOG(3) << indent(depth) << "    refined: |C'|=" << C_refined.size()
            << ", removed u from candidate set";

    bool empty = false;
    for (const auto& kv : C_refined)
      if (kv.second.empty()) {
        empty = true;
        break;
      }
    if (empty) continue;

    auto D_next = D;

    S.push_back({u, v});
    RRSplitRec(S, C_refined, D_next, Q, G, S_best, depth + 1);
    S.pop_back();

    VLOG(3) << indent(depth) << "  - backtrack v=" << v;
  }

  VLOG(3) << indent(depth) << "[exit] depth=" << depth;
}

}  // namespace

MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff) {
  const auto start = std::chrono::steady_clock::now();
  VLOG(0) << "MCS start: G1 nodes=" << graph1.nodes.size()
          << " edges=" << graph1.edges.size()
          << " | G2 nodes=" << graph2.nodes.size()
          << " edges=" << graph2.edges.size()
          << " | cutoff=" << mcs_cutoff;

  g_stop = false;
  g_best_size = 0;
  g_mcs_cutoff = mcs_cutoff;
  g_total_nodes = graph1.nodes.size();

  auto initial_candidates = BuildInitialCandidates(graph1, graph2);
  VLOG(1) << "Initial candidate variables=" << initial_candidates.size();

  // Check cutoff condition before starting search
  int total_nodes = graph1.nodes.size();
  if (mcs_cutoff >= 0 && total_nodes <= mcs_cutoff) {
    VLOG(0) << "MCS cutoff: total nodes (" << total_nodes
            << ") <= cutoff (" << mcs_cutoff
            << "), skipping MCS (will use GED for all nodes)";
    MCSResult empty_result;
    empty_result.size = 0;
    for (int i = 0; i < total_nodes; ++i) {
      empty_result.unmatched_g1.push_back(i);
    }
    for (int j = 0; j < (int)graph2.nodes.size(); ++j) {
      empty_result.unmatched_g2.push_back(j);
    }
    return empty_result;
  }

  State S, S_best;
  RRSplitRec(S, initial_candidates, {}, graph1, graph2, S_best, 0);

  MCSResult result;
  result.mapping = S_best;
  result.size = S_best.size();

  // Check if cutoff threshold is met
  int remaining_nodes = total_nodes - result.size;
  if (mcs_cutoff >= 0 && remaining_nodes <= mcs_cutoff) {
    VLOG(0) << "MCS cutoff reached: remaining unmatched nodes (" << remaining_nodes
            << ") <= cutoff (" << mcs_cutoff << "), stopping MCS early";
  }

  VLOG(1) << "MCS found with " << result.size << " matched nodes";

  std::unordered_set<int> matched_g1, matched_g2;
  for (const auto& [u, v] : S_best) {
    matched_g1.insert(u);
    matched_g2.insert(v);
    VLOG(2) << "  " << graph1.nodes[u].name << " -> " << graph2.nodes[v].name;
  }

  for (int i = 0; i < (int)graph1.nodes.size(); ++i) {
    if (matched_g1.find(i) == matched_g1.end()) {
      result.unmatched_g1.push_back(i);
      VLOG(3) << "  " << graph1.nodes[i].name << " -> unmatched";
    }
  }

  for (int j = 0; j < (int)graph2.nodes.size(); ++j) {
    if (matched_g2.find(j) == matched_g2.end()) {
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
std::unordered_map<int, int> GetBoundaryNodes(const MCSResult& mcs,
                                              const XLSGraph& graph1,
                                              const XLSGraph& graph2) {
  std::unordered_map<int, int> boundary_nodes;
  if (mcs.mapping.empty()) return boundary_nodes;

  std::unordered_set<int> mcs_g1, mcs_g2;
  for (const auto& [u, v] : mcs.mapping) {
    mcs_g1.insert(u);
    mcs_g2.insert(v);
  }

  for (const auto& [u, v] : mcs.mapping) {
    bool is_boundary = false;

    for (int n1 : graph1.get_neighbors(u)) {
      if (!mcs_g1.count(n1)) {
        is_boundary = true;
        break;
      }
    }

    if (!is_boundary) {
      for (int n2 : graph2.get_neighbors(v)) {
        if (!mcs_g2.count(n2)) {
          is_boundary = true;
          break;
        }
      }
    }

    if (is_boundary) boundary_nodes[u] = v;
  }

  VLOG(1) << "Identified " << boundary_nodes.size()
          << " boundary nodes (MCS nodes with ≥1 non-MCS neighbor)";
  return boundary_nodes;
}

}  // namespace mcs
