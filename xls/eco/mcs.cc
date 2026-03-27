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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"

// Maximum Common Subgraph search based on the RRSplit paper:
//   Kaiqiang Yu, Kaixin Wang, Cheng Long, Laks Lakshmanan, and Reynold Cheng.
//   "Fast Maximum Common Subgraph Search: A Redundancy-Reduced Backtracking
//   Approach." Proc. ACM Manag. Data 3, 3, Article 160 (2025).
//   https://doi.org/10.1145/3725404
//
// The paper studies unlabeled, undirected simple graphs. XLS ECO operates on
// directed, labeled IR graphs with edge indices, so this implementation keeps
// the paper's search structure (candidate partitions, exclusion set D,
// maximality reduction, and tighter upper bound) while strengthening the local
// compatibility tests. In practice:
//   * adjacency/non-adjacency checks become exact forward/backward
//     edge-index-profile checks;
//   * initial candidate classes are filtered by exact node labels, signatures,
//     and in/out degrees; and
//   * the pipeline adds optional cutoff / timeout / plateau controls to bound
//   MCS runtime before GED.

namespace mcs {
namespace {

using CandidateClass = internal::CandidateClass;
using CandidatePartition = internal::CandidatePartition;
using EquivalenceInfo = internal::EquivalenceInfo;
using ExclusionSet = internal::ExclusionSet;

// Search-global state around Algorithm 2, (S* from the paper).
struct SearchContext {
  bool stop = false;
  bool mcs_optimal = true;
  int best_size = 0;
  int mcs_cutoff = -1;
  int mcs_timeout_sec = -1;
  int plateau_search_node_threshold = 0;
  int total_nodes = 0;
  int search_nodes_since_best = 0;
  std::chrono::steady_clock::time_point start_time;
  State best_mapping;
};

// Stricter compatibility key for initial candidate partitioning compared to the
// paper, this is required for the directed/labeled/indexed nature of XLS IR
// graphs.
struct CompatibilityKey {
  std::size_t label = 0;
  std::size_t signature = 0;
  int indegree = 0;
  int outdegree = 0;

  bool operator==(const CompatibilityKey& other) const {
    return label == other.label && signature == other.signature &&
           indegree == other.indegree && outdegree == other.outdegree;
  }
};

struct CompatibilityKeyHash {
  std::size_t operator()(const CompatibilityKey& key) const {
    std::size_t seed = key.label;
    seed ^= key.signature + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    seed ^= static_cast<std::size_t>(key.indegree) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    seed ^= static_cast<std::size_t>(key.outdegree) + 0x9e3779b97f4a7c15ULL +
            (seed << 6) + (seed >> 2);
    return seed;
  }
};

// Directed/indexed generalization of the paper's "same adjacency to the chosen
// pair" relation. For simple graphs, Equation (4) and Equation (13) only need
// adjacency vs. non-adjacency; here we preserve exact forward/backward edge
// indices so refinement and maximality checks remain valid for XLS graphs.
struct RelationProfile {
  std::vector<int> forward;
  std::vector<int> backward;

  bool operator==(const RelationProfile& other) const {
    return forward == other.forward && backward == other.backward;
  }

  bool operator<(const RelationProfile& other) const {
    return std::tie(forward, backward) <
           std::tie(other.forward, other.backward);
  }
};

// Avoid repeated refinement and maximality checks
class RelationProfileCache {
 public:
  explicit RelationProfileCache(const XLSGraph& graph) : graph_(graph) {}

  RelationProfile Get(int from, int to) const {
    const std::uint64_t key = EncodePair(from, to);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }

    RelationProfile profile{
        .forward = graph_.get_edges_between(from, to),
        .backward = graph_.get_edges_between(to, from),
    };
    std::sort(profile.forward.begin(), profile.forward.end());
    std::sort(profile.backward.begin(), profile.backward.end());
    cache_.insert({key, profile});
    return profile;
  }

 private:
  static std::uint64_t EncodePair(int u, int v) {
    return (static_cast<std::uint64_t>(static_cast<unsigned int>(u)) << 32) |
           static_cast<std::uint64_t>(static_cast<unsigned int>(v));
  }

  const XLSGraph& graph_;
  mutable absl::flat_hash_map<std::uint64_t, RelationProfile> cache_;
};

inline std::string Indent(int depth) {
  return "[d=" + std::to_string(depth) + "] ";
}

CompatibilityKey MakeCompatibilityKey(const XLSGraph& graph, int node_index) {
  const XLSNode& node = graph.nodes[node_index];
  // This initial compatibility gate is stricter than the paper: we require
  // exact node label, exact full-neighborhood signature, and exact in/out
  // degree equality before two nodes can share a candidate class. We keep
  // this stronger filter for the XLS ECO pipeline to make the MCS->GED handoff
  // conservative, which helps avoid correctness issues and edge-case matching
  // bugs on directed/indexed-edge IR graphs.
  return CompatibilityKey{
      .label = node.label,
      .signature = node.signature,
      .indegree =
          static_cast<int>(graph.get_incoming_neighbors(node_index).size()),
      .outdegree =
          static_cast<int>(graph.get_outgoing_neighbors(node_index).size()),
  };
}

std::vector<CompatibilityKey> BuildCompatibilityKeys(const XLSGraph& graph) {
  std::vector<CompatibilityKey> keys;
  keys.reserve(graph.nodes.size());
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    keys.push_back(MakeCompatibilityKey(graph, i));
  }
  return keys;
}

// Canonical ordering for candidate classes so branching and tests are stable.
void SortCandidatePartition(CandidatePartition& candidates) {
  std::sort(candidates.begin(), candidates.end(),
            [](const CandidateClass& lhs, const CandidateClass& rhs) {
              const int lhs_q = lhs.query_nodes.empty()
                                    ? std::numeric_limits<int>::max()
                                    : lhs.query_nodes.front();
              const int rhs_q = rhs.query_nodes.empty()
                                    ? std::numeric_limits<int>::max()
                                    : rhs.query_nodes.front();
              if (lhs_q != rhs_q) {
                return lhs_q < rhs_q;
              }
              const int lhs_t = lhs.target_nodes.empty()
                                    ? std::numeric_limits<int>::max()
                                    : lhs.target_nodes.front();
              const int rhs_t = rhs.target_nodes.empty()
                                    ? std::numeric_limits<int>::max()
                                    : rhs.target_nodes.front();
              if (lhs_t != rhs_t) {
                return lhs_t < rhs_t;
              }
              if (lhs.query_nodes.size() != rhs.query_nodes.size()) {
                return lhs.query_nodes.size() < rhs.query_nodes.size();
              }
              return lhs.target_nodes.size() < rhs.target_nodes.size();
            });
}

// Builds the initial candidate partition P(C). The paper starts from V_Q x V_G
// and refines recursively (Section 3 / Equation (4)); here we pre-split that
// space by CompatibilityKey so only obviously compatible XLS nodes ever
// share a candidate class.
CandidatePartition BuildInitialCandidatePartition(const XLSGraph& query,
                                                  const XLSGraph& target) {
  struct Bucket {
    std::vector<int> query_nodes;
    std::vector<int> target_nodes;
  };

  absl::flat_hash_map<CompatibilityKey, Bucket, CompatibilityKeyHash> buckets;

  for (int u = 0; u < static_cast<int>(query.nodes.size()); ++u) {
    buckets[MakeCompatibilityKey(query, u)].query_nodes.push_back(u);
  }
  for (int v = 0; v < static_cast<int>(target.nodes.size()); ++v) {
    buckets[MakeCompatibilityKey(target, v)].target_nodes.push_back(v);
  }

  CandidatePartition candidates;
  candidates.reserve(buckets.size());
  for (auto& [key, bucket] : buckets) {
    if (bucket.query_nodes.empty() || bucket.target_nodes.empty()) {
      continue;
    }
    candidates.push_back(CandidateClass{
        .query_nodes = std::move(bucket.query_nodes),
        .target_nodes = std::move(bucket.target_nodes),
    });
  }

  SortCandidatePartition(candidates);
  return candidates;
}

// Definition 5 / Equation (6), adapted from simple-graph adjacency to the
// stronger relation profiles required by directed, indexed XLS graphs.
bool AreStructurallyEquivalent(int lhs, int rhs, const XLSGraph& query,
                               const RelationProfileCache& query_cache) {
  for (int w = 0; w < static_cast<int>(query.nodes.size()); ++w) {
    if (!(query_cache.Get(lhs, w) == query_cache.Get(rhs, w))) {
      return false;
    }
  }
  return true;
}

// Computes the structural-equivalence classes Psi(u) from Definition 5 /
// Equation (7) in Section 4.1 on the query side. RRSplit uses these classes in
// the node-equivalence reductions from Section 4.1 and in the tighter upper
// bound from Section 4.3.
EquivalenceInfo ComputeStructuralEquivalence(const XLSGraph& query) {
  const std::vector<CompatibilityKey> keys = BuildCompatibilityKeys(query);
  absl::flat_hash_map<CompatibilityKey, std::vector<int>, CompatibilityKeyHash>
      buckets;
  for (int u = 0; u < static_cast<int>(query.nodes.size()); ++u) {
    buckets[keys[u]].push_back(u);
  }

  EquivalenceInfo info;
  info.class_id.assign(query.nodes.size(), -1);

  RelationProfileCache query_cache(query);
  for (auto& [key, bucket] : buckets) {
    std::vector<int> representatives;
    std::sort(bucket.begin(), bucket.end());
    for (int u : bucket) {
      bool assigned = false;
      for (int rep : representatives) {
        if (AreStructurallyEquivalent(u, rep, query, query_cache)) {
          info.class_id[u] = info.class_id[rep];
          info.members[info.class_id[rep]].push_back(u);
          assigned = true;
          break;
        }
      }

      if (!assigned) {
        const int next_id = static_cast<int>(info.members.size());
        info.class_id[u] = next_id;
        info.members.push_back({u});
        representatives.push_back(u);
      }
    }
  }

  return info;
}

// Equation (4): refine the candidate partition after committing to pair (u, v).
// In the paper this splits each Xi x Yi by neighbor/non-neighbor relationships;
// for XLS graphs we split by exact RelationProfile to preserve directionality
// and edge-index semantics.
CandidatePartition RefineCandidatePartition(
    const CandidatePartition& candidates, int u, int v,
    const RelationProfileCache& query_cache,
    const RelationProfileCache& target_cache) {
  CandidatePartition refined;

  for (const CandidateClass& candidate_class : candidates) {
    std::map<RelationProfile, std::vector<int>> query_groups;
    std::map<RelationProfile, std::vector<int>> target_groups;

    for (int x : candidate_class.query_nodes) {
      if (x == u) {
        continue;
      }
      query_groups[query_cache.Get(u, x)].push_back(x);
    }

    for (int y : candidate_class.target_nodes) {
      if (y == v) {
        continue;
      }
      target_groups[target_cache.Get(v, y)].push_back(y);
    }

    for (const auto& [profile, query_nodes] : query_groups) {
      auto target_it = target_groups.find(profile);
      if (target_it == target_groups.end()) {
        continue;
      }
      refined.push_back(CandidateClass{
          .query_nodes = query_nodes,
          .target_nodes = target_it->second,
      });
    }
  }

  SortCandidatePartition(refined);
  return refined;
}

// Algorithm 2, line 19 / Section 4.1 second-group reduction:
// when excluding branching node u, also remove every query node in Psi(u),
// because any solution using one of them would be cs-isomorphic to a solution
// that has already been or will be explored earlier (Lemma 3).
CandidatePartition RemoveEquivalentQueryNodes(
    const CandidatePartition& candidates, int u,
    const EquivalenceInfo& equivalence) {
  std::vector<char> remove_mask(equivalence.class_id.size(), 0);
  for (int equiv_u : equivalence.members[equivalence.class_id[u]]) {
    remove_mask[equiv_u] = 1;
  }

  CandidatePartition filtered;
  filtered.reserve(candidates.size());
  for (const CandidateClass& candidate_class : candidates) {
    CandidateClass next_class;
    next_class.target_nodes = candidate_class.target_nodes;
    next_class.query_nodes.reserve(candidate_class.query_nodes.size());
    for (int x : candidate_class.query_nodes) {
      if (!remove_mask[x]) {
        next_class.query_nodes.push_back(x);
      }
    }
    if (!next_class.query_nodes.empty() && !next_class.target_nodes.empty()) {
      filtered.push_back(std::move(next_class));
    }
  }

  SortCandidatePartition(filtered);
  return filtered;
}

// Algorithm 2, lines 15-16 / Section 4.1 first-group reduction. If D already
// contains (u', v) for some u' in Psi(u), then exploring (u, v) would only
// rediscover a cs-isomorphic solution that was found earlier (Lemma 2).
bool ShouldPruneByExclusion(int u, int v, const ExclusionSet& excluded,
                            const EquivalenceInfo& equivalence) {
  for (int equiv_u : equivalence.members[equivalence.class_id[u]]) {
    auto it = excluded.find(equiv_u);
    if (it != excluded.end() && it->second.contains(v)) {
      return true;
    }
  }
  return false;
}

// Maintains the exclusion set D as described in Section 4.1. For a chosen
// query node u, later branches record which target nodes were already
// paired with equivalent query-side choices.
void AddExcludedValues(ExclusionSet& excluded, int u,
                       const std::vector<int>& target_nodes) {
  if (target_nodes.empty()) {
    return;
  }
  auto& target_set = excluded[u];
  for (int v : target_nodes) {
    target_set.insert(v);
  }
}

// Equation (13) / Lemma 4.
bool SatisfiesMaximalityReduction(const CandidatePartition& candidates, int u,
                                  int v,
                                  const RelationProfileCache& query_cache,
                                  const RelationProfileCache& target_cache) {
  for (const CandidateClass& candidate_class : candidates) {
    if (candidate_class.query_nodes.empty() ||
        candidate_class.target_nodes.empty()) {
      continue;
    }

    auto query_it = std::find_if(candidate_class.query_nodes.begin(),
                                 candidate_class.query_nodes.end(),
                                 [u](int x) { return x != u; });
    auto target_it = std::find_if(candidate_class.target_nodes.begin(),
                                  candidate_class.target_nodes.end(),
                                  [v](int y) { return y != v; });

    if (query_it == candidate_class.query_nodes.end() ||
        target_it == candidate_class.target_nodes.end()) {
      continue;
    }

    const RelationProfile query_profile = query_cache.Get(u, *query_it);
    for (int x : candidate_class.query_nodes) {
      if (x == u) {
        continue;
      }
      if (!(query_cache.Get(u, x) == query_profile)) {
        return false;
      }
    }

    const RelationProfile target_profile = target_cache.Get(v, *target_it);
    for (int y : candidate_class.target_nodes) {
      if (y == v) {
        continue;
      }
      if (!(target_cache.Get(v, y) == target_profile)) {
        return false;
      }
    }

    if (!(query_profile == target_profile)) {
      return false;
    }
  }

  return true;
}

// Section 4.3 / Equation (18) and Equation (19).
int ComputeUpperBound(const State& partial_mapping,
                      const CandidatePartition& candidates,
                      const ExclusionSet& excluded,
                      const EquivalenceInfo& equivalence) {
  int upper_bound = static_cast<int>(partial_mapping.size());

  for (const CandidateClass& candidate_class : candidates) {
    if (candidate_class.query_nodes.empty() ||
        candidate_class.target_nodes.empty()) {
      continue;
    }

    // The paper remarks that different representatives for Xi can give
    // different bounds; it suggests random choice as a cheap trade-off.
    // We instead pick the first sorted query node to keep runs deterministic.
    const int representative = candidate_class.query_nodes.front();
    const int representative_class = equivalence.class_id[representative];

    // Equation (16): X_L are the query nodes in the representative's
    // structural-equivalence class; X_R are the rest of the class.
    int x_l = 0;
    for (int x : candidate_class.query_nodes) {
      if (equivalence.class_id[x] == representative_class) {
        ++x_l;
      }
    }
    const int x_r = static_cast<int>(candidate_class.query_nodes.size()) - x_l;

    // Equation (17): Y_L are target nodes already blocked by D for Psi(u);
    // Y_R are the remaining target nodes.
    int y_l = 0;
    for (int y : candidate_class.target_nodes) {
      if (ShouldPruneByExclusion(representative, y, excluded, equivalence)) {
        ++y_l;
      }
    }
    const int y_size = static_cast<int>(candidate_class.target_nodes.size());
    const int y_r = y_size - y_l;

    upper_bound +=
        std::min(x_r, y_size) + std::min({x_l, y_r, std::max(y_size - x_r, 0)});
  }

  return upper_bound;
}

// Branching heuristic for Algorithm 2, line 8: pick the most constrained
// candidate class first.
int SelectBranchClassIndex(const CandidatePartition& candidates) {
  int best_index = -1;
  std::size_t best_y = std::numeric_limits<std::size_t>::max();
  std::size_t best_x = std::numeric_limits<std::size_t>::max();
  int best_u = std::numeric_limits<int>::max();

  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    const CandidateClass& candidate_class = candidates[i];
    if (candidate_class.query_nodes.empty() ||
        candidate_class.target_nodes.empty()) {
      continue;
    }

    const std::size_t y_size = candidate_class.target_nodes.size();
    const std::size_t x_size = candidate_class.query_nodes.size();
    const int min_u = candidate_class.query_nodes.front();

    if (y_size < best_y || (y_size == best_y && x_size < best_x) ||
        (y_size == best_y && x_size == best_x && min_u < best_u)) {
      best_index = i;
      best_y = y_size;
      best_x = x_size;
      best_u = min_u;
    }
  }

  return best_index;
}

constexpr int kPlateauSearchNodeMultiplier = 25;

// Not part of the paper.
bool MaybeStopForTimeout(SearchContext& ctx, int depth) {
  if (ctx.mcs_timeout_sec < 0) {
    return false;
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - ctx.start_time)
                           .count();
  if (elapsed < ctx.mcs_timeout_sec) {
    return false;
  }

  ctx.stop = true;
  VLOG(0) << Indent(depth) << "[timeout-stop] approximate MCS stop after "
          << elapsed << "s (timeout=" << ctx.mcs_timeout_sec << "s)";
  return true;
}

// Optional approximation mode: if we have not improved S* for long enough,
// stop and hand the residual graphs to GED. This is outside the paper.
bool MaybeStopForPlateau(SearchContext& ctx, int depth) {
  if (ctx.mcs_optimal || ctx.best_size <= 0) {
    return false;
  }
  if (ctx.search_nodes_since_best < ctx.plateau_search_node_threshold) {
    return false;
  }

  ctx.stop = true;
  VLOG(0) << Indent(depth) << "[plateau-stop] approximate MCS stop after "
          << ctx.search_nodes_since_best
          << " search nodes without improvement (threshold="
          << ctx.plateau_search_node_threshold << ")";
  return true;
}

// Maintains the best complete partial solution seen so far, i.e. S* in
// Algorithm 2. The cutoff logic is an ECO-specific early-stop condition.
bool MaybeUpdateBest(const State& current, const XLSGraph& query,
                     SearchContext& ctx, int depth) {
  const int current_size = static_cast<int>(current.size());
  if (current_size <= ctx.best_size) {
    return false;
  }

  ctx.best_mapping = current;
  ctx.best_size = current_size;
  ctx.search_nodes_since_best = 0;

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

// Main Recursive RRSplit search corresponding to Algorithm 2 in Section 4.4:
void RRSplitRec(State& partial_mapping, const CandidatePartition& candidates,
                const ExclusionSet& excluded, const XLSGraph& query,
                const XLSGraph& target, const EquivalenceInfo& equivalence,
                const RelationProfileCache& query_cache,
                const RelationProfileCache& target_cache, SearchContext& ctx,
                int depth) {
  if (ctx.stop) {
    return;
  }

  VLOG(3) << Indent(depth) << "[enter] depth=" << depth
          << ", partial_size=" << partial_mapping.size()
          << ", best_size=" << ctx.best_size
          << ", #candidate_classes=" << candidates.size();

  // Algorithm 2, line 5.
  MaybeUpdateBest(partial_mapping, query, ctx, depth);
  if (ctx.stop) {
    return;
  }

  // Extensions: bounded-stop controls for MCS preprocessing.
  ++ctx.search_nodes_since_best;
  if (MaybeStopForTimeout(ctx, depth)) {
    return;
  }
  if (MaybeStopForPlateau(ctx, depth)) {
    return;
  }

  if (candidates.empty()) {
    // Algorithm 2, line 6.
    VLOG(3) << Indent(depth) << "[leaf] no candidate classes remain";
    return;
  }

  // Algorithm 2, line 7.
  const int ub =
      ComputeUpperBound(partial_mapping, candidates, excluded, equivalence);
  if (ub <= ctx.best_size) {
    VLOG(3) << Indent(depth) << "[prune] upper_bound=" << ub
            << " <= best_size=" << ctx.best_size;
    return;
  }

  const int branch_class_index = SelectBranchClassIndex(candidates);
  if (branch_class_index < 0) {
    VLOG(3) << Indent(depth) << "[leaf] no branching class";
    return;
  }

  // Algorithm 2, line 8.
  const CandidateClass& branch_class = candidates[branch_class_index];
  const int u = branch_class.query_nodes.front();
  const std::vector<int>& y_nodes = branch_class.target_nodes;

  VLOG(3) << Indent(depth) << "[branch] class=" << branch_class_index
          << " u=" << u << " (" << query.nodes[u].name << ")"
          << " |X|=" << branch_class.query_nodes.size()
          << " |Y|=" << branch_class.target_nodes.size();

  // Algorithm 2, lines 9-11.
  for (int v : y_nodes) {
    if (SatisfiesMaximalityReduction(candidates, u, v, query_cache,
                                     target_cache)) {
      CandidatePartition refined =
          RefineCandidatePartition(candidates, u, v, query_cache, target_cache);
      ExclusionSet next_excluded = excluded;
      std::vector<int> other_target_nodes;
      other_target_nodes.reserve(y_nodes.size());
      for (int other_v : y_nodes) {
        if (other_v != v) {
          other_target_nodes.push_back(other_v);
        }
      }
      AddExcludedValues(next_excluded, u, other_target_nodes);

      State next_mapping = partial_mapping;
      next_mapping.emplace_back(u, v);
      RRSplitRec(next_mapping, refined, next_excluded, query, target,
                 equivalence, query_cache, target_cache, ctx, depth + 1);
      VLOG(3) << Indent(depth) << "[max-red] selected unique maximality branch";
      return;
    }
  }

  // Algorithm 2, lines 12-18. `tried_target_nodes` tracks the paper's
  // D ∪ {u} x (Y \ Y_temp): every time we move past a target node, later
  // siblings record that pairing so equivalent query nodes can prune it.
  std::vector<int> tried_target_nodes;
  tried_target_nodes.reserve(y_nodes.size());
  for (int v : y_nodes) {
    if (ctx.stop) {
      return;
    }

    if (ShouldPruneByExclusion(u, v, excluded, equivalence)) {
      VLOG(3) << Indent(depth) << "  - skip v=" << v << " ("
              << target.nodes[v].name << ") [excluded by Psi(u)]";
      tried_target_nodes.push_back(v);
      continue;
    }

    CandidatePartition refined =
        RefineCandidatePartition(candidates, u, v, query_cache, target_cache);
    ExclusionSet next_excluded = excluded;
    AddExcludedValues(next_excluded, u, tried_target_nodes);

    partial_mapping.push_back({u, v});
    RRSplitRec(partial_mapping, refined, next_excluded, query, target,
               equivalence, query_cache, target_cache, ctx, depth + 1);
    partial_mapping.pop_back();

    tried_target_nodes.push_back(v);
  }

  // Algorithm 2, line 19: recurse on C \ Psi(u). We exclude the whole
  // equivalence class, not just u itself, because remaining solutions would be
  // redundant under Lemma 3.
  CandidatePartition second_group =
      RemoveEquivalentQueryNodes(candidates, u, equivalence);
  RRSplitRec(partial_mapping, second_group, excluded, query, target,
             equivalence, query_cache, target_cache, ctx, depth + 1);

  VLOG(3) << Indent(depth) << "[exit] depth=" << depth;
}

// The paper optimizes node-set size. The ECO pipeline also needs a concrete
// matched-edge set so the later GED/patch stages can preserve edge identities.
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

}  // namespace

namespace internal {

EquivalenceInfo ComputeStructuralEquivalenceForTesting(const XLSGraph& query) {
  return ComputeStructuralEquivalence(query);
}

CandidatePartition RefineCandidatePartitionForTesting(
    const CandidatePartition& candidates, int u, int v, const XLSGraph& query,
    const XLSGraph& target) {
  RelationProfileCache query_cache(query);
  RelationProfileCache target_cache(target);
  return RefineCandidatePartition(candidates, u, v, query_cache, target_cache);
}

CandidatePartition RemoveEquivalentQueryNodesForTesting(
    const CandidatePartition& candidates, int u,
    const EquivalenceInfo& equivalence) {
  return RemoveEquivalentQueryNodes(candidates, u, equivalence);
}

bool ShouldPruneByExclusionForTesting(int u, int v,
                                      const ExclusionSet& excluded,
                                      const EquivalenceInfo& equivalence) {
  return ShouldPruneByExclusion(u, v, excluded, equivalence);
}

bool SatisfiesMaximalityReductionForTesting(
    const CandidatePartition& candidates, int u, int v, const XLSGraph& query,
    const XLSGraph& target) {
  RelationProfileCache query_cache(query);
  RelationProfileCache target_cache(target);
  return SatisfiesMaximalityReduction(candidates, u, v, query_cache,
                                      target_cache);
}

int ComputeUpperBoundForTesting(const State& partial_mapping,
                                const CandidatePartition& candidates,
                                const ExclusionSet& excluded,
                                const EquivalenceInfo& equivalence) {
  return ComputeUpperBound(partial_mapping, candidates, excluded, equivalence);
}

}  // namespace internal

// Entry point for the XLS ECO adaptation of RRSplit. We always use the smaller
// graph as the paper's query side, build the initial candidate partition and
// query-side equivalence classes once, and then run the recursive search.
MCSResult SolveMCS(const XLSGraph& graph1, const XLSGraph& graph2,
                   int mcs_cutoff, bool mcs_optimal, int mcs_timeout_sec) {
  const auto start = std::chrono::steady_clock::now();
  const bool swapped = graph1.nodes.size() > graph2.nodes.size();
  const XLSGraph& query = swapped ? graph2 : graph1;
  const XLSGraph& target = swapped ? graph1 : graph2;
  const int smaller_graph_nodes = static_cast<int>(query.nodes.size());
  const int plateau_search_node_threshold =
      smaller_graph_nodes * kPlateauSearchNodeMultiplier;
  VLOG(0) << "MCS start: G1 nodes=" << graph1.nodes.size()
          << " edges=" << graph1.edges.size()
          << " | G2 nodes=" << graph2.nodes.size()
          << " edges=" << graph2.edges.size() << " | cutoff=" << mcs_cutoff
          << " | optimal=" << mcs_optimal
          << (mcs_timeout_sec >= 0
                  ? " | timeout=" + std::to_string(mcs_timeout_sec)
                  : "")
          << " | plateau_search_node_threshold="
          << plateau_search_node_threshold;

  if (swapped) {
    VLOG(1) << "Using G2 as the internal MCS query side because it is smaller";
  }

  const int total_nodes = static_cast<int>(query.nodes.size());
  CandidatePartition initial_candidates =
      BuildInitialCandidatePartition(query, target);
  EquivalenceInfo equivalence = ComputeStructuralEquivalence(query);
  RelationProfileCache query_cache(query);
  RelationProfileCache target_cache(target);

  VLOG(1) << "Initial candidate classes=" << initial_candidates.size();

  if (mcs_cutoff >= 0 && total_nodes <= mcs_cutoff) {
    VLOG(0) << "MCS cutoff: total nodes (" << total_nodes << ") <= cutoff ("
            << mcs_cutoff << "), skipping MCS (will use GED for all nodes)";

    MCSResult empty_result;
    empty_result.size = 0;

    for (int i = 0; i < static_cast<int>(graph1.nodes.size()); ++i) {
      empty_result.unmatched_g1.push_back(i);
    }
    for (int j = 0; j < static_cast<int>(graph2.nodes.size()); ++j) {
      empty_result.unmatched_g2.push_back(j);
    }
    return empty_result;
  }

  SearchContext ctx{
      .stop = false,
      .mcs_optimal = mcs_optimal,
      .best_size = 0,
      .mcs_cutoff = mcs_cutoff,
      .mcs_timeout_sec = mcs_timeout_sec,
      .plateau_search_node_threshold = plateau_search_node_threshold,
      .total_nodes = total_nodes,
      .search_nodes_since_best = 0,
      .start_time = start,
      .best_mapping = {},
  };

  State partial_mapping;
  RRSplitRec(partial_mapping, initial_candidates, ExclusionSet{}, query, target,
             equivalence, query_cache, target_cache, ctx, 0);

  MCSResult result;
  if (swapped) {
    result.mapping.reserve(ctx.best_mapping.size());
    for (const auto& [u, v] : ctx.best_mapping) {
      result.mapping.emplace_back(v, u);
    }
  } else {
    result.mapping = ctx.best_mapping;
  }
  result.size = static_cast<int>(result.mapping.size());
  PopulateMatchedEdges(graph1, graph2, result.mapping, result);

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
  for (const auto& [u, v] : result.mapping) {
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

// Pipeline utility: identify MCS pairs that sit on the cut boundary, i.e. a
// matched node whose neighborhood still touches an unmatched node.
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
