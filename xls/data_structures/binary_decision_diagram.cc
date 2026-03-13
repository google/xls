// Copyright 2020 The XLS Authors
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

#include "xls/data_structures/binary_decision_diagram.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {

namespace {
#ifdef NDEBUG
constexpr bool kDebug = false;
#else
constexpr bool kDebug = true;
#endif
}  // namespace

BinaryDecisionDiagram::BinaryDecisionDiagram()
    : BinaryDecisionDiagram(kDefaultMaxPaths) {}

BinaryDecisionDiagram::BinaryDecisionDiagram(int64_t max_paths)
    : free_node_head_(2), nodes_size_(2), max_paths_(max_paths) {
  // Leaf node 0.
  nodes_.push_back(BddNode(BddVariable(-1), BddNodeIndex(-1), BddNodeIndex(-1),
                           /*p=*/1));
  // Leaf node 1.
  nodes_.push_back(BddNode(BddVariable(-1), BddNodeIndex(-1), BddNodeIndex(-1),
                           /*p=*/1));
}

BddNodeIndex BinaryDecisionDiagram::CreateNode(BddNode node) {
  int32_t slot = free_node_head_.value();
  CHECK_LE(slot, nodes_.size())
      << "free list head is far outside of the available ids";
  if (slot == nodes_.size()) {
    VLOG(2) << "Extending nodes_ to " << (nodes_.size() * 2) << " elements";
    nodes_.resize(
        nodes_.size() * 2,
        BddNode(kFreeNodeVariable, kNextFreeIsConsecutive, BddNodeIndex(0), 0));
  }
  DCHECK(IsFreeNode(nodes_.at(slot))) << " at slot " << slot;
  free_node_head_ =
      NextFree(GetNextFreeNode(nodes_.at(slot)), BddNodeIndex(slot));
  ++nodes_size_;
  nodes_[slot] = node;
  return BddNodeIndex(slot);
}

BddNodeIndex BinaryDecisionDiagram::CreateVariableBaseNode(BddVariable var) {
  const BddNodeIndex high = one();
  const BddNodeIndex low = zero();
  const int32_t paths = 2;
  return CreateNode(BddNode(var, high, low, paths));
}

BddNodeIndex BinaryDecisionDiagram::GetOrCreateNode(BddVariable var,
                                                    BddNodeIndex high,
                                                    BddNodeIndex low) {
  if (low == high) {
    return low;
  }

  // If low == 0 and high == 1, then this is a variable base node and is kept
  // in the variable_base_nodes_ vector.
  if (low == zero() && high == one()) {
    return variable_base_nodes_[var.value()].value();
  }

  // Otherwise, this is a normal node and is kept in node_map_.
  NodeKey key = std::make_tuple(var, high, low);
  auto it = node_map_.lazy_emplace(key, [&](const auto& ctor) {
    // Compute the number of paths that the new node will have to the terminal
    // nodes 0 and 1. Use int64s to avoid overflowing and saturate at INT32_MAX.
    int32_t paths =
        std::min(static_cast<int64_t>(GetNode(low).path_count) +
                     GetNode(high).path_count,
                 static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    BddNodeIndex node_index = CreateNode(BddNode(var, high, low, paths));
    ctor(key, node_index);
  });
  return it->second;
}
size_t BinaryDecisionDiagram::DynamicIteCache::approximate_memory_use() const {
  if (std::holds_alternative<ExactMap>(cache_)) {
    return std::get<ExactMap>(cache_).capacity() * sizeof(ExactMap::value_type);
  } else {
    return std::get<LossyArray>(cache_).capacity() * sizeof(IteCacheEntry);
  }
}

size_t BinaryDecisionDiagram::DynamicIteCache::Index(
    BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false) const {
  return absl::HashOf(cond, if_true, if_false) % kLossyCacheSize;
}

std::optional<BddNodeIndex> BinaryDecisionDiagram::DynamicIteCache::Get(
    BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false) const {
  if (const ExactMap* map = std::get_if<ExactMap>(&cache_)) {
    auto it = map->find(std::make_tuple(cond, if_true, if_false));
    if (it != map->end()) {
      return it->second;
    }
    return std::nullopt;
  } else {
    const LossyArray& array = std::get<LossyArray>(cache_);
    const IteCacheEntry& entry = array[Index(cond, if_true, if_false)];
    if (entry.cond == cond && entry.if_true == if_true &&
        entry.if_false == if_false) {
      return entry.result;
    }
    return std::nullopt;
  }
}

void BinaryDecisionDiagram::DynamicIteCache::TransitionToLossy() {
  ExactMap& map = std::get<ExactMap>(cache_);
  LossyArray new_array(kLossyCacheSize);

  // Note: if multiple entries from the exact map hash to the same slot, to
  //       avoid non-determinism, we will keep the last one in lexicographic
  //       order.
  for (const auto& [key, result] : map) {
    const auto& [cond, if_true, if_false] = key;
    size_t index = Index(cond, if_true, if_false);
    if (new_array[index] == IteCacheEntry() ||
        key > std::tie(new_array[index].cond, new_array[index].if_true,
                       new_array[index].if_false)) {
      new_array[Index(cond, if_true, if_false)] = IteCacheEntry{
          .cond = cond,
          .if_true = if_true,
          .if_false = if_false,
          .result = result,
      };
    }
  }
  cache_ = std::move(new_array);
}

void BinaryDecisionDiagram::DynamicIteCache::Insert(BddNodeIndex cond,
                                                    BddNodeIndex if_true,
                                                    BddNodeIndex if_false,
                                                    BddNodeIndex result) {
  if (ExactMap* map = std::get_if<ExactMap>(&cache_)) {
    map->try_emplace(std::make_tuple(cond, if_true, if_false), result);
    if (map->size() > kCutoverThreshold) {
      TransitionToLossy();
    }
  } else {
    LossyArray& array = std::get<LossyArray>(cache_);
    array[Index(cond, if_true, if_false)] = IteCacheEntry{
        .cond = cond,
        .if_true = if_true,
        .if_false = if_false,
        .result = result,
    };
  }
}

void BinaryDecisionDiagram::DynamicIteCache::GarbageCollect(
    const InlineBitmap& live_nodes) {
  if (ExactMap* map = std::get_if<ExactMap>(&cache_)) {
    absl::erase_if(*map, [&live_nodes](const auto& entry) {
      const auto& [key, value] = entry;
      const auto& [cond, if_true, if_false] = key;
      return !live_nodes.Get(cond.value()) ||
             !live_nodes.Get(if_true.value()) ||
             !live_nodes.Get(if_false.value()) ||
             (value != kInfeasible && !live_nodes.Get(value.value()));
    });
  } else {
    // We're already accepting a lossy cache; if we remove *all* entries, then
    // we've removed all dead entries.
    LossyArray& array = std::get<LossyArray>(cache_);
    std::fill(array.begin(), array.end(), IteCacheEntry{});
  }
}

std::optional<BddNodeIndex> BinaryDecisionDiagram::IfThenElseTrivial(
    BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false) {
  if (cond == one()) {
    return if_true;
  }
  if (cond == zero()) {
    return if_false;
  }
  if (if_true == one() && if_false == zero()) {
    return cond;
  }
  if (if_true == if_false) {
    return if_true;
  }
  if (cond == kInfeasible || if_true == kInfeasible ||
      if_false == kInfeasible) {
    return kInfeasible;
  }
  return std::nullopt;
}

BinaryDecisionDiagram::ITEDecomposition BinaryDecisionDiagram::DecomposeITE(
    BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false) {
  ITEDecomposition result;

  // First, find the lowest-index variable amongst all expressions. In all
  // paths through the BDD the variable indices are strictly increasing.
  const BddNode& cond_node = GetNode(cond);
  result.min_var = cond_node.variable;

  const BddNode* if_true_node = nullptr;
  if (if_true != zero() && if_true != one()) {
    if_true_node = &GetNode(if_true);
    result.min_var = std::min(result.min_var, if_true_node->variable);
  }

  const BddNode* if_false_node = nullptr;
  if (if_false != zero() && if_false != one()) {
    if_false_node = &GetNode(if_false);
    result.min_var = std::min(result.min_var, if_false_node->variable);
  }

  // Perform a Shannon expansion about the variable where Shannon expansion is
  // the identity:
  //
  //   F(x0, x1, ..) = !x0 && F(0, x1, ...) + x0 && F(1, x1, ...)
  //
  if (cond_node.variable == result.min_var) {
    result.true_cofactor.cond = cond_node.high;
    result.false_cofactor.cond = cond_node.low;
  } else {
    result.true_cofactor.cond = cond;
    result.false_cofactor.cond = cond;
  }

  if (if_true_node != nullptr && if_true_node->variable == result.min_var) {
    result.true_cofactor.if_true = if_true_node->high;
    result.false_cofactor.if_true = if_true_node->low;
  } else {
    result.true_cofactor.if_true = if_true;
    result.false_cofactor.if_true = if_true;
  }

  if (if_false_node != nullptr && if_false_node->variable == result.min_var) {
    result.true_cofactor.if_false = if_false_node->high;
    result.false_cofactor.if_false = if_false_node->low;
  } else {
    result.true_cofactor.if_false = if_false;
    result.false_cofactor.if_false = if_false;
  }

  return result;
}

BddNodeIndex BinaryDecisionDiagram::IfThenElse(BddNodeIndex cond,
                                               BddNodeIndex if_true,
                                               BddNodeIndex if_false) {
  std::optional<BddNodeIndex> initial_result =
      IfThenElseTrivial(cond, if_true, if_false);
  if (initial_result.has_value()) {
    return *initial_result;
  }

  std::optional<BddNodeIndex> cached = ite_cache_.Get(cond, if_true, if_false);
  if (cached.has_value()) {
    return *cached;
  }

  // The expression is non-trivial and has not been computed before.
  // Recursively decompose the expression by peeling away the first variable
  // and performing a Shannon decomposition.
  ITEDecomposition decomposition = DecomposeITE(cond, if_true, if_false);

  BddNodeIndex true_cofactor = IfThenElse(decomposition.true_cofactor.cond,
                                          decomposition.true_cofactor.if_true,
                                          decomposition.true_cofactor.if_false);
  if (true_cofactor == kInfeasible) {
    ite_cache_.Insert(cond, if_true, if_false, kInfeasible);
    return kInfeasible;
  }

  BddNodeIndex false_cofactor = IfThenElse(
      decomposition.false_cofactor.cond, decomposition.false_cofactor.if_true,
      decomposition.false_cofactor.if_false);
  if (false_cofactor == kInfeasible) {
    ite_cache_.Insert(cond, if_true, if_false, kInfeasible);
    return kInfeasible;
  }

  if (true_cofactor == false_cofactor) {
    if (path_count(true_cofactor) > max_paths_) {
      ite_cache_.Insert(cond, if_true, if_false, kInfeasible);
      return kInfeasible;
    }

    ite_cache_.Insert(cond, if_true, if_false, true_cofactor);
    return true_cofactor;
  }

  SaturatedResult<int64_t> total_path_count =
      SaturatingAdd(path_count(true_cofactor), path_count(false_cofactor));
  if (total_path_count.did_overflow || total_path_count.result > max_paths_) {
    ite_cache_.Insert(cond, if_true, if_false, kInfeasible);
    return kInfeasible;
  }

  BddNodeIndex result =
      GetOrCreateNode(decomposition.min_var, true_cofactor, false_cofactor);
  ite_cache_.Insert(cond, if_true, if_false, result);
  return result;
}

std::optional<bool> BinaryDecisionDiagram::IfThenElseConstant(
    BddNodeIndex cond, BddNodeIndex if_true, BddNodeIndex if_false) {
  auto to_constant = [&](BddNodeIndex node) -> std::optional<bool> {
    if (node == zero()) {
      return false;
    } else if (node == one()) {
      return true;
    } else {
      return std::nullopt;
    }
  };

  std::optional<BddNodeIndex> initial_result =
      IfThenElseTrivial(cond, if_true, if_false);
  if (initial_result.has_value()) {
    if (*initial_result == kInfeasible) {
      return std::nullopt;
    }
    return to_constant(*initial_result);
  }

  if (if_true == zero() && if_false == one()) {
    std::optional<bool> c = to_constant(cond);
    if (c.has_value()) {
      return !*c;
    }
    return std::nullopt;
  }

  std::optional<BddNodeIndex> cached = ite_cache_.Get(cond, if_true, if_false);
  if (cached.has_value()) {
    return to_constant(*cached);
  }

  // The expression is non-trivial and has not been computed before.
  // Recursively decompose the expression by peeling away the first variable
  // and performing a Shannon decomposition.
  ITEDecomposition decomposition = DecomposeITE(cond, if_true, if_false);

  std::optional<bool> then_branch = IfThenElseConstant(
      decomposition.true_cofactor.cond, decomposition.true_cofactor.if_true,
      decomposition.true_cofactor.if_false);
  if (!then_branch.has_value()) {
    return std::nullopt;
  }
  std::optional<bool> else_branch = IfThenElseConstant(
      decomposition.false_cofactor.cond, decomposition.false_cofactor.if_true,
      decomposition.false_cofactor.if_false);
  if (else_branch != then_branch) {
    return std::nullopt;
  }

  ite_cache_.Insert(cond, if_true, if_false, *then_branch ? one() : zero());
  return then_branch;
}

template <typename T>
void ReserveVector(int64_t new_size, std::vector<T>& vec) {
  constexpr int64_t kGrowthFactor = 2;
  if (new_size > vec.capacity()) {
    vec.reserve(std::max<int64_t>(new_size, kGrowthFactor * vec.capacity()));
  }
}

BddNodeIndex BinaryDecisionDiagram::NewVariable() {
  BddVariable var = BddVariable(variable_base_nodes_.size());
  BddNodeIndex index = CreateVariableBaseNode(var);
  variable_base_nodes_.push_back(index);
  return index;
}

std::vector<BddNodeIndex> BinaryDecisionDiagram::NewVariables(int64_t count) {
  // This is a bulk-insert API but we need to be fast for two cases:
  //   1. Count is large.
  //   2. Count is small (and called many times).
  //
  // For case 1, we would typically call std::vector::reserve(). But this would
  // pessimize case 2 because std::vector's growth strategy never kicks in, so
  // we would reallocate many more times than the naive push_back loop would.
  //
  // C++ guidance[0] is to use the std::vector::insert(begin, end) API for
  // bulk-inserts, but that really makes the code more complicated (and is
  // slightly less efficient for case 1).
  //
  // Instead we manually reserve() but to
  //   max(count+size(), kGrowthFactor*capacity())
  // to optimize for both cases.
  //
  // [0] https://en.cppreference.com/w/cpp/container/vector/reserve
  ReserveVector(variable_base_nodes_.size() + count, variable_base_nodes_);

  std::vector<BddNodeIndex> indexes;
  indexes.reserve(count);
  int64_t next_var = variable_base_nodes_.size();
  for (int64_t i = 0; i < count; ++i) {
    BddNodeIndex index = CreateVariableBaseNode(BddVariable(next_var++));
    variable_base_nodes_.push_back(index);
    indexes.push_back(index);
  }
  return indexes;
}

BddNodeIndex BinaryDecisionDiagram::Not(BddNodeIndex expr) {
  return IfThenElse(expr, zero(), one());
}

BddNodeIndex BinaryDecisionDiagram::Or(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElse(a, one(), b);
}

BddNodeIndex BinaryDecisionDiagram::And(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElse(a, b, zero());
}

BddNodeIndex BinaryDecisionDiagram::Implies(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElse(a, b, one());
}

bool BinaryDecisionDiagram::DoesImply(BddNodeIndex a, BddNodeIndex b) {
  return IfThenElseConstant(a, b, one()) == true;
}

absl::StatusOr<bool> BinaryDecisionDiagram::Evaluate(
    BddNodeIndex expr,
    const absl::flat_hash_map<BddNodeIndex, bool>& variable_values) const {
  if (expr == kInfeasible) {
    return absl::InvalidArgumentError("BDD expression is infeasible");
  }
  BddNodeIndex result = expr;
  VLOG(2) << "Evaluating node: " << static_cast<int64_t>(expr);
  VLOG(2) << "  expression = " << ToStringDnf(expr, /*minterm_limit=*/5);
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "  variable values: ";
    std::vector<BddNodeIndex> variables;
    for (const auto& pair : variable_values) {
      variables.push_back(pair.first);
    }
    std::sort(variables.begin(), variables.end());
    for (BddNodeIndex node : variables) {
      VLOG(3) << "    variable " << GetNode(node).variable << ": "
              << variable_values.at(node);
    }
  }
  while (result != zero() && result != one()) {
    BddNodeIndex var_node = GetVariableBaseNode(GetNode(result).variable);
    if (!variable_values.contains(var_node)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Missing value for BDD variable %d (node index %d)",
                          GetNode(result).variable.value(), var_node.value()));
    }
    result = variable_values.at(var_node) ? GetNode(result).high
                                          : GetNode(result).low;
  }
  VLOG(2) << "  result = " << (result == one() ? true : false);
  return result == one();
}

void BinaryDecisionDiagram::ToStringDnfHelper(BddNodeIndex expr,
                                              int64_t* minterms_to_emit,
                                              std::vector<std::string>* terms,
                                              std::string* str) const {
  if (*minterms_to_emit < 0) {
    return;
  }

  if (expr == zero()) {
    return;
  }
  if (expr == one()) {
    if (*minterms_to_emit == 0) {
      absl::StrAppend(str, " + ...");
    } else {
      if (str->empty()) {
        absl::StrAppend(str, absl::StrJoin(*terms, "."));
      } else {
        absl::StrAppend(str, " + ", absl::StrJoin(*terms, "."));
      }
    }
    --*minterms_to_emit;
    return;
  }

  const BddNode& node = GetNode(expr);
  terms->push_back(absl::StrCat("x", node.variable.value()));
  ToStringDnfHelper(node.high, minterms_to_emit, terms, str);
  terms->back() = absl::StrCat("!x", node.variable.value());
  ToStringDnfHelper(node.low, minterms_to_emit, terms, str);
  terms->pop_back();
}

std::string BinaryDecisionDiagram::ToStringDnf(BddNodeIndex expr,
                                               int64_t minterm_limit) const {
  if (expr == kInfeasible) {
    return "infeasible";
  }
  if (expr == zero()) {
    return "0";
  }
  if (expr == one()) {
    return "1";
  }
  std::string result;
  std::vector<std::string> terms;
  int64_t minterms_to_emit =
      minterm_limit == 0 ? std::numeric_limits<int64_t>::max() : minterm_limit;
  ToStringDnfHelper(expr, &minterms_to_emit, &terms, &result);
  return result;
}

absl::StatusOr<std::vector<BddNodeIndex>> BinaryDecisionDiagram::GarbageCollect(
    absl::Span<BddNodeIndex const> roots, double gc_threshold) {
  if (nodes_size_ * gc_threshold < prev_nodes_size_) {
    // Cannot save enough
    VLOG(2) << "Skipping GC of BDD since not enough new nodes were added to "
               "get over threshold (current "
            << nodes_size_ << ", last was at: " << prev_nodes_size_ << ")";
    return std::vector<BddNodeIndex>();
  }
  VLOG(2) << "Performing GC of BDD with " << roots.size() << " roots";
  // Mark all live nodes.
  InlineBitmap live_nodes(nodes_.size());
  InlineBitmap live_variables(variable_base_nodes_.size());
  // Avoid having to deal with the leafs.
  live_nodes.Set(zero().value());
  live_nodes.Set(one().value());
  std::deque<BddNodeIndex> worklist;
  for (BddNodeIndex root : roots) {
    worklist.push_back(root);
  }
  int64_t cnt_live = 2;
  while (!worklist.empty()) {
    BddNodeIndex node = worklist.front();
    worklist.pop_front();
    if (live_nodes.Get(node.value())) {
      continue;
    }
    const BddNode& bdd_node = GetNode(node);
    ++cnt_live;
    live_nodes.Set(node.value());
    live_variables.Set(bdd_node.variable.value());
    if (bdd_node.high != one() && bdd_node.high != zero()) {
      worklist.push_back(bdd_node.high);
    }
    if (bdd_node.low != one() && bdd_node.low != zero()) {
      worklist.push_back(bdd_node.low);
    }
  }
  VLOG(2) << "GC found " << cnt_live << "/" << nodes_size_ << " ("
          << (cnt_live * 100.0 / nodes_size_) << "%) live nodes.";
  if (static_cast<double>(cnt_live) / nodes_size_ >= gc_threshold ||
      cnt_live == nodes_size_) {
    VLOG(2) << "Skipping GC because insufficient dead nodes found.";
    return std::vector<BddNodeIndex>();
  }

  // Get all the dead nodes in a list.
  std::vector<BddNodeIndex> dead;
  XLS_RET_CHECK_GT(nodes_size_, cnt_live) << "Bad size";
  dead.reserve(nodes_size_ - cnt_live);
  for (auto [idx, live] : iter::enumerate(live_nodes)) {
    if (!live && !IsFreeNode(nodes_.at(idx))) {
      dead.push_back(BddNodeIndex(idx));
    }
  }
  // build a new free-list prefix.
  // Don't use iter::sliding_window because this is faster.
  for (int i = 0; i < dead.size(); ++i) {
    BddNodeIndex first = dead[i];
    BddNodeIndex second =
        i + 1 < dead.size() ? dead[i + 1] : BddNodeIndex(free_node_head_);
    SetFreeNode(nodes_[first.value()], second);
  }
  free_node_head_ = dead.front();

  // Clear the references to deleted variables.
  for (auto [idx, live] : iter::enumerate(live_variables)) {
    if (!live) {
      variable_base_nodes_[idx] = std::nullopt;
    }
  }

  ite_cache_.GarbageCollect(live_nodes);

  // clear out the node_map_ of references to deleted things.
  absl::erase_if(node_map_, [&live_nodes](const auto& entry) {
    const auto& [key, value] = entry;
    return !live_nodes.Get(value.value());
  });
  nodes_size_ = cnt_live;
  prev_nodes_size_ = nodes_size_;
  if constexpr (kDebug) {
    auto head = free_node_head_;
    auto it = dead.begin();
    InlineBitmap debug_seen(nodes_.size());
    while (head.value() < nodes_.size() && head != kNextFreeIsConsecutive) {
      CHECK(!debug_seen.Get(head.value())) << " loop in free list at " << head;
      debug_seen.Set(head.value());
      if (it != dead.end()) {
        CHECK_EQ(head, *it);
        ++it;
      }
      CHECK(IsFreeNode(nodes_[head.value()]));
      auto cur_next = GetNextFreeNode(nodes_[head.value()]);
      if (cur_next == kNextFreeIsConsecutive) {
        break;
      }
      head = NextFree(cur_next, head);
    }
  }
  return std::move(dead);
}

}  // namespace xls
