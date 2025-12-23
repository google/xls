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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/chain.hpp"
#include "cppitertools/enumerate.hpp"
#include "cppitertools/sliding_window.hpp"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {

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
  auto slot = free_node_head_;
  CHECK_LE(slot.value(), nodes_.size())
      << "free list head is far outside of the available ids";
  if (slot.value() == nodes_.size()) {
    VLOG(2) << "Extending nodes_ to " << (nodes_.size() * 2) << " elements";
    nodes_.resize(nodes_.size() * 2, FreeListNode());
  }
  CHECK(std::holds_alternative<FreeListNode>(nodes_.at(slot.value())))
      << " at slot " << slot;
  free_node_head_ =
      std::get<FreeListNode>(nodes_.at(slot.value())).next_free(slot);
  ++nodes_size_;
  nodes_[slot.value()] = node;
  return BddNodeIndex(slot.value());
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

BddNodeIndex BinaryDecisionDiagram::Restrict(BddNodeIndex expr, BddVariable var,
                                             bool value) {
  if (expr == zero() || expr == one()) {
    return expr;
  }

  const BddNode& node = GetNode(expr);
  CHECK_LE(var, node.variable);
  if (node.variable == var) {
    return value ? node.high : node.low;
  }
  return expr;
}

BddNodeIndex BinaryDecisionDiagram::IfThenElse(BddNodeIndex cond,
                                               BddNodeIndex if_true,
                                               BddNodeIndex if_false) {
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
  auto key = std::make_tuple(cond, if_true, if_false);
  auto it = ite_map_.find(key);
  if (it != ite_map_.end()) {
    return it->second;
  }

  // The expression is non-trivial and has not been computed before.
  // Recursively decompose the expression by peeling away the first variable
  // and performing a Shannon decomposition.

  // First, find the lowest-index variable amongst all expressions. In all
  // paths through the BDD the variable indices are strictly increasing.
  BddVariable min_var = GetNode(cond).variable;
  // Only non-leaf nodes (not zero or one) have associated variables.
  if (if_true != zero() && if_true != one()) {
    min_var = std::min(min_var, GetNode(if_true).variable);
  }
  if (if_false != zero() && if_false != one()) {
    min_var = std::min(min_var, GetNode(if_false).variable);
  }

  // Perform a Shannon expansion about the variable where Shannon expansion is
  // the identity:
  //
  //   F(x0, x1, ..) = !x0 && F(0, x1, ...) + x0 && F(1, x1, ...)
  //
  BddNodeIndex true_cofactor = IfThenElse(Restrict(cond, min_var, true),
                                          Restrict(if_true, min_var, true),
                                          Restrict(if_false, min_var, true));
  BddNodeIndex false_cofactor = IfThenElse(Restrict(cond, min_var, false),
                                           Restrict(if_true, min_var, false),
                                           Restrict(if_false, min_var, false));

  if (true_cofactor == kInfeasible || false_cofactor == kInfeasible) {
    return kInfeasible;
  }
  if (true_cofactor == false_cofactor) {
    if (path_count(true_cofactor) > max_paths_) {
      return kInfeasible;
    }
    ite_map_[key] = true_cofactor;
    return true_cofactor;
  }
  SaturatedResult<int64_t> total_path_count =
      SaturatingAdd(path_count(true_cofactor), path_count(false_cofactor));
  if (total_path_count.did_overflow || total_path_count.result > max_paths_) {
    return kInfeasible;
  }
  BddNodeIndex expr = GetOrCreateNode(min_var, true_cofactor, false_cofactor);
  ite_map_[key] = expr;
  return expr;
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
    if (!live && std::holds_alternative<BddNode>(nodes_.at(idx))) {
      dead.push_back(BddNodeIndex(idx));
    }
  }
  // build a new free-list prefix.
  for (auto vs : iter::sliding_window(
           iter::chain(dead, std::vector<BddNodeIndex>{BddNodeIndex(
                                 free_node_head_.value())}),
           2)) {
    BddNodeIndex first = vs[0];
    BddNodeIndex second = vs[1];
    nodes_[first.value()] = FreeListNode(FreeListNode::Index(second.value()));
  }
  free_node_head_ = FreeListNode::Index(dead.front().value());

  // Clear the references to deleted variables.
  for (auto [idx, live] : iter::enumerate(live_variables)) {
    if (!live) {
      variable_base_nodes_[idx] = std::nullopt;
    }
  }

  // Clear the ite_map_ of references to deleted things.
  for (auto it = ite_map_.begin(); it != ite_map_.end();) {
    const auto& [key, value] = *it;
    const auto& [cond, if_true, if_false] = key;
    if (!live_nodes.Get(cond.value()) || !live_nodes.Get(if_true.value()) ||
        !live_nodes.Get(if_false.value()) || !live_nodes.Get(value.value())) {
      ite_map_.erase(it++);
    } else {
      ++it;
    }
  }
  // clear out the node_map_ of references to deleted things.
  for (auto it = node_map_.begin(); it != node_map_.end();) {
    const auto& [key, value] = *it;
    if (!live_nodes.Get(value.value())) {
      node_map_.erase(it++);
    } else {
      ++it;
    }
  }
  nodes_size_ = cnt_live;
  prev_nodes_size_ = nodes_size_;
  auto head = free_node_head_;
  auto it = dead.begin();
  InlineBitmap debug_seen(nodes_.size());
  while (head.value() < nodes_.size() &&
         head != FreeListNode::kNextIsConsecutive) {
    CHECK(!debug_seen.Get(head.value()))
        << " loop in free list at " << head.value();
    debug_seen.Set(head.value());
    if (it != dead.end()) {
      CHECK_EQ(head, FreeListNode::Index(it->value()));
      ++it;
    }
    CHECK(std::holds_alternative<FreeListNode>(nodes_[head.value()]));
    auto cur = std::get<FreeListNode>(nodes_[head.value()]);
    if (cur.raw_next() == FreeListNode::kNextIsConsecutive) {
      break;
    }
    head = cur.next_free(head);
  }
  return std::move(dead);
}

}  // namespace xls
