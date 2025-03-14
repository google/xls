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

#ifndef XLS_PASSES_VISIBILITY_ANALYSIS_H_
#define XLS_PASSES_VISIBILITY_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/partial_information.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/query_engine.h"

namespace xls {

// Encapsulates a condition (a node holding a particular value) which can be
// assumed to be true at particular points in the graph.
struct Condition {
  Node* node;
  PartialInformation partial;

  std::string ToString() const {
    if (partial.IsUnconstrained()) {
      return absl::StrFormat("%s (unconstrained)", node->GetName());
    }
    if (partial.IsImpossible()) {
      return absl::StrFormat("%s (impossible)", node->GetName());
    }
    return absl::StrFormat("%s in %s", node->GetName(), partial.ToString());
  }

  bool operator==(const Condition& other) const {
    return node == other.node && partial == other.partial;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Condition& c) {
    return H::combine(std::move(h), c.node, c.partial);
  }
};

// A comparison functor for ordering Conditions. The functor orders conditions
// based on the topological order of the respective node, and then based on the
// condition's value. The motivation for ordering based on topological order is
// that the size of the set of conditions considered at any one time is bounded
// which means some conditions must be discarded.  We preferentially discard
// conditions of nodes which may be far away from the node being considered for
// transformation. Topological sort order gives a measure of distance for this
// purpose.
class ConditionCmp {
 public:
  explicit ConditionCmp(
      std::shared_ptr<absl::flat_hash_map<Node*, int64_t>> topo_index)
      : topo_index_(std::move(topo_index)) {}

  ConditionCmp(const ConditionCmp&) = default;
  ConditionCmp& operator=(const ConditionCmp&) = default;

  bool operator()(const Condition& a, const Condition& b) const {
    return topo_index_->at(a.node) < topo_index_->at(b.node);
  }

 private:
  std::shared_ptr<absl::flat_hash_map<Node*, int64_t>> topo_index_;
};

// A set of Conditions which can be assumed to hold at a particular point in the
// graph.
class ConditionSet {
 private:
  // The maximum size limit on the number of conditions to avoid superlinear
  // behavior.
  static constexpr int64_t kMaxConditions = 64;
  using ConditionVector = absl::InlinedVector<Condition, kMaxConditions + 1>;

 public:
  explicit ConditionSet(
      std::shared_ptr<absl::flat_hash_map<Node*, int64_t>> topo_index)
      : condition_cmp_(topo_index) {}

  ConditionSet(const ConditionSet&) = default;
  ConditionSet(ConditionSet&&) = default;
  ConditionSet& operator=(const ConditionSet&) = default;

  // Limit to conditions that are also true in `other`.
  void Intersect(const ConditionSet& other);

  // Add all conditions in `other` to this set.
  void Union(const ConditionSet& other);

  // Adds a condition to the set.  Note: it is possible to add conflicting
  // conditions to the set (pred==0 and pred==1). This is an indication that the
  // node is necessarily dead. We will merge the partial information of the
  // conditions, and record that the result is impossible.
  void AddCondition(const Condition& condition);

  absl::Span<const Condition> conditions() const { return conditions_; }

  std::optional<Condition> condition(Node* node) const {
    auto it = absl::c_lower_bound(
        conditions_,
        Condition{.node = node,
                  .partial =
                      PartialInformation::Unconstrained(node->BitCountOrDie())},
        condition_cmp_);
    if (it == conditions_.end() || it->node != node) {
      return std::nullopt;
    }
    return *it;
  }

  bool empty() const { return conditions_.empty(); }

  bool impossible() const {
    return absl::c_any_of(conditions_, [](const Condition& condition) {
      return condition.partial.IsImpossible();
    });
  }

  std::string ToString() const {
    std::vector<std::string> pieces;
    pieces.reserve(conditions_.size());
    for (const Condition& condition : conditions_) {
      pieces.push_back(condition.ToString());
    }
    return absl::StrCat("(", absl::StrJoin(pieces, " & "), ")");
  }

  // Returns the conditions as predicates
  std::vector<std::pair<TreeBitLocation, bool>> GetPredicates() const;

  // Returns the conditions as givens
  absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan> GetAsGivens()
      const;

  bool operator==(const ConditionSet& other) const {
    return conditions_ == other.conditions_;
  }

 private:
  ConditionCmp condition_cmp_;

  // Kept sorted at all times (according to `condition_cmp_`), retaining
  // only unique elements.
  ConditionVector conditions_;
};

struct TrackedValue {
  Node* node = nullptr;
  int64_t operand_no = -1;

  bool IsNode() const { return operand_no == -1; }
  bool IsEdge() const { return operand_no != -1; }

  bool operator==(const TrackedValue& other) const {
    return node == other.node && operand_no == other.operand_no;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TrackedValue& t) {
    return H::combine(std::move(h), t.node, t.operand_no);
  }
};

class VisibilityConditions {
 public:
  explicit VisibilityConditions(
      absl::flat_hash_map<TrackedValue, ConditionSet> results)
      : results_(std::move(results)) {}

  const ConditionSet& GetVisibilityConditions(Node* node) {
    return results_.at(TrackedValue{.node = node, .operand_no = -1});
  }

  const ConditionSet& GetVisibilityConditionsForEdge(Node* node,
                                                     int64_t operand_no) {
    return results_.at(TrackedValue{.node = node, .operand_no = operand_no});
  }

 private:
  absl::flat_hash_map<TrackedValue, ConditionSet> results_;
};

absl::StatusOr<VisibilityConditions> RunVisibilityAnalysis(
    FunctionBase* f, OptimizationContext& context, bool use_bdd);

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_ANALYSIS_H_
