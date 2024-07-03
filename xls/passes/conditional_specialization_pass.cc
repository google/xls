// Copyright 2021 The XLS Authors
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

#include "xls/passes/conditional_specialization_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Encapsulates a condition (a node holding a particular value) which can be
// assumed to be true at particular points in the graph.
struct Condition {
  Node* node;
  TernaryVector value;

  std::string ToString() const {
    return absl::StrFormat("%s==%s", node->GetName(), xls::ToString(value));
  }

  bool operator==(const Condition& other) const {
    return node == other.node && value == other.value;
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
  explicit ConditionCmp(const absl::flat_hash_map<Node*, int64_t>& topo_index)
      : topo_index_(&topo_index) {}

  ConditionCmp(const ConditionCmp&) = default;
  ConditionCmp& operator=(const ConditionCmp&) = default;

  bool operator()(const Condition& a, const Condition& b) const {
    return std::pair(topo_index_->at(a.node), a.value) <
           std::pair(topo_index_->at(b.node), b.value);
  }

 private:
  const absl::flat_hash_map<Node*, int64_t>* topo_index_;
};

// A set of Conditions which can be assumed to hold at a particular point in the
// graph.
class ConditionSet {
 public:
  // Use a btree to hold the conditions for a well defined order (determined by
  // ConditionCmp).
  using ConditionBTree = absl::btree_set<Condition, ConditionCmp>;

  explicit ConditionSet(const absl::flat_hash_map<Node*, int64_t>& topo_index)
      : condition_cmp_(topo_index), conditions_(condition_cmp_) {}

  ConditionSet(const ConditionSet&) = default;
  ConditionSet(ConditionSet&&) = default;
  ConditionSet& operator=(const ConditionSet&) = default;

  // Perform a set intersection with this set and `other` and assign the result
  // to this set.
  void Intersect(const ConditionSet& other) {
    ConditionBTree original = std::move(conditions_);
    conditions_.clear();
    std::set_intersection(other.conditions_.begin(), other.conditions_.end(),
                          original.begin(), original.end(),
                          std::inserter(conditions_, conditions_.begin()),
                          condition_cmp_);
    // Intersection should not increase set size.
    CHECK_LE(conditions_.size(), kMaxConditions);
  }

  // Adds a condition to the set.  Note: it is possible to add conflicting
  // conditions to the set (pred==0 and pred==1). This is an indication that the
  // node is necessarily dead. Arbitrary transformation to dead code is legal so
  // arbitrarily picking one of the conflicting conditions and transforming
  // based on it is fine.
  void AddCondition(const Condition& condition) {
    VLOG(4) << absl::StreamFormat(
        "ConditionSet for (%s, %s) : %s", condition.node->GetName(),
        xls::ToString(condition.value), this->ToString());
    CHECK(!condition.node->Is<Literal>());
    conditions_.insert(condition);
    // The conditions are ordering in topological sort order (based on
    // Condition.node) and transformation occurs in reverse topological sort
    // order so the most distant conditions should be at the end of the
    // condition set.  Just pop the last condition off the end if it exceeds the
    // limit.
    if (conditions_.size() > kMaxConditions) {
      conditions_.erase(std::next(conditions_.end(), -1));
    }
    CHECK_LE(conditions_.size(), kMaxConditions);
  }

  const ConditionBTree& conditions() const { return conditions_; }

  std::string ToString() const {
    std::vector<std::string> pieces;
    for (const Condition& condition : conditions_) {
      pieces.push_back(condition.ToString());
    }
    return absl::StrCat("(", absl::StrJoin(pieces, " & "), ")");
  }

  // Returns the conditions as predicates
  std::vector<std::pair<TreeBitLocation, bool>> GetPredicates() const {
    std::vector<std::pair<TreeBitLocation, bool>> predicates;
    for (const Condition& condition : conditions()) {
      for (int64_t i = 0; i < condition.node->BitCountOrDie(); ++i) {
        if (condition.value[i] == TernaryValue::kUnknown) {
          continue;
        }
        bool bit_value = (condition.value[i] == TernaryValue::kKnownOne);
        predicates.push_back({TreeBitLocation{condition.node, i}, bit_value});
      }
    }
    return predicates;
  }

 private:
  // The maximum size limit on the number of conditions to avoid superlinear
  // behavior.
  static constexpr int64_t kMaxConditions = 64;
  ConditionCmp condition_cmp_;
  ConditionBTree conditions_;
};

// A map containing the set of conditions which can be assumed at each node (and
// at some edges). An example where a condition would be assigned to an edge
// rather than a node is a case arm of a select. In this case, a selector value
// can be assumed on the edge from the case operand to the select operation. In
// general, the condition cannot be assumed on the source node of the edge (case
// operand) because the source node may be used outside the case arm expression.
class ConditionMap {
 public:
  explicit ConditionMap(FunctionBase* f) {
    std::vector<Node*> topo_sort = TopoSort(f);
    for (int64_t i = 0; i < topo_sort.size(); ++i) {
      topo_index_[topo_sort[i]] = i;
      // Initially all node conditions are empty.
      node_conditions_.emplace(topo_sort[i], topo_index_);
    }
  }

  // Returns the condition set for the given node. Returns a mutable reference
  // as this is the mechanism for setting condition sets of nodes.
  ConditionSet& GetNodeConditionSet(Node* node) {
    return node_conditions_.at(node);
  }

  // Sets the condition set for the given edge where the edge extends to `node`
  // and operand index `operand_no`.
  void SetEdgeConditionSet(Node* node, int64_t operand_no,
                           ConditionSet condition_set) {
    VLOG(4) << absl::StrFormat("Setting conditions on %s->%s (operand %d): %s",
                               node->operand(operand_no)->GetName(),
                               node->GetName(), operand_no,
                               condition_set.ToString());
    std::pair<Node*, int64_t> key = {node, operand_no};
    CHECK(!edge_conditions_.contains(key));
    edge_conditions_.insert({key, std::move(condition_set)});
  }

  // Returns the conditions which can be assumed along the edge to `node` from
  // its operand index `operand_no`.
  const ConditionSet& GetEdgeConditionSet(Node* node, int64_t operand_no) {
    std::pair<Node*, int64_t> key = {node, operand_no};
    if (!edge_conditions_.contains(key)) {
      // There are no special conditions for this edge. Return the conditions on
      // the target of the edge which necessarily hold on the edge as well.
      return node_conditions_.at(node);
    }
    return edge_conditions_.at(key);
  }

  // Returns the conditions which can be assumed along the edge(s) from node to
  // user. This interface is asymmetric to SetEdgeCondition (which takes a node
  // and operand number) to make it easier to use because at a particular node
  // you have easy access to the user list but not the operand number(s)
  // associated with each user.
  const ConditionSet& GetEdgeConditionSet(Node* node, Node* user) {
    // Find the unique (if there is one) operand number of user which
    // corresponds to node.
    std::optional<int64_t> operand_index;
    for (int64_t i = 0; i < user->operand_count(); ++i) {
      if (user->operand(i) == node) {
        if (operand_index.has_value()) {
          // `node` appears in multiple operands of `user`. Return the
          // assumptions that can be made at the node `user` itself. This is
          // typically not a strong conditions as might be assuming along the
          // edges.
          return GetNodeConditionSet(user);
        }
        operand_index = i;
      }
    }
    CHECK(operand_index.has_value()) << absl::StreamFormat(
        "%s is not a user of %s", user->GetName(), node->GetName());
    return GetEdgeConditionSet(user, *operand_index);
  }

  std::string ToString() const {
    std::stringstream os;
    os << "Node conditions:\n";
    for (const auto& [node, cond_set] : node_conditions_) {
      if (!cond_set.conditions().empty()) {
        os << absl::StrFormat("[%s]: %s", node->ToString(), cond_set.ToString())
           << "\n";
      }
    }
    os << "Edge conditions:\n";
    for (const auto& [key, cond_set] : edge_conditions_) {
      if (!cond_set.conditions().empty()) {
        os << absl::StrFormat("[%s, %i]: %s", std::get<0>(key)->ToString(),
                              std::get<1>(key), cond_set.ToString())
           << "\n";
      }
    }
    return os.str();
  }

 private:
  // Index of each node in the function base in a topological sort.
  absl::flat_hash_map<Node*, int64_t> topo_index_;

  // Set of conditions which might be assumed at each node.
  absl::flat_hash_map<Node*, ConditionSet> node_conditions_;

  // Set of conditions which might be assumed at some edges. The key defines an
  // edge as (node, operand_no). If no key exists for an edge, then there are no
  // special conditions for the edge, and the conditions for the edge are the
  // same as the node.
  absl::flat_hash_map<std::pair<Node*, int64_t>, ConditionSet> edge_conditions_;
};

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<Bits> ImpliedNodeValue(const ConditionSet& condition_set,
                                     Node* node,
                                     const QueryEngine& query_engine) {
  for (const Condition& condition : condition_set.conditions()) {
    if (condition.node == node && ternary_ops::IsFullyKnown(condition.value)) {
      VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                    condition_set.ToString(), node->GetName(),
                                    xls::ToString(condition.value));
      return ternary_ops::ToKnownBitsValues(condition.value);
    }
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<Bits> implied_value =
      query_engine.ImpliedNodeValue(predicates, node);

  if (implied_value.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%v", condition_set.ToString(),
                                  node->GetName(), implied_value.value());
  }
  return implied_value;
}

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<TernaryVector> ImpliedNodeTernary(
    const ConditionSet& condition_set, Node* node,
    const QueryEngine& query_engine) {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  TernaryVector result(node->BitCountOrDie(), TernaryValue::kUnknown);
  for (const Condition& condition : condition_set.conditions()) {
    if (condition.node == node) {
      VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                    condition_set.ToString(), node->GetName(),
                                    xls::ToString(condition.value));
      CHECK_OK(ternary_ops::UpdateWithUnion(result, condition.value));
    }
  }
  if (ternary_ops::IsFullyKnown(result)) {
    return result;
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<TernaryVector> implied_ternary =
      query_engine.ImpliedNodeTernary(predicates, node);
  if (implied_ternary.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%s", condition_set.ToString(),
                                  node->GetName(),
                                  xls::ToString(*implied_ternary));
    CHECK_OK(ternary_ops::UpdateWithUnion(result, *implied_ternary));
  }

  return result;
}

// Returns the case arm node of the given select which is selected when the
// selector has the given value.
Node* GetSelectedCase(Select* select, const Bits& selector_value) {
  if (bits_ops::UGreaterThanOrEqual(selector_value, select->cases().size())) {
    return select->default_value().value();
  }
  // It is safe to convert to uint64_t because of the above check against cases
  // size.
  return select->get_case(selector_value.ToUint64().value());
}

std::optional<Node*> GetSelectedCase(PrioritySelect* select,
                                     const TernaryVector& selector_value) {
  for (int64_t i = 0; i < select->cases().size(); ++i) {
    if (selector_value[i] == TernaryValue::kUnknown) {
      // We can't be sure which case is selected.
      return std::nullopt;
    }
    if (selector_value[i] == TernaryValue::kKnownOne) {
      return select->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return select->default_value();
}

struct ZeroValue : std::monostate {};
std::optional<std::variant<Node*, ZeroValue>> GetSelectedCase(
    OneHotSelect* ohs, const TernaryVector& selector_value) {
  if (!ternary_ops::IsFullyKnown(selector_value)) {
    // We can't be sure which case is selected.
    return std::nullopt;
  }
  Bits selector_bits = ternary_ops::ToKnownBitsValues(selector_value);
  if (selector_bits.PopCount() > 1) {
    // We aren't selecting just one state.
    return std::nullopt;
  }
  for (int64_t i = 0; i < selector_value.size(); ++i) {
    if (selector_bits.Get(i)) {
      return ohs->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return ZeroValue{};
}

}  // namespace

absl::StatusOr<bool> ConditionalSpecializationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  if (use_bdd_) {
    query_engines.push_back(std::make_unique<BddQueryEngine>(
        BddFunction::kDefaultPathLimit, IsCheapForBdds));
  }

  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  ConditionMap condition_map(f);

  // Iterate backwards through the graph because we add conditions at the case
  // arm operands of selects and propagate them upwards through the expressions
  // which compute the case arm.
  bool changed = false;
  for (Node* node : ReverseTopoSort(f)) {
    ConditionSet& set = condition_map.GetNodeConditionSet(node);
    VLOG(4) << absl::StreamFormat("Considering node %s: %s", node->GetName(),
                                  set.ToString());

    if (node->Is<Invoke>()) {
      // The contents of an invoke may be side-effecting (e.g., the invoked
      // function might contain an assert), so don't assume any conditions for
      // this node or its predecessors.
      VLOG(4) << absl::StreamFormat(
          "Node %s is an invoke and could be side-effecting", node->GetName());
      continue;
    }
    if (OpIsSideEffecting(node->op()) && !node->Is<Send>() &&
        !node->Is<Next>()) {
      // Inputs to side-effecting operations should not change so don't assume
      // any conditions for this node or it's predecessors.
      VLOG(4) << absl::StreamFormat("Node %s is side-effecting",
                                    node->GetName());
      continue;
    }

    // Compute the intersection of the condition sets of the users of this node.
    //
    // If this node has an implicit use then we can't propagate any conditions
    // from the users because this value is unconditionally live and therefore
    // its computed value should not be changed.
    if (!f->HasImplicitUse(node)) {
      VLOG(4) << absl::StreamFormat(
          "%s has no implicit use, computing intersection of conditions of "
          "users",
          node->GetName());
      bool first_user = true;
      for (Node* user : node->users()) {
        if (first_user) {
          set = condition_map.GetEdgeConditionSet(node, user);
        } else {
          set.Intersect(condition_map.GetEdgeConditionSet(node, user));
        }
        first_user = false;
      }
    }

    // If the only user of this node is a single select arm then add a condition
    // based on the selector value for this arm.
    if (node->Is<Select>()) {
      Select* select = node->As<Select>();
      // Don't bother specializing if the selector is a literal as this
      // results in a useless condition where we assume a literal has a
      // literal value.
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know the selector is exactly
          // `case_no`.
          edge_set.AddCondition(Condition{
              .node = select->selector(),
              .value = ternary_ops::BitsToTernary(
                  UBits(case_no, select->selector()->BitCountOrDie())),
          });
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
      }
    }
    if (node->Is<OneHotSelect>()) {
      OneHotSelect* select = node->As<OneHotSelect>();
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know the corresponding bit of the
          // selector is one.
          TernaryVector selector_value(select->selector()->BitCountOrDie(),
                                       TernaryValue::kUnknown);
          selector_value[case_no] = TernaryValue::kKnownOne;
          edge_set.AddCondition(Condition{
              .node = select->selector(),
              .value = selector_value,
          });
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
      }
    }
    if (node->Is<PrioritySelect>()) {
      PrioritySelect* select = node->As<PrioritySelect>();
      if (!select->selector()->Is<Literal>()) {
        for (int64_t case_no = 0; case_no < select->cases().size(); ++case_no) {
          ConditionSet edge_set = set;
          // If this case is selected, we know all the bits of the selector up
          // to and including `case_no`; all higher-priority bits are zero, and
          // this case's bit is one.
          Bits known_bits = Bits(select->selector()->BitCountOrDie());
          known_bits.SetRange(0, case_no + 1);
          Bits known_bits_values =
              Bits::PowerOfTwo(case_no, select->selector()->BitCountOrDie());
          edge_set.AddCondition(Condition{
              .node = select->selector(),
              .value =
                  ternary_ops::FromKnownBits(known_bits, known_bits_values),
          });
          condition_map.SetEdgeConditionSet(node, case_no + 1,
                                            std::move(edge_set));
        }
        ConditionSet edge_set = set;
        // If the default value is selected, we know all the bits of the
        // selector are zero.
        edge_set.AddCondition(Condition{
            .node = select->selector(),
            .value = TernaryVector(select->selector()->BitCountOrDie(),
                                   TernaryValue::kKnownZero),
        });
        condition_map.SetEdgeConditionSet(node, select->cases().size() + 1,
                                          std::move(edge_set));
      }
    }
    // The operands of single-bit logical operations may not be observable
    // depending on the value of the *other* operands. For example, given:
    //
    //    OR(X, Y, Z)
    //
    // The value X is not observable if Y or Z is one, so we can assume in the
    // computation of X that Y and Z are zero. Similar assumptions can be made
    // for other logical operations (NOR, NAD, NAND).
    //
    // This can only be used when a (BDD) query engine is *not* used because as
    // soon as the graph is transformed the query engine becomes stale. This is
    // not the case with the select-based transformations because of the mutual
    // exclusion of the assumed conditions.
    //
    // TODO(b/323003986): Incrementally update the BDD.
    if ((node->op() == Op::kOr || node->op() == Op::kNor ||
         node->op() == Op::kAnd || node->op() == Op::kNand) &&
        node->BitCountOrDie() == 1 && !use_bdd_) {
      // The value you can assume other operands have in the computation of this
      // operand.
      TernaryValue assumed_operand_value =
          (node->op() == Op::kOr || node->op() == Op::kNor)
              ? TernaryValue::kKnownZero
              : TernaryValue::kKnownOne;
      for (int64_t i = 0; i < node->operand_count(); ++i) {
        if (node->operand(i)->Is<Literal>()) {
          continue;
        }
        ConditionSet edge_set = set;
        for (int64_t j = 0; j < node->operand_count(); ++j) {
          if (i != j && !node->operand(j)->Is<Literal>()) {
            edge_set.AddCondition(Condition{.node = node->operand(j),
                                            .value = {assumed_operand_value}});
          }
        }
        condition_map.SetEdgeConditionSet(node, i, std::move(edge_set));
      }
    }

    if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      if (!send->predicate().has_value()) {
        continue;
      }

      Node* predicate = *send->predicate();
      if (predicate->Is<Literal>()) {
        continue;
      }

      VLOG(4) << absl::StreamFormat(
          "%s is a conditional send, assuming predicate %s is true for data %s",
          node->GetName(), predicate->GetName(), send->data()->GetName());

      ConditionSet edge_set = set;
      edge_set.AddCondition(
          Condition{.node = predicate, .value = {TernaryValue::kKnownOne}});
      condition_map.SetEdgeConditionSet(node, Send::kDataOperand,
                                        std::move(edge_set));
    }

    if (node->Is<Next>()) {
      Next* next = node->As<Next>();
      if (!next->predicate().has_value()) {
        continue;
      }

      Node* predicate = *next->predicate();
      if (predicate->Is<Literal>()) {
        continue;
      }

      VLOG(4) << absl::StreamFormat(
          "%s is a conditional next_value, assuming predicate %s is true for "
          "data %s",
          node->GetName(), predicate->GetName(), next->value()->GetName());

      ConditionSet edge_set = set;
      edge_set.AddCondition(
          Condition{.node = predicate, .value = {TernaryValue::kKnownOne}});
      condition_map.SetEdgeConditionSet(node, Next::kValueOperand,
                                        std::move(edge_set));
    }

    VLOG(4) << absl::StreamFormat("Conditions for %s : %s", node->GetName(),
                                  set.ToString());

    // Now specialize any operands (if possible) based on the conditions.
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      Node* operand = node->operand(operand_no);

      if (operand->Is<Literal>()) {
        // Operand is already a literal. Nothing to do.
        continue;
      }

      const ConditionSet& edge_set =
          condition_map.GetEdgeConditionSet(operand, node);
      VLOG(4) << absl::StrFormat("Conditions on edge %s -> %s: %s",
                                 operand->GetName(), node->GetName(),
                                 edge_set.ToString());

      // First check to see if the condition set directly implies a value for
      // the operand. If so replace with the implied value.
      if (std::optional<Bits> implied_value =
              ImpliedNodeValue(edge_set, operand, query_engine);
          implied_value.has_value()) {
        VLOG(3) << absl::StreamFormat("Replacing operand %d of %s with %v",
                                      operand_no, node->GetName(),
                                      implied_value.value());
        XLS_ASSIGN_OR_RETURN(
            Literal * literal,
            f->MakeNode<Literal>(operand->loc(), Value(implied_value.value())));
        XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(operand_no, literal));
        changed = true;
        continue;
      }

      // If `operand` is a select and any condition set of `node` implies the
      // selector value then we can wire the respective implied case directly to
      // that user. For example:
      //
      //         a   b                 a     b
      //          \ /                   \   / \
      //  s|t ->  sel0          s|t ->  sel0    \
      //           | \     =>            |      |
      //        c  |  \                  d   c  |
      //         \ |   d                      \ |
      //  s   ->  sel1                   s -> sel1
      //           |                            |
      //
      // This pattern is not handled elsewhere because `sel0` has other uses
      // than `sel1` so `sel0` does not inherit the selector condition `s==1`.
      //
      // It may be possible to bypass multiple selects so walk the edge up the
      // graph as far as possible. For example, in the diagram above `b` may
      // also be a select with a selector whose value is implied by `s`.
      //
      // This also applies to ANDs, ORs, and XORs, if the condition set implies
      // that all but one operand is the identity for the operation.
      if (operand->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                         Op::kOr, Op::kXor})) {
        std::optional<Node*> replacement;
        Node* src = operand;
        while (src->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                          Op::kOr, Op::kXor})) {
          if (src->Is<Select>()) {
            Select* select = src->As<Select>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<Bits> implied_selector =
                ImpliedNodeValue(edge_set, select->selector(), query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            Node* implied_case =
                GetSelectedCase(select, implied_selector.value());
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %v",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                implied_selector.value());
            replacement = implied_case;
            src = implied_case;
          } else if (src->Is<PrioritySelect>()) {
            PrioritySelect* select = src->As<PrioritySelect>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector =
                ImpliedNodeTernary(edge_set, select->selector(), query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            std::optional<Node*> implied_case =
                GetSelectedCase(select, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                xls::ToString(*implied_selector));
            src = *implied_case;
            replacement = src;
          } else if (src->Is<OneHotSelect>()) {
            XLS_RET_CHECK(src->Is<OneHotSelect>());
            OneHotSelect* ohs = src->As<OneHotSelect>();
            if (ohs->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector =
                ImpliedNodeTernary(edge_set, ohs->selector(), query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            for (int64_t case_no = 0; case_no < ohs->cases().size();
                 ++case_no) {
              if (implied_selector.value()[case_no] ==
                  TernaryValue::kKnownZero) {
                continue;
              }

              // This case could be selected - but if it's definitely zero when
              // selected, then we can ignore it.
              std::optional<Bits> implied_case =
                  ImpliedNodeValue(condition_map.GetEdgeConditionSet(
                                       ohs, /*operand_no=*/case_no + 1),
                                   ohs->cases()[case_no], query_engine);
              if (implied_case.has_value() && implied_case->IsZero()) {
                implied_selector.value()[case_no] = TernaryValue::kKnownZero;
              }
            }
            std::optional<std::variant<Node*, ZeroValue>> implied_case =
                GetSelectedCase(ohs, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(), ohs->selector()->GetName(),
                ohs->GetName(), xls::ToString(*implied_selector));
            if (std::holds_alternative<Node*>(*implied_case)) {
              src = std::get<Node*>(*implied_case);
            } else {
              XLS_RET_CHECK(std::holds_alternative<ZeroValue>(*implied_case));
              XLS_ASSIGN_OR_RETURN(
                  src,
                  f->MakeNode<Literal>(src->loc(), ZeroOfType(src->GetType())));
            }
            replacement = src;
          } else {
            XLS_RET_CHECK(src->OpIn({Op::kAnd, Op::kOr, Op::kXor}));
            auto is_identity = [&](const Bits& b) {
              if (src->op() == Op::kAnd) {
                return b.IsAllOnes();
              }
              return b.IsZero();
            };
            NaryOp* bitwise_op = src->As<NaryOp>();
            std::optional<Node*> nonidentity_operand = std::nullopt;
            for (Node* potential_src : bitwise_op->operands()) {
              XLS_RET_CHECK(potential_src->GetType()->IsBits());
              std::optional<Bits> implied_src =
                  ImpliedNodeValue(edge_set, potential_src, query_engine);
              if (implied_src.has_value() && is_identity(*implied_src)) {
                continue;
              }
              if (nonidentity_operand.has_value()) {
                // There's more than one potentially-non-zero operand; we're
                // done, there's nothing to do.
                nonidentity_operand = std::nullopt;
                break;
              }
              nonidentity_operand = potential_src;
            }
            if (!nonidentity_operand.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply that bitwise operation "
                "%s has only one non-identity operand: %s",
                operand->GetName(), node->GetName(), bitwise_op->GetName(),
                nonidentity_operand.value()->GetName());
            src = *nonidentity_operand;
            replacement = src;
          }
        }
        if (replacement.has_value()) {
          VLOG(3) << absl::StreamFormat(
              "Replacing operand %d of %s with %s due to implied selector "
              "value(s)",
              operand_no, node->GetName(), replacement.value()->GetName());
          XLS_RETURN_IF_ERROR(
              node->ReplaceOperandNumber(operand_no, replacement.value()));
          changed = true;
        }
      }
    }
  }

  return changed;
}

XLS_REGISTER_MODULE_INITIALIZER(cond_spec, {
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(true)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(Bdd)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(false)", false));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(noBdd)", false));
});

}  // namespace xls
