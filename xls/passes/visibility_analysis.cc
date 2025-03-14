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

#include "xls/passes/visibility_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/partial_ops.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

void ConditionSet::Intersect(const ConditionSet& other) {
  auto it = conditions_.begin();
  auto other_it = other.conditions_.begin();
  while (it != conditions_.end() && other_it != other.conditions_.end()) {
    if (condition_cmp_(*it, *other_it)) {
      // No condition on it->node in `other`, so remove it from this set.
      it = conditions_.erase(it);
    } else if (condition_cmp_(*other_it, *it)) {
      // No condition on other_it->node in `this`, so ignore it.
      ++other_it;
    } else {
      // Take the meet of the conditions.
      it->partial.MeetWith(other_it->partial);
      if (it->partial.IsUnconstrained()) {
        it = conditions_.erase(it);
      } else {
        ++it;
      }
      ++other_it;
    }
  }
  // Remove all trailing conditions from `this` that were not in `other`.
  conditions_.erase(it, conditions_.end());
}

void ConditionSet::Union(const ConditionSet& other) {
  ConditionVector original = std::move(conditions_);
  conditions_.clear();
  auto original_it = original.begin();
  auto other_it = other.conditions_.begin();
  while (
      conditions_.size() < kMaxConditions &&
      (original_it != original.end() || other_it != other.conditions_.end())) {
    if (other_it == other.conditions_.end() ||
        (original_it != original.end() &&
         condition_cmp_(*original_it, *other_it))) {
      // No condition on original_it->node in `other`, so just insert the
      // original.
      conditions_.push_back(std::move(*original_it));
      ++original_it;
    } else if (original_it == original.end() ||
               (other_it != other.conditions_.end() &&
                condition_cmp_(*other_it, *original_it))) {
      // No condition on other_it->node in the original, so insert the other.
      conditions_.push_back(*other_it);
      ++other_it;
    } else {
      // Take the join of the conditions, and insert the result.
      Condition condition = std::move(*original_it);
      condition.partial.JoinWith(other_it->partial);
      conditions_.push_back(std::move(condition));
      ++original_it;
      ++other_it;
    }
  }
}

void ConditionSet::AddCondition(const Condition& condition) {
  VLOG(4) << absl::StreamFormat("ConditionSet for (%s) : %s",
                                condition.ToString(), this->ToString());
  CHECK(!condition.node->Is<Literal>());
  auto it = absl::c_lower_bound(conditions_, condition, condition_cmp_);
  if (it != conditions_.end() && it->node == condition.node) {
    it->partial.JoinWith(condition.partial);
    return;
  }
  conditions_.insert(it, condition);

  // The conditions are ordered in topological sort order (based on
  // Condition.node) and transformation occurs in reverse topological sort
  // order so the most distant conditions should be at the end of the
  // condition set. Just pop the last condition off the end if it exceeds the
  // limit.
  if (conditions_.size() > kMaxConditions) {
    conditions_.pop_back();
  }
  CHECK_LE(conditions_.size(), kMaxConditions);
}

std::vector<std::pair<TreeBitLocation, bool>> ConditionSet::GetPredicates()
    const {
  std::vector<std::pair<TreeBitLocation, bool>> predicates;
  for (const Condition& condition : conditions()) {
    std::optional<TernarySpan> ternary = condition.partial.Ternary();
    if (!ternary.has_value()) {
      continue;
    }
    for (int64_t i = 0; i < condition.node->BitCountOrDie(); ++i) {
      if (ternary->at(i) == TernaryValue::kUnknown) {
        continue;
      }
      bool bit_value = (ternary->at(i) == TernaryValue::kKnownOne);
      predicates.push_back({TreeBitLocation{condition.node, i}, bit_value});
    }
  }
  return predicates;
}

absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan>
ConditionSet::GetAsGivens() const {
  absl::btree_map<Node*, ValueKnowledge, Node::NodeIdLessThan> givens;
  for (const Condition& condition : conditions()) {
    if (condition.partial.IsUnconstrained()) {
      continue;
    }

    ValueKnowledge& given = givens[condition.node];
    if (condition.partial.Ternary().has_value()) {
      if (given.ternary.has_value()) {
        if (absl::Status merged = ternary_ops::UpdateWithUnion(
                given.ternary->Get({}), *condition.partial.Ternary());
            !merged.ok()) {
          // This is impossible, as the conditions contradict each other. For
          // now, we can't do anything about this; it might be worth finding a
          // way to propagate this information.
          VLOG(1) << "Proved this condition set is impossible: " << ToString();
          return {};
        }
      } else {
        given.ternary = TernaryTree::CreateSingleElementTree(
            condition.node->GetType(), *condition.partial.Ternary());
      }
    }
    if (condition.partial.Range().has_value()) {
      IntervalSet range = *condition.partial.Range();
      if (given.intervals.has_value()) {
        range = IntervalSet::Intersect(range, given.intervals->Get({}));
      }
      given.intervals = IntervalSetTree::CreateSingleElementTree(
          condition.node->GetType(), std::move(range));
    }
  }
  return givens;
}

namespace {

class ImpliedConditionCache {
 public:
  ImpliedConditionCache(FunctionBase* f, OptimizationContext& context,
                        QueryEngine* query_engine)
      : f_(f),
        query_engine_(query_engine),
        topo_index_(std::make_shared<absl::flat_hash_map<Node*, int64_t>>()) {
    int64_t index = 0;
    for (Node* node : context.TopoSort(f_)) {
      CHECK(topo_index_->emplace(node, index++).second);
    }
  }

  const ConditionSet& GetImplied(const Condition& condition);

  ConditionSet GetEmpty() const { return ConditionSet(topo_index_); }

  void Clear() {
    cache_.clear();
    topo_index_->clear();
  }

 private:
  FunctionBase* f_;

  friend class VisibilityAnalysis;
  void SetQueryEngine(QueryEngine* query_engine) {
    query_engine_ = query_engine;
  }
  QueryEngine* query_engine_;

  // Index of each node in the function base in a topological sort.
  std::shared_ptr<absl::flat_hash_map<Node*, int64_t>> topo_index_;

  absl::flat_hash_map<Condition, std::unique_ptr<ConditionSet>> cache_;
};

const ConditionSet& ImpliedConditionCache::GetImplied(
    const Condition& condition) {
  if (auto it = cache_.find(condition); it != cache_.end()) {
    return *it->second;
  }
  auto [it, _] =
      cache_.emplace(condition, std::make_unique<ConditionSet>(topo_index_));
  ConditionSet& implied_conditions = *it->second;

  implied_conditions.AddCondition(condition);

  if (condition.node->op() == Op::kNot &&
      condition.partial.Ternary().has_value() &&
      !condition.node->operand(0)->Is<Literal>()) {
    Node* operand = condition.node->operand(0);

    VLOG(4) << "Lifting a known negated value: not(" << operand->GetName()
            << ") == " << xls::ToString(*condition.partial.Ternary());

    implied_conditions.Union(GetImplied(Condition{
        .node = operand, .partial = partial_ops::Not(condition.partial)}));
  }

  if (condition.node->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor}) &&
      condition.partial.Ternary().has_value()) {
    TernarySpan value = *condition.partial.Ternary();
    TernaryVector lifted_value =
        condition.node->OpIn({Op::kNand, Op::kNor})
            ? ternary_ops::Not(value)
            : TernaryVector(value.begin(), value.end());
    TernaryValue lifted_bit = condition.node->OpIn({Op::kAnd, Op::kNand})
                                  ? TernaryValue::kKnownOne
                                  : TernaryValue::kKnownZero;
    TernaryValue non_lifted_bit = ternary_ops::Not(lifted_bit);

    if (absl::c_contains(lifted_value, lifted_bit)) {
      for (int64_t i = 0; i < lifted_value.size(); ++i) {
        if (lifted_value[i] == non_lifted_bit) {
          lifted_value[i] = TernaryValue::kUnknown;
        }
      }
      VLOG(4) << "Lifting known bits; " << OpToString(condition.node->op())
              << "("
              << absl::StrJoin(condition.node->operands(), ", ",
                               [](std::string* out, Node* node) {
                                 absl::StrAppend(out, node->GetName());
                               })
              << ") == " << xls::ToString(value)
              << ", so all operands must match: "
              << xls::ToString(lifted_value);
      for (Node* operand : condition.node->operands()) {
        if (operand->Is<Literal>()) {
          continue;
        }
        implied_conditions.Union(GetImplied(Condition{
            .node = operand, .partial = PartialInformation(lifted_value)}));
      }
    }
  }

  bool lhs_rhs_equal = false;
  bool lhs_rhs_unequal = false;
  if (condition.node->OpIn({Op::kEq, Op::kNe}) &&
      condition.partial.Range().has_value() &&
      condition.partial.Range()->IsPrecise()) {
    lhs_rhs_equal = (condition.node->op() == Op::kEq) ==
                    (condition.partial.Range()->GetPreciseValue()->IsOne());
    lhs_rhs_unequal = (condition.node->op() == Op::kNe) ==
                      (condition.partial.Range()->GetPreciseValue()->IsOne());
  }
  if (lhs_rhs_equal && condition.node->operand(0)->GetType()->IsBits()) {
    Node* lhs = condition.node->operand(0);
    Node* rhs = condition.node->operand(1);
    CHECK(rhs->GetType()->IsBits());

    VLOG(4) << "Converting a known equality to direct conditions: "
            << lhs->GetName() << " == " << rhs->GetName();

    std::optional<SharedLeafTypeTree<TernaryVector>> lhs_ternary =
        query_engine_->GetTernary(lhs);
    IntervalSet lhs_range = query_engine_->GetIntervals(lhs).Get({});
    bool info_on_lhs = (lhs_ternary.has_value() &&
                        !ternary_ops::AllUnknown(lhs_ternary->Get({}))) ||
                       !lhs_range.IsMaximal();
    if (!rhs->Is<Literal>() && info_on_lhs) {
      implied_conditions.Union(GetImplied(
          Condition{.node = rhs,
                    .partial = {lhs_ternary.has_value()
                                    ? std::make_optional(lhs_ternary->Get({}))
                                    : std::nullopt,
                                lhs_range}}));
    }

    std::optional<SharedLeafTypeTree<TernaryVector>> rhs_ternary =
        query_engine_->GetTernary(rhs);
    IntervalSet rhs_range = query_engine_->GetIntervals(rhs).Get({});
    bool info_on_rhs = (rhs_ternary.has_value() &&
                        !ternary_ops::AllUnknown(rhs_ternary->Get({}))) ||
                       !rhs_range.IsMaximal();
    if (!lhs->Is<Literal>() && info_on_rhs) {
      implied_conditions.Union(GetImplied(
          Condition{.node = lhs,
                    .partial = {rhs_ternary.has_value()
                                    ? std::make_optional(rhs_ternary->Get({}))
                                    : std::nullopt,
                                rhs_range}}));
    }
  }

  if (lhs_rhs_unequal && condition.node->operand(0)->GetType()->IsBits()) {
    Node* lhs = condition.node->operand(0);
    Node* rhs = condition.node->operand(1);
    CHECK(rhs->GetType()->IsBits());

    VLOG(4) << "Converting a known inequality to direct conditions: "
            << lhs->GetName() << " != " << rhs->GetName();

    if (std::optional<Value> lhs_value = query_engine_->KnownValue(lhs);
        !rhs->Is<Literal>() && rhs->GetType()->IsBits() &&
        lhs_value.has_value()) {
      implied_conditions.Union(GetImplied(Condition{
          .node = rhs,
          .partial =
              PartialInformation(IntervalSet::Punctured(lhs_value->bits()))}));
    }
    if (std::optional<Value> rhs_value = query_engine_->KnownValue(rhs);
        !lhs->Is<Literal>() && lhs->GetType()->IsBits() &&
        rhs_value.has_value()) {
      implied_conditions.Union(GetImplied(Condition{
          .node = lhs,
          .partial =
              PartialInformation(IntervalSet::Punctured(rhs_value->bits()))}));
    }
  }

  return implied_conditions;
}

class VisibilityAnalysis
    : public LazyDagCache<TrackedValue, ConditionSet>::DagProvider,
      public ChangeListener {
 private:
  using CacheState = LazyDagCache<TrackedValue, ConditionSet>::CacheState;

 public:
  VisibilityAnalysis(FunctionBase* f, OptimizationContext& context,
                     bool use_bdd);
  ~VisibilityAnalysis() override { f_->UnregisterChangeListener(this); }

  VisibilityAnalysis(const VisibilityAnalysis&) = delete;
  VisibilityAnalysis& operator=(const VisibilityAnalysis&) = delete;
  VisibilityAnalysis(VisibilityAnalysis&&) = delete;
  VisibilityAnalysis& operator=(VisibilityAnalysis&&) = delete;

  const ConditionSet& GetVisibilityConditions(Node* node) {
    return *cache_.QueryValue(TrackedValue{.node = node, .operand_no = -1});
  }

  const ConditionSet& GetVisibilityConditionsForEdge(Node* node,
                                                     int64_t operand_no) {
    return *cache_.QueryValue(
        TrackedValue{.node = node, .operand_no = operand_no});
  }

  std::string GetName(const TrackedValue& tracked) const override {
    if (tracked.IsNode()) {
      return tracked.node->GetName();
    }
    return absl::StrFormat("%s (operand %d)", tracked.node->GetName(),
                           tracked.operand_no);
  }
  std::vector<TrackedValue> GetInputs(
      const TrackedValue& tracked) const override;
  std::vector<TrackedValue> GetUsers(
      const TrackedValue& tracked) const override;

  absl::StatusOr<ConditionSet> ComputeValue(
      const TrackedValue& key,
      absl::Span<const ConditionSet* const> input_values) const override;

  // This isn't actually a lazy cache; we need to invalidate it on approximately
  // every change, as any change to the topological order affects the structure
  // of our ConditionSets.
  void NodeAdded(Node* node) override {
    cache_.ClearNonForced();
    condition_cache_.Clear();
  }
  void NodeDeleted(Node* node) override {
    cache_.ClearNonForced();
    condition_cache_.Clear();
  }
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override {
    cache_.ClearNonForced();
    condition_cache_.Clear();
  }
  void OperandRemoved(Node* node, Node* old_operand) override {
    cache_.ClearNonForced();
    condition_cache_.Clear();
  }
  void OperandAdded(Node* node) override {
    cache_.ClearNonForced();
    condition_cache_.Clear();
  }

  absl::Status EagerlyPopulate(
      absl::Span<const TrackedValue> topo_sorted_values) {
    return cache_.EagerlyPopulate(topo_sorted_values);
  }

 private:
  FunctionBase* f_;
  mutable ImpliedConditionCache condition_cache_;
  LazyDagCache<TrackedValue, ConditionSet> cache_;

  std::unique_ptr<QueryEngine> query_engine_;
};

VisibilityAnalysis::VisibilityAnalysis(FunctionBase* f,
                                       OptimizationContext& context,
                                       bool use_bdd)
    : f_(f),
      condition_cache_(f, context,
                       /*query_engine=*/nullptr),
      cache_(this) {
  CHECK(f_ != nullptr);
  f_->RegisterChangeListener(this);
  if (use_bdd) {
    query_engine_ = UnionQueryEngine::UniquePtrOf(
        StatelessQueryEngine(),
        context.GetForwardingQueryEngine<BddQueryEngine>(f));
    CHECK_OK(query_engine_->Populate(f));
  } else {
    query_engine_ = std::make_unique<StatelessQueryEngine>();
  }
  condition_cache_.SetQueryEngine(query_engine_.get());
}

std::vector<TrackedValue> VisibilityAnalysis::GetInputs(
    const TrackedValue& tracked) const {
  if (tracked.IsEdge()) {
    // The only point that affects a tracked edge directly is the receiving
    // node.
    return {TrackedValue{.node = tracked.node, .operand_no = -1}};
  }
  std::vector<TrackedValue> consumers;
  consumers.reserve(tracked.node->users().size());
  for (Node* user : tracked.node->users()) {
    for (int64_t operand_no = 0; operand_no < user->operand_count();
         ++operand_no) {
      if (user->operand(operand_no) == tracked.node) {
        consumers.push_back(
            TrackedValue{.node = user, .operand_no = operand_no});
      }
    }
  }
  return consumers;
}

std::vector<TrackedValue> VisibilityAnalysis::GetUsers(
    const TrackedValue& tracked) const {
  if (tracked.IsEdge()) {
    // The only consumer of a tracked edge is the sending node.
    return {TrackedValue{.node = tracked.node->operand(tracked.operand_no),
                         .operand_no = -1}};
  }
  std::vector<TrackedValue> producers;
  producers.reserve(tracked.node->operands().size());
  for (int64_t operand_no = 0; operand_no < tracked.node->operand_count();
       ++operand_no) {
    producers.push_back(
        TrackedValue{.node = tracked.node, .operand_no = operand_no});
  }
  return producers;
}

absl::StatusOr<ConditionSet> VisibilityAnalysis::ComputeValue(
    const TrackedValue& key,
    absl::Span<const ConditionSet* const> input_values) const {
  Node* node = key.node;
  int64_t operand_no = key.operand_no;

  if (key.IsNode()) {
    if (node->Is<Invoke>()) {
      // The contents of an invoke may be side-effecting (e.g., the invoked
      // function might contain an assert), so don't assume any conditions for
      // this node or its predecessors.
      VLOG(4) << absl::StreamFormat(
          "Node %s is an invoke and could be side-effecting", node->GetName());
      return condition_cache_.GetEmpty();
    }
    if (OpIsSideEffecting(node->op()) &&
        !node->OpIn({Op::kSend, Op::kStateRead, Op::kNext})) {
      // Inputs to side-effecting operations should not change so don't assume
      // any conditions for this node or it's predecessors.
      VLOG(4) << absl::StreamFormat("Node %s is side-effecting",
                                    node->GetName());
      return condition_cache_.GetEmpty();
    }

    // If this node has an implicit use then we can't propagate any conditions
    // from the users because this value is unconditionally live and therefore
    // its computed value should not be changed. Similarly, if this node is a
    // Send then the data input can only be controlled by the predicate, not by
    // any successors of the Send.
    //
    // In addition, if this node is a StateRead's predicate, then its value can
    // affect throughput and so shouldn't be changed.
    if (f_->HasImplicitUse(node) || node->Is<Send>() ||
        absl::c_any_of(node->users(),
                       [](Node* user) { return user->Is<StateRead>(); })) {
      return condition_cache_.GetEmpty();
    }
  }

  // We start by intersecting the conditions of all users (if any).
  ConditionSet set = condition_cache_.GetEmpty();
  if (!input_values.empty()) {
    set = *input_values.front();
    for (const ConditionSet* user_set : input_values.subspan(1)) {
      set.Intersect(*user_set);
    }
  }

  // If we're at a node, we're done.
  if (key.IsNode()) {
    return set;
  }

  // Since we're at an edge, we need to add in the conditions under which this
  // node looks at this operand.

  // Don't bother specializing if the selector is a literal as this results in
  // a useless condition where we assume a literal has a literal value.
  if (node->Is<Select>() && !node->As<Select>()->selector()->Is<Literal>() &&
      operand_no > Select::kSelectorOperand) {
    Select* select = node->As<Select>();
    int64_t case_no = operand_no - 1;
    if (case_no < select->cases().size()) {
      set.Union(condition_cache_.GetImplied(Condition{
          .node = select->selector(),
          .partial = PartialInformation(IntervalSet::Precise(
              UBits(case_no, select->selector()->BitCountOrDie()))),
      }));
    } else {
      // This is the default case, matched whenever the selector is larger
      // than the last case index.
      set.Union(condition_cache_.GetImplied(Condition{
          .node = select->selector(),
          .partial = PartialInformation(IntervalSet::Of(
              {{UBits(select->cases().size(),
                      select->selector()->BitCountOrDie()),
                Bits::AllOnes(select->selector()->BitCountOrDie())}})),
      }));
    }
  }
  if (node->Is<OneHotSelect>() &&
      !node->As<OneHotSelect>()->selector()->Is<Literal>() &&
      operand_no > OneHotSelect::kSelectorOperand) {
    OneHotSelect* select = node->As<OneHotSelect>();
    int64_t case_no = operand_no - 1;
    // If this case is selected, we know the corresponding bit of the
    // selector is one.
    TernaryVector selector_value(select->selector()->BitCountOrDie(),
                                 TernaryValue::kUnknown);
    selector_value[case_no] = TernaryValue::kKnownOne;
    set.Union(condition_cache_.GetImplied(Condition{
        .node = select->selector(),
        .partial = PartialInformation(selector_value),
    }));
  }
  if (node->Is<PrioritySelect>() &&
      !node->As<PrioritySelect>()->selector()->Is<Literal>() &&
      operand_no > PrioritySelect::kSelectorOperand) {
    PrioritySelect* select = node->As<PrioritySelect>();
    int64_t case_no = operand_no - 1;
    // If this case is selected, we know all the bits of the selector up
    // to and including `case_no`; all higher-priority bits are zero, and
    // (unless this is the default case) this case's bit is one.
    if (case_no < select->cases().size()) {
      Bits known_bits = Bits(select->selector()->BitCountOrDie());
      known_bits.SetRange(0, case_no + 1);
      Bits known_bits_values =
          Bits::PowerOfTwo(case_no, select->selector()->BitCountOrDie());
      set.Union(condition_cache_.GetImplied(
          Condition{.node = select->selector(),
                    .partial = PartialInformation(ternary_ops::FromKnownBits(
                        known_bits, known_bits_values))}));
    } else {
      // Default case, selected when all bits of the selector are zero.
      set.Union(condition_cache_.GetImplied(Condition{
          .node = select->selector(),
          .partial = PartialInformation(
              IntervalSet::Precise(Bits(select->selector()->BitCountOrDie()))),
      }));
    }
  }

  if (node->Is<ArrayUpdate>() &&
      operand_no == ArrayUpdate::kUpdateValueOperand) {
    ArrayUpdate* update = node->As<ArrayUpdate>();
    Type* array_type = update->array_to_update()->GetType();
    for (Node* index : update->indices()) {
      if (index->Is<Literal>()) {
        continue;
      }

      const int64_t array_size = array_type->AsArrayOrDie()->size();
      if (Bits::MinBitCountUnsigned(array_size) > index->BitCountOrDie()) {
        continue;
      }

      // ArrayUpdate is a no-op if any index is out of range; as such, it
      // only cares about the update value if all indices are in range.
      set.Union(condition_cache_.GetImplied(Condition{
          .node = index,
          .partial = PartialInformation(IntervalSet::Of({Interval::RightOpen(
              UBits(0, index->BitCountOrDie()),
              UBits(array_size, index->BitCountOrDie()))})),
      }));

      array_type = array_type->AsArrayOrDie()->element_type();
    }
  }

  if (node->Is<Send>() && operand_no == Send::kDataOperand &&
      node->As<Send>()->predicate().has_value() &&
      !(*node->As<Send>()->predicate())->Is<Literal>()) {
    Send* send = node->As<Send>();
    set.Union(condition_cache_.GetImplied(Condition{
        .node = *send->predicate(),
        .partial = PartialInformation(IntervalSet::Precise(UBits(1, 1)))}));
  }

  if (node->Is<Next>() &&
      (operand_no == Next::kValueOperand ||
       operand_no == Next::kStateReadOperand) &&
      node->As<Next>()->predicate().has_value() &&
      !(*node->As<Next>()->predicate())->Is<Literal>()) {
    Next* next = node->As<Next>();
    set.Union(condition_cache_.GetImplied(Condition{
        .node = *next->predicate(),
        .partial = PartialInformation(IntervalSet::Precise(UBits(1, 1)))}));
  }

  return set;
}

}  // namespace

absl::StatusOr<VisibilityConditions> RunVisibilityAnalysis(
    FunctionBase* f, OptimizationContext& context, bool use_bdd) {
  std::vector<TrackedValue> keys;
  int64_t key_count = absl::c_accumulate(
      context.ReverseTopoSort(f), 0, [&](int64_t count, Node* node) {
        return count + 1 + node->operand_count();
      });
  keys.reserve(key_count);
  for (Node* node : context.ReverseTopoSort(f)) {
    keys.push_back(TrackedValue{.node = node, .operand_no = -1});
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      keys.push_back(TrackedValue{.node = node, .operand_no = operand_no});
    }
  }

  absl::flat_hash_map<TrackedValue, ConditionSet> results;
  results.reserve(keys.size());
  VisibilityAnalysis analysis(f, context, use_bdd);
  XLS_RETURN_IF_ERROR(analysis.EagerlyPopulate(keys));
  for (const TrackedValue& key : keys) {
    if (key.IsNode()) {
      results.emplace(key, analysis.GetVisibilityConditions(key.node));
    } else {
      results.emplace(key, analysis.GetVisibilityConditionsForEdge(
                               key.node, key.operand_no));
    }
  }
  return VisibilityConditions(std::move(results));
}

}  // namespace xls
