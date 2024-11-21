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

#include "xls/passes/bdd_query_engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_function.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"

namespace xls {

class BddQueryEngine::AssumingBddQueryEngine final : public QueryEngine {
 public:
  AssumingBddQueryEngine(const BddQueryEngine* query_engine,
                         BddNodeIndex assumption)
      : query_engine_(query_engine), assumption_(assumption) {}
  AssumingBddQueryEngine(std::shared_ptr<BddQueryEngine> query_engine,
                         BddNodeIndex assumption)
      : query_engine_storage_(std::move(query_engine)),
        query_engine_(query_engine_storage_.get()),
        assumption_(assumption) {}

  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;
  bool IsTracked(Node* node) const override;
  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override;
  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override;
  std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const override;
  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override;
  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override;
  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override;
  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override;
  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override;
  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override;
  bool IsKnown(const TreeBitLocation& bit) const override;
  std::optional<bool> KnownValue(const TreeBitLocation& bit) const override;
  bool IsAllZeros(Node* n) const override;
  bool IsAllOnes(Node* n) const override;
  bool IsFullyKnown(Node* n) const override;

 private:
  std::shared_ptr<BddQueryEngine> query_engine_storage_;
  const BddQueryEngine* query_engine_;

  BddNodeIndex assumption_;
};

absl::StatusOr<ReachedFixpoint>
BddQueryEngine::AssumingBddQueryEngine::Populate(FunctionBase* f) {
  return absl::UnimplementedError("Cannot populate forwarding engine!");
}

bool BddQueryEngine::AssumingBddQueryEngine::IsTracked(Node* node) const {
  return query_engine_->IsTracked(node);
}
std::optional<SharedLeafTypeTree<TernaryVector>>
BddQueryEngine::AssumingBddQueryEngine::GetTernary(Node* node) const {
  if (!query_engine_->IsTracked(node)) {
    return std::nullopt;
  }
  absl::StatusOr<TernaryTree> ltt = TernaryTree::CreateFromFunction(
      node->GetType(),
      [&](Type* leaf_type, absl::Span<const int64_t> tree_index)
          -> absl::StatusOr<TernaryVector> {
        TernaryVector ternary(leaf_type->GetFlatBitCount(),
                              TernaryValue::kUnknown);
        for (int64_t bit_index = 0; bit_index < leaf_type->GetFlatBitCount();
             ++bit_index) {
          std::optional<BddNodeIndex> bit = query_engine_->GetBddNode(
              TreeBitLocation(node, bit_index, tree_index));
          if (!bit.has_value()) {
            continue;
          }
          if (query_engine_->Implies(assumption_, *bit)) {
            ternary[bit_index] = TernaryValue::kKnownOne;
          } else if (query_engine_->Implies(assumption_,
                                            query_engine_->bdd().Not(*bit))) {
            ternary[bit_index] = TernaryValue::kKnownZero;
          }
        }
        return ternary;
      });
  if (!ltt.ok()) {
    return std::nullopt;
  }
  return std::move(ltt).value().AsShared();
};

std::unique_ptr<QueryEngine>
BddQueryEngine::AssumingBddQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  std::unique_ptr<AssumingBddQueryEngine> specialized(
      down_cast<AssumingBddQueryEngine*>(
          query_engine_->SpecializeGivenPredicate(state).release()));
  specialized->assumption_ =
      query_engine_->bdd().And(assumption_, specialized->assumption_);
  return specialized;
}

std::unique_ptr<QueryEngine>
BddQueryEngine::AssumingBddQueryEngine::SpecializeGiven(
    const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const {
  std::unique_ptr<AssumingBddQueryEngine> result;
  if (query_engine_storage_ != nullptr) {
    result = std::make_unique<AssumingBddQueryEngine>(query_engine_storage_,
                                                      assumption_);
  } else {
    result =
        std::make_unique<AssumingBddQueryEngine>(query_engine_, assumption_);
  }
  BddNodeIndex new_assumption =
      query_engine_->bdd().And(assumption_, result->assumption_);
  if (query_engine_->ExceedsPathLimit(result->assumption_)) {
    // Restore the original assumption, with no updates.
    result->assumption_ = assumption_;
  } else {
    result->assumption_ = new_assumption;
  }
  return result;
}

LeafTypeTree<IntervalSet> BddQueryEngine::AssumingBddQueryEngine::GetIntervals(
    Node* node) const {
  // The default QueryEngine::GetIntervals implementation will inherit the
  // specialization from GetTernary.
  return QueryEngine::GetIntervals(node);
}

bool BddQueryEngine::AssumingBddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return query_engine_->AtMostOneTrue(bits, assumption_);
}

bool BddQueryEngine::AssumingBddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return query_engine_->AtLeastOneTrue(bits, assumption_);
}

bool BddQueryEngine::AssumingBddQueryEngine::Implies(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = query_engine_->GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = query_engine_->GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return query_engine_->Implies(query_engine_->bdd().And(*a_bdd, assumption_),
                                *b_bdd);
}

std::optional<Bits> BddQueryEngine::AssumingBddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return query_engine_->ImpliedNodeValue(predicate_bit_values, node,
                                         assumption_);
}

std::optional<TernaryVector>
BddQueryEngine::AssumingBddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return query_engine_->ImpliedNodeTernary(predicate_bit_values, node,
                                           assumption_);
}

bool BddQueryEngine::AssumingBddQueryEngine::KnownEquals(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  return query_engine_->KnownEquals(a, b, assumption_);
}

bool BddQueryEngine::AssumingBddQueryEngine::KnownNotEquals(
    const TreeBitLocation& a, const TreeBitLocation& b) const {
  return query_engine_->KnownNotEquals(a, b, assumption_);
}

bool BddQueryEngine::AssumingBddQueryEngine::IsKnown(
    const TreeBitLocation& bit) const {
  return query_engine_->IsKnown(bit, assumption_);
}
std::optional<bool> BddQueryEngine::AssumingBddQueryEngine::KnownValue(
    const TreeBitLocation& bit) const {
  return query_engine_->KnownValue(bit, assumption_);
}
bool BddQueryEngine::AssumingBddQueryEngine::IsAllZeros(Node* n) const {
  return query_engine_->IsAllZeros(n, assumption_);
}
bool BddQueryEngine::AssumingBddQueryEngine::IsAllOnes(Node* n) const {
  return query_engine_->IsAllOnes(n, assumption_);
}
bool BddQueryEngine::AssumingBddQueryEngine::IsFullyKnown(Node* n) const {
  return query_engine_->IsFullyKnown(n, assumption_);
}

absl::StatusOr<ReachedFixpoint> BddQueryEngine::Populate(FunctionBase* f) {
  XLS_ASSIGN_OR_RETURN(bdd_function_,
                       BddFunction::Run(f, path_limit_, node_filter_));
  // Construct the Bits objects indication which bit values are statically known
  // for each node and what those values are (0 or 1) if known.
  BinaryDecisionDiagram& bdd = this->bdd();
  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      absl::InlinedVector<bool, 1> known_bits;
      absl::InlinedVector<bool, 1> bits_values;
      for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
        if (GetBddNode(TreeBitLocation(node, i)) == bdd.zero()) {
          known_bits.push_back(true);
          bits_values.push_back(false);
        } else if (GetBddNode(TreeBitLocation(node, i)) == bdd.one()) {
          known_bits.push_back(true);
          bits_values.push_back(true);
        } else {
          known_bits.push_back(false);
          bits_values.push_back(false);
        }
      }
      if (!known_bits_.contains(node)) {
        known_bits_[node] = Bits(known_bits.size());
        bits_values_[node] = Bits(bits_values.size());
      }
      Bits new_known_bits(known_bits);
      Bits new_bits_values(bits_values);
      // TODO(taktoa): check for inconsistency
      Bits ored_known_bits = bits_ops::Or(known_bits_[node], new_known_bits);
      Bits ored_bits_values = bits_ops::Or(bits_values_[node], new_bits_values);
      if ((ored_known_bits != known_bits_[node]) ||
          (ored_bits_values != bits_values_[node])) {
        rf = ReachedFixpoint::Changed;
      }
      known_bits_[node] = ored_known_bits;
      bits_values_[node] = ored_bits_values;
    }
  }
  return rf;
}

std::unique_ptr<QueryEngine> BddQueryEngine::SpecializeGivenPredicate(
    const absl::flat_hash_set<PredicateState>& state) const {
  absl::flat_hash_map<Node*, ValueKnowledge> givens;
  for (const PredicateState& ps : state) {
    if (ps.IsBasePredicate()) {
      continue;
    }
    if (ps.IsSelectPredicate()) {
      // Add the assumption that this arm of the select is taken.
      //
      // We need to build the equality assertion.
      Node* selector = ps.selector();
      CHECK(selector->GetType()->IsBits());
      if (ps.node()->Is<Select>()) {
        if (ps.IsDefaultArm()) {
          // The selector can be assumed to be past the end of the cases...
          // unless it doesn't fit, in which case this is impossible.
          int64_t num_cases = ps.node()->As<Select>()->cases().size();
          if (Bits::MinBitCountUnsigned(num_cases) <=
              selector->BitCountOrDie()) {
            IntervalSet intervals(selector->BitCountOrDie());
            intervals.AddInterval(
                Interval::Closed(UBits(ps.node()->As<Select>()->cases().size(),
                                       selector->BitCountOrDie()),
                                 Bits::AllOnes(selector->BitCountOrDie())));
            intervals.Normalize();

            if (auto it = givens.find(selector);
                it != givens.end() && it->second.intervals.has_value()) {
              intervals = IntervalSet::Intersect(intervals,
                                                 it->second.intervals->Get({}));
            }
            givens[selector].intervals =
                IntervalSetTree::CreateSingleElementTree(selector->GetType(),
                                                         std::move(intervals));
          } else {
            // TODO(epastor): Use the impossibility of the default case somehow.
          }
        } else {
          // The selector can be assumed equal to the arm index... unless it
          // doesn't fit, in which case this is impossible.
          if (Bits::MinBitCountUnsigned(ps.arm_index()) <=
              selector->BitCountOrDie()) {
            IntervalSet intervals = IntervalSet::Precise(
                UBits(ps.arm_index(), selector->BitCountOrDie()));
            if (auto it = givens.find(selector);
                it != givens.end() && it->second.intervals.has_value()) {
              intervals = IntervalSet::Intersect(intervals,
                                                 it->second.intervals->Get({}));
            }
            givens[selector].intervals =
                IntervalSetTree::CreateSingleElementTree(selector->GetType(),
                                                         std::move(intervals));
          } else {
            // TODO(epastor): Use the impossibility of this arm somehow.
          }
        }
      } else if (ps.node()->Is<PrioritySelect>()) {
        // PrioritySelect has a selector which is a vector of bits.
        Node* selector = ps.selector();
        CHECK(selector->GetType()->IsBits());
        if (ps.IsDefaultArm()) {
          // The default arm is taken iff all bits are zero.
          IntervalSet intervals =
              IntervalSet::Precise(UBits(0, selector->BitCountOrDie()));
          if (auto it = givens.find(selector);
              it != givens.end() && it->second.intervals.has_value()) {
            intervals = IntervalSet::Intersect(intervals,
                                               it->second.intervals->Get({}));
          }
          givens[selector].intervals = IntervalSetTree::CreateSingleElementTree(
              selector->GetType(), std::move(intervals));
        } else {
          // The arm is taken iff the corresponding bit is one and all previous
          // bits are zero.
          CHECK_LT(ps.arm_index(), selector->BitCountOrDie());
          TernaryVector ternary(selector->BitCountOrDie(),
                                TernaryValue::kUnknown);
          for (int64_t i = 0; i < ps.arm_index(); ++i) {
            ternary[i] = TernaryValue::kKnownZero;
          }
          ternary[ps.arm_index()] = TernaryValue::kKnownOne;

          if (auto it = givens.find(selector);
              it != givens.end() && it->second.ternary.has_value()) {
            if (!ternary_ops::UpdateWithUnion(ternary,
                                              it->second.ternary->Get({}))
                     .ok()) {
              // TODO(epastor): Use the impossibility of this arm somehow.
            }
          }
          givens[selector].ternary = TernaryTree::CreateSingleElementTree(
              selector->GetType(), std::move(ternary));
        }
      } else if (ps.node()->Is<OneHotSelect>()) {
        // OneHotSelect has a selector which is a vector of bits.
        Node* selector = ps.selector();
        CHECK(selector->GetType()->IsBits());
        // The arm is taken iff the corresponding bit is one.
        CHECK_LT(ps.arm_index(), selector->BitCountOrDie());
        TernaryVector ternary(selector->BitCountOrDie(),
                              TernaryValue::kUnknown);
        ternary[ps.arm_index()] = TernaryValue::kKnownOne;
        if (auto it = givens.find(selector);
            it != givens.end() && it->second.ternary.has_value()) {
          if (!ternary_ops::UpdateWithUnion(ternary,
                                            it->second.ternary->Get({}))
                   .ok()) {
            // TODO(epastor): Use the impossibility of this arm somehow.
          }
        }
        givens[selector].ternary = TernaryTree::CreateSingleElementTree(
            selector->GetType(), std::move(ternary));
      } else {
        LOG(FATAL) << "Unsupported select type: " << ps.node();
      }
    } else if (ps.IsArrayUpdatePredicate()) {
      LOG(ERROR) << "ArrayUpdatePredicate for: " << ps.node();
      // Add an assumption that the indices are in or out of bounds.
      std::vector<BddNodeIndex> in_bounds_checks;
      in_bounds_checks.reserve(ps.indices().size());
      Type* array_type = ps.node()->GetType();
      for (Node* index : ps.indices()) {
        CHECK(array_type->IsArray());
        const int64_t array_bound = array_type->AsArrayOrDie()->size();

        CHECK(index->GetType()->IsBits());
        if (index->BitCountOrDie() >= Bits::MinBitCountUnsigned(array_bound)) {
          // Could be out of bounds; add the check.
          IntervalSet intervals(index->BitCountOrDie());
          intervals.AddInterval(
              Interval::RightOpen(UBits(0, index->BitCountOrDie()),
                                  UBits(array_bound, index->BitCountOrDie())));
          intervals.Normalize();

          if (auto it = givens.find(index);
              it != givens.end() && it->second.intervals.has_value()) {
            intervals = IntervalSet::Intersect(intervals,
                                               it->second.intervals->Get({}));
          }
          givens[index].intervals = IntervalSetTree::CreateSingleElementTree(
              index->GetType(), std::move(intervals));
        }

        array_type = array_type->AsArrayOrDie()->element_type();
      }
    }
  }
  return SpecializeGiven(givens);
}

std::unique_ptr<QueryEngine> BddQueryEngine::SpecializeGiven(
    const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const {
  std::vector<BddNodeIndex> assumptions;

  SaturatingBddEvaluator evaluator(path_limit_, &bdd());
  for (const auto& [node, value_knowledge] : givens) {
    if (value_knowledge.ternary.has_value()) {
      VLOG(2) << "Specializing on " << node->GetName() << " = "
              << value_knowledge.ternary->ToString(
                     [](TernarySpan span) { return xls::ToString(span); });
      CHECK_OK(leaf_type_tree::ForEachIndex(
          value_knowledge.ternary->AsView(),
          [&](Type*, TernarySpan ternary,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            // Iterate from high to low bits; that way, if we prune some due to
            // exceeding the path limit, we will preferentially retain
            // high-order bit information.
            for (int64_t bit_index = ternary.size() - 1; bit_index >= 0;
                 --bit_index) {
              if (ternary_ops::IsUnknown(ternary[bit_index])) {
                continue;
              }
              std::optional<BddNodeIndex> bit =
                  GetBddNode(TreeBitLocation(node, bit_index, tree_index));
              if (!bit.has_value()) {
                continue;
              }
              assumptions.push_back(ternary[bit_index] ==
                                            TernaryValue::kKnownOne
                                        ? *bit
                                        : bdd().Not(*bit));
            }
            return absl::OkStatus();
          }));
    }
    if (value_knowledge.intervals.has_value()) {
      VLOG(2) << "Specializing on " << node->GetName() << " in "
              << value_knowledge.intervals->ToString();
      CHECK_OK(leaf_type_tree::ForEachIndex(
          value_knowledge.intervals->AsView(),
          [&](Type*, const IntervalSet& intervals,
              absl::Span<const int64_t> tree_index) -> absl::Status {
            std::vector<SaturatingBddNodeIndex> bits;
            bits.reserve(intervals.BitCount());
            for (int64_t i = 0; i < intervals.BitCount(); ++i) {
              std::optional<BddNodeIndex> bit =
                  GetBddNode(TreeBitLocation(node, i, tree_index));
              if (!bit.has_value()) {
                return absl::OkStatus();
              }
              bits.push_back(*bit);
            }

            SaturatingBddNodeVector in_interval_checks;
            for (const Interval& interval : intervals.Intervals()) {
              if (std::optional<Bits> precise_value =
                      interval.GetPreciseValue();
                  precise_value.has_value()) {
                SaturatingBddNodeIndex is_value = evaluator.Equals(
                    bits, evaluator.BitsToVector(*precise_value));
                if (HasTooManyPaths(is_value)) {
                  VLOG(3) << "SpecializeGiven exceeded path limit of "
                          << path_limit_ << " on: " << node->GetName() << " in "
                          << interval.ToString() << " (precise)";
                  continue;
                }
                in_interval_checks.push_back(is_value);
                continue;
              }

              SaturatingBddNodeIndex lower_bound =
                  evaluator.UGreaterThanOrEqual(
                      bits, evaluator.BitsToVector(interval.LowerBound()));
              if (HasTooManyPaths(lower_bound)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (lower bound)";
                continue;
              }
              SaturatingBddNodeIndex upper_bound = evaluator.ULessThanOrEqual(
                  bits, evaluator.BitsToVector(interval.UpperBound()));
              if (HasTooManyPaths(upper_bound)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (upper bound)";
                continue;
              }
              SaturatingBddNodeIndex in_interval =
                  evaluator.And(lower_bound, upper_bound);
              if (HasTooManyPaths(in_interval)) {
                VLOG(3) << "SpecializeGiven exceeded path limit of "
                        << path_limit_ << " on: " << node->GetName() << " in "
                        << interval.ToString() << " (joint)";
                continue;
              }
              in_interval_checks.push_back(in_interval);
            }
            SaturatingBddNodeIndex in_intervals =
                evaluator.OrReduce(in_interval_checks).front();
            if (HasTooManyPaths(in_intervals)) {
              VLOG(3) << "SpecializeGiven exceeded path limit of "
                      << path_limit_ << " on: " << node->GetName() << " in "
                      << intervals.ToString();
              return absl::OkStatus();
            }
            assumptions.push_back(ToBddNode(in_intervals));

            return absl::OkStatus();
          }));
    }
  }

  bool already_exceeded_limit = false;
  BddNodeIndex assumed = absl::c_accumulate(
      assumptions, bdd().one(), [&](BddNodeIndex a, BddNodeIndex b) {
        BddNodeIndex result = bdd().And(a, b);
        if (ExceedsPathLimit(result)) {
          if (!already_exceeded_limit) {
            VLOG(3) << "SpecializeGiven exceeded path limit of " << path_limit_;
            already_exceeded_limit = true;
          }
          return a;
        }
        return result;
      });
  return std::make_unique<AssumingBddQueryEngine>(this, assumed);
}
bool BddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return AtMostOneTrue(bits, std::nullopt);
}
bool BddQueryEngine::AtMostOneTrue(
    absl::Span<TreeBitLocation const> bits,
    std::optional<BddNodeIndex> assumption) const {
  // Computing this property is quadratic (at least) so limit the width.
  const int64_t kMaxWidth = 64;
  if (bits.size() > kMaxWidth) {
    return false;
  }

  BddNodeIndex result = bdd().zero();
  for (const TreeBitLocation& loc : bits) {
    if (!IsTracked(loc.node())) {
      return false;
    }
  }

  // Compute the OR-reduction of a pairwise AND of all bits. If this value is
  // zero then no two bits can be simultaneously true. Equivalently: at most one
  // bit is true.
  for (int64_t i = 0; i < bits.size(); ++i) {
    std::optional<BddNodeIndex> i_bdd = GetBddNode(bits[i]);
    if (!i_bdd.has_value()) {
      return false;
    }
    for (int64_t j = i + 1; j < bits.size(); ++j) {
      std::optional<BddNodeIndex> j_bdd = GetBddNode(bits[j]);
      if (!j_bdd.has_value()) {
        return false;
      }
      result = bdd().Or(result, bdd().And(*i_bdd, *j_bdd));
      if (ExceedsPathLimit(result)) {
        VLOG(3) << "AtMostOneTrue exceeded path limit of " << path_limit_;
        return false;
      }
    }
  }
  if (assumption.has_value()) {
    // If we have an assumption, we only care about the result when it's true;
    // we ignore all cases when it's false.
    result = bdd().And(*assumption, result);
  }
  return result == bdd().zero();
}

std::optional<SharedLeafTypeTree<TernaryVector>> BddQueryEngine::GetTernary(
    Node* node, BddNodeIndex assumption) const {
  if (!IsTracked(node)) {
    return std::nullopt;
  }
  absl::StatusOr<TernaryTree> ltt = TernaryTree::CreateFromFunction(
      node->GetType(),
      [&](Type* leaf_type, absl::Span<const int64_t> tree_index)
          -> absl::StatusOr<TernaryVector> {
        TernaryVector ternary(leaf_type->GetFlatBitCount(),
                              TernaryValue::kUnknown);
        for (int64_t bit_index = 0; bit_index < leaf_type->GetFlatBitCount();
             ++bit_index) {
          std::optional<BddNodeIndex> bit =
              GetBddNode(TreeBitLocation(node, bit_index, tree_index));
          if (!bit.has_value()) {
            continue;
          }
          if (Implies(assumption, *bit)) {
            ternary[bit_index] = TernaryValue::kKnownOne;
          } else if (Implies(assumption, bdd().Not(*bit))) {
            ternary[bit_index] = TernaryValue::kKnownZero;
          }
        }
        return ternary;
      });
  if (!ltt.ok()) {
    return std::nullopt;
  }
  return std::move(ltt).value().AsShared();
}

bool BddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits) const {
  return AtLeastOneTrue(bits, std::nullopt);
}
bool BddQueryEngine::AtLeastOneTrue(
    absl::Span<TreeBitLocation const> bits,
    std::optional<BddNodeIndex> assumption) const {
  // At least one bit is true is equivalent to an OR-reduction of all the bits.
  SaturatingBddEvaluator evaluator(path_limit_, &bdd());
  SaturatingBddNodeVector bdd_bits;
  for (const TreeBitLocation& location : bits) {
    if (!IsTracked(location.node())) {
      return false;
    }
    std::optional<BddNodeIndex> bdd_node = GetBddNode(location);
    if (!bdd_node.has_value()) {
      return false;
    }
    bdd_bits.push_back(*bdd_node);
  }
  SaturatingBddNodeIndex or_reduce = evaluator.OrReduce(bdd_bits).front();
  if (HasTooManyPaths(or_reduce)) {
    VLOG(3) << "AtLeastOneTrue exceeded path limit of " << path_limit_;
    return false;
  }
  BddNodeIndex result = ToBddNode(or_reduce);
  if (assumption.has_value()) {
    // If we have an assumption, we only care about the result when it's true;
    // we ignore all cases when it's false.
    result = bdd().Or(bdd().Not(*assumption), ToBddNode(result));
  }
  return result == bdd().one();
}

bool BddQueryEngine::Implies(const BddNodeIndex& a,
                             const BddNodeIndex& b) const {
  // A implies B  <=>  !(A && !B)
  BddNodeIndex result = bdd().And(a, bdd().Not(b));
  return result == bdd().zero();
}

bool BddQueryEngine::Implies(const TreeBitLocation& a,
                             const TreeBitLocation& b) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  return Implies(*a_bdd, *b_bdd);
}

std::optional<Bits> BddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return ImpliedNodeValue(predicate_bit_values, node,
                          /*assumption=*/std::nullopt);
}

std::optional<Bits> BddQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node, std::optional<BddNodeIndex> assumption) const {
  std::optional<TernaryVector> implied_ternary =
      ImpliedNodeTernary(predicate_bit_values, node, assumption);
  if (!implied_ternary.has_value() ||
      !ternary_ops::IsFullyKnown(*implied_ternary)) {
    return std::nullopt;
  }
  return ternary_ops::ToKnownBitsValues(*implied_ternary);
}

std::optional<TernaryVector> BddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node) const {
  return ImpliedNodeTernary(predicate_bit_values, node,
                            /*assumption=*/std::nullopt);
}

std::optional<TernaryVector> BddQueryEngine::ImpliedNodeTernary(
    absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
    Node* node, std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(node) || !node->GetType()->IsBits()) {
    return std::nullopt;
  }

  // Create a Bdd node for the predicate_bit_values.
  SaturatingBddEvaluator evaluator(path_limit_, &bdd());
  SaturatingBddNodeVector bdd_predicate_bits;
  for (const auto& [conjuction_bit_location, conjunction_value] :
       predicate_bit_values) {
    std::optional<BddNodeIndex> conjuction_bit =
        GetBddNode(conjuction_bit_location);
    if (!conjuction_bit.has_value()) {
      // Skip this predicate; we don't recognize the node, so we can't see the
      // effects of assuming it.
      continue;
    }
    bdd_predicate_bits.push_back(
        conjunction_value ? *conjuction_bit : bdd().Not(*conjuction_bit));
  }
  SaturatingBddNodeIndex bdd_predicate =
      evaluator.And(assumption.value_or(bdd().one()),
                    evaluator.AndReduce(bdd_predicate_bits).front());
  if (HasTooManyPaths(bdd_predicate)) {
    return std::nullopt;
  }
  BddNodeIndex bdd_predicate_bit = ToBddNode(bdd_predicate);

  // If the predicate evaluates to false, we can't determine
  // what node value it implies. That is, !predicate || node_bit
  // evaluates to true for both node_bit == 1 and == 0.
  if (bdd_predicate_bit == bdd().zero()) {
    return std::nullopt;
  }

  enum class ImpliedValue : std::uint8_t {
    kNotAnalyzable,
    kUnknown,
    kImpliedTrue,
    kImpliedFalse
  };
  auto implied_value = [&](int node_idx) -> std::optional<TernaryValue> {
    std::optional<BddNodeIndex> bdd_node_bit =
        GetBddNode(TreeBitLocation(node, node_idx));
    if (!bdd_node_bit.has_value()) {
      return std::nullopt;
    }
    if (Implies(bdd_predicate_bit, *bdd_node_bit)) {
      return TernaryValue::kKnownOne;
    }
    if (Implies(bdd_predicate_bit, bdd().Not(*bdd_node_bit))) {
      return TernaryValue::kKnownZero;
    }
    return TernaryValue::kUnknown;
  };

  // Check if bdd_predicate_bit implies anything about the node.
  CHECK(node->GetType()->IsBits());
  TernaryVector ternary(node->BitCountOrDie(), TernaryValue::kUnknown);
  for (int node_idx = 0; node_idx < node->BitCountOrDie(); ++node_idx) {
    std::optional<TernaryValue> bit_value = implied_value(node_idx);
    if (!bit_value.has_value()) {
      // Missing information; we can't determine the node value.
      return std::nullopt;
    }
    ternary[node_idx] = *bit_value;
  }
  if (ternary_ops::AllUnknown(ternary)) {
    return std::nullopt;
  }
  return ternary;
}

bool BddQueryEngine::KnownEquals(const TreeBitLocation& a,
                                 const TreeBitLocation& b) const {
  return KnownEquals(a, b,
                     /*assumption=*/std::nullopt);
}

bool BddQueryEngine::KnownEquals(const TreeBitLocation& a,
                                 const TreeBitLocation& b,
                                 std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  if (assumption.has_value()) {
    a_bdd = bdd().And(*a_bdd, *assumption);
    b_bdd = bdd().And(*b_bdd, *assumption);
  }
  return *a_bdd == *b_bdd;
}

bool BddQueryEngine::KnownNotEquals(const TreeBitLocation& a,
                                    const TreeBitLocation& b) const {
  return KnownNotEquals(a, b,
                        /*assumption=*/std::nullopt);
}

bool BddQueryEngine::KnownNotEquals(
    const TreeBitLocation& a, const TreeBitLocation& b,
    std::optional<BddNodeIndex> assumption) const {
  if (!IsTracked(a.node()) || !IsTracked(b.node())) {
    return false;
  }
  std::optional<BddNodeIndex> a_bdd = GetBddNode(a);
  if (!a_bdd.has_value()) {
    return false;
  }
  std::optional<BddNodeIndex> b_bdd = GetBddNode(b);
  if (!b_bdd.has_value()) {
    return false;
  }
  if (assumption.has_value()) {
    a_bdd = bdd().And(*a_bdd, *assumption);
    b_bdd = bdd().And(*b_bdd, *assumption);
  }
  return *a_bdd == bdd().Not(*b_bdd);
}

bool BddQueryEngine::IsKnown(const TreeBitLocation& bit,
                             std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsKnown(bit);
  }
  return KnownValue(bit, assumption).has_value();
}

std::optional<bool> BddQueryEngine::KnownValue(
    const TreeBitLocation& bit, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return KnownValue(bit);
  }
  if (!IsTracked(bit.node())) {
    return std::nullopt;
  }
  std::optional<BddNodeIndex> bdd_node = GetBddNode(bit);
  if (!bdd_node.has_value()) {
    return std::nullopt;
  }
  if (assumption.has_value()) {
    bdd_node = bdd().And(*bdd_node, *assumption);
  }

  if (bdd_node == bdd().one()) {
    return true;
  } else if (bdd_node == bdd().zero()) {
    return false;
  } else {
    return std::nullopt;
  }
}

std::optional<Value> BddQueryEngine::KnownValue(
    Node* node, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return KnownValue(node);
  }

  if (!IsTracked(node)) {
    return std::nullopt;
  }

  std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
      GetTernary(node, *assumption);
  if (!ternary.has_value() ||
      !absl::c_all_of(ternary->elements(), [](const TernaryVector& v) {
        return ternary_ops::IsFullyKnown(v);
      })) {
    return std::nullopt;
  }

  absl::StatusOr<LeafTypeTree<Value>> value =
      leaf_type_tree::MapIndex<Value, TernaryVector>(
          ternary->AsView(),
          [](Type* leaf_type, const TernaryVector& v,
             absl::Span<const int64_t>) -> absl::StatusOr<Value> {
            if (leaf_type->IsToken()) {
              return Value::Token();
            }
            CHECK(leaf_type->IsBits());
            return Value(ternary_ops::ToKnownBitsValues(v));
          });
  CHECK_OK(value.status());
  absl::StatusOr<Value> result = LeafTypeTreeToValue(value->AsView());
  CHECK_OK(result.status());
  return *result;
}

bool BddQueryEngine::IsAllZeros(Node* n,
                                std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsAllZeros(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownZero(v);
         });
}

bool BddQueryEngine::IsAllOnes(Node* n,
                               std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsAllOnes(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsKnownOne(v);
         });
}
bool BddQueryEngine::IsFullyKnown(
    Node* n, std::optional<BddNodeIndex> assumption) const {
  if (!assumption.has_value()) {
    return IsFullyKnown(n);
  }
  if (!IsTracked(n) || TypeHasToken(n->GetType())) {
    return false;
  }
  std::optional<SharedLeafTypeTree<TernaryVector>> ternary_value =
      GetTernary(n, *assumption);
  return ternary_value.has_value() &&
         absl::c_all_of(ternary_value->elements(), [](const TernaryVector& v) {
           return ternary_ops::IsFullyKnown(v);
         });
}

}  // namespace xls
