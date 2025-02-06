// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_PROC_STATE_RANGE_QUERY_ENGINE_H_
#define XLS_PASSES_PROC_STATE_RANGE_QUERY_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"
#include "xls/ir/ternary.h"
#include "xls/passes/predicate_state.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/range_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {

// A range query engine that additionally does and incorporates temporal
// reasoning about proc state evolution.
//
// TODO(allight): Currently this assumes that each proc state element is
// disconnected from all others. This is generally good enough to get good
// results but running to a fixed-point would be better. That has the
// disadvantage of making the analysis take even longer.
class ProcStateRangeQueryEngine final : public QueryEngine {
 public:
  // If only_analyze_proc_states then analysis will only continue until we get
  // some bounds on the proc state elements themselves. This is useful for (eg)
  // proc_state_narrowing.
  ProcStateRangeQueryEngine()
      : ternary_(std::make_unique<TernaryQueryEngine>()),
        range_(std::make_unique<RangeQueryEngine>()),
        inner_(UnownedUnionQueryEngine({ternary_.get(), range_.get()})) {}
  ProcStateRangeQueryEngine(ProcStateRangeQueryEngine&&) = default;
  ProcStateRangeQueryEngine(const ProcStateRangeQueryEngine&) = delete;
  ProcStateRangeQueryEngine& operator=(const ProcStateRangeQueryEngine&) =
      delete;
  ProcStateRangeQueryEngine& operator=(ProcStateRangeQueryEngine&&) = default;
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override;

  // Returns true if this analysis will be able to give any additional
  // information over normal range analysis. The query engine can still be
  // populated if this is false but it is no different than a union of ternary
  // and range analyses.
  inline static bool CanAnalyzeProcStateEvolution(FunctionBase* f) {
    if (!f->IsProc()) {
      return false;
    }
    Proc* p = f->AsProcOrDie();
    if (!p->next_values().empty()) {
      return true;
    }
    for (int64_t i = 0; i < p->GetStateElementCount(); ++i) {
      if (p->GetNextStateElement(i) != p->GetStateRead(p->GetStateElement(i))) {
        return false;
      }
    }
    return true;
  }

  LeafTypeTree<IntervalSet> GetIntervals(Node* node) const override {
    return inner_.GetIntervals(node);
  }

  bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return inner_.AtMostOneTrue(bits);
  }

  bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const override {
    return inner_.AtLeastOneTrue(bits);
  }

  bool KnownEquals(const TreeBitLocation& a,
                   const TreeBitLocation& b) const override {
    return inner_.KnownEquals(a, b);
  }

  bool KnownNotEquals(const TreeBitLocation& a,
                      const TreeBitLocation& b) const override {
    return inner_.KnownNotEquals(a, b);
  }

  bool Implies(const TreeBitLocation& a,
               const TreeBitLocation& b) const override {
    return inner_.Implies(a, b);
  }

  std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return inner_.ImpliedNodeValue(predicate_bit_values, node);
  }

  std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const override {
    return inner_.ImpliedNodeTernary(predicate_bit_values, node);
  }

  IntervalSetTree GetIntervalSetTree(Node* node) const {
    CHECK(range_);
    return range_->GetIntervalSetTree(node);
  }

  absl::StatusOr<IntervalSetTreeView> GetIntervalSetTreeView(Node* node) const {
    XLS_RET_CHECK(range_);
    return range_->GetIntervalSetTreeView(node);
  }

  Bits MaxUnsignedValue(Node* n) const override {
    return inner_.MaxUnsignedValue(n);
  }

  Bits MinUnsignedValue(Node* n) const override {
    return inner_.MinUnsignedValue(n);
  }

  // Returns whether any information is available for this node.
  bool IsTracked(Node* node) const override { return inner_.IsTracked(node); }

  std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const override {
    return inner_.GetTernary(node);
  }

  std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const override {
    return inner_.SpecializeGivenPredicate(state);
  }

 private:
  // Actual range results from the proc-state aware analysis.
  std::unique_ptr<TernaryQueryEngine> ternary_;
  std::unique_ptr<RangeQueryEngine> range_;
  UnownedUnionQueryEngine inner_;
};

}  // namespace xls

#endif  // XLS_PASSES_PROC_STATE_RANGE_QUERY_ENGINE_H_
