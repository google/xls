// Copyright 2023 The XLS Authors
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

#ifndef XLS_PASSES_PREDICATE_STATE_H_
#define XLS_PASSES_PREDICATE_STATE_H_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/ternary.h"

namespace xls {

// Special value denoting the 'default' arm.
struct DefaultArm : public std::monostate {
  template <typename H>
  friend H AbslHashValue(H h, const DefaultArm& a) {
    return H::combine(std::move(h), std::monostate{});
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const DefaultArm& arm) {
    absl::Format(&sink, "DEFAULT");
  }
};

// Special value denoting the 'in bounds' arm.
struct InBoundsArm : public std::monostate {
  template <typename H>
  friend H AbslHashValue(H h, const InBoundsArm& a) {
    return H::combine(std::move(h), std::monostate{});
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const InBoundsArm& arm) {
    absl::Format(&sink, "IN_BOUNDS");
  }
};

// Known values that a particular predicate state implies.
struct SelectorValues {
  IntervalSet range;
  TernaryVector ternary;
};

// Abstraction representing the state of some operation where access to a value
// is predicated.
class PredicateState {
 public:
  using ArmT = std::variant<int64_t, DefaultArm, InBoundsArm>;
  static constexpr ArmT kDefaultArm{DefaultArm{}};
  static constexpr ArmT kInBoundsArm{InBoundsArm{}};
  using SelectT = std::variant<Select*, OneHotSelect*, PrioritySelect*,
                               ArrayUpdate*, std::nullptr_t>;
  PredicateState() : node_(nullptr), index_(kDefaultArm) {}
  PredicateState(SelectT node, ArmT index) : node_(node), index_(index) {}
  PredicateState(const PredicateState&) = default;
  PredicateState& operator=(const PredicateState&) = default;
  PredicateState(PredicateState&&) = default;
  PredicateState& operator=(PredicateState&&) = default;

  // Does this state represent no selects guarding.
  bool IsBasePredicate() const {
    return std::holds_alternative<std::nullptr_t>(node_);
  }

  // Does this state represent a predicate for a select.
  bool IsSelectPredicate() const {
    return std::holds_alternative<Select*>(node_) ||
           std::holds_alternative<OneHotSelect*>(node_) ||
           std::holds_alternative<PrioritySelect*>(node_);
  }

  // Get the values of the selector implied by this predicate being selected.
  absl::StatusOr<SelectorValues> SelectorRange() const {
    XLS_RET_CHECK(IsSelectPredicate());
    Node* selector = node()->operand(0);
    if (std::holds_alternative<Select*>(node_)) {
      if (IsDefaultArm()) {
        auto intervals = IntervalSet::Of(
            {Interval::Open(UBits(cases_count(), selector->BitCountOrDie()),
                            Bits::AllOnes(selector->BitCountOrDie()))});
        auto tern = interval_ops::ExtractTernaryVector(intervals);
        return SelectorValues{.range = std::move(intervals),
                              .ternary = std::move(tern)};
      } else {
        auto intervals =
            IntervalSet::Precise(UBits(arm_index(), selector->BitCountOrDie()));
        auto tern = ternary_ops::BitsToTernary(
            UBits(arm_index(), selector->BitCountOrDie()));
        return SelectorValues{.range = std::move(intervals),
                              .ternary = std::move(tern)};
      }
    }
    if (std::holds_alternative<OneHotSelect*>(node_)) {
      if (IsDefaultArm()) {
        return SelectorValues{
            .range = IntervalSet::Precise(UBits(0, selector->BitCountOrDie())),
            .ternary = TernaryVector(selector->BitCountOrDie(),
                                     TernaryValue::kKnownZero)};
      } else {
        TernaryVector vec(selector->BitCountOrDie(), TernaryValue::kUnknown);
        vec[arm_index()] = TernaryValue::kKnownOne;
        return SelectorValues{.range = IntervalSet::Precise(UBits(
                                  1 << arm_index(), selector->BitCountOrDie())),
                              .ternary = std::move(vec)};
      }
    }
    if (IsDefaultArm()) {
      return SelectorValues{
          .range = IntervalSet::Precise(UBits(0, selector->BitCountOrDie())),
          .ternary = TernaryVector(selector->BitCountOrDie(),
                                   TernaryValue::kKnownZero)};
    } else {
      TernaryVector vec(selector->BitCountOrDie(), TernaryValue::kKnownZero);
      vec[arm_index()] = TernaryValue::kKnownOne;
      for (int64_t i = arm_index() - 1; i >= 0; --i) {
        vec[i] = TernaryValue::kKnownZero;
      }
      auto interval = interval_ops::FromTernary(vec);
      return SelectorValues{.range = std::move(interval),
                            .ternary = std::move(vec)};
    }
  }

  int64_t cases_count() const {
    CHECK(IsSelectPredicate());
    if (std::holds_alternative<Select*>(node_)) {
      return std::get<Select*>(node_)->cases().size();
    }
    if (std::holds_alternative<OneHotSelect*>(node_)) {
      return std::get<OneHotSelect*>(node_)->cases().size();
    }
    return std::get<PrioritySelect*>(node_)->cases().size();
  }

  // Does this state represent a predicate for an array update.
  bool IsArrayUpdatePredicate() const {
    return std::holds_alternative<ArrayUpdate*>(node_);
  }

  // Is the arm the 'default' arm (assuming that's even meaningful for the
  // node).
  bool IsDefaultArm() const { return kDefaultArm == index_; }

  // Is the arm the 'in bounds' arm (assuming that's even meaningful for the
  // node).
  bool IsInBoundsArm() const { return kInBoundsArm == index_; }

  // The select this predicate represents as a node.
  Node* node() const {
    return absl::visit([](auto v) -> Node* { return v; }, node_);
  }

  absl::Span<Node* const> indices() const {
    CHECK(!IsBasePredicate());
    return absl::visit(
        xls::Visitor{[&](ArrayUpdate* u) -> absl::Span<Node* const> {
                       return u->indices();
                     },
                     [&](auto n) -> absl::Span<Node* const> {
                       return absl::Span<Node* const>();
                     }},
        node_);
  }

  // The value which controls the select
  Node* selector() const {
    CHECK(!IsBasePredicate());
    if (std::holds_alternative<ArrayUpdate*>(node_)) {
      return nullptr;
    }
    // All selects have selector as op(0)
    return node()->operand(0);
  }

  Node* value() const {
    CHECK(!IsBasePredicate());
    return absl::visit(xls::Visitor{[&](Select* s) -> Node* {
                                      return IsDefaultArm()
                                                 ? s->default_value().value()
                                                 : s->get_case(arm_index());
                                    },
                                    [&](OneHotSelect* s) -> Node* {
                                      CHECK(!IsDefaultArm());
                                      return s->get_case(arm_index());
                                    },
                                    [&](PrioritySelect* s) -> Node* {
                                      CHECK(!IsDefaultArm());
                                      return s->get_case(arm_index());
                                    },
                                    [&](ArrayUpdate* s) -> Node* {
                                      if (IsDefaultArm()) {
                                        return s->array_to_update();
                                      }
                                      return nullptr;
                                    },
                                    [](std::nullptr_t) -> Node* {
                                      LOG(FATAL) << "Unreachable";
                                      return nullptr;
                                    }},
                       node_);
  }

  // The arm index this predicate protects
  ArmT arm() const { return index_; }
  int64_t arm_index() const {
    CHECK(std::holds_alternative<int64_t>(index_));
    return std::get<int64_t>(index_);
  }

  friend bool operator==(const PredicateState& x, const PredicateState& y) {
    return (x.node_ == y.node_) && (x.index_ == y.index_);
  }
  friend bool operator!=(const PredicateState& x, const PredicateState& y) {
    return !(x == y);
  }

  friend std::strong_ordering operator<=>(const PredicateState& x,
                                          const PredicateState& y) {
    if (std::strong_ordering node_cmp =
            Node::NodeIdLessThan().Compare(x.node(), y.node());
        node_cmp != std::strong_ordering::equal) {
      return node_cmp;
    }
    if (std::strong_ordering arm_type_cmp = x.arm().index() <=> y.arm().index();
        arm_type_cmp != std::strong_ordering::equal) {
      return arm_type_cmp;
    }
    return x.arm_index() <=> y.arm_index();
  }

  template <typename H>
  friend H AbslHashValue(H h, const PredicateState& ps) {
    return H::combine(std::move(h), ps.node_, ps.index_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PredicateState& state) {
    if (state.IsBasePredicate()) {
      absl::Format(&sink, "PredicateState[Base]");
    } else {
      absl::visit(
          [&](auto index) {
            absl::Format(&sink, "PredicateState[%v: arm: %v]", *state.node(),
                         index);
          },
          state.index_);
    }
  }

 private:
  SelectT node_;
  ArmT index_;
};
}  // namespace xls

#endif  // XLS_PASSES_PREDICATE_STATE_H_
