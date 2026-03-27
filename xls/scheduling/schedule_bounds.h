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

#ifndef XLS_SCHEDULING_SCHEDULE_BOUNDS_H_
#define XLS_SCHEDULING_SCHEDULE_BOUNDS_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/visitor.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace sched {

// An abstraction holding lower and upper bounds for each node in a
// function. The bounds are constraints on cycles in which a node may be
// scheduled.
class ScheduleBounds {
 public:
  // By default how many cycles we are willing to even attempt to schedule in.
  // We set this to 1 billion since that is far larger than any even
  // unreasonable design would require and is still small enough that we don't
  // need to deal with overflow issues when implementing bounds.
  //
  // TODO(allight): It would be nice to make this fully overflow-safe.
  static constexpr int64_t kDefaultMaxUpperBound = 1000000000;

  // A simplified representation of a scheduling constraint. These are always
  // expressed in terms of relations between individual nodes.
  //
  // A constraint is one of the following:
  // * A `NodeInCycleConstraint` that constrains a node to be scheduled in a
  //   specific cycle.
  // * A `NodeDifferenceConstraint` that constrains the 'subject' node to be
  // between 'min_after' and 'max_after' cycles after the 'anchor' node
  // (inclusive on both ends).
  // * A `LastStageConstraint` that constrains a node to be scheduled in the
  //   last stage of the pipeline.
  class NodeSchedulingConstraint {
   public:
    // Constraint that a node must be in the last stage of the pipeline
    struct LastStageConstraint {
      Node* node;

      template <typename Sink>
      friend void AbslStringify(Sink& sink,
                                const LastStageConstraint& constraint) {
        absl::Format(&sink, "%s:last_stage", constraint.node->GetName());
      }
      friend std::ostream& operator<<(std::ostream& os,
                                      const LastStageConstraint& constraint) {
        os << absl::StreamFormat("%v", constraint);
        return os;
      }
    };
    struct NodeDifferenceConstraint {
      // The node which defines where 'subject' can be placed.
      Node* const anchor;
      // This node is constrained to be scheduled at least
      // `min_difference` cycles after the first node and at most
      // `max_difference` cycles after the first node.
      Node* const subject;
      // The minimum number of cycles the second node must be scheduled after
      // the first node. Must be non-negative.
      int64_t min_after;
      int64_t max_after;

      template <typename Sink>
      friend void AbslStringify(Sink& sink,
                                const NodeDifferenceConstraint& constraint) {
        absl::Format(&sink, "%s - %s ∈ [%d, %d]", constraint.subject->GetName(),
                     constraint.anchor->GetName(), constraint.min_after,
                     constraint.max_after);
      }
      friend std::ostream& operator<<(
          std::ostream& os, const NodeDifferenceConstraint& constraint) {
        os << absl::StreamFormat("%v", constraint);
        return os;
      }
    };
    using InnerConstraints =
        std::variant<NodeInCycleConstraint, NodeDifferenceConstraint,
                     LastStageConstraint>;

    NodeSchedulingConstraint(const InnerConstraints& inner_constraints,
                             bool is_lower_bound_constraint = true)
        : inner_constraints_(inner_constraints),
          is_lower_bound_constraint_(is_lower_bound_constraint) {}

    bool is_lower_bound_constraint() const {
      return is_lower_bound_constraint_;
    }
    operator const InnerConstraints&() const { return inner_constraints_; }
    // Get the node which is constrained.
    Node* subject() const {
      return std::visit(
          Visitor{
              [](const LastStageConstraint& lsc) { return lsc.node; },
              [](const NodeDifferenceConstraint& ndc) { return ndc.subject; },
              [](const NodeInCycleConstraint& nicc) { return nicc.GetNode(); },
          },
          inner_constraints_);
    }

    template <typename T>
      requires(std::is_same_v<T, NodeInCycleConstraint> ||
               std::is_same_v<T, NodeDifferenceConstraint> ||
               std::is_same_v<T, LastStageConstraint>)
    bool Is() const {
      return std::holds_alternative<T>(inner_constraints_);
    }

    template <typename T>
      requires(std::is_same_v<T, NodeInCycleConstraint> ||
               std::is_same_v<T, NodeDifferenceConstraint> ||
               std::is_same_v<T, LastStageConstraint>)
    T As() const {
      CHECK(Is<T>()) << "Unexpected constraint type";
      return std::get<T>(inner_constraints_);
    }

    NodeSchedulingConstraint Reverse(int64_t max_upper_bound) const {
      return std::visit(
          Visitor{
              [&](const LastStageConstraint& lsc) {
                // Last stage constraints need to be handled differently for
                // upper and lower bounds.
                return NodeSchedulingConstraint(lsc,
                                                !is_lower_bound_constraint_);
              },
              [&](const NodeInCycleConstraint& nicc) {
                return NodeSchedulingConstraint(
                    NodeInCycleConstraint{nicc.GetNode(),
                                          max_upper_bound - nicc.GetCycle()},
                    !is_lower_bound_constraint_);
              },
              [&](const NodeDifferenceConstraint& ndc) {
                return NodeSchedulingConstraint(
                    NodeDifferenceConstraint{
                        .anchor = ndc.subject,
                        .subject = ndc.anchor,
                        .min_after = ndc.min_after,
                        .max_after = ndc.max_after,
                    },
                    !is_lower_bound_constraint_);
              },
          },
          inner_constraints_);
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink,
                              const NodeSchedulingConstraint& constraint) {
      std::visit(Visitor{
                     [&](const LastStageConstraint& lsc) {
                       absl::Format(&sink, "%v", lsc);
                     },
                     [&](const NodeDifferenceConstraint& ndc) {
                       absl::Format(&sink, "%v", ndc);
                     },
                     [&](const NodeInCycleConstraint& nicc) {
                       absl::Format(&sink, "%v", nicc);
                     },
                 },
                 constraint.inner_constraints_);
    }
    friend std::ostream& operator<<(
        std::ostream& os, const NodeSchedulingConstraint& constraint) {
      os << absl::StreamFormat("%v", constraint);
      return os;
    }

   private:
    InnerConstraints inner_constraints_;
    bool is_lower_bound_constraint_ = true;
  };

  // Returns a object with the lower bounds of each node set to the earliest
  // possible cycle which satisfies dependency and clock period constraints.
  // Similarly, upper bounds are set to the latest possible cycle The upper
  // bounds of nodes with no uses (leaf nodes) are set to the maximum lower
  // bound of any node. Note this never considers nodes 'dead-after-synthesis'
  // if they are returned by the schedule_util.h 'GetDeadAfterSynthesisNodes'
  // function.
  static absl::StatusOr<ScheduleBounds> ComputeAsapAndAlapBounds(
      FunctionBase* f, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator,
      std::optional<int64_t> ii = std::nullopt,
      absl::Span<SchedulingConstraint const> constraints = {},
      int64_t max_upper_bound = kDefaultMaxUpperBound);
  // Same as ComputeAsapAndAlapBounds but takes NodeSchedulingConstraints
  // directly. This is for testing.
  static absl::StatusOr<ScheduleBounds> ComputeAsapAndAlapBounds(
      FunctionBase* f, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator,
      absl::Span<NodeSchedulingConstraint const> constraints,
      int64_t max_upper_bound = kDefaultMaxUpperBound);

  // Upon construction all parameters have lower and upper bounds of 0. All
  // other nodes have a lower bound of 1 and an upper bound of
  // 'max_upper_bound' (defaults to kDefaultMaxUpperBound or 1 billion).
  static absl::StatusOr<ScheduleBounds> Create(
      ScheduleGraph graph, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator,
      std::optional<int64_t> ii = std::nullopt,
      absl::Span<SchedulingConstraint const> constraints = {},
      int64_t max_upper_bound = kDefaultMaxUpperBound);

  ScheduleBounds(const ScheduleBounds& other) = default;
  ScheduleBounds(ScheduleBounds&& other) = default;
  ScheduleBounds& operator=(const ScheduleBounds& other) = default;
  ScheduleBounds& operator=(ScheduleBounds&& other) = default;

  // Resets node bounds to their initial unconstrained values.
  void Reset();

  const ScheduleGraph& graph() const { return graph_; }

  // Return the lower/upper bound of the given node.
  int64_t lb(Node* node) const { return bounds_.at(node).after_min; }
  int64_t ub(Node* node) const {
    return max_upper_bound_ - bounds_.at(node).before_max;
  }

  // Return the lower and upper bound as a pair (lower bound is first element).
  std::pair<int64_t, int64_t> bounds(Node* node) const {
    NodeBound b = bounds_.at(node);
    return std::make_pair(b.after_min, max_upper_bound_ - b.before_max);
  }

  // Sets the lower bound of the given node to the maximum of its existing value
  // and the given value. Raises a ResourceExhaustedError if the new value
  // results in infeasible bounds (lower bound is greater than upper bound).
  absl::Status TightenNodeLb(Node* node, int64_t value) {
    VLOG(2) << "TightenNodeLb: " << node->GetName() << " to " << value;
    return TightenNodeLbInternal(node, value).status();
  }

  // Sets the upper bound of the given node to the minimum of its existing value
  // and the given value. Raises a ResourceExhaustedError if the new value
  // results in infeasible bounds (lower bound is greater than upper bound).
  absl::Status TightenNodeUb(Node* node, int64_t value) {
    VLOG(2) << "TightenNodeUb: " << node->GetName() << " to " << value;
    return TightenNodeUbInternal(node, value).status();
  }

  // Returns the maximum lower (upper) bound of any node in the function.
  int64_t max_lower_bound() const { return max_lower_bound_; }
  int64_t min_upper_bound() const {
    return max_upper_bound_ - max_before_upper_bound_;
  }

  std::string ToString() const;

  // Uses the set bounds and constraints to attempt to tighten the bounds of
  // each node such that all constraints, including explicit constraints,
  // clock-period and dependency constraints, are satisfied. If 'fuel' is given,
  // then the constraint propagation will stop after 'fuel' constraint
  // applications. If 'fuel' is not given, then the bounds will be propagated
  // until no more tightening is possible. Returns an error if the bounds are
  // infeasible or fail to converge to a constraint-compatible state.
  //
  // Should be called after calling TightenNodeLb (TightenNodeUb) to propagate
  // the tightened bound throughout the graph. This method only tightens bounds
  // (increases lower bounds and decreases upper bounds). Returns an error if
  // propagation results in infeasible bounds (lower bound is greater than upper
  // bound for a node).
  absl::Status PropagateBounds(std::optional<int64_t> fuel = 6);

  // Force all constraints to be given in terms of nodes. Public for testing.
  static absl::StatusOr<std::vector<NodeSchedulingConstraint>>
  ConvertSchedulingConstraints(
      ScheduleGraph& graph, absl::Span<const SchedulingConstraint> constraints,
      std::optional<int64_t> ii, int64_t max_upper_bound);

  // Create using NodeSchedulingConstraints. Public for testing.
  //
  // All constraints must be constraints suitable for 'lower-bound' propagation.
  static absl::StatusOr<ScheduleBounds> Create(
      ScheduleGraph graph, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator,
      absl::Span<NodeSchedulingConstraint const> constraints,
      int64_t max_upper_bound = kDefaultMaxUpperBound);

  // Add a new constraint to the set of constraints.
  void AddConstraint(NodeSchedulingConstraint constraint) {
    auto lower_bound = constraint.is_lower_bound_constraint()
                           ? constraint
                           : constraint.Reverse(max_upper_bound_);
    auto upper_bound = constraint.is_lower_bound_constraint()
                           ? constraint.Reverse(max_upper_bound_)
                           : constraint;
    constraints_by_subject_lower_bound_[lower_bound.subject()].push_back(
        lower_bound);
    constraints_by_subject_upper_bound_[upper_bound.subject()].push_back(
        upper_bound);
    VLOG(2) << "Added constraint: " << constraint << " to "
            << lower_bound.subject()->GetName();
  }
  // Add a new lower-bound constraint to the set of constraints.
  void AddConstraint(NodeSchedulingConstraint::InnerConstraints constraint,
                     bool is_lower_bound_constraint = true) {
    AddConstraint(
        NodeSchedulingConstraint(constraint, is_lower_bound_constraint));
  }

  // A node's bounds, stored as a {lower, upper} pair.
  struct NodeBound {
    // How many cycles after the 0'th cycle a node must be.
    int64_t after_min = 0;
    // How many cycles before the last'th cycle a node must be.
    int64_t before_max = 0;
  };

 private:
  ScheduleBounds(
      ScheduleGraph graph, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator,
      absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
          constraints_by_subject_lower_bound,
      absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
          constraints_by_subject_upper_bound,
      int64_t max_upper_bound)
      : graph_(std::move(graph)),
        clock_period_ps_(clock_period_ps),
        delay_estimator_(&delay_estimator),
        constraints_by_subject_lower_bound_(
            std::move(constraints_by_subject_lower_bound)),
        constraints_by_subject_upper_bound_(
            std::move(constraints_by_subject_upper_bound)),
        max_upper_bound_(max_upper_bound) {
    Reset();
  }

  absl::StatusOr<int64_t> GetDelay(Node* node) const;

  // Updates the lower (upper) bounds of each node such that dependency and
  // clock period constraints are met for every node. Should be called after
  // calling TightenNodeLb (TightenNodeUb) to propagate the tightened bound
  // throughout the graph. This method only tightens bounds (increases lower
  // bounds and decreases upper bounds). Returns an error if propagation results
  // in infeasible bounds (lower bound is greater than upper bound for a node).
  absl::StatusOr<bool> PropagateLowerBounds();
  absl::StatusOr<bool> PropagateUpperBounds();
  absl::StatusOr<bool> PropagateConstraints();
  absl::Status CheckConstraints(std::optional<int64_t> fuel, bool stabilized);
  absl::Status CheckBasicBounds();
  absl::StatusOr<bool> TightenNodeLbInternal(Node* node, int64_t value) {
    if (value > ub(node)) {
      return absl::ResourceExhaustedError(
          absl::StrFormat("Unable to tighten the lower bound of node %s to %d.",
                          node->GetName(), value));
    }
    int64_t initial = lb(node);
    bounds_.at(node).after_min = std::max(lb(node), value);
    max_lower_bound_ = std::max(max_lower_bound_, value);
    return initial < value;
  }
  absl::StatusOr<bool> TightenNodeUbInternal(Node* node, int64_t value) {
    if (value < lb(node)) {
      return absl::ResourceExhaustedError(
          absl::StrFormat("Unable to tighten the upper bound of node %s to %d "
                          "(current lb: %d).",
                          node->GetName(), value, lb(node)));
    }
    int64_t initial = ub(node);
    bounds_.at(node).before_max = max_upper_bound_ - std::min(initial, value);
    VLOG(3) << "TightenNodeUbInternal: " << node->GetName() << " to " << value
            << " initial: " << initial << " new ub: " << ub(node);
    return initial > value;
  }

  ScheduleGraph graph_;
  int64_t clock_period_ps_;
  const DelayEstimator* delay_estimator_;

  // Map from each node to all of the scheduling constraints that the node is
  // subject to.
  absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
      constraints_by_subject_lower_bound_;
  // Constraints specifically on the upper bound. This reverses the
  // direction of the NodeDifferenceConstraint to put the subject first.
  absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
      constraints_by_subject_upper_bound_;

  // The bounds of each node stored as a {lower, upper} pair.
  absl::flat_hash_map<Node*, NodeBound> bounds_;
  absl::flat_hash_map<Node*, int64_t> explicit_upper_bounds_;

  int64_t max_lower_bound_;
  int64_t max_before_upper_bound_;
  // The maximum upper bound of any node in the function. This is what we use as
  // the zero point for upper bounds.
  // Const after construction.
  int64_t max_upper_bound_ = kDefaultMaxUpperBound;
};

class ScheduleBoundsAnnotator : public IrAnnotator {
 public:
  explicit ScheduleBoundsAnnotator(const ScheduleBounds& bounds)
      : bounds_(std::move(bounds)) {}
  Annotation NodeAnnotation(Node* node) const override;

 private:
  const ScheduleBounds& bounds_;
};

}  // namespace sched
}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULE_BOUNDS_H_
