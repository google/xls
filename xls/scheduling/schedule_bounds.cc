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

#include "xls/scheduling/schedule_bounds.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/combinations.hpp"
#include "cppitertools/imap.hpp"
#include "cppitertools/sliding_window.hpp"
#include "xls/common/iterator_range.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/channel.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/schedule_util.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace sched {
namespace {

using NodeSchedulingConstraint = ScheduleBounds::NodeSchedulingConstraint;
using LastStageConstraint = NodeSchedulingConstraint::LastStageConstraint;
using NodeDifferenceConstraint =
    NodeSchedulingConstraint::NodeDifferenceConstraint;

class ConstraintConverter {
 public:
  struct ChannelOps {
    absl::InlinedVector<ChannelNode*, 1> send_ops;
    absl::InlinedVector<ChannelNode*, 1> recv_ops;

    absl::InlinedVector<ChannelNode*, 1>& GetDirection(
        ChannelDirection direction) {
      return direction == ChannelDirection::kSend ? send_ops : recv_ops;
    }
  };
  static absl::StatusOr<ConstraintConverter> Create(ScheduleGraph& graph,
                                                    std::optional<int64_t> ii,
                                                    int64_t max_upper_bound) {
    std::variant<Package*, FunctionBase*> scope = graph.ir_scope();
    XLS_RET_CHECK(std::holds_alternative<FunctionBase*>(scope));
    FunctionBase* fb = std::get<FunctionBase*>(scope);
    ConstraintConverter converter(graph, ii, max_upper_bound, fb);
    XLS_RETURN_IF_ERROR(converter.Init());
    return converter;
  }

  absl::Status AddConstraint(const SchedulingConstraint& constraint) {
    return std::visit([&](const auto& c) { return AddSpecificConstraint(c); },
                      constraint);
  }

  absl::Status AddSpecificConstraint(const NodeInCycleConstraint& nic) {
    node_constraints_.emplace_back(nic);
    return absl::OkStatus();
  }
  absl::Status AddSpecificConstraint(const IOConstraint& io) {
    XLS_RET_CHECK(fb_->IsProc());
    XLS_RET_CHECK_GE(io.MinimumLatency(), 0);
    XLS_RET_CHECK_GE(io.MaximumLatency(), 0);
    if (!channel_name_map_.contains(io.SourceChannel()) ||
        !channel_name_map_.contains(io.TargetChannel())) {
      // Must be from a different proc?
      return absl::OkStatus();
    }
    const absl::InlinedVector<ChannelNode*, 1>& source_nodes =
        channel_ops_.at(channel_name_map_.at(io.SourceChannel()))
            .GetDirection(io.SourceDirection());
    const absl::InlinedVector<ChannelNode*, 1>& sink_nodes =
        channel_ops_.at(channel_name_map_.at(io.TargetChannel()))
            .GetDirection(io.TargetDirection());
    // We need to ensure we see 'anchor's before 'subjects'. This is
    // normally handled by the topo sort defining an order but this
    // isn't necessarily required (nb it is almost certainly a bug
    // if there isn't some dependency order between them). While for
    // the SDC scheduler the constraint solver will handle this fine
    // we need to choose an order quickly. We simply update the
    // topo-order to force the source before the sink.
    if (source_nodes.empty() && sink_nodes.empty()) {
      // Must be a different proc?
      return absl::OkStatus();
    }
    XLS_RET_CHECK(!source_nodes.empty() && !sink_nodes.empty())
        << "IO constraint stretches across procs.";
    // If the source **could** come after the sink we add a refinement
    // to force it to come before. NB We need to do this even in the
    // case where it happens to be in the right order initially since
    // other io_constraints could add dependencies which may make the
    // refinement order them differently.
    for (ChannelNode* source : source_nodes) {
      for (ChannelNode* sink : sink_nodes) {
        if (!forward_dependency_analysis_.IsDependent(source, sink)) {
          XLS_RET_CHECK(!forward_dependency_analysis_.IsDependent(sink, source))
              << "IO constraint source and sink are not ordered "
                 "topologically. Constraint asks that "
              << source->ToString() << " comes before " << sink->ToString()
              << " but the dependency analysis says this is impossible.";
          topo_refinements_[source].push_back(sink);
        }
        XLS_RET_CHECK_LE(io.MinimumLatency(), max_upper_bound_)
            << "Minimum latency is longer than the entire pipeline "
               "is "
               "allowed to be";
        if (io.MaximumLatency() > max_upper_bound_) {
          LOG(WARNING) << "Lowering maximum latency to " << max_upper_bound_
                       << " for " << io;
        }
        node_constraints_.emplace_back(NodeDifferenceConstraint{
            .anchor = source,
            .subject = sink,
            .min_after = std::min(io.MinimumLatency(), max_upper_bound_),
            .max_after = std::min(io.MaximumLatency(), max_upper_bound_),
        });
      }
    }

    return absl::OkStatus();
  }
  absl::Status AddSpecificConstraint(
      const RecvsFirstSendsLastConstraint& rfsl) {
    for (const auto& [_, channel_ops] : channel_ops_) {
      for (ChannelNode* recv : channel_ops.recv_ops) {
        node_constraints_.emplace_back(NodeInCycleConstraint{recv, 0});
      }
      for (ChannelNode* send : channel_ops.send_ops) {
        node_constraints_.emplace_back(LastStageConstraint{send});
      }
    }
    return absl::OkStatus();
  }

  absl::Status AddSpecificConstraint(const SendThenRecvConstraint& strc) {
    std::vector<ChannelNode*> sends;
    std::vector<ChannelNode*> receives;
    for (const auto& [chan, ops] : channel_ops_) {
      sends.reserve(sends.size() + ops.send_ops.size());
      receives.reserve(receives.size() + ops.recv_ops.size());
      sends.insert(sends.end(), ops.send_ops.begin(), ops.send_ops.end());
      receives.insert(receives.end(), ops.recv_ops.begin(), ops.recv_ops.end());
    }
    XLS_RET_CHECK(strc.MinimumLatency() >= 0);
    for (ChannelNode* send : sends) {
      XLS_RET_CHECK(send->Is<Send>()) << send->ToString();
      for (ChannelNode* recv : receives) {
        XLS_RET_CHECK(recv->Is<Receive>()) << recv->ToString();
        if (forward_dependency_analysis_.IsDependent(send, recv)) {
          node_constraints_.emplace_back(NodeDifferenceConstraint{
              .anchor = send,
              .subject = recv,
              .min_after = strc.MinimumLatency(),
              .max_after = max_upper_bound_,
          });
        }
      }
    }
    return absl::OkStatus();
  }

  absl::Status AddSpecificConstraint(const BackedgeConstraint& other) {
    if (!fb_->IsProc() || !ii_) {
      return absl::OkStatus();
    }
    for (Next* next : fb_->AsProcOrDie()->next_values()) {
      node_constraints_.emplace_back(NodeDifferenceConstraint{
          .anchor = next->state_read(),
          .subject = next,
          .min_after = 0,
          .max_after = *ii_ - 1,
      });
    }
    return absl::OkStatus();
  }
  absl::Status AddSpecificConstraint(const SameChannelConstraint& other) {
    if (other.MinimumLatency() == 0) {
      return absl::OkStatus();
    }
    // NB: At this point all kProvenMutuallyExclusive channels have
    // already been merged so there should not be more than a single
    // operation on them.
    for (const auto& [channel_ref, nodes] : channel_ops_) {
      if (nodes.send_ops.size() < 2 && nodes.recv_ops.size() < 2) {
        continue;
      }
      std::optional<ChannelStrictness> strictness =
          ChannelRefStrictness(channel_ref);
      if (strictness == ChannelStrictness::kArbitraryStaticOrder) {
        // Partition keeping the order within each partition.
        for (const auto& type : {nodes.send_ops, nodes.recv_ops}) {
          for (const auto& group : iter::sliding_window(type, 2)) {
            node_constraints_.emplace_back(NodeDifferenceConstraint{
                .anchor = group[0],
                .subject = group[1],
                .min_after = other.MinimumLatency(),
                .max_after = max_upper_bound_,
            });
          }
        }
      } else {
        // TODO(allight): It's weird that proven-mutex channels with
        // multiple sends or receives end up here but it can happen when
        // merge_on_mutual_exclusion is disabled. Match sdc by forcing these
        // into different cycles.
        //
        // XLS_RET_CHECK(strictness !=
        //               ChannelStrictness::kProvenMutuallyExclusive)
        //     << "Proven mutually exclusive channels should have been
        //     "
        //        "merged by now but channel "
        //     << ChannelRefToString(channel_ref)
        //     << " still has sends: ["
        //     << absl::StrJoin(nodes.sends, ", ",
        //                      [](std::string* out, ChannelNode* n) {
        //                        absl::StrAppend(out, n->ToString());
        //                      })
        //     << "] and recvs: ["
        //     << absl::StrJoin(nodes.recvs, ", ",
        //                      [](std::string* out, ChannelNode* n) {
        //                        absl::StrAppend(out, n->ToString());
        //                      })
        //     << "]";
        // TODO(allight): It would be nice to do the same trick that
        // sdc_scheduler does where it avoids adding a constraint if a
        // dominator of the node is already in the constraint set.
        // Since constraints for asap are somewhat cheaper this isn't
        // as big a deal. For simplicity just add constraints to every
        // pair which are dependent.
        for (const auto& group : {nodes.send_ops, nodes.recv_ops}) {
          for (const auto& combo : iter::combinations(group, 2)) {
            if (forward_dependency_analysis_.IsDependent(combo[0], combo[1])) {
              node_constraints_.emplace_back(NodeDifferenceConstraint{
                  .anchor = combo[0],
                  .subject = combo[1],
                  .min_after = other.MinimumLatency(),
                  .max_after = max_upper_bound_,
              });
            }
            XLS_RET_CHECK(
                !forward_dependency_analysis_.IsDependent(combo[1], combo[0]))
                << "Mismatch between dependency analysis and topo "
                   "sort for "
                << combo[0]->ToString() << " vs " << combo[1]->ToString();
          }
        }
      }
    }
    return absl::OkStatus();
  }
  absl::Status AddSpecificConstraint(const DifferenceConstraint& dc) {
    // TODO(allight): It would be good to support this. Though on the
    // other hand this constraint type doesn't seem to be used at all
    // outside of ECO flows so maybe we can just remove it.
    return absl::UnimplementedError(
        "Unsupported constraint type: DifferenceConstraint");
  }

  absl::StatusOr<std::vector<NodeSchedulingConstraint>> Finalize() && {
    // Refine the topo sort if needed to ensure that io-constraints can be
    // satisfied in a single pass.
    if (!topo_refinements_.empty()) {
      XLS_RETURN_IF_ERROR(graph_.RefineTopoSort(topo_refinements_))
          << "Failed to refine topo sort for io-constraints.";
    }
    return std::move(node_constraints_);
  }

 private:
  ConstraintConverter(ScheduleGraph& graph, std::optional<int64_t> ii,
                      int64_t max_upper_bound, FunctionBase* fb)
      : graph_(graph), ii_(ii), max_upper_bound_(max_upper_bound), fb_(fb) {
    node_constraints_.reserve(graph_.nodes().size());
  }

  absl::Status Init() {
    XLS_RET_CHECK(fb_ != nullptr);
    XLS_RETURN_IF_ERROR(forward_dependency_analysis_.Attach(fb_).status());
    if (fb_->IsProc()) {
      int64_t channel_cnt = fb_->AsProcOrDie()->channel_interfaces().size() +
                            fb_->AsProcOrDie()->package()->channels().size();
      channel_ops_.reserve(channel_cnt);
      channel_name_map_.reserve(channel_cnt);
      for (const ScheduleNode& sched_node : graph_.nodes()) {
        Node* node = sched_node.node;
        if (!node->Is<Send>() && !node->Is<Receive>()) {
          continue;
        }
        ChannelNode* channel_node = node->As<ChannelNode>();
        XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                             channel_node->GetChannelRef());
        channel_name_map_.emplace(channel_node->channel_name(), channel_ref);
        channel_ops_[channel_ref]
            .GetDirection(channel_node->direction())
            .push_back(channel_node);
      }
    }
    return absl::OkStatus();
  }

  ScheduleGraph& graph_;
  std::optional<int64_t> ii_;
  int64_t max_upper_bound_;
  FunctionBase* fb_;
  NodeForwardDependencyAnalysis forward_dependency_analysis_;
  // Additional constraints to the topo sort needed to ensure that bounds
  // tightening can be done in a single pass. This makes it so io-constraints
  // force the 'source' node to come before the 'sink' node in the topo sort.
  absl::flat_hash_map<Node*, std::vector<Node*>> topo_refinements_;
  std::vector<NodeSchedulingConstraint> node_constraints_;
  absl::flat_hash_map<ChannelRef, ChannelOps> channel_ops_;
  absl::flat_hash_map<std::string, ChannelRef> channel_name_map_;
};

}  // namespace

/* static */ absl::StatusOr<std::vector<NodeSchedulingConstraint>>
ScheduleBounds::ConvertSchedulingConstraints(
    ScheduleGraph& graph, absl::Span<const SchedulingConstraint> constraints,
    std::optional<int64_t> ii, int64_t max_upper_bound) {
  XLS_ASSIGN_OR_RETURN(auto converter,
                       ConstraintConverter::Create(graph, ii, max_upper_bound));
  // TODO(allight): This whole conversion is not done very efficiently. The size
  // of the io-constraint list is unlikely to be very large however even on
  // enormous designs so it should be fine. For better performance though we
  // might want to run through the list of nodes only once and check each
  // constraint for applicability.
  VLOG(2) << "For entity: " << graph.name();
  VLOG(2) << "  Max upper bound: " << max_upper_bound;
  VLOG(2) << "  Converting constraints: ["
          << absl::StrJoin(constraints, ", ",
                           [](std::string* out, const SchedulingConstraint& c) {
                             std::ostringstream ss;
                             std::visit([&](const auto& var) { ss << var; }, c);
                             absl::StrAppend(out, ss.str());
                           })
          << "]";
  // As a slight perf optimization do a single scan for channel nodes if we have
  // io constraints.
  for (const SchedulingConstraint& constraint : constraints) {
    XLS_RETURN_IF_ERROR(converter.AddConstraint(constraint))
        << "Could not add constraint for " << graph.name();
  }
  XLS_ASSIGN_OR_RETURN(std::vector<NodeSchedulingConstraint> node_constraints,
                       std::move(converter).Finalize());
  VLOG(2) << "  Node constraints are: ["
          << absl::StrJoin(node_constraints, ", ") << "]";
  return node_constraints;
}

/* static */ absl::StatusOr<ScheduleBounds> ScheduleBounds::Create(
    ScheduleGraph graph, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, std::optional<int64_t> ii,
    absl::Span<const SchedulingConstraint> constraints,
    int64_t max_upper_bound) {
  XLS_ASSIGN_OR_RETURN(std::vector<NodeSchedulingConstraint> node_constraints,
                       ScheduleBounds::ConvertSchedulingConstraints(
                           graph, constraints, ii, max_upper_bound));
  return ScheduleBounds::Create(std::move(graph), clock_period_ps,
                                delay_estimator, std::move(node_constraints),
                                max_upper_bound);
}

/* static */ absl::StatusOr<ScheduleBounds> ScheduleBounds::Create(
    ScheduleGraph graph, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator,
    absl::Span<NodeSchedulingConstraint const> constraints,
    int64_t max_upper_bound) {
  absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
      constraints_by_subject_lower_bound;
  absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>
      constraints_by_subject_upper_bound;
  for (const NodeSchedulingConstraint& constraint : constraints) {
    NodeSchedulingConstraint lower = constraint.is_lower_bound_constraint()
                                         ? constraint
                                         : constraint.Reverse(max_upper_bound);
    NodeSchedulingConstraint upper = constraint.is_lower_bound_constraint()
                                         ? constraint.Reverse(max_upper_bound)
                                         : constraint;
    constraints_by_subject_lower_bound[lower.subject()].emplace_back(lower);
    constraints_by_subject_upper_bound[upper.subject()].emplace_back(upper);
  }
  return ScheduleBounds(std::move(graph), clock_period_ps, delay_estimator,
                        std::move(constraints_by_subject_lower_bound),
                        std::move(constraints_by_subject_upper_bound),
                        max_upper_bound);
}

void ScheduleBounds::Reset() {
  max_lower_bound_ = 0;
  bounds_.clear();
  for (const ScheduleNode& sn : graph_.nodes()) {
    bounds_[sn.node] = {.after_min = 0, .before_max = 0};
  }
}

std::string ScheduleBounds::ToString() const {
  std::string out = "Bounds:\n";
  if (!bounds_.empty()) {
    absl::StatusOr<std::vector<Node*>> topo_sort_nodes =
        TopoSort(bounds_.begin()->first->function_base());
    if (!topo_sort_nodes.ok()) {
      return absl::StrFormat("ERROR: could not topo sort: %s\n",
                             topo_sort_nodes.status().message());
    }
    for (Node* node : *topo_sort_nodes) {
      if (bounds_.contains(node)) {
        absl::StrAppendFormat(&out, "  %s : [%d, %d]\n", node->GetName(),
                              lb(node), ub(node));
      }
    }
  }
  return out;
}

absl::StatusOr<int64_t> ScheduleBounds::GetDelay(Node* node) const {
  // Treat nodes that will be dead after synthesis as having a delay of 0.
  if (graph_.GetScheduleNode(node).is_dead_after_synthesis) {
    return 0;
  }
  return delay_estimator_->GetOperationDelayInPs(node);
}

namespace {
struct CycleInfo {
  // upper or lower bound depending on the context.
  int64_t bound;
  int64_t in_cycle_delay;
};
enum class BoundType { kLower, kUpper };

absl::Status TightenBound(int64_t& bound, int64_t value) {
  XLS_RET_CHECK_LE(bound, value)
      << "Bound " << bound << " is already greater than " << value;
  bound = value;
  return absl::OkStatus();
}

// Returns the new max bound if any bounds were changed or nullopt if nothing
// changed.
template <BoundType kBoundType>
absl::StatusOr<std::optional<int64_t>> PropagateGenericBounds(
    /*in-out*/ absl::flat_hash_map<Node*, ScheduleBounds::NodeBound>& bounds,
    const ScheduleGraph& graph,
    const absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>&
        constraints_by_subject,
    int64_t clock_period_ps, const DelayEstimator& delay_estimator,
    int64_t max_upper_bound) {
  VLOG(4) << "PropagateInternalBounds()";
  auto get_delay = [&](const ScheduleNode& node) -> absl::StatusOr<int64_t> {
    if (node.is_dead_after_synthesis) {
      return 0;
    }
    return delay_estimator.GetOperationDelayInPs(node.node);
  };
  auto cur_bound = [&](const auto& node) -> int64_t& {
    Node* real_node;
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(node)>,
                                 ScheduleNode>) {
      real_node = node.node;
    } else {
      static_assert(std::is_same_v<std::remove_cvref_t<decltype(node)>, Node*>);
      real_node = node;
    }
    ScheduleBounds::NodeBound& bound = bounds.at(real_node);
    if constexpr (kBoundType == BoundType::kLower) {
      return bound.after_min;
    } else {
      return bound.before_max;
    }
  };
  auto predecessors = [&](const ScheduleNode& node) {
    auto to_schedule_node = [&](Node* n) -> const ScheduleNode& {
      return graph.GetScheduleNode(n);
    };
    if constexpr (kBoundType == BoundType::kLower) {
      return iter::imap(to_schedule_node, node.predecessors);
    } else {
      return iter::imap(to_schedule_node, node.successors);
    }
  };
  auto nodes = [&]() {
    if constexpr (kBoundType == BoundType::kLower) {
      return xabsl::make_range(graph.nodes().begin(), graph.nodes().end());
    } else {
      return xabsl::make_range(graph.nodes().rbegin(), graph.nodes().rend());
    }
  }();
  bool changed = false;
  // The delay in picoseconds from the beginning of a cycle to the start of
  // the node.
  absl::flat_hash_map<Node*, int64_t> in_cycle_delay;

  int64_t max_bound = 0;
  auto add_bound = [&](int64_t bound) {
    max_bound = std::max(max_bound, bound);
  };

  // Compute the lower bound of each node based on the lower bounds of the
  // operands of the node.
  auto cycle_by_operands =
      [&](const ScheduleNode& node) -> absl::StatusOr<CycleInfo> {
    int64_t node_in_cycle_delay = 0;
    int64_t& node_cb = cur_bound(node);
    for (const ScheduleNode& operand : predecessors(node)) {
      int64_t operand_cb = cur_bound(operand);
      if (operand_cb < node_cb) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(int64_t operand_delay, get_delay(operand));
      if (operand_cb > node_cb) {
        VLOG(4) << absl::StreamFormat(
            "    tightened lb to %d because of operand %s", operand_cb,
            operand.node->GetName());
        XLS_RETURN_IF_ERROR(TightenBound(node_cb, operand_cb));
        changed = true;
        node_in_cycle_delay = in_cycle_delay.at(operand.node) + operand_delay;
        continue;
      }
      int64_t min_delay = operand.node->Is<MinDelay>()
                              ? operand.node->As<MinDelay>()->delay()
                              : 0;
      if (operand_cb + min_delay > node_cb) {
        VLOG(4) << absl::StreamFormat(
            "    tightened lb to %d because of operand %s", operand_cb,
            operand.node->GetName());
        XLS_RETURN_IF_ERROR(TightenBound(node_cb, operand_cb + min_delay));
        changed = true;
        node_in_cycle_delay = 0;
        continue;
      }
      node_in_cycle_delay = std::max(
          node_in_cycle_delay, in_cycle_delay.at(operand.node) + operand_delay);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay, get_delay(node));
    if (node_delay > clock_period_ps) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Node %s has a greater delay (%dps) than the clock period (%dps)",
          node.node->GetName(), node_delay, clock_period_ps));
    }
    if (node_in_cycle_delay + node_delay > clock_period_ps) {
      // Node does not fit in this cycle. Move to next cycle.
      VLOG(4) << "    overflows clock period, tightened lb to " << node_cb + 1;
      XLS_RETURN_IF_ERROR(TightenBound(node_cb, node_cb + 1));
      changed = true;
      node_in_cycle_delay = 0;
    }
    add_bound(node_cb);
    return CycleInfo{.bound = node_cb, .in_cycle_delay = node_in_cycle_delay};
  };
  for (const ScheduleNode& node : nodes) {
    int64_t& node_in_cycle_delay = in_cycle_delay[node.node];
    VLOG(4) << absl::StreamFormat("  %s : original bound=%d",
                                  node.node->GetName(), cur_bound(node));
    XLS_ASSIGN_OR_RETURN((const auto [cycle, op_in_cycle_delay]),
                         cycle_by_operands(node));
    node_in_cycle_delay = op_in_cycle_delay;
    if (constraints_by_subject.contains(node.node)) {
      for (const NodeSchedulingConstraint& constraint :
           constraints_by_subject.at(node.node)) {
        if (constraint.Is<LastStageConstraint>()) {
          // We handle these later by pushing them to the last possible cycle
          // in PropagateGenericConstraints and trying again.
        } else if (constraint.Is<NodeInCycleConstraint>()) {
          auto nicc = constraint.As<NodeInCycleConstraint>();
          XLS_RET_CHECK_GE(nicc.GetCycle(), cycle)
              << "Constraint "
              << (kBoundType == BoundType::kLower
                      ? constraint
                      : constraint.Reverse(max_upper_bound))
              << " is not compatible with " << node.node->ToString()
              << " due to delay requiring "
              << (kBoundType == BoundType::kLower ? "a later" : "an earlier")
              << " cycle ("
              << (kBoundType == BoundType::kLower ? cycle
                                                  : max_upper_bound - cycle)
              << ") than constraint";
          if (nicc.GetCycle() > cycle) {
            // Push to a later cycle.
            node_in_cycle_delay = 0;
            XLS_RETURN_IF_ERROR(TightenBound(cur_bound(node), nicc.GetCycle()));
            add_bound(cur_bound(node));
            changed = true;
          }
        } else {
          // This is a min-only update of constraints. We don't try to push
          // things forward until PropagateConstraints
          XLS_RET_CHECK(constraint.Is<NodeDifferenceConstraint>());
          auto ndc = constraint.As<NodeDifferenceConstraint>();
          int64_t anchor_cb = cur_bound(ndc.anchor);
          if (anchor_cb + ndc.min_after > cycle) {
            // Push the subject later.
            XLS_RETURN_IF_ERROR(
                TightenBound(cur_bound(node), anchor_cb + ndc.min_after));
            add_bound(cur_bound(node));
            node_in_cycle_delay = 0;
            changed = true;
          }
        }
      }
    }
  }
  return changed ? std::make_optional(max_bound) : std::nullopt;
}

template <BoundType kBoundType>
absl::StatusOr<bool> PropagateGenericConstraints(
    /*in-out*/ absl::flat_hash_map<Node*, ScheduleBounds::NodeBound>& bounds,
    const ScheduleGraph& graph,
    const absl::flat_hash_map<Node*, std::vector<NodeSchedulingConstraint>>&
        constraints_by_subject,
    int64_t max_found_bound) {
  VLOG(4) << "PropagateGenericConstraints()";
  auto cur_bound = [&](auto node) -> int64_t& {
    Node* real_node;
    if constexpr (std::is_same_v<decltype(node), ScheduleNode>) {
      real_node = node.node;
    } else {
      static_assert(std::is_same_v<decltype(node), Node*>);
      real_node = node;
    }
    ScheduleBounds::NodeBound& bound = bounds.at(real_node);
    if constexpr (kBoundType == BoundType::kLower) {
      return bound.after_min;
    } else {
      return bound.before_max;
    }
  };
  bool changed = false;
  std::vector<LastStageConstraint> last_stage_constraints;
  for (const auto& [node, constraints] : constraints_by_subject) {
    for (const NodeSchedulingConstraint& constraint : constraints) {
      if (constraint.Is<LastStageConstraint>()) {
        int64_t& subject = cur_bound(graph.GetScheduleNode(node));
        if constexpr (kBoundType == BoundType::kLower) {
          int64_t last_stage = max_found_bound;
          // Make sure that subject is within the bounds of the last stage.
          if (last_stage > subject) {
            changed = true;
            XLS_RETURN_IF_ERROR(TightenBound(subject, last_stage));
          }
        } else {
          // Upper bounds we clean up at the very end by forcing everything to
          // have an upper bound no higher than the earliest upper bound found
          // here.
          last_stage_constraints.push_back(
              constraint.As<LastStageConstraint>());
        }
      } else if (constraint.Is<NodeDifferenceConstraint>()) {
        auto ndc = constraint.As<NodeDifferenceConstraint>();
        int64_t& anchor = cur_bound(ndc.anchor);
        int64_t subject = cur_bound(ndc.subject);
        // Make sure that subject is within the bounds of the anchor.
        if (anchor + ndc.max_after < subject) {
          // pull forward so that subject is just within the bounds of the
          // anchor.
          changed = true;
          XLS_RETURN_IF_ERROR(TightenBound(anchor, subject - ndc.max_after));
        }
      } else {
        // These are handled by the generic bounds function.
        XLS_RET_CHECK(constraint.Is<NodeInCycleConstraint>()) << constraint;
      }
    }
  }
  if constexpr (kBoundType == BoundType::kUpper) {
    if (!last_stage_constraints.empty()) {
      // The largest is the latest the last stage can be.
      int64_t earliest_bound = 0;
      for (const auto& constraint : last_stage_constraints) {
        earliest_bound = std::max(earliest_bound, cur_bound(constraint.node));
      }
      for (const ScheduleNode& node : graph.nodes()) {
        if (earliest_bound > cur_bound(node)) {
          changed = true;
          XLS_RETURN_IF_ERROR(TightenBound(cur_bound(node), earliest_bound));
        }
      }
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ScheduleBounds::PropagateLowerBounds() {
  VLOG(4) << "PropagateLowerBounds()";
  XLS_ASSIGN_OR_RETURN(
      std::optional<int64_t> changed_push,
      PropagateGenericBounds<BoundType::kLower>(
          bounds_, graph_, constraints_by_subject_lower_bound_,
          clock_period_ps_, *delay_estimator_, max_upper_bound_));
  XLS_ASSIGN_OR_RETURN(
      bool changed_pull,
      PropagateGenericConstraints<BoundType::kLower>(
          bounds_, graph_, constraints_by_subject_lower_bound_,
          /*max_found_bound=*/changed_push.value_or(max_lower_bound_)));
  XLS_ASSIGN_OR_RETURN(
      std::optional<int64_t> changed_push2,
      PropagateGenericBounds<BoundType::kLower>(
          bounds_, graph_, constraints_by_subject_lower_bound_,
          clock_period_ps_, *delay_estimator_, max_upper_bound_));
  max_lower_bound_ =
      changed_push2.value_or(changed_push.value_or(max_lower_bound_));
  return changed_push || changed_pull || changed_push2;
}

absl::StatusOr<bool> ScheduleBounds::PropagateUpperBounds() {
  VLOG(4) << "PropagateUpperBounds()";
  XLS_ASSIGN_OR_RETURN(
      std::optional<int64_t> changed_push,
      PropagateGenericBounds<BoundType::kUpper>(
          bounds_, graph_, constraints_by_subject_upper_bound_,
          clock_period_ps_, *delay_estimator_, max_upper_bound_));
  XLS_ASSIGN_OR_RETURN(
      bool changed_pull,
      PropagateGenericConstraints<BoundType::kUpper>(
          bounds_, graph_, constraints_by_subject_upper_bound_,
          /*max_found_bound=*/changed_push.value_or(max_before_upper_bound_)));
  XLS_ASSIGN_OR_RETURN(
      std::optional<int64_t> changed_push2,
      PropagateGenericBounds<BoundType::kUpper>(
          bounds_, graph_, constraints_by_subject_upper_bound_,
          clock_period_ps_, *delay_estimator_, max_upper_bound_));
  max_before_upper_bound_ =
      changed_push2.value_or(changed_push.value_or(max_before_upper_bound_));
  return changed_push || changed_pull || changed_push2;
}

absl::Status ScheduleBounds::PropagateBounds(std::optional<int64_t> fuel) {
  bool stable = false;
  // Since upper and lower bounds don't actually affect one another we could do
  // them in separate loops. That would make determining whether the bounds are
  // stable harder though and probably wouldn't save much time.
  for (int64_t i = 0; !fuel.has_value() || i < *fuel; ++i) {
    VLOG(2) << "  Doing propagate bounds iteration: " << i;
    XLS_ASSIGN_OR_RETURN(bool changed_lb, PropagateLowerBounds());
    XLS_ASSIGN_OR_RETURN(bool changed_ub, PropagateUpperBounds());
    if (!changed_lb && !changed_ub) {
      stable = true;
      break;
    }
    // Check for lower-bound being beyond upper bound. This means we've failed
    // to schedule (probably because we're trying to push a node to the end but
    // its user is too long.)
    XLS_RETURN_IF_ERROR(CheckBasicBounds());
  }
  XLS_RETURN_IF_ERROR(CheckConstraints(fuel, stable));
  return absl::OkStatus();
}

absl::Status ScheduleBounds::CheckBasicBounds() {
  for (const auto& [node, bounds] : bounds_) {
    if (bounds.after_min > max_upper_bound_) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Lower bound (%d) of %s is greater than maximum pipeline length of "
          "%d",
          bounds.after_min, node->ToString(), max_upper_bound_));
    }
    if (bounds.before_max > max_upper_bound_) {
      return absl::ResourceExhaustedError(
          absl::StrFormat("Upper bound (%d) of %s is greater than maximum "
                          "allowable pipeline length of %d",
                          bounds.before_max - max_upper_bound_,
                          node->ToString(), max_upper_bound_));
    }
    if (bounds.after_min + bounds.before_max > max_upper_bound_) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Upper bound (%d) of %s is less than lower bound (%d)",
          max_upper_bound_ - bounds.before_max, node->ToString(),
          bounds.after_min));
    }
  }
  return absl::OkStatus();
}

absl::Status ScheduleBounds::CheckConstraints(std::optional<int64_t> fuel,
                                              bool stabilized) {
  auto main_result = [&]() -> absl::Status {
    int64_t before_max_last_stage_num =
        absl::c_min_element(bounds_, [](const auto& a, const auto& b) {
          return a.second.before_max < b.second.before_max;
        })->second.before_max;
    for (const auto& [node, bounds] : bounds_) {
      if (bounds.after_min > max_upper_bound_) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Lower bound (%d) of %s is greater than maximum pipeline length of "
            "%d",
            bounds.after_min, node->ToString(), max_upper_bound_));
      }
      if (bounds.before_max > max_upper_bound_) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Upper bound (%d) of %s is greater than maximum pipeline length of "
            "%d",
            bounds.before_max - max_upper_bound_, node->ToString(),
            max_upper_bound_));
      }
      if (bounds.before_max + bounds.after_min > max_upper_bound_) {
        return absl::ResourceExhaustedError(absl::StrFormat(
            "Upper bound (%d) of %s is less than lower bound (%d)",
            max_upper_bound_ - bounds.before_max, node->ToString(),
            bounds.after_min));
      }
      if (absl::c_any_of(node->operands(), [&](Node* operand) {
            return bounds.after_min < bounds_.at(operand).after_min;
          })) {
        return absl::InternalError(absl::StrFormat(
            "Lower bound (%d) of %s is less than lower bound of an operand",
            bounds.after_min, node->ToString()));
      }
      if (absl::c_any_of(node->users(), [&](Node* user) {
            return bounds.before_max < bounds_.at(user).before_max;
          })) {
        return absl::InternalError(absl::StrFormat(
            "Upper bound (%d) of %s is greater than upper bound of a successor",
            max_upper_bound_ - bounds.before_max, node->ToString()));
      }

      if (constraints_by_subject_lower_bound_.contains(node)) {
        for (const NodeSchedulingConstraint& constraint :
             constraints_by_subject_lower_bound_.at(node)) {
          if (constraint.Is<NodeDifferenceConstraint>()) {
            auto ndc = constraint.As<NodeDifferenceConstraint>();
            if (bounds.after_min <
                ndc.min_after + bounds_.at(ndc.anchor).after_min) {
              return absl::InternalError(absl::StrFormat(
                  "Lower bound (%d) of %s potentially "
                  "incompatible with constraint "
                  "%v due to anchor being at a lower bound of %d",
                  lb(node), node->ToString(), constraint, lb(ndc.anchor)));
            }
            if (bounds.after_min >
                ndc.max_after + bounds_.at(ndc.anchor).after_min) {
              return absl::InternalError(absl::StrFormat(
                  "Lower bound (%d) of %s potentially "
                  "incompatible with constraint "
                  "%v due to anchor being at a lower bound of %d",
                  lb(node), node->ToString(), constraint, lb(ndc.anchor)));
            }
          } else if (constraint.Is<NodeInCycleConstraint>()) {
            auto nicc = constraint.As<NodeInCycleConstraint>();
            if (lb(node) != nicc.GetCycle()) {
              return absl::InternalError(absl::StrFormat(
                  "Lower bound (%d) of %s potentially incompatible with "
                  "constraint %v",
                  lb(node), node->ToString(), constraint));
            }
          } else {
            XLS_RET_CHECK(constraint.Is<LastStageConstraint>()) << constraint;
            if (bounds.after_min != max_lower_bound_) {
              return absl::InternalError(absl::StrFormat(
                  "Lower bound (%d) of %s incompatible with constraint %v due "
                  "to being greater than max lower bound of %d",
                  lb(node), node->ToString(), constraint, max_lower_bound_));
            }
          }
        }
      }
      if (constraints_by_subject_upper_bound_.contains(node)) {
        for (const NodeSchedulingConstraint& constraint :
             constraints_by_subject_upper_bound_.at(node)) {
          if (constraint.Is<NodeDifferenceConstraint>()) {
            auto ndc = constraint.As<NodeDifferenceConstraint>();
            if (bounds.before_max <
                ndc.min_after + bounds_.at(ndc.anchor).before_max) {
              return absl::InternalError(absl::StrFormat(
                  "Upper bound (%d) of %s potentially "
                  "incompatible with constraint "
                  "%v due to anchor being at a lower bound of %d",
                  ub(node), node->ToString(),
                  constraint.Reverse(max_lower_bound_), ub(ndc.anchor)));
            }
            if (bounds.before_max >
                ndc.max_after + bounds_.at(ndc.anchor).before_max) {
              return absl::InternalError(absl::StrFormat(
                  "Upper bound (%d) of %s potentially "
                  "incompatible with constraint "
                  "%v due to anchor being at a lower bound of %d",
                  ub(node), node->ToString(),
                  constraint.Reverse(max_lower_bound_), ub(ndc.anchor)));
            }
          } else if (constraint.Is<NodeInCycleConstraint>()) {
            auto nicc = constraint.As<NodeInCycleConstraint>();
            if (bounds.before_max != nicc.GetCycle()) {
              return absl::InternalError(absl::StrFormat(
                  "Upper bound (%d) of %s potentially incompatible with "
                  "constraint %v",
                  ub(node), node->ToString(),
                  // NB Reversing the constraint so the cycle number is in the
                  // actual cycle number space.
                  constraint.Reverse(max_upper_bound_)));
            }
          } else {
            XLS_RET_CHECK(constraint.Is<LastStageConstraint>()) << constraint;
            if (bounds.before_max != before_max_last_stage_num) {
              return absl::InternalError(absl::StrFormat(
                  "Upper bound (%d) of %s incompatible with constraint %v due "
                  "to being greater than max upper bound of %d",
                  ub(node), node->ToString(), constraint,
                  max_upper_bound_ - before_max_last_stage_num));
            }
          }
        }
      }
    }
    return absl::OkStatus();
  }();
  if (main_result.ok() || stabilized) {
    return main_result;
  }
  return xabsl::StatusBuilder(main_result).SetPrepend()
         << "Constraints failed to converge after " << *fuel
         << " iterations. Result is not usable because: ";
}

namespace {
absl::Status PropagateBasicAsapAndAlapBounds(ScheduleBounds& b) {
  XLS_RETURN_IF_ERROR(b.PropagateBounds());
  VLOG(4) << "Setting all upper bounds to max-lower-bound "
          << b.max_lower_bound();
  for (const ScheduleNode& sn : b.graph().nodes()) {
    XLS_RETURN_IF_ERROR(b.TightenNodeUb(sn.node, b.max_lower_bound()));
  }
  XLS_RETURN_IF_ERROR(b.PropagateBounds());
  return absl::OkStatus();
}
}  // namespace

/* static */ absl::StatusOr<ScheduleBounds>
ScheduleBounds::ComputeAsapAndAlapBounds(
    FunctionBase* f, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, std::optional<int64_t> ii,
    absl::Span<const SchedulingConstraint> constraints,
    int64_t max_upper_bound) {
  VLOG(4) << "ComputeAsapAndAlapBounds()";
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> dead_after_synthesis,
                       GetDeadAfterSynthesisNodes(f));
  XLS_ASSIGN_OR_RETURN(ScheduleGraph graph,
                       ScheduleGraph::Create(f, dead_after_synthesis));
  XLS_ASSIGN_OR_RETURN(auto bounds, ScheduleBounds::Create(
                                        graph, clock_period_ps, delay_estimator,
                                        ii, constraints, max_upper_bound));
  XLS_RETURN_IF_ERROR(PropagateBasicAsapAndAlapBounds(bounds));
  return bounds;
}

/* static */ absl::StatusOr<ScheduleBounds>
ScheduleBounds::ComputeAsapAndAlapBounds(
    FunctionBase* f, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator,
    absl::Span<NodeSchedulingConstraint const> constraints,
    int64_t max_upper_bound) {
  VLOG(4) << "ComputeAsapAndAlapBoundsDirect()";
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> dead_after_synthesis,
                       GetDeadAfterSynthesisNodes(f));
  XLS_ASSIGN_OR_RETURN(ScheduleGraph graph,
                       ScheduleGraph::Create(f, dead_after_synthesis));
  XLS_ASSIGN_OR_RETURN(auto bounds, ScheduleBounds::Create(
                                        graph, clock_period_ps, delay_estimator,
                                        constraints, max_upper_bound));
  XLS_RETURN_IF_ERROR(PropagateBasicAsapAndAlapBounds(bounds));
  return bounds;
}

Annotation ScheduleBoundsAnnotator::NodeAnnotation(Node* node) const {
  if (!bounds_.graph().contains(node)) {
    return {};
  }
  return Annotation{.prefix = absl::StrFormat("[%d, %d]", bounds_.lb(node),
                                              bounds_.ub(node))};
}

}  // namespace sched
}  // namespace xls
