// Copyright 2026 The XLS Authors
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

#include "xls/scheduling/asap_scheduler.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

// Returns the nodes of `f` which must be scheduled in the first stage of a
// pipeline. For functions this is parameters.
std::vector<Node*> FirstStageNodes(FunctionBase* f) {
  if (Function* function = dynamic_cast<Function*>(f)) {
    return std::vector<Node*>(function->params().begin(),
                              function->params().end());
  }

  return {};
}
// Returns the nodes of `f` which must be scheduled in the final stage of a
// pipeline. For functions this is the return value.
std::vector<Node*> FinalStageNodes(FunctionBase* f) {
  if (Function* function = dynamic_cast<Function*>(f)) {
    // If the return value is a parameter, then we do not force the return value
    // to be scheduled in the final stage because, as a parameter, the node must
    // be in the first stage.
    if (function->return_value()->Is<Param>()) {
      return {};
    }
    return {function->return_value()};
  }

  return {};
}

}  // namespace

// Tighten `bounds` to the ASAP/ALAP bounds for each node. If `schedule_length`
// is given, then the ALAP bounds are computed with the given length. Otherwise,
// we use the minimum viable pipeline length, per the ASAP bounds.
//
// Both schedules will be feasible if no other scheduling constraints are
// applied.
/* static */ absl::Status ASAPScheduler::TightenBounds(
    sched::ScheduleBounds& bounds, FunctionBase* f,
    std::optional<int64_t> schedule_length) {
  // If we have a schedule length give everything that upper bound.
  if (schedule_length.has_value()) {
    for (Node* node : f->nodes()) {
      XLS_RETURN_IF_ERROR(
          bounds.TightenNodeUb(node, schedule_length.value() - 1));
    }
  }
  // If we have a schedule_length (and therefore a small-ish max stage) we can
  // try to propagate forever since we will hit that schedule length before
  // long.
  std::optional<int64_t> propagation_fuel =
      schedule_length ? std::nullopt : std::make_optional<int64_t>(6);
  XLS_RETURN_IF_ERROR(bounds.PropagateBounds(propagation_fuel))
      << "Failed to schedule bounds for " << f->name() << ".";
  VLOG(5) << "ASAP bounds: "
          << f->DumpIr(sched::ScheduleBoundsAnnotator(bounds));
  return absl::OkStatus();
}
absl::Status ASAPScheduler::GenerateHelpfulError(
    absl::Status&& orig_status, std::optional<int64_t> pipeline_stages,
    int64_t clock_period_ps, std::optional<int64_t> worst_case_throughput) {
  xabsl::StatusBuilder status(std::move(orig_status));
  // Try to figure out what the actual required stages are.
  if (pipeline_stages.has_value()) {
    auto no_length = ComputeBounds(/*pipeline_stages=*/std::nullopt,
                                   clock_period_ps, worst_case_throughput);
    no_length.IgnoreError();
    if (no_length.ok()) {
      return (status << absl::StrFormat(
                  "Function %s cannot be scheduled in %d "
                  "stages. Computed minimum is --pipeline_stages=%d",
                  graph().name(), pipeline_stages.value(),
                  no_length->max_lower_bound() + 1))
          .SetCode(absl::StatusCode::kInvalidArgument);
    }
    return (status << "Function " << graph().name()
                   << " cannot be scheduled in any number of stages via ASAP "
                      "bounds. Some constraints may be unsatisfiable or "
                      "require a full SDC schedule to resolve.")
        .SetCode(absl::StatusCode::kResourceExhausted);
  }
  return status;
}

absl::StatusOr<ScheduleCycleMap> ASAPScheduler::Schedule(
    std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
    SchedulingFailureBehavior failure_behavior,
    std::optional<int64_t> worst_case_throughput) {
  XLS_ASSIGN_OR_RETURN(
      sched::ScheduleBounds bounds,
      ComputeBounds(pipeline_stages, clock_period_ps, worst_case_throughput));
  ScheduleCycleMap cycle_map;
  cycle_map.reserve(graph().nodes().size());
  XLS_RET_CHECK(!pipeline_stages.has_value() || pipeline_stages == 1 ||
                pipeline_stages >= bounds.max_lower_bound() + 1)
      << "Pipeline stages must be at least 1 or greater than or equal to "
         "the number of stages in the function. pipeline_stages: "
      << (pipeline_stages ? *pipeline_stages : -1)
      << " max_lower_bound: " << bounds.max_lower_bound();
  // Just schedule everything as soon as possible.
  for (const auto& sn : graph().nodes()) {
    cycle_map[sn.node] = bounds.lb(sn.node);
  }
  return cycle_map;
}

absl::StatusOr<sched::ScheduleBounds> ASAPScheduler::ComputeBounds(
    std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
    std::optional<int64_t> worst_case_throughput, bool get_helpful_error,
    int64_t max_upper_bound) {
  XLS_RET_CHECK(std::holds_alternative<FunctionBase*>(graph_.ir_scope()));
  auto* f = std::get<FunctionBase*>(graph_.ir_scope());
  // TODO(allight): This actually creates a copy of graph_ since it needs to
  // have its topo-sort changed as constraints are added. Since that list is
  // held with the graph it needs its own mutable copy. Since the graph is not
  // actually terribly large and this copy is only done once per call this is
  // fine.
  VLOG(5) << "ASAP scheduler: graph: " << graph_.name();
  VLOG(5) << "                clock_period_ps: " << clock_period_ps;
  VLOG(5) << "                worst_case_throughput: "
          << worst_case_throughput.value_or(-1);
  VLOG(5) << "                constraints: "
          << absl::StrJoin(constraints_, ", ");
  XLS_ASSIGN_OR_RETURN(
      auto bounds, sched::ScheduleBounds::Create(
                       graph_, clock_period_ps, delay_estimator_,
                       worst_case_throughput, constraints_, max_upper_bound));
  // Add first and last stage constraints.
  using LastStageConstraint =
      sched::ScheduleBounds::NodeSchedulingConstraint::LastStageConstraint;
  for (Node* n : FirstStageNodes(f)) {
    bounds.AddConstraint(NodeInCycleConstraint{n, 0});
  }
  for (Node* n : FinalStageNodes(f)) {
    bounds.AddConstraint(LastStageConstraint{n});
  }
  absl::Status tighten_bounds_status =
      TightenBounds(bounds, f, pipeline_stages);
  if (!tighten_bounds_status.ok()) {
    if (get_helpful_error) {
      return GenerateHelpfulError(std::move(tighten_bounds_status),
                                  pipeline_stages, clock_period_ps,
                                  worst_case_throughput);
    }
    return tighten_bounds_status;
  }
  return bounds;
}

}  // namespace xls
