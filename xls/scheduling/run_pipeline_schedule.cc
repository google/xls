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

#include "xls/scheduling/run_pipeline_schedule.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_builder.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_search.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/fdo/delay_manager.h"
#include "xls/fdo/iterative_sdc_scheduler.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/asap_scheduler.h"
#include "xls/scheduling/min_cut_scheduler.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/random_scheduler.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/schedule_util.h"
#include "xls/scheduling/scheduler.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/sdc_scheduler.h"

namespace xls {

namespace {

// Tighten `bounds` to the ASAP/ALAP bounds for each node. If `schedule_length`
// is given, then the ALAP bounds are computed with the given length. Otherwise,
// we use the minimum viable pipeline length, per the ASAP bounds.
//
// Both schedules will be feasible if no other scheduling constraints are
// applied.
absl::Status TightenBounds(sched::ScheduleBounds& bounds, FunctionBase* f,
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
  return absl::OkStatus();
}

// Returns the critical path through the given nodes (ordered topologically),
// ignoring nodes that will be dead after synthesis.
absl::StatusOr<int64_t> ComputeCriticalPath(
    const ScheduleGraph& graph, const DelayEstimator& delay_estimator) {
  int64_t function_cp = 0;
  absl::flat_hash_map<Node*, int64_t> node_cp;
  for (const auto& sn : graph.nodes()) {
    if (sn.is_dead_after_synthesis) {
      continue;
    }
    int64_t node_start = 0;
    for (Node* operand : sn.node->operands()) {
      node_start = std::max(node_start, node_cp[operand]);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(sn.node));
    node_cp[sn.node] = node_start + node_delay;
    function_cp = std::max(function_cp, node_cp[sn.node]);
  }
  return function_cp;
}

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages. If
// `target_clock_period_ps` is specified, will not try to check lower clock
// periods than this.
absl::StatusOr<int64_t> FindMinimumClockPeriod(
    const ScheduleGraph& graph, std::optional<int64_t> pipeline_stages,
    std::optional<int64_t> worst_case_throughput,
    const DelayEstimator& delay_estimator, Scheduler& scheduler,
    SchedulingFailureBehavior failure_behavior,
    std::optional<int64_t> target_clock_period_ps = std::nullopt) {
  VLOG(4) << "FindMinimumClockPeriod()";
  VLOG(4) << "  graph = " << graph.name();
  VLOG(4) << "  scheduler = " << scheduler.name();
  VLOG(4) << "  pipeline stages = "
          << (pipeline_stages.has_value() ? absl::StrCat(*pipeline_stages)
                                          : "(unspecified)");
  VLOG(4) << "  worst case throughput = "
          << (worst_case_throughput.has_value()
                  ? absl::StrCat(*worst_case_throughput)
                  : "(unspecified)");
  VLOG(4) << "  target clock period = "
          << (target_clock_period_ps.has_value()
                  ? absl::StrCat(*target_clock_period_ps)
                  : "(unspecified)");
  XLS_ASSIGN_OR_RETURN(int64_t function_cp_ps,
                       ComputeCriticalPath(graph, delay_estimator));

  // The upper bound of the search is simply the critical path of the entire
  // function, and the lower bound is the critical path delay evenly distributed
  // across our pipeline stages (rounded up). It's possible the upper bound is
  // the best you can do if there exists a single operation with delay equal to
  // the critical-path delay of the function.
  int64_t pessimistic_clk_period_ps = std::max(int64_t{1}, function_cp_ps);
  int64_t optimistic_clk_period_ps = 1;
  if (pipeline_stages.has_value()) {
    optimistic_clk_period_ps =
        std::max(optimistic_clk_period_ps,
                 CeilOfRatio(function_cp_ps, *pipeline_stages));
  }
  if (target_clock_period_ps.has_value()) {
    // Don't check any clock period less than the specified target.
    optimistic_clk_period_ps =
        std::max(optimistic_clk_period_ps, *target_clock_period_ps);
    pessimistic_clk_period_ps =
        std::max(pessimistic_clk_period_ps, *target_clock_period_ps);
  }
  VLOG(4) << absl::StreamFormat("Binary searching over interval [%d, %d]",
                                optimistic_clk_period_ps,
                                pessimistic_clk_period_ps);

  // Check that it is in fact possible to
  // schedule this function at all; if not, return a useful error.
  XLS_RETURN_IF_ERROR(scheduler
                          .Schedule(pipeline_stages, pessimistic_clk_period_ps,
                                    failure_behavior, worst_case_throughput)
                          .status())
          .SetPrepend()
      << absl::StrFormat(
             "Impossible to schedule %s %s as specified with clock period %d",
             (graph.IsSingleProc() ? "proc" : "function"), graph.name(),
             pessimistic_clk_period_ps);

  // Don't waste time explaining infeasibility for the failing points in the
  // search.
  failure_behavior.explain_infeasibility = false;
  int64_t min_clk_period_ps = BinarySearchMinTrue(
      optimistic_clk_period_ps, pessimistic_clk_period_ps,
      [&](int64_t clk_period_ps) {
        auto sched =
            scheduler.Schedule(pipeline_stages, clk_period_ps, failure_behavior,
                               worst_case_throughput);
        VLOG(4) << "FindMinClockPeriod(" << graph.name()
                << ", pipeline_stages=" << pipeline_stages.value_or(-1)
                << ", worst_case_throughput="
                << worst_case_throughput.value_or(-1)
                << ", clk_period_ps=" << clk_period_ps
                << "): " << sched.status();
        return sched.ok();
      },
      BinarySearchAssumptions::kEndKnownTrue);
  VLOG(4) << "minimum clock period = " << min_clk_period_ps;

  return min_clk_period_ps;
}

// Returns the minimum inverse worst-case throughput for which it is feasible to
// schedule the function into a pipeline with the given number of stages and
// target clock period.
absl::StatusOr<int64_t> FindMinimumWorstCaseThroughput(
    Proc* proc, std::optional<int64_t> pipeline_stages, int64_t clock_period_ps,
    Scheduler& scheduler, SchedulingFailureBehavior failure_behavior) {
  VLOG(4) << "FindMinimumWorstCaseThroughput()";
  VLOG(4) << "  pipeline stages = "
          << (pipeline_stages.has_value() ? absl::StrCat(*pipeline_stages)
                                          : "(unspecified)")
          << ", clock period = " << clock_period_ps << " ps";

  // Check that it is in fact possible to schedule this function at all, with no
  // worst-case throughput bound; if not, return a useful error.
  XLS_ASSIGN_OR_RETURN(
      ScheduleCycleMap schedule_cycle_map,
      scheduler.Schedule(pipeline_stages, clock_period_ps, failure_behavior,
                         /*worst_case_throughput=*/std::nullopt),
      _.SetPrepend() << absl::StrFormat(
          "Impossible to schedule proc %s as specified; proc ", proc->name()));

  // Extract the worst-case throughput from this schedule as an upper bound.
  int64_t pessimistic_worst_case_throughput = 1;
  for (Next* next : proc->next_values()) {
    Node* state_read = next->state_read();
    const int64_t backedge_length =
        schedule_cycle_map[next] - schedule_cycle_map[state_read];
    pessimistic_worst_case_throughput =
        std::max(pessimistic_worst_case_throughput, backedge_length + 1);
  }
  VLOG(4) << absl::StreamFormat(
      "Schedules at worst-case throughput %d; now binary searching over "
      "interval [1, %d]",
      pessimistic_worst_case_throughput, pessimistic_worst_case_throughput);

  // Don't waste time explaining infeasibility for the failing points in the
  // search.
  failure_behavior.explain_infeasibility = false;
  int64_t min_worst_case_throughput = BinarySearchMinTrue(
      1, pessimistic_worst_case_throughput,
      [&](int64_t worst_case_throughput) {
        CHECK(!failure_behavior.explain_infeasibility);
        auto sched = scheduler.Schedule(
            pipeline_stages, clock_period_ps, failure_behavior,
            /*worst_case_throughput=*/worst_case_throughput);
        VLOG(4) << "FindMinimumWorstCaseThroughput("
                << absl::StreamFormat(
                       "pipeline_stages=%d, clock_period_ps=%d, "
                       "worst_case_throughput=%d",
                       pipeline_stages.value_or(-1), clock_period_ps,
                       worst_case_throughput)
                << "): " << sched.status();
        sched.IgnoreError();
        return sched.ok();
      },
      BinarySearchAssumptions::kEndKnownTrue);
  VLOG(4) << "minimum worst-case throughput = " << min_worst_case_throughput;

  return min_worst_case_throughput;
}

// Returns true if the given node communicates via a channel which is on the
// interface of the top-level proc.
// TODO(meheff): Revisit whether this is the right thing to do.  This will
// return true if any instance of the proc has the channel as an I/O. With
// multiple instantiations, this could lead to unexpected behavior.
bool IsExternalIoNode(ChannelNode* node,
                      const std::optional<const ProcElaboration*> elab) {
  Proc* proc = node->function_base()->AsProcOrDie();
  if (proc->is_new_style_proc()) {
    // Channels are proc-scoped.
    absl::StatusOr<ChannelInterface*> channel_interface =
        proc->GetChannelInterface(node->As<ChannelNode>()->channel_name(),
                                  node->Is<Send>()
                                      ? ChannelDirection::kSend
                                      : ChannelDirection::kReceive);
    CHECK_OK(channel_interface.status());
    CHECK(elab.has_value());
    for (ChannelInstance* channel_instance :
         (*elab)->GetInstancesOfChannelInterface(channel_interface.value())) {
      if ((*elab)->IsTopInterfaceChannel(channel_instance)) {
        return true;
      }
    }
    return false;
  }

  // Channels are globally scoped.
  absl::StatusOr<Channel*> chan = GetChannelUsedByNode(node);
  CHECK_OK(chan.status());
  return (*chan)->supported_ops() != ChannelOps::kSendReceive;
}

absl::StatusOr<int64_t> ApplyClockMargin(const SchedulingOptions& options,
                                         int64_t clock_period_ps) {
  if (!options.clock_margin_percent().has_value()) {
    return clock_period_ps;
  }

  int64_t original_clock_period_ps = clock_period_ps;
  clock_period_ps -=
      (clock_period_ps * *options.clock_margin_percent() + 50) / 100;
  if (clock_period_ps <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Clock period non-positive (%dps) after adjusting for margin. "
        "Original clock period: %dps, clock margin: %d%%",
        clock_period_ps, original_clock_period_ps,
        *options.clock_margin_percent()));
  }
  return clock_period_ps;
}

// Handles the case where scheduling fails, providing more informative error
// messages based on the failure type and options.
absl::Status HandleScheduleFailure(
    absl::Status schedule_status, const ScheduleGraph& graph,
    const SchedulingOptions& options,
    std::optional<int64_t> worst_case_throughput,
    const DelayEstimator& io_delay_added,
    std::optional<int64_t>& min_clock_period_ps_for_tracing,
    int64_t clock_period_ps, Scheduler& bounds_scheduler) {
  if (absl::IsInvalidArgument(schedule_status)) {
    // The scheduler was able to explain the failure; report it up without
    // further analysis.
    return schedule_status;
  }
  if (options.clock_period_ps().has_value()) {
    // The user specified a specific clock period; see if we can confirm
    // that that's the issue.

    if (options.minimize_clock_on_failure().value_or(true)) {
      // Find the smallest clock period that would have worked.
      LOG(LEVEL(options.recover_after_minimizing_clock().value_or(false)
                    ? absl::LogSeverity::kWarning
                    : absl::LogSeverity::kError))
          << "Unable to schedule with the specified clock period ("
          << clock_period_ps
          << " ps); finding the shortest feasible clock period...";
      int64_t target_clock_period_ps = clock_period_ps + 1;
      absl::StatusOr<int64_t> min_clock_period_ps = FindMinimumClockPeriod(
          graph, options.pipeline_stages(), worst_case_throughput,
          io_delay_added, bounds_scheduler, options.failure_behavior(),
          target_clock_period_ps);
      VLOG(4) << "FindMinimumClockPeriod() returned " << min_clock_period_ps;
      if (min_clock_period_ps.ok()) {
        min_clock_period_ps_for_tracing = *min_clock_period_ps;
        // Just increasing the clock period suffices.
        return absl::InvalidArgumentError(
            absl::StrFormat("cannot achieve the specified clock period. Try "
                            "`--clock_period_ps=%d`.",
                            *min_clock_period_ps));
      }
      if (absl::IsInvalidArgument(min_clock_period_ps.status())) {
        // We failed with an explained error at the longest possible clock
        // period. Report this error up, adding that the clock period will
        // also need to be increased - though we don't know by how much.
        return xabsl::StatusBuilder(std::move(min_clock_period_ps).status())
                   .SetPrepend()
               << absl::StrFormat(
                      "cannot achieve the specified clock period; try "
                      "increasing `--clock_period_ps`. Also, ");
      }
      // We fail with an unexplained error even at the longest possible
      // clock period. Report the original error.
      return schedule_status;
    }

    // Check if just increasing the clock period would have helped.
    XLS_ASSIGN_OR_RETURN(int64_t pessimistic_clock_period_ps,
                         ComputeCriticalPath(graph, io_delay_added));
    // Make a copy of failure behavior with explain_feasibility true- we
    // always want to produce an error message because this we are
    // re-running the scheduler for its error message.
    SchedulingFailureBehavior pessimistic_failure_behavior =
        options.failure_behavior();
    pessimistic_failure_behavior.explain_infeasibility = true;
    absl::Status pessimistic_status =
        bounds_scheduler
            .Schedule(options.pipeline_stages(), pessimistic_clock_period_ps,
                      pessimistic_failure_behavior)
            .status();
    if (pessimistic_status.ok()) {
      // Just increasing the clock period suffices.
      return absl::InvalidArgumentError(
          "cannot achieve the specified clock period. Try increasing "
          "`--clock_period_ps`.");
    }
    if (absl::IsInvalidArgument(pessimistic_status)) {
      // We failed with an explained error at the pessimistic clock period.
      // Report this error up, adding that the clock period will also need
      // to be increased - though we don't know by how much.
      return xabsl::StatusBuilder(std::move(pessimistic_status)).SetPrepend()
             << absl::StrFormat(
                    "cannot achieve the specified clock period; try "
                    "increasing `--clock_period_ps`. Also, ");
    }
    return pessimistic_status;
  }
  return schedule_status;
}

absl::StatusOr<PipelineSchedule> RunIterativeSDCSchedule(
    FunctionBase* f, const SchedulingOptions& options, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator,
    const synthesis::Synthesizer* synthesizer) {
  if (!options.clock_period_ps().has_value()) {
    return absl::UnimplementedError(
        "Iterative SDC scheduling is only supported when a clock period is "
        "specified.");
  }

  IterativeSDCSchedulingOptions isdc_options;
  isdc_options.synthesizer = synthesizer;
  isdc_options.iteration_number = options.fdo_iteration_number();
  isdc_options.delay_driven_path_number =
      options.fdo_delay_driven_path_number();
  isdc_options.fanout_driven_path_number =
      options.fdo_fanout_driven_path_number();
  isdc_options.stochastic_ratio = options.fdo_refinement_stochastic_ratio();
  isdc_options.path_evaluate_strategy = options.fdo_path_evaluate_strategy();

  XLS_ASSIGN_OR_RETURN(DelayManager delay_manager,
                       DelayManager::Create(f, delay_estimator));
  XLS_ASSIGN_OR_RETURN(
      ScheduleCycleMap cycle_map,
      ScheduleByIterativeSDC(f, options.pipeline_stages(), clock_period_ps,
                             delay_manager, options.constraints(), isdc_options,
                             options.failure_behavior()));

  // Use delay manager for scheduling timing verification.
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Create(f, cycle_map, options.pipeline_stages()));
  XLS_RETURN_IF_ERROR(schedule.Verify());
  XLS_RETURN_IF_ERROR(schedule.VerifyTiming(clock_period_ps, delay_manager));
  XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                 f->GetInitiationInterval()));

  XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
  return schedule;
}

// NB For backwards compatibility reasons the flag options for worst case
// throughput/II are rather confusing. Specifically, a value of 0 means that
// the scheduler is free to choose any WCT it wants and a value of nullopt
// means the value must be exactly 1.
//
// To make everything after this clearer we change this interpretation to have
// std::nullopt mean "no throughput requirements" and `Some(X)` mean a
// throughput requirement of X. Therefore a value of 0 from now on is invalid.
std::optional<int64_t> GetWorstCaseThroughputSetting(
    const SchedulingOptions& options, FunctionBase* f) {
  if (!f->IsProc()) {
    VLOG(4) << "No II is possible because this is not a proc.";
    return std::nullopt;
  }
  std::optional<int64_t> worst_case_throughput =
      options.worst_case_throughput();
  if (worst_case_throughput && *worst_case_throughput == 0) {
    VLOG(4) << "Setting II to unconstrained from options.";
    worst_case_throughput = std::nullopt;
  } else if (!worst_case_throughput) {
    // No throughput requirement from options, See if the proc has one.
    //
    // NB This is only reachable if the scheduler is being configured from
    // something other than scheduling_options_flags.cc and has complicated
    // backwards-compat behaviors.
    //
    // In the future we are likely to move towards this being the primary way of
    // configuring things however and we may want to reconsider how it is
    // interpreted if/when that happens.
    //
    // TODO(allight): It may be worthwhile to go through and figure out what
    // tests/other tools depend on this behavior to rationalize it all.
    if (f->IsProc() && f->GetInitiationInterval()) {
      // Now convert this to the new interpretation.
      worst_case_throughput = f->GetInitiationInterval();
      if (worst_case_throughput == 0) {
        VLOG(4) << "Setting II to unconstrained from proc setting.";
        worst_case_throughput = std::nullopt;
      } else {
        VLOG(4) << "Setting II to " << *worst_case_throughput
                << " from proc setting.";
      }
    } else if (options.minimize_worst_case_throughput()) {
      VLOG(4) << "Setting II to unconstrained since no options or proc setting "
                 "and MinimizeWorstCaseThroughput is true.";
      worst_case_throughput = std::nullopt;
    } else {
      VLOG(4) << "Setting II to 1 since no options or proc setting and "
                 "MinimizeWorstCaseThroughput is false.";
      worst_case_throughput = 1;
    }
  } else {
    VLOG(4) << "Setting II to " << *worst_case_throughput
            << " from worst_case_throughput option.";
  }
  return worst_case_throughput;
}

// Set the WCT (either calculated or from options) to the function. For
// compatibility we need this to be in the same strange format as the option
// setting. Specifically where nullopt means a WCT of 1, and a value of 0 means
// unconstrained and all other values are WCT of the value.
//
// TODO(allight): Go through and update all users to the internal format where
// std::nullopt means unconstrained and any value is an exact WCT.
void SetWorstCaseThroughput(std::optional<int64_t> worst_case_throughput,
                            FunctionBase* f) {
  if (!f->IsProc()) {
    return;
  }
  if (worst_case_throughput.has_value()) {
    if (worst_case_throughput == 1) {
      f->AsProcOrDie()->ClearInitiationInterval();
    } else {
      f->AsProcOrDie()->SetInitiationInterval(*worst_case_throughput);
    }
  } else {
    f->AsProcOrDie()->SetInitiationInterval(0);
  }
}

absl::StatusOr<PipelineSchedule> RunPipelineScheduleInternal(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options,
    const std::optional<const ProcElaboration*> elab,
    const synthesis::Synthesizer* synthesizer) {
  if (!options.pipeline_stages().has_value() &&
      !options.clock_period_ps().has_value()) {
    return absl::InvalidArgumentError(
        "Pipeline scheduling requires either --pipeline_stages or "
        "--clock_period_ps to be specified; see "
        "https://google.github.io/xls/codegen_options/"
        "#pipelining-and-scheduling-options for details.");
  }

  if (f->IsProc() && f->AsProcOrDie()->is_new_style_proc() &&
      !elab.has_value()) {
    return absl::InvalidArgumentError(
        "Pipeline scheduling of a proc with proc-scoped channels requires an "
        "elaboration.");
  }

  // NB For backwards compatibility reasons the flag options for worst case
  // throughput/II are rather confusing. Specifically, a value of 0 means that
  // the scheduler is free to choose any WCT it wants and a value of nullopt
  // means the value must be exactly 1.
  //
  // To make everything after this clearer we change this interpretation to have
  // std::nullopt mean "no throughput requirements" and `Some(X)` mean a
  // throughput requirement of X. Therefore a value of 0 from now on is invalid.
  std::optional<int64_t> worst_case_throughput =
      GetWorstCaseThroughputSetting(options, f);

  VLOG(4) << "Computed WCT is "
          << (worst_case_throughput.has_value()
                  ? absl::StrCat(*worst_case_throughput)
                  : "unconstrained");

  int64_t input_delay = options.additional_input_delay_ps().value_or(0);
  int64_t output_delay = options.additional_output_delay_ps().value_or(0);
  // Sends and receives each have inputs and outputs from the flow control
  // signals, so the max of input and output delays is the amount needed for
  // each channel.
  int64_t max_io_delay = std::max(input_delay, output_delay);

  DecoratingDelayEstimator io_delay_added(
      "io_delay_added", delay_estimator, [&](Node* node, int64_t base_delay) {
        if (node->Is<ChannelNode>()) {
          std::string_view channel_name =
              node->As<ChannelNode>()->channel_name();
          std::string channel_op = absl::StrCat(
              channel_name, ":",
              node->As<ChannelNode>()->direction() == ChannelDirection::kSend
                  ? "send"
                  : "recv");
          int64_t channel_delay_ps = 0;
          if (std::optional<int64_t> op_delay =
                  options.additional_channel_delay_ps(channel_op);
              op_delay.has_value()) {
            channel_delay_ps = *op_delay;
          } else if (std::optional<int64_t> channel_delay =
                         options.additional_channel_delay_ps(channel_name);
                     channel_delay.has_value()) {
            channel_delay_ps = *channel_delay;
          }

          if (IsExternalIoNode(node->As<ChannelNode>(), elab)) {
            return base_delay + max_io_delay + channel_delay_ps;
          }

          return base_delay + channel_delay_ps;
        }
        if (node->function_base()->IsFunction()) {
          if (node->Is<Param>()) {
            return base_delay + input_delay;
          }
          if (node->function_base()->AsFunctionOrDie()->return_value() ==
              node) {
            return base_delay + output_delay;
          }
        }
        return base_delay;
      });

  if (worst_case_throughput == 0) {
    VLOG(4) << "Worst case throughput explicitly set to 0 (unbounded).";
    worst_case_throughput = std::nullopt;
  }

  if (options.pipeline_stages() == 1 &&
      !options.clock_period_ps().has_value() &&
      !options.failure_behavior().explain_infeasibility) {
    // No scheduling to be done, and there's no way to violate timing; just
    // schedule everything (other than literals) in the first cycle.
    ScheduleCycleMap cycle_map;
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes, TopoSort(f));
    for (Node* node : topo_sort_nodes) {
      if (!IsUntimed(node)) {
        cycle_map[node] = 0;
      }
    }
    XLS_ASSIGN_OR_RETURN(PipelineSchedule schedule,
                         PipelineSchedule::Create(f, std::move(cycle_map),
                                                  options.pipeline_stages()));
    XLS_RETURN_IF_ERROR(schedule.Verify());
    XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                   f->GetInitiationInterval()));

    XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
    return schedule;
  }

  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> dead_after_synthesis,
                       GetDeadAfterSynthesisNodes(f));
  XLS_ASSIGN_OR_RETURN(ScheduleGraph graph,
                       ScheduleGraph::Create(f, dead_after_synthesis));

  // std::unique_ptr<SDCScheduler> sdc_scheduler;
  std::unique_ptr<Scheduler> scheduler;
  auto create_scheduler =
      [&](SchedulingStrategy strategy,
          std::optional<double> dynamic_throughput_objective_weight,
          bool check_feasibility)
      -> absl::StatusOr<std::unique_ptr<Scheduler>> {
    std::unique_ptr<Scheduler> s;
    switch (strategy) {
      case SchedulingStrategy::ASAP: {
        s = std::make_unique<ASAPScheduler>(graph, io_delay_added);
        break;
      }
      case SchedulingStrategy::MIN_CUT: {
        s = std::make_unique<MinCutScheduler>(graph, io_delay_added);
        break;
      }
      case SchedulingStrategy::SDC: {
        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<SDCScheduler> sdc_scheduler,
            SDCScheduler::Create(graph, io_delay_added, options));
        sdc_scheduler->SetCheckFeasibility(check_feasibility);
        sdc_scheduler->SetDynamicThroughputObjectiveWeight(
            dynamic_throughput_objective_weight);
        s = std::move(sdc_scheduler);
        break;
      }
      case SchedulingStrategy::RANDOM: {
        s = std::make_unique<RandomScheduler>(
            graph, io_delay_added,
            absl::BitGen(std::seed_seq{options.seed().value_or(0)}));
        break;
      }
    }
    VLOG(4) << "Constraints are : ["
            << absl::StrJoin(options.constraints(), ",") << "]";
    XLS_RETURN_IF_ERROR(s->AddConstraints(options.constraints()));
    return std::move(s);
  };
  auto initialize_scheduler = [&]() -> absl::Status {
    if (scheduler == nullptr) {
      XLS_ASSIGN_OR_RETURN(
          scheduler,
          create_scheduler(options.strategy(),
                           options.dynamic_throughput_objective_weight(),
                           /*check_feasibility=*/false));
    }
    VLOG(4) << "Using scheduler " << scheduler->name()
            << " for primary scheduling.";
    return absl::OkStatus();
  };
  std::unique_ptr<Scheduler> bounds_scheduler;
  auto initialize_bounds_scheduler = [&]() -> absl::Status {
    if (bounds_scheduler == nullptr) {
      // Even if we are SDC we ignore the dynamic_throughput_objective_weight
      // for bounds.
      XLS_ASSIGN_OR_RETURN(
          bounds_scheduler,
          create_scheduler(options.find_bounds_strategy(),
                           /*dynamic_throughput_objective_weight=*/std::nullopt,
                           /*check_feasibility=*/true));
    }
    VLOG(4) << "Using scheduler " << bounds_scheduler->name() << " for bounds.";
    return absl::OkStatus();
  };

  std::optional<int64_t> min_clock_period_ps_for_tracing;
  int64_t clock_period_ps;
  if (!options.clock_period_ps().has_value() ||
      (options.minimize_clock_on_failure().value_or(false) &&
       options.recover_after_minimizing_clock().value_or(false))) {
    // We don't know the exact target clock period - either none was provided,
    // or we want to fall back to the minimum feasible clock period if the
    // target is infeasible. Determine the minimum clock period for which the
    // function can be scheduled in the given pipeline length (adjusted for the
    // user-specified relaxation percent if needed), and use that if it's
    // smaller than the target clock period.
    //
    // TODO(allight): Ideally we'd always use the ASAP scheduler but it is
    // possible to create graphs that are not ASAP schedulable. For now just
    // use the configured bounds strategy.
    VLOG(4) << "Finding min clock due to "
            << (!options.clock_period_ps().has_value()
                    ? "no target clock period"
                    : "minimize_clock_on_failure and recover_after_minimizing_"
                      "clock");
    XLS_RETURN_IF_ERROR(initialize_bounds_scheduler());
    XLS_ASSIGN_OR_RETURN(
        min_clock_period_ps_for_tracing,
        FindMinimumClockPeriod(graph, options.pipeline_stages(),
                               worst_case_throughput, io_delay_added,
                               *bounds_scheduler, options.failure_behavior()));

    // Pad the minimum clock period to account for the clock margin.
    if (options.clock_margin_percent().has_value()) {
      if (*options.clock_margin_percent() >= 100) {
        return absl::InvalidArgumentError(
            "Clock margin percent must be less than 100.");
      }
      *min_clock_period_ps_for_tracing =
          (*min_clock_period_ps_for_tracing * 100 + 50) /
          (100 - *options.clock_margin_percent());
    }

    // Set the clock period to the minimum feasible value.
    clock_period_ps = *min_clock_period_ps_for_tracing;

    if (options.period_relaxation_percent().has_value()) {
      // Apply the user-specified relaxation to allow less evenly distributed
      // slack.
      int64_t relaxation_percent = options.period_relaxation_percent().value();
      int64_t adjustment = ((clock_period_ps * relaxation_percent + 50) / 100);
      VLOG(4) << "Applying period relaxation of " << adjustment
              << " ps to clock period of " << clock_period_ps
              << " ps. Total is: " << (clock_period_ps + adjustment) << " ps.";
      clock_period_ps += adjustment;
    }

    if (options.clock_period_ps().has_value()) {
      // If the user specified a clock period, and it's at least as long as our
      // relaxed minimum clock period, use that instead; no need to squeeze the
      // stages for a tighter clock than the user's target.
      if (VLOG_IS_ON(4) && clock_period_ps < *options.clock_period_ps()) {
        VLOG(4) << "User-specified clock period of "
                << *options.clock_period_ps() << " is greater than minimum "
                << "feasible clock period of " << clock_period_ps
                << " ps; using user-specified clock period.";
      }
      clock_period_ps = std::max(clock_period_ps, *options.clock_period_ps());
    }

    if (options.clock_period_ps().has_value() &&
        clock_period_ps > *options.clock_period_ps()) {
      CHECK(options.minimize_clock_on_failure().value_or(false));
      CHECK(options.recover_after_minimizing_clock().value_or(false));
      LOG(WARNING) << "Target clock period was " << *options.clock_period_ps()
                   << ", but shortest feasible clock period (after any "
                      "specified relaxation) is "
                   << clock_period_ps
                   << " ps; continuing with clock period = " << clock_period_ps
                   << " ps.";
    }
    VLOG(4) << "Got clock period of " << clock_period_ps << " from minimum.";
  } else {
    VLOG(4) << "Got clock period of " << options.clock_period_ps().value()
            << " from options.";
    clock_period_ps = options.clock_period_ps().value();
  }

  XLS_ASSIGN_OR_RETURN(clock_period_ps,
                       ApplyClockMargin(options, clock_period_ps));

  // Only actually attempt to minimize WCT if we are (1) told we can do so,
  // (2) we don't have an explicitly chosen WCT already, and (3) we have a
  // constraint that actually cares about WCT.
  // TODO(allight): Does it actually make sense to ever not have a back-edge
  // constraint with a WCT setting?
  if (options.minimize_worst_case_throughput() && f->IsProc() &&
      !worst_case_throughput &&
      absl::c_any_of(
          options.constraints(), [](const SchedulingConstraint& constraint) {
            return std::holds_alternative<BackedgeConstraint>(constraint);
          })) {
    XLS_RETURN_IF_ERROR(initialize_bounds_scheduler());
    absl::StatusOr<int64_t> wct = FindMinimumWorstCaseThroughput(
        f->AsProcOrDie(), options.pipeline_stages(), clock_period_ps,
        *bounds_scheduler,
        /*failure_behavior=*/{.explain_infeasibility = false});
    if (wct.ok()) {
      worst_case_throughput = *wct;
      LOG(INFO) << "Minimized worst-case throughput for proc '" << f->name()
                << "': " << *worst_case_throughput;
    } else {
      VLOG(2) << "Failed to minimize worst-case throughput for proc '"
              << f->name() << "', continuing to allow normal error handling: "
              << wct.status();
    }
  }

  // Worst case throughput is either known from the options, the proc itself, or
  // minimization by now. Notate it in the proc itself.
  //
  // Using a helper since other parts of codegen/sched/other tools use the same
  // nullopt interpretation as the scheduling_options flag. See comment on
  // function for more information.
  SetWorstCaseThroughput(worst_case_throughput, f);

  // TODO(allight): Rewrite FDO into the scheduler API.
  if (options.use_fdo() && options.strategy() == SchedulingStrategy::SDC) {
    if (f->IsProc()) {
      XLS_RET_CHECK_EQ(f->AsProcOrDie()->GetInitiationInterval().has_value(),
                       worst_case_throughput.has_value())
          << " ii: " << f->AsProcOrDie()->GetInitiationInterval().value_or(-1)
          << " wct: " << worst_case_throughput.value_or(-1);
      if (worst_case_throughput) {
        XLS_RET_CHECK_EQ(*f->AsProcOrDie()->GetInitiationInterval(),
                         *worst_case_throughput);
      }
    }
    return RunIterativeSDCSchedule(f, options, clock_period_ps, delay_estimator,
                                   synthesizer);
  } else if (options.use_fdo()) {
    return absl::InvalidArgumentError(
        "FDO is only supported with SDC strategy.");
  }
  VLOG(4) << "Starting primary scheduling. Failure behavior is: "
          << options.failure_behavior().ToProto().ShortDebugString();
  ScheduleCycleMap cycle_map;
  XLS_RETURN_IF_ERROR(initialize_scheduler());
  absl::StatusOr<ScheduleCycleMap> schedule_cycle_map =
      scheduler->Schedule(options.pipeline_stages(), clock_period_ps,
                          options.failure_behavior(), worst_case_throughput);

  if (!schedule_cycle_map.ok()) {
    XLS_RETURN_IF_ERROR(initialize_bounds_scheduler());
    return HandleScheduleFailure(std::move(schedule_cycle_map).status(), graph,
                                 options, worst_case_throughput, io_delay_added,
                                 min_clock_period_ps_for_tracing,
                                 clock_period_ps, *bounds_scheduler);
  }
  cycle_map = *std::move(schedule_cycle_map);

  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      PipelineSchedule::Create(f, cycle_map, options.pipeline_stages(),
                               min_clock_period_ps_for_tracing));
  XLS_VLOG_LINES(5, "Schedule\n" + schedule.ToString());
  XLS_RETURN_IF_ERROR(schedule.Verify());
  XLS_RETURN_IF_ERROR(schedule.VerifyTiming(clock_period_ps, io_delay_added));
  XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                 f->GetInitiationInterval()));

  return schedule;
}

}  // namespace

absl::StatusOr<PipelineSchedule> RunPipelineSchedule(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options,
    std::optional<const ProcElaboration*> elab) {
  return RunPipelineScheduleInternal(f, delay_estimator, options, elab,
                                     /*synthesizer=*/nullptr);
}

absl::StatusOr<PipelineSchedule> RunPipelineScheduleWithFdo(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options, const synthesis::Synthesizer& synthesizer,
    std::optional<const ProcElaboration*> elab) {
  return RunPipelineScheduleInternal(f, delay_estimator, options, elab,
                                     &synthesizer);
}

absl::StatusOr<PackageSchedule> RunSynchronousPipelineSchedule(
    Package* package, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options, const ProcElaboration& elab) {
  // TODO(https://github.com/google/xls/issues/2175): Synchronous scheduling
  // currently only supports a clock period option.
  XLS_RET_CHECK(options.clock_period_ps().has_value());
  XLS_RET_CHECK(!options.pipeline_stages().has_value());
  XLS_RET_CHECK(!options.worst_case_throughput().has_value());
  XLS_RET_CHECK(!options.clock_margin_percent().has_value());
  XLS_RET_CHECK(!options.period_relaxation_percent().has_value());
  XLS_RET_CHECK(!options.use_fdo());
  XLS_RET_CHECK(!options.additional_input_delay_ps().has_value());
  XLS_RET_CHECK(!options.additional_output_delay_ps().has_value());
  XLS_RET_CHECK(options.strategy() == SchedulingStrategy::SDC);

  // TODO(https://github.com/google/xls/issues/2175): Current limitations are
  // that all sends and receives mush be unconditional and procs can only be
  // instantiated once.
  for (Proc* proc : elab.procs()) {
    if (elab.GetInstances(proc).size() != 1) {
      return absl::UnimplementedError(
          absl::StrFormat("Proc `%s` is instantiated more than once which is "
                          "not supported with synchronous procs.",
                          proc->name()));
    }
    for (Node* node : proc->nodes()) {
      if (node->Is<ChannelNode>() &&
          node->As<ChannelNode>()->predicate().has_value()) {
        return absl::UnimplementedError(
            absl::StrFormat("Send/receive node `%s` is predicated which is not "
                            "supported with synchronous procs.",
                            node->GetName()));
      }
    }
  }

  XLS_ASSIGN_OR_RETURN(
      ScheduleGraph graph,
      ScheduleGraph::CreateSynchronousGraph(package, /*loopback_channels=*/{},
                                            elab, /*dead_after_synthesis=*/{}));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SDCScheduler> sdc_scheduler,
                       SDCScheduler::Create(graph, delay_estimator, options));
  XLS_RETURN_IF_ERROR(sdc_scheduler->AddConstraints(options.constraints()));

  XLS_ASSIGN_OR_RETURN(
      ScheduleCycleMap cycle_map,
      sdc_scheduler->Schedule(std::nullopt, *options.clock_period_ps(),
                              options.failure_behavior()));

  // Convert the global cycle map of all nodes in the package to individual
  // schedules for each proc.
  //
  // * For the top-level proc, the schedule cycle for each node X is the cycle
  //   in the global schedule (cycle_map[X]).
  //
  // * For all other procs, the schedule cycle for a node X is cycle_map[X]
  //   minus the minimum cycle for any node in the proc.
  //
  // The computed adjustments for each proc are saved in `synchronous_offsets`.
  absl::flat_hash_map<FunctionBase*, int64_t> synchronous_offsets;
  // Iterate through the proc hierarchy from the bottom up.
  for (auto it = elab.procs().crbegin(); it != elab.procs().crend(); ++it) {
    Proc* proc = *it;

    int64_t earliest_stage;
    if (proc == elab.top()->proc() || proc->node_count() == 0) {
      earliest_stage = 0;
    } else {
      earliest_stage = std::numeric_limits<int64_t>::max();
      for (Node* node : proc->nodes()) {
        if (IsUntimed(node)) {
          continue;
        }
        earliest_stage = std::min(earliest_stage, cycle_map.at(node));
      }
    }
    synchronous_offsets[proc] = earliest_stage;
  }

  PackageSchedule::ScheduleMap schedule_map;
  for (Proc* proc : elab.procs()) {
    absl::flat_hash_map<Node*, int64_t> proc_cycle_map;
    for (Node* node : proc->nodes()) {
      if (IsUntimed(node)) {
        continue;
      }
      proc_cycle_map[node] = cycle_map.at(node) - synchronous_offsets.at(proc);
    }
    XLS_ASSIGN_OR_RETURN(schedule_map[proc],
                         PipelineSchedule::Create(proc, proc_cycle_map));
  }

  PackageSchedule package_schedule(package, std::move(schedule_map),
                                   std::move(synchronous_offsets));
  VLOG(2) << package_schedule.ToString();
  return package_schedule;
}

}  // namespace xls
