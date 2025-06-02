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
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/log_severity.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
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
#include "xls/scheduling/min_cut_scheduler.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/schedule_util.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/sdc_scheduler.h"

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

// Tighten `bounds` to the ASAP/ALAP bounds for each node. If `schedule_length`
// is given, then the ALAP bounds are computed with the given length. Otherwise,
// we use the minimum viable pipeline length, per the ASAP bounds.
//
// Both schedules will be feasible if no other scheduling constraints are
// applied.
absl::Status TightenBounds(sched::ScheduleBounds& bounds, FunctionBase* f,
                           std::optional<int64_t> schedule_length) {
  // Initially compute the lower bounds of all nodes.
  XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());

  int64_t upper_bound;
  if (schedule_length.has_value()) {
    if (schedule_length.value() <= bounds.max_lower_bound()) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Cannot be scheduled in %d stages. Computed lower bound is %d.",
          schedule_length.value(), bounds.max_lower_bound() + 1));
    }
    upper_bound = schedule_length.value() - 1;
  } else {
    upper_bound = bounds.max_lower_bound();
  }

  // Set the lower bound of nodes which must be in the final stage to
  // `upper_bound`
  bool rerun_lb_propagation = false;
  for (Node* node : FinalStageNodes(f)) {
    if (bounds.lb(node) != upper_bound) {
      XLS_RETURN_IF_ERROR(bounds.TightenNodeLb(node, upper_bound));
      if (!node->users().empty()) {
        rerun_lb_propagation = true;
      }
    }
  }

  // If fixing nodes in the final stage changed any lower bounds then
  // repropagate the lower bounds.
  if (rerun_lb_propagation) {
    XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
  }

  if (bounds.max_lower_bound() > upper_bound) {
    return absl::ResourceExhaustedError(
        absl::StrFormat("Impossible to schedule %s %s; the following node(s) "
                        "must be scheduled in the final cycle but that is "
                        "impossible due to users of these node(s): %s",
                        (f->IsProc() ? "proc" : "function"), f->name(),
                        absl::StrJoin(FinalStageNodes(f), ", ")));
  }

  // Set and propagate upper bounds.
  for (Node* node : f->nodes()) {
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, upper_bound));
  }
  for (Node* node : FirstStageNodes(f)) {
    if (bounds.lb(node) > 0) {
      return absl::ResourceExhaustedError(absl::StrFormat(
          "Impossible to schedule %s %s; node `%s` must be scheduled in the "
          "first cycle but that is impossible due to the node's operand(s)",
          (f->IsProc() ? "Proc" : "Function"), f->name(), node->GetName()));
    }
    XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, 0));
  }
  XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());

  return absl::OkStatus();
}

// Returns the critical path through the given nodes (ordered topologically),
// ignoring nodes that will be dead after synthesis.
absl::StatusOr<int64_t> ComputeCriticalPath(
    absl::Span<Node* const> topo_sort,
    const absl::flat_hash_set<Node*>& dead_after_synthesis,
    const DelayEstimator& delay_estimator) {
  int64_t function_cp = 0;
  absl::flat_hash_map<Node*, int64_t> node_cp;
  for (Node* node : topo_sort) {
    if (dead_after_synthesis.contains(node)) {
      continue;
    }
    int64_t node_start = 0;
    for (Node* operand : node->operands()) {
      node_start = std::max(node_start, node_cp[operand]);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    node_cp[node] = node_start + node_delay;
    function_cp = std::max(function_cp, node_cp[node]);
  }
  return function_cp;
}
absl::StatusOr<int64_t> ComputeCriticalPath(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  return ComputeCriticalPath(TopoSort(f), GetDeadAfterSynthesisNodes(f),
                             delay_estimator);
}

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages. If
// `target_clock_period_ps` is specified, will not try to check lower clock
// periods than this.
absl::StatusOr<int64_t> FindMinimumClockPeriod(
    FunctionBase* f, std::optional<int64_t> pipeline_stages,
    std::optional<int64_t> worst_case_throughput,
    const DelayEstimator& delay_estimator, SDCScheduler& scheduler,
    SchedulingFailureBehavior failure_behavior,
    std::optional<int64_t> target_clock_period_ps = std::nullopt) {
  VLOG(4) << "FindMinimumClockPeriod()";
  VLOG(4) << "  pipeline stages = "
          << (pipeline_stages.has_value() ? absl::StrCat(*pipeline_stages)
                                          : "(unspecified)");
  XLS_ASSIGN_OR_RETURN(int64_t function_cp_ps,
                       ComputeCriticalPath(f, delay_estimator));

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
                 (function_cp_ps + *pipeline_stages - 1) / *pipeline_stages);
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
                                    failure_behavior,
                                    /*check_feasibility=*/true,
                                    worst_case_throughput)
                          .status())
          .SetPrepend()
      << absl::StrFormat("Impossible to schedule %s %s as specified; ",
                         (f->IsProc() ? "proc" : "function"), f->name());

  // Don't waste time explaining infeasibility for the failing points in the
  // search.
  failure_behavior.explain_infeasibility = false;
  int64_t min_clk_period_ps = BinarySearchMinTrue(
      optimistic_clk_period_ps, pessimistic_clk_period_ps,
      [&](int64_t clk_period_ps) {
        return scheduler
            .Schedule(pipeline_stages, clk_period_ps, failure_behavior,
                      /*check_feasibility=*/true, worst_case_throughput)
            .ok();
      },
      BinarySearchAssumptions::kEndKnownTrue);
  VLOG(4) << "minimum clock period = " << min_clk_period_ps;

  return min_clk_period_ps;
}

// Returns the minimum inverse worst-case throughput for which it is feasible to
// schedule the function into a pipeline with the given number of stages and
// target clock period.
absl::StatusOr<int64_t> FindMinimumWorstCaseThroughput(
    FunctionBase* f, std::optional<int64_t> pipeline_stages,
    int64_t clock_period_ps, SDCScheduler& scheduler,
    SchedulingFailureBehavior failure_behavior) {
  VLOG(4) << "FindMinimumWorstCaseThroughput()";
  VLOG(4) << "  pipeline stages = "
          << (pipeline_stages.has_value() ? absl::StrCat(*pipeline_stages)
                                          : "(unspecified)")
          << ", clock period = " << clock_period_ps << " ps";

  Proc* proc = f->AsProcOrDie();

  // Check that it is in fact possible to schedule this function at all, with no
  // worst-case throughput bound; if not, return a useful error.
  XLS_ASSIGN_OR_RETURN(
      ScheduleCycleMap schedule_cycle_map,
      scheduler.Schedule(pipeline_stages, clock_period_ps, failure_behavior,
                         /*check_feasibility=*/true,
                         /*worst_case_throughput=*/0),
      _.SetPrepend() << absl::StrFormat(
          "Impossible to schedule %s %s as specified; ",
          (f->IsProc() ? "proc" : "function"), f->name()));

  // Extract the worst-case throughput from this schedule as an upper bound.
  int64_t pessimistic_worst_case_throughput = 1;
  if (f->IsProc()) {
    for (Next* next : proc->next_values()) {
      Node* state_read = next->state_read();
      const int64_t backedge_length =
          schedule_cycle_map[next] - schedule_cycle_map[state_read];
      pessimistic_worst_case_throughput =
          std::max(pessimistic_worst_case_throughput, backedge_length + 1);
    }
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
        return scheduler
            .Schedule(pipeline_stages, clock_period_ps, failure_behavior,
                      /*check_feasibility=*/true,
                      /*worst_case_throughput=*/worst_case_throughput)
            .ok();
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

  int64_t input_delay = options.additional_input_delay_ps().value_or(0);
  int64_t output_delay = options.additional_output_delay_ps().value_or(0);
  // Sends and receives each have inputs and outputs from the flow control
  // signals, so the max of input and output delays is the amount needed for
  // each channel.
  int64_t max_io_delay = std::max(input_delay, output_delay);

  DecoratingDelayEstimator io_delay_added(
      "io_delay_added", delay_estimator, [&](Node* node, int64_t base_delay) {
        if (node->Is<ChannelNode>()) {
          if (IsExternalIoNode(node->As<ChannelNode>(), elab)) {
            return base_delay + max_io_delay;
          }
          return base_delay;
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

  if (options.worst_case_throughput().has_value()) {
    f->SetInitiationInterval(*options.worst_case_throughput());
  }

  if (options.pipeline_stages() == 1 &&
      !options.clock_period_ps().has_value() &&
      !options.failure_behavior().explain_infeasibility) {
    // No scheduling to be done, and there's no way to violate timing; just
    // schedule everything in the first cycle.
    ScheduleCycleMap cycle_map;
    for (Node* node : TopoSort(f)) {
      cycle_map[node] = 0;
    }
    PipelineSchedule schedule =
        PipelineSchedule(f, std::move(cycle_map), options.pipeline_stages());
    XLS_RETURN_IF_ERROR(schedule.Verify());
    XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                   f->GetInitiationInterval()));

    XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
    return schedule;
  }

  std::unique_ptr<SDCScheduler> sdc_scheduler;
  auto initialize_sdc_scheduler = [&]() -> absl::Status {
    if (sdc_scheduler == nullptr) {
      XLS_ASSIGN_OR_RETURN(sdc_scheduler,
                           SDCScheduler::Create(f, io_delay_added));
      XLS_RETURN_IF_ERROR(sdc_scheduler->AddConstraints(options.constraints()));
    }
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
    // NOTE: We currently use the SDC scheduler to determine the minimum clock
    //       period (if not specified), even if we're not using it for the final
    //       schedule.
    XLS_RETURN_IF_ERROR(initialize_sdc_scheduler());
    XLS_ASSIGN_OR_RETURN(
        min_clock_period_ps_for_tracing,
        FindMinimumClockPeriod(
            f, options.pipeline_stages(),
            /*worst_case_throughput=*/f->IsProc() ? f->GetInitiationInterval()
                                                  : std::nullopt,
            io_delay_added, *sdc_scheduler, options.failure_behavior()));

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
      clock_period_ps += (clock_period_ps * relaxation_percent + 50) / 100;
    }

    if (options.clock_period_ps().has_value()) {
      // If the user specified a clock period, and it's at least as long as our
      // relaxed minimum clock period, use that instead; no need to squeeze the
      // stages for a tighter clock than the user's target.
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
  } else {
    clock_period_ps = options.clock_period_ps().value();
  }

  XLS_ASSIGN_OR_RETURN(clock_period_ps,
                       ApplyClockMargin(options, clock_period_ps));

  std::optional<int64_t> worst_case_throughput = std::nullopt;
  if (options.minimize_worst_case_throughput().value_or(false) && f->IsProc() &&
      f->GetInitiationInterval().value_or(1) <= 0 &&
      absl::c_any_of(
          options.constraints(), [](const SchedulingConstraint& constraint) {
            return std::holds_alternative<BackedgeConstraint>(constraint);
          })) {
    // NOTE: We currently use the SDC scheduler to minimize the worst-case
    //       throughput (if minimization is requested), even if we're not using
    //       it for the final schedule.
    XLS_RETURN_IF_ERROR(initialize_sdc_scheduler());
    absl::StatusOr<int64_t> wct = FindMinimumWorstCaseThroughput(
        f, options.pipeline_stages(), clock_period_ps, *sdc_scheduler,
        /*failure_behavior=*/{.explain_infeasibility = false});
    if (wct.ok()) {
      worst_case_throughput = *wct;
      f->AsProcOrDie()->SetInitiationInterval(*wct);
      LOG(INFO) << "Minimized worst-case throughput for proc '" << f->name()
                << "': " << *worst_case_throughput;
    } else {
      VLOG(2) << "Failed to minimize worst-case throughput for proc '"
              << f->name() << "', continuing to allow normal error handling: "
              << wct.status();
    }
  }

  ScheduleCycleMap cycle_map;
  if (options.strategy() == SchedulingStrategy::SDC) {
    // Enable iterative SDC scheduling when use_fdo is true
    if (options.use_fdo()) {
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
      isdc_options.path_evaluate_strategy =
          options.fdo_path_evaluate_strategy();

      DelayManager delay_manager(f, delay_estimator);
      XLS_ASSIGN_OR_RETURN(
          cycle_map,
          ScheduleByIterativeSDC(f, options.pipeline_stages(), clock_period_ps,
                                 delay_manager, options.constraints(),
                                 isdc_options, options.failure_behavior()));

      // Use delay manager for scheduling timing verification.
      auto schedule = PipelineSchedule(f, cycle_map, options.pipeline_stages());
      XLS_RETURN_IF_ERROR(schedule.Verify());
      XLS_RETURN_IF_ERROR(
          schedule.VerifyTiming(clock_period_ps, delay_manager));
      XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(
          options.constraints(), f->GetInitiationInterval()));

      XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
      return schedule;
    }

    XLS_RETURN_IF_ERROR(initialize_sdc_scheduler());
    absl::StatusOr<ScheduleCycleMap> schedule_cycle_map =
        sdc_scheduler->Schedule(options.pipeline_stages(), clock_period_ps,
                                options.failure_behavior(),
                                /*check_feasibility=*/false,
                                worst_case_throughput,
                                options.dynamic_throughput_objective_weight());
    if (!schedule_cycle_map.ok()) {
      if (absl::IsInvalidArgument(schedule_cycle_map.status())) {
        // The scheduler was able to explain the failure; report it up without
        // further analysis.
        return std::move(schedule_cycle_map).status();
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
          XLS_RETURN_IF_ERROR(initialize_sdc_scheduler());
          absl::StatusOr<int64_t> min_clock_period_ps = FindMinimumClockPeriod(
              f, options.pipeline_stages(), worst_case_throughput,
              io_delay_added, *sdc_scheduler, options.failure_behavior(),
              target_clock_period_ps);
          if (min_clock_period_ps.ok()) {
            min_clock_period_ps_for_tracing = *min_clock_period_ps;
            // Just increasing the clock period suffices.
            return absl::InvalidArgumentError(absl::StrFormat(
                "cannot achieve the specified clock period. Try "
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
          return std::move(schedule_cycle_map).status();
        }

        // Check if just increasing the clock period would have helped.
        XLS_ASSIGN_OR_RETURN(int64_t pessimistic_clock_period_ps,
                             ComputeCriticalPath(f, io_delay_added));
        // Make a copy of failure behavior with explain_feasibility true- we
        // always want to produce an error message because this we are
        // re-running the scheduler for its error message.
        SchedulingFailureBehavior pessimistic_failure_behavior =
            options.failure_behavior();
        pessimistic_failure_behavior.explain_infeasibility = true;
        XLS_RETURN_IF_ERROR(initialize_sdc_scheduler());
        absl::Status pessimistic_status =
            sdc_scheduler
                ->Schedule(options.pipeline_stages(),
                           pessimistic_clock_period_ps,
                           pessimistic_failure_behavior,
                           /*check_feasibility=*/true)
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
          return xabsl::StatusBuilder(std::move(pessimistic_status))
                     .SetPrepend()
                 << absl::StrFormat(
                        "cannot achieve the specified clock period; try "
                        "increasing `--clock_period_ps`. Also, ");
        }
        return pessimistic_status;
      }
      return schedule_cycle_map.status();
    }
    cycle_map = *std::move(schedule_cycle_map);
  } else {
    // Run an initial ASAP/ALAP scheduling pass, which we'll refine with the
    // chosen scheduler.
    sched::ScheduleBounds bounds(f, TopoSort(f), clock_period_ps,
                                 io_delay_added);
    XLS_RETURN_IF_ERROR(TightenBounds(bounds, f, options.pipeline_stages()));

    if (options.strategy() == SchedulingStrategy::MIN_CUT) {
      XLS_ASSIGN_OR_RETURN(
          cycle_map,
          MinCutScheduler(
              f,
              options.pipeline_stages().value_or(bounds.max_lower_bound() + 1),
              clock_period_ps, io_delay_added, &bounds, options.constraints()));
    } else if (options.strategy() == SchedulingStrategy::RANDOM) {
      std::mt19937_64 gen(options.seed().value_or(0));

      cycle_map = ScheduleCycleMap();
      for (Node* node : TopoSort(f)) {
        int64_t cycle = absl::Uniform<int64_t>(
            absl::IntervalClosed, gen, bounds.lb(node), bounds.ub(node));
        XLS_RETURN_IF_ERROR(bounds.TightenNodeLb(node, cycle));
        XLS_RETURN_IF_ERROR(bounds.PropagateLowerBounds());
        XLS_RETURN_IF_ERROR(bounds.TightenNodeUb(node, cycle));
        XLS_RETURN_IF_ERROR(bounds.PropagateUpperBounds());
        cycle_map[node] = cycle;
      }
    } else {
      XLS_RET_CHECK(options.strategy() == SchedulingStrategy::ASAP);
      XLS_RET_CHECK(!options.pipeline_stages().has_value() ||
                    options.pipeline_stages() == 1);
      // Just schedule everything as soon as possible.
      for (Node* node : f->nodes()) {
        cycle_map[node] = bounds.lb(node);
      }
    }
  }

  auto schedule = PipelineSchedule(f, cycle_map, options.pipeline_stages(),
                                   min_clock_period_ps_for_tracing);
  XLS_RETURN_IF_ERROR(schedule.Verify());
  XLS_RETURN_IF_ERROR(schedule.VerifyTiming(clock_period_ps, io_delay_added));
  XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                 f->GetInitiationInterval()));

  XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
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

absl::StatusOr<PackagePipelineSchedules> RunSynchronousPipelineSchedule(
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
                       SDCScheduler::Create(graph, delay_estimator));
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
  //   minus the minimum cycle for any node in the proc including nodes of procs
  //   instantiated (transitively) by the proc.
  PackagePipelineSchedules schedules;
  absl::flat_hash_map<Proc*, int64_t> first_proc_cycle;
  // Iterate through the proc hierarchy from the bottom up.
  for (auto it = elab.procs().crbegin(); it != elab.procs().crend(); ++it) {
    Proc* proc = *it;
    XLS_ASSIGN_OR_RETURN(ProcInstance * proc_instance,
                         elab.GetUniqueInstance(proc));

    int64_t earliest_stage = std::numeric_limits<int64_t>::max();
    for (const std::unique_ptr<ProcInstance>& instantiated_proc_instance :
         proc_instance->instantiated_procs()) {
      earliest_stage =
          std::min(earliest_stage,
                   first_proc_cycle.at(instantiated_proc_instance->proc()));
    }
    for (Node* node : proc->nodes()) {
      earliest_stage = std::min(earliest_stage, cycle_map.at(node));
    }
    first_proc_cycle[proc] = earliest_stage;
  }

  for (Proc* proc : elab.procs()) {
    absl::flat_hash_map<Node*, int64_t> proc_cycle_map;
    if (proc == elab.top()->proc()) {
      for (Node* node : proc->nodes()) {
        proc_cycle_map[node] = cycle_map.at(node);
      }
    } else {
      for (Node* node : proc->nodes()) {
        proc_cycle_map[node] = cycle_map.at(node) - first_proc_cycle.at(proc);
      }
    }
    schedules.insert({proc, PipelineSchedule(proc, proc_cycle_map)});
  }

  return schedules;
}

}  // namespace xls
