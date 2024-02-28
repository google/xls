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
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_search.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/fdo/delay_manager.h"
#include "xls/fdo/iterative_sdc_scheduler.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/op.h"
#include "xls/scheduling/min_cut_scheduler.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/schedule_bounds.h"
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

// Returns the critical path through the given nodes (ordered topologically).
absl::StatusOr<int64_t> ComputeCriticalPath(
    absl::Span<Node* const> topo_sort, const DelayEstimator& delay_estimator) {
  int64_t function_cp = 0;
  absl::flat_hash_map<Node*, int64_t> node_cp;
  for (Node* node : topo_sort) {
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
  return ComputeCriticalPath(TopoSort(f).AsVector(), delay_estimator);
}

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages. If
// `target_clock_period_ps` is specified, will not try to check lower clock
// periods than this.
absl::StatusOr<int64_t> FindMinimumClockPeriod(
    FunctionBase* f, std::optional<int64_t> pipeline_stages,
    const DelayEstimator& delay_estimator, SDCScheduler& scheduler,
    SchedulingFailureBehavior failure_behavior,
    std::optional<int64_t> target_clock_period_ps = std::nullopt) {
  XLS_VLOG(4) << "FindMinimumClockPeriod()";
  XLS_VLOG(4) << "  pipeline stages = "
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
  XLS_VLOG(4) << absl::StreamFormat("Binary searching over interval [%d, %d]",
                                    optimistic_clk_period_ps,
                                    pessimistic_clk_period_ps);

  // Check that it is in fact possible to
  // schedule this function at all; if not, return a useful error.
  XLS_RETURN_IF_ERROR(scheduler
                          .Schedule(pipeline_stages, pessimistic_clk_period_ps,
                                    failure_behavior,
                                    /*check_feasibility=*/true)
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
            .Schedule(pipeline_stages, clk_period_ps,
                      failure_behavior,
                      /*check_feasibility=*/true)
            .ok();
      },
      BinarySearchAssumptions::kEndKnownTrue);
  XLS_VLOG(4) << "minimum clock period = " << min_clk_period_ps;

  return min_clk_period_ps;
}

}  // namespace

absl::StatusOr<PipelineSchedule> RunPipelineSchedule(
    FunctionBase* f, const DelayEstimator& delay_estimator,
    const SchedulingOptions& options,
    const synthesis::Synthesizer* synthesizer) {
  int64_t input_delay = options.additional_input_delay_ps().has_value()
                            ? options.additional_input_delay_ps().value()
                            : 0;

  DecoratingDelayEstimator input_delay_added(
      "input_delay_added", delay_estimator,
      [input_delay](Node* node, int64_t base_delay) {
        return node->op() == Op::kReceive ? base_delay + input_delay
                                          : base_delay;
      });

  if (options.worst_case_throughput().has_value()) {
    f->SetInitiationInterval(*options.worst_case_throughput());
  }

  std::unique_ptr<SDCScheduler> sdc_scheduler;
  if (!options.clock_period_ps().has_value() ||
      options.strategy() == SchedulingStrategy::SDC) {
    // We currently use the SDC scheduler to determine the minimum clock period
    // (if not specified), even if we're not using it for the final schedule.
    XLS_ASSIGN_OR_RETURN(sdc_scheduler,
                         SDCScheduler::Create(f, input_delay_added));
    XLS_RETURN_IF_ERROR(sdc_scheduler->AddConstraints(options.constraints()));
  }

  int64_t clock_period_ps;
  if (options.clock_period_ps().has_value()) {
    clock_period_ps = *options.clock_period_ps();

    if (options.clock_margin_percent().has_value()) {
      int64_t original_clock_period_ps = clock_period_ps;
      clock_period_ps -=
          (clock_period_ps * options.clock_margin_percent().value() + 50) / 100;
      if (clock_period_ps <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Clock period non-positive (%dps) after adjusting for margin. "
            "Original clock period: %dps, clock margin: %d%%",
            clock_period_ps, original_clock_period_ps,
            *options.clock_margin_percent()));
      }
    }
  } else {
    XLS_RET_CHECK(options.pipeline_stages().has_value());
    // A pipeline length is specified, but no target clock period. Determine
    // the minimum clock period for which the function can be scheduled in the
    // given pipeline length.
    CHECK(sdc_scheduler != nullptr);
    XLS_ASSIGN_OR_RETURN(
        clock_period_ps,
        FindMinimumClockPeriod(f, options.pipeline_stages(), input_delay_added,
                               *sdc_scheduler, options.failure_behavior()));

    if (options.period_relaxation_percent().has_value()) {
      int64_t relaxation_percent = options.period_relaxation_percent().value();

      clock_period_ps += (clock_period_ps * relaxation_percent + 50) / 100;
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

    absl::StatusOr<ScheduleCycleMap> schedule_cycle_map =
        sdc_scheduler->Schedule(options.pipeline_stages(), clock_period_ps,
                                options.failure_behavior());
    if (!schedule_cycle_map.ok()) {
      if (absl::IsInvalidArgument(schedule_cycle_map.status())) {
        // The scheduler was able to explain the failure; report it up.
        return std::move(schedule_cycle_map).status();
      }
      if (options.clock_period_ps().has_value()) {
        // The scheduler was unable to explain the failure internally, and the
        // user specified a specific clock period.

        if (options.minimize_clock_on_failure().value_or(true)) {
          // Find the smallest clock period that would have worked.
          XLS_LOG(ERROR)
              << "Unable to schedule with the specified clock period; finding "
                 "the shortest feasible clock period...";
          int64_t target_clock_period_ps = clock_period_ps + 1;
          absl::StatusOr<int64_t> min_clock_period_ps = FindMinimumClockPeriod(
              f, options.pipeline_stages(), input_delay_added, *sdc_scheduler,
              options.failure_behavior(), target_clock_period_ps);
          if (min_clock_period_ps.ok()) {
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
                             ComputeCriticalPath(f, input_delay_added));
        // Make a copy of failure behavior with explain_feasibility true- we
        // always want to produce an error message because this we are
        // re-running the scheduler for its error message.
        SchedulingFailureBehavior pessimistic_failure_behavior =
            options.failure_behavior();
        pessimistic_failure_behavior.explain_infeasibility = true;
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
    sched::ScheduleBounds bounds(f, TopoSort(f).AsVector(), clock_period_ps,
                                 input_delay_added);
    XLS_RETURN_IF_ERROR(TightenBounds(bounds, f, options.pipeline_stages()));

    if (options.strategy() == SchedulingStrategy::MIN_CUT) {
      XLS_ASSIGN_OR_RETURN(cycle_map,
                           MinCutScheduler(f,
                                           options.pipeline_stages().value_or(
                                               bounds.max_lower_bound() + 1),
                                           clock_period_ps, input_delay_added,
                                           &bounds, options.constraints()));
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
      XLS_RET_CHECK(!options.pipeline_stages().has_value());
      // Just schedule everything as soon as possible.
      for (Node* node : f->nodes()) {
        if (node->Is<MinDelay>()) {
          return absl::InternalError(
              "The ASAP scheduler doesn't support min_delay nodes.");
        }
        cycle_map[node] = bounds.lb(node);
      }
    }
  }

  auto schedule = PipelineSchedule(f, cycle_map, options.pipeline_stages());
  XLS_RETURN_IF_ERROR(schedule.Verify());
  XLS_RETURN_IF_ERROR(
      schedule.VerifyTiming(clock_period_ps, input_delay_added));
  XLS_RETURN_IF_ERROR(schedule.VerifyConstraints(options.constraints(),
                                                 f->GetInitiationInterval()));

  XLS_VLOG_LINES(3, "Schedule\n" + schedule.ToString());
  return schedule;
}

}  // namespace xls
