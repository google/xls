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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

// Returns the minimum clock period in picoseconds for which it is feasible to
// schedule the function into a pipeline with the given number of stages.
absl::StatusOr<int64_t> FindMinimumClockPeriod(
    FunctionBase* f, int64_t pipeline_stages,
    const DelayEstimator& delay_estimator, SDCScheduler& scheduler) {
  XLS_VLOG(4) << "FindMinimumClockPeriod()";
  XLS_VLOG(4) << "  pipeline stages = " << pipeline_stages;
  auto topo_sort_it = TopoSort(f);
  std::vector<Node*> topo_sort(topo_sort_it.begin(), topo_sort_it.end());
  XLS_ASSIGN_OR_RETURN(int64_t function_cp_ps,
                       ComputeCriticalPath(topo_sort, delay_estimator));
  // The lower bound of the search is the critical path delay evenly distributed
  // across all stages (rounded up), and the upper bound is simply the critical
  // path of the entire function. It's possible this upper bound is the best you
  // can do if there exists a single operation with delay equal to the
  // critical-path delay of the function.
  int64_t optimistic_clk_period_ps =
      (function_cp_ps + pipeline_stages - 1) / pipeline_stages;
  int64_t pessimistic_clk_period_ps = function_cp_ps;
  XLS_VLOG(4) << absl::StreamFormat("Binary searching over interval [%d, %d]",
                                    optimistic_clk_period_ps,
                                    pessimistic_clk_period_ps);

  // Check that it is in fact possible to schedule this function at all; if not,
  // return a useful error.
  XLS_RETURN_IF_ERROR(scheduler
                          .Schedule(pipeline_stages, pessimistic_clk_period_ps,
                                    /*check_feasibility=*/true,
                                    /*explain_infeasibility=*/true)
                          .status())
          .SetPrepend()
      << absl::StrFormat("Impossible to schedule %s %s as specified; ",
                         (f->IsProc() ? "proc" : "function"), f->name());

  int64_t min_clk_period_ps = BinarySearchMinTrue(
      optimistic_clk_period_ps, pessimistic_clk_period_ps,
      [&](int64_t clk_period_ps) {
        return scheduler
            .Schedule(pipeline_stages, clk_period_ps,
                      /*check_feasibility=*/true,
                      /*explain_infeasibility=*/false)
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
    XLS_CHECK(sdc_scheduler != nullptr);
    XLS_ASSIGN_OR_RETURN(
        clock_period_ps,
        FindMinimumClockPeriod(f, *options.pipeline_stages(), input_delay_added,
                               *sdc_scheduler));

    if (options.period_relaxation_percent().has_value()) {
      int64_t relaxation_percent = options.period_relaxation_percent().value();

      clock_period_ps += (clock_period_ps * relaxation_percent + 50) / 100;
    }
  }

  ScheduleCycleMap cycle_map;
  if (options.strategy() == SchedulingStrategy::SDC) {
    // Enable iterative SDC scheduling when iteration number is larger than 1.
    if (options.fdo_iteration_number() > 1) {
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
          cycle_map, ScheduleByIterativeSDC(
                         f, options.pipeline_stages(), clock_period_ps,
                         delay_manager, options.constraints(), isdc_options));

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

    XLS_ASSIGN_OR_RETURN(
        cycle_map,
        sdc_scheduler->Schedule(options.pipeline_stages(), clock_period_ps));
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

      for (Node* node : TopoSort(f)) {
        int64_t lower_bound = bounds.lb(node);
        int64_t upper_bound = bounds.ub(node);
        std::uniform_int_distribution<int64_t> distrib(lower_bound,
                                                       upper_bound);
        int64_t cycle = distrib(gen);
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
