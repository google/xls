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

#include "xls/fdo/iterative_sdc_scheduler.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fdo/delay_manager.h"
#include "xls/fdo/node_cut.h"
#include "xls/fdo/synthesizer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/scheduling/schedule_util.h"
#include "xls/scheduling/scheduling_options.h"
#include "ortools/math_opt/cpp/math_opt.h"

namespace xls {

namespace {

using DelayMap = absl::flat_hash_map<Node *, int64_t>;
using NodeSet = absl::flat_hash_set<Node *>;

namespace math_opt = ::operations_research::math_opt;

// Gets the cut of the given path. For instance, given the following graph:
// A   B
//  \ /
//   C   D
//    \ /
//     E
// Cut {A, B, D} rooted at E is the cut of path C-E. Note that here we demands
// the given path not beginning with Param operation.
NodeCut GetPathCut(const std::vector<Node *> &full_path) {
  absl::flat_hash_set<Node *> leaves;
  for (int64_t i = 0; i < full_path.size(); ++i) {
    Node *node = full_path[i];
    Node *critical_operand = i == 0 ? nullptr : full_path[i - 1];
    for (Node *operand : node->operands()) {
      if (operand != critical_operand) {
        leaves.insert(operand);
      }
    }
  }
  return NodeCut(full_path.back(), leaves);
}

// Gets the critical path (path with the longest delay) given the source and
// target nodes in PathInfo.
absl::Status GetFullPaths(const std::vector<PathInfo> &paths,
                          const DelayManager &delay_manager,
                          std::vector<NodeSet> &full_paths,
                          absl::flat_hash_set<NodeCut> &evaluated_cuts) {
  for (auto [delay, source, target] : paths) {
    XLS_ASSIGN_OR_RETURN(std::vector<Node *> full_path,
                         delay_manager.GetFullCriticalPath(source, target));
    full_paths.emplace_back(NodeSet(full_path.begin(), full_path.end()));
    evaluated_cuts.emplace(GetPathCut(full_path));
  }
  return absl::OkStatus();
}

// Expands all the given paths to maximum cones and return. The mapping from
// root node to maximum cuts must have already been stored in "max_cut_map".
absl::Status GetMaxCones(const std::vector<PathInfo> &paths,
                         const NodeCutMap &max_cut_map,
                         std::vector<NodeSet> &max_cones,
                         absl::flat_hash_set<NodeCut> &evaluated_cuts) {
  for (auto [delay, source, target] : paths) {
    XLS_RET_CHECK(max_cut_map.contains(target));
    const NodeCut &cut = max_cut_map.at(target);
    const NodeSet &cone = cut.GetNodeCone();
    XLS_RET_CHECK(cone.contains(source));
    max_cones.emplace_back(cone);
    evaluated_cuts.emplace(cut);
  }
  return absl::OkStatus();
}

// Expands all the given paths to windows and return. A window can have multiple
// outputs, while a cone must only have one output.
//
// Implementation note: We first expand each path to their maximum cones. Then,
// we traverse each cone and if the leaves of the current cone is a subset of an
// exist cone/window (or vice versa), they are merged into a new window.
absl::Status GetMergedWindows(const std::vector<PathInfo> &paths,
                              const NodeCutMap &max_cut_map,
                              std::vector<NodeSet> &merged_windows,
                              absl::flat_hash_set<NodeCut> &evaluated_cuts) {
  std::vector<NodeSet> leaves_list;
  for (auto [delay, source, target] : paths) {
    XLS_RET_CHECK(max_cut_map.contains(target));
    const NodeCut &cut = max_cut_map.at(target);
    const NodeSet &cone = cut.GetNodeCone();
    XLS_RET_CHECK(cone.contains(source));
    evaluated_cuts.emplace(cut);

    bool merge_flag = false;
    for (int64_t i = 0; i < merged_windows.size(); ++i) {
      NodeSet &exist_leaves = leaves_list[i];
      NodeSet &exist_window = merged_windows[i];

      // Merge the cone into the window if one of them is a subset of
      // another.
      if (std::all_of(
              exist_leaves.begin(), exist_leaves.end(),
              [&](Node *node) { return cut.leaves().contains(node); })) {
        exist_leaves = cut.leaves();
        merge_flag = true;
      } else if (std::all_of(
                     cut.leaves().begin(), cut.leaves().end(),
                     [&](Node *node) { return exist_leaves.contains(node); })) {
        merge_flag = true;
      }
      // TODO(hanchenye): 2023-08-14 The third case is they have overlap by more
      // than a certain percentage.
      if (merge_flag) {
        exist_window.insert(cone.begin(), cone.end());
        break;
      }
    }

    // Otherwise, create a new window.
    if (!merge_flag) {
      merged_windows.emplace_back(cone);
      leaves_list.emplace_back(cut.leaves());
    }
  }
  return absl::OkStatus();
}

// Refines the delay estimations recorded in the given delay manager with
// low-level feedback from the given synthesizer.
//
// Implementation note: A number of subgraphs are extracted from the given
// function or proc and passed to the synthesizer for synthesis and static
// timing analysis (STA). Then, the results are fed back to the delay manager to
// refine its estimations. We use node cut as signature to distinguish different
// subgraphs. The evaluated subgraphs are recorded in "evaluated_cuts" to avoid
// duplicated evaluation.
absl::Status RefineDelayEstimations(
    FunctionBase *f, const ScheduleCycleMap &cycle_map,
    DelayManager &delay_manager, absl::flat_hash_set<NodeCut> &evaluated_cuts,
    int64_t min_pipeline_length, const IterativeSDCSchedulingOptions &options,
    absl::BitGenRef bit_gen) {
  XLS_ASSIGN_OR_RETURN(
      NodeCutMap cut_map,
      EnumerateMaxCutInSchedule(f, min_pipeline_length, cycle_map));

  auto except_evaluated_cuts = [&](Node *source, Node *target) {
    if (options.path_evaluate_strategy == PathEvaluateStrategy::PATH) {
      absl::StatusOr<std::vector<Node *>> path =
          delay_manager.GetFullCriticalPath(source, target);
      CHECK_OK(path);
      return evaluated_cuts.contains(GetPathCut(*path));
    }
    CHECK(cut_map.contains(target));
    return evaluated_cuts.contains(cut_map.at(target));
  };

  auto get_normalized_fanout = [](Node *source, Node *target) {
    return static_cast<float>(target->GetType()->GetFlatBitCount()) /
           std::max(static_cast<float>(0.00001),
                    static_cast<float>(target->users().size()));
  };

  // Collect longest paths and enumerate cuts from the pipeline schedule of
  // the current iteration.
  PathExtractOptions path_extract_options;
  path_extract_options.cycle_map = &cycle_map;
  path_extract_options.exclude_single_node_path = true;
  XLS_ASSIGN_OR_RETURN(
      std::vector<PathInfo> targeted_paths,
      delay_manager.GetTopNPathsStochastically(
          options.delay_driven_path_number, options.stochastic_ratio,
          path_extract_options, bit_gen,
          /*except=*/except_evaluated_cuts));

  // Extract fan-out driven paths.
  XLS_ASSIGN_OR_RETURN(
      const std::vector<PathInfo> &fanout_driven_paths,
      delay_manager.GetTopNPathsStochastically(
          options.fanout_driven_path_number, options.stochastic_ratio,
          path_extract_options, bit_gen, /*except=*/except_evaluated_cuts,
          /*score=*/get_normalized_fanout));

  for (auto path_info : fanout_driven_paths) {
    targeted_paths.push_back(path_info);
  }

  LOG(INFO) << "Number of paths to evaluate is " << targeted_paths.size();
  for (auto [delay, source, target] : targeted_paths) {
    LOG(INFO) << "(" << delay << "ps) Source " << source->GetName() << " ("
              << cycle_map.at(source) << ") to target " << target->GetName()
              << " (" << cycle_map.at(target) << ")";
  }

  // Extract nodes from paths with given strategy.
  std::vector<NodeSet> nodes_list;
  if (options.path_evaluate_strategy == PathEvaluateStrategy::PATH) {
    XLS_RET_CHECK_OK(GetFullPaths(targeted_paths, delay_manager, nodes_list,
                                  evaluated_cuts));
  } else if (options.path_evaluate_strategy == PathEvaluateStrategy::CONE) {
    XLS_RET_CHECK_OK(
        GetMaxCones(targeted_paths, cut_map, nodes_list, evaluated_cuts));
  } else if (options.path_evaluate_strategy == PathEvaluateStrategy::WINDOW) {
    XLS_RET_CHECK_OK(
        GetMergedWindows(targeted_paths, cut_map, nodes_list, evaluated_cuts));
  }

  XLS_ASSIGN_OR_RETURN(
      std::vector<int64_t> delay_list,
      options.synthesizer->SynthesizeNodesConcurrentlyAndGetDelays(nodes_list));

  VLOG(1) << "Number of modules generated is " << nodes_list.size();
  for (int64_t j = 0; j < delay_list.size(); ++j) {
    const NodeSet &nodes = nodes_list[j];
    int64_t delay = delay_list[j];
    LOG(INFO) << "(Updated delay: " << delay << "ps) Nodes: "
              << absl::StrJoin(nodes, ", ", [](std::string *out, Node *n) {
                   absl::StrAppend(out, n->GetName());
                   absl::StrAppend(out, "-->");
                   absl::StrAppend(out, n->GetUsersString());
                 });

    // Update delays in the delay manager with the synthesis results.
    for (Node *source : nodes) {
      for (Node *target : nodes) {
        XLS_RET_CHECK_OK(delay_manager.SetCriticalPathDelay(
            source, target, delay, /*if_shorter=*/true, /*if_exist=*/true));
      }
    }
  }
  delay_manager.PropagateDelays();
  return absl::OkStatus();
}

absl::Status BuildError(IterativeSDCSchedulingModel &model,
                        const math_opt::SolveResult &result,
                        SchedulingFailureBehavior failure_behavior) {
  CHECK_NE(result.termination.reason, math_opt::TerminationReason::kOptimal);

  if (failure_behavior.explain_infeasibility &&
      (result.termination.reason == math_opt::TerminationReason::kInfeasible ||
       result.termination.reason ==
           math_opt::TerminationReason::kInfeasibleOrUnbounded)) {
    XLS_RETURN_IF_ERROR(model.AddSlackVariables(
        failure_behavior.infeasible_per_state_backedge_slack_pool));
    XLS_ASSIGN_OR_RETURN(
        math_opt::SolveResult result_with_slack,
        math_opt::Solve(model.UnderlyingModel(), math_opt::SolverType::kGlop));
    if (result_with_slack.termination.reason ==
            math_opt::TerminationReason::kOptimal ||
        result_with_slack.termination.reason ==
            math_opt::TerminationReason::kFeasible) {
      XLS_RETURN_IF_ERROR(
          model.ExtractError(result_with_slack.variable_values()));
    }
  }

  // We don't know why the solver failed to find an optimal solution to our LP
  // problem; it could be an infeasibility issue (which needs more work to
  // analyze), a timeout, a precision error, or more. For now, just return a
  // simple error hinting at the problem.
  return absl::InternalError(
      absl::StrCat("The problem does not have an optimal solution; solver "
                   "terminated with ",
                   math_opt::EnumToString(result.termination.reason)));
}

}  // namespace

absl::Status IterativeSDCSchedulingModel::AddTimingConstraints(
    int64_t clock_period_ps) {
  absl::flat_hash_map<Node *, std::vector<Node *>> delay_constraints =
      delay_manager_.GetPathsOverDelayThreshold(clock_period_ps);

  int64_t number_constraints = 0;
  for (const auto &p : delay_constraints) {
    Node *source = p.first;
    for (Node *target : p.second) {
      number_constraints++;
      DiffAtLeastConstraint(target, source, 1, "timing");
      VLOG(2) << "Setting timing constraint: "
              << absl::StrFormat("1 ≤ %s - %s", target->GetName(),
                                 source->GetName());
    }
  }
  VLOG(2) << "Number of timing constraints added: " << number_constraints;
  return absl::OkStatus();
}

static absl::Status UpdateStats(const ScheduleCycleMap &prev_cycle_map,
                                const ScheduleCycleMap &cycle_map,
                                FunctionBase *f, int64_t iter) {
  // Suppress output when rerunning scheduling
  // See https://github.com/google/xls/issues/1107
  static int64_t previous_iter = -1;
  if (iter < previous_iter) {
    previous_iter = INT64_MAX;
    return absl::OkStatus();
  }
  previous_iter = iter;

  // Show changes (nodes that moved from one stage to another)
  for (auto [node, prev_cycle] : prev_cycle_map) {
    if (cycle_map.at(node) != prev_cycle) {
      LOG(INFO) << "*** Node " << node->GetName()
                << " MOVED STAGES: " << prev_cycle << "-->"
                << cycle_map.at(node) << "\n";
    }
  }

  // Display cycle_map histogram (nodes per cycle)
  std::map<int64_t, int64_t> histo;
  for (auto &[node, cycle] : cycle_map) {
    if (!node->Is<Param>() && !node->Is<Literal>()) {
      ++histo[cycle];
    }
  }
  for (auto &[cycle, node_count] : histo) {
    LOG(INFO) << "Stage " << cycle << ": " << node_count << " nodes";
  }

  // Count flops at pipeline stage crossings
  int64_t crossing_bits = 0;
  for (auto [node, cycle] : cycle_map) {
    auto users = node->users();
    int64_t next_cycle = cycle + 1;
    if (std::any_of(users.cbegin(), users.cend(),
                    [next_cycle, &cycle_map](Node *n) {
                      return cycle_map.at(n) >= next_cycle;
                    })) {
      crossing_bits += node->GetType()->GetFlatBitCount();
    }
  }
  LOG(INFO) << "FLOPS: " << crossing_bits << "\n";

  return absl::OkStatus();
}

absl::StatusOr<ScheduleCycleMap> ScheduleByIterativeSDC(
    FunctionBase *f, std::optional<int64_t> pipeline_stages,
    int64_t clock_period_ps, DelayManager &delay_manager,
    absl::Span<const SchedulingConstraint> constraints,
    const IterativeSDCSchedulingOptions &options,
    const SchedulingFailureBehavior failure_behavior) {
  VLOG(3) << "SDCScheduler()";
  VLOG(3) << "  pipeline stages = "
          << (pipeline_stages.has_value()
                  ? absl::StrCat(pipeline_stages.value())
                  : "(unspecified)");
  XLS_VLOG_LINES(4, f->DumpIr());

  if (options.iteration_number < 1 || options.delay_driven_path_number < 0 ||
      options.fanout_driven_path_number < 0 ||
      options.stochastic_ratio <= 0.0 || options.stochastic_ratio > 1.0) {
    return absl::InvalidArgumentError(
        "invalid iteration_number or delay_driven_path_number or "
        "fanout_driven_path_number or stochastic_ratio");
  }

  if (options.synthesizer == nullptr) {
    return absl::InvalidArgumentError("synthesizer is not ready");
  }

  ScheduleCycleMap cycle_map;
  absl::flat_hash_set<NodeCut> evaluated_cuts;
  std::mt19937_64 bit_gen;
  absl::flat_hash_set<Node *> dead_after_synthesis =
      GetDeadAfterSynthesisNodes(f);
  for (int64_t i = 0; i < options.iteration_number; ++i) {
    IterativeSDCSchedulingModel model(f, dead_after_synthesis, delay_manager);

    for (const SchedulingConstraint &constraint : constraints) {
      XLS_RETURN_IF_ERROR(model.AddSchedulingConstraint(constraint));
    }

    for (Node *node : f->nodes()) {
      for (Node *user : node->users()) {
        XLS_RETURN_IF_ERROR(model.AddDefUseConstraints(node, user));
      }
      if (f->IsFunction() && f->HasImplicitUse(node)) {
        XLS_RETURN_IF_ERROR(model.AddDefUseConstraints(node, std::nullopt));
      }
    }

    XLS_RETURN_IF_ERROR(model.AddTimingConstraints(clock_period_ps));

    int64_t min_pipeline_length = 1;
    model.SetPipelineLength(pipeline_stages);
    if (pipeline_stages.has_value()) {
      min_pipeline_length = *pipeline_stages;
    } else {
      // Find the minimum feasible pipeline length.
      model.MinimizePipelineLength();
      XLS_ASSIGN_OR_RETURN(
          const math_opt::SolveResult result_with_minimized_pipeline_length,
          math_opt::Solve(model.UnderlyingModel(),
                          math_opt::SolverType::kGlop));
      if (result_with_minimized_pipeline_length.termination.reason !=
          math_opt::TerminationReason::kOptimal) {
        return BuildError(model, result_with_minimized_pipeline_length,
                          failure_behavior);
      }
      XLS_ASSIGN_OR_RETURN(
          min_pipeline_length,
          model.ExtractPipelineLength(
              result_with_minimized_pipeline_length.variable_values()));
      model.SetPipelineLength(min_pipeline_length);
    }

    model.SetObjective();

    XLS_ASSIGN_OR_RETURN(
        math_opt::SolveResult result,
        math_opt::Solve(model.UnderlyingModel(), math_opt::SolverType::kGlop));

    if (result.termination.reason != math_opt::TerminationReason::kOptimal) {
      return BuildError(model, result, failure_behavior);
    }

    // Extract scheduling results to the cycle map.
    ScheduleCycleMap prev_cycle_map = cycle_map;
    XLS_ASSIGN_OR_RETURN(cycle_map,
                         model.ExtractResult(result.variable_values()));
    PathExtractOptions path_extract_options;
    path_extract_options.cycle_map = &cycle_map;
    XLS_RET_CHECK_OK(UpdateStats(prev_cycle_map, cycle_map, f, i));

    // Report the current estimated critical path delay.
    XLS_ASSIGN_OR_RETURN(PathInfo critical_path,
                         delay_manager.GetLongestPath(path_extract_options));
    auto [critical_delay, critical_source, critical_target] = critical_path;
    if (critical_delay > 0) {
      LOG(INFO) << "SDC iteration " << i << " critical path delay is "
                << critical_delay << "ps: " << critical_source->GetName()
                << " -> " << critical_target->GetName();
    }

    // Run delay estimation refinement except the last iteration.
    if (i != options.iteration_number - 1) {
      XLS_RET_CHECK_OK(
          RefineDelayEstimations(f, cycle_map, delay_manager, evaluated_cuts,
                                 min_pipeline_length, options, bit_gen));
    }
  }
  return cycle_map;
}

}  // namespace xls
