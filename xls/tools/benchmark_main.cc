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

#include <numeric>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node_iterator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/scheduling/pipeline_schedule.h"

// TODO(meheff): These codegen flags are duplicated from codegen_main. Might be
// easier to wrap all the options into a proto or something codegen_main and
// this could consume. It would make it less likely that the benchmark and
// actual generated Verilog would diverge.
ABSL_FLAG(
    int64_t, clock_period_ps, 0,
    "The number of picoseconds in a cycle to use when generating a pipeline "
    "(codegen). Cannot be specified with --pipeline_stages. If both this "
    "flag value and --pipeline_stages are zero then codegen is not"
    "performed.");
ABSL_FLAG(
    int64_t, pipeline_stages, 0,
    "The number of stages in the generated pipeline when performing codegen. "
    "Cannot be specified with --clock_period_ps. If both this flag value and "
    "--clock_period_ps are zero then codegen is not performed.");
ABSL_FLAG(int64_t, clock_margin_percent, 0,
          "The percentage of clock period to set aside as a margin to ensure "
          "timing is met. Effectively, this lowers the clock period by this "
          "percentage amount for the purposes of scheduling. Must be specified "
          "with --clock_period_ps");
ABSL_FLAG(bool, show_known_bits, false,
          "Show known bits as determined via the query engine.");
ABSL_FLAG(std::string, entry, "",
          "Entry function to use in lieu of the default.");
ABSL_FLAG(std::string, delay_model, "",
          "Delay model name to use from registry.");

namespace xls {
namespace {

std::string KnownBitString(Node* node, const QueryEngine& query_engine) {
  if (!node->GetType()->IsBits()) {
    return "?";
  }
  return query_engine.ToString(node);
}

void PrintNodeBreakdown(Function* f) {
  std::cout << absl::StreamFormat("Entry function (%s) node count: %d nodes\n",
                                  f->name(), f->node_count());
  std::vector<Op> ops;
  absl::flat_hash_map<Op, int64_t> op_count;
  for (Node* node : f->nodes()) {
    if (!op_count.contains(node->op())) {
      ops.push_back(node->op());
    }
    op_count[node->op()] += 1;
  }
  std::sort(ops.begin(), ops.end(),
            [&](Op a, Op b) { return op_count.at(a) > op_count.at(b); });
  std::cout << "Breakdown by op of all nodes in the graph:" << std::endl;
  for (Op op : ops) {
    std::cout << absl::StreamFormat("  %15s : %5d (%5.2f%%)\n", OpToString(op),
                                    op_count.at(op),
                                    100.0 * op_count.at(op) / f->node_count());
  }
}

// Run the standard pipeline on the given package and prints stats about the
// passes and execution time.
absl::Status RunOptimizationAndPrintStats(Package* package) {
  std::unique_ptr<CompoundPass> pipeline = CreateStandardPassPipeline();

  absl::Time start = absl::Now();
  PassResults pass_results;
  XLS_RETURN_IF_ERROR(
      pipeline->Run(package, PassOptions(), &pass_results).status());
  absl::Duration total_time = absl::Now() - start;
  auto to_ms = [](absl::Duration d) { return d / absl::Milliseconds(1); };
  std::cout << absl::StreamFormat("Optimization time: %dms\n",
                                  to_ms(total_time));
  std::cout << absl::StreamFormat("Dynamic pass count: %d\n",
                                  pass_results.invocations.size());

  // Aggregate run times by the pass name and print a table of the aggregate
  // execution time of each pass in decending order.
  absl::flat_hash_map<std::string, absl::Duration> pass_times;
  absl::flat_hash_map<std::string, int64_t> pass_counts;
  absl::flat_hash_map<std::string, int64_t> changed_counts;
  for (const PassInvocation& invocation : pass_results.invocations) {
    pass_times[invocation.pass_name] += invocation.run_duration;
    ++pass_counts[invocation.pass_name];
    changed_counts[invocation.pass_name] += invocation.ir_changed ? 1 : 0;
  }
  std::vector<std::string> pass_names;
  for (const auto& pair : pass_times) {
    pass_names.push_back(pair.first);
  }
  std::sort(pass_names.begin(), pass_names.end(),
            [&](const std::string& a, const std::string& b) {
              return pass_times.at(a) > pass_times.at(b);
            });
  std::cout << "Pass run durations (# of times pass changed IR / # of times "
               "pass was run):"
            << std::endl;
  for (const std::string& name : pass_names) {
    std::cout << absl::StreamFormat(
        "  %-20s : %-5dms (%3d / %3d)\n", name, to_ms(pass_times.at(name)),
        changed_counts.at(name), pass_counts.at(name));
  }
  return absl::OkStatus();
}

absl::Status PrintCriticalPath(
    Function* f, const QueryEngine& query_engine,
    const DelayEstimator& delay_estimator,
    absl::optional<int64_t> effective_clock_period_ps) {
  XLS_ASSIGN_OR_RETURN(
      std::vector<CriticalPathEntry> critical_path,
      AnalyzeCriticalPath(f, effective_clock_period_ps, delay_estimator));
  std::cout << absl::StrFormat("Return value delay: %dps\n",
                               critical_path.front().path_delay_ps);
  std::cout << absl::StrFormat("Critical path entry count: %d\n",
                               critical_path.size());

  absl::flat_hash_map<Op, std::pair<int64_t, int64_t>> op_to_sum;
  std::cout << "Critical path:" << std::endl;
  std::cout << CriticalPathToString(
      critical_path, [&query_engine](Node* n) -> std::string {
        if (absl::GetFlag(FLAGS_show_known_bits)) {
          return absl::StrFormat(
              "        %s <= [%s]\n", KnownBitString(n, query_engine),
              absl::StrJoin(
                  n->operands(), ", ", [&](std::string* out, Node* n) {
                    absl::StrAppend(out, KnownBitString(n, query_engine));
                  }));
        } else {
          return "";
        }
      });

  for (CriticalPathEntry& entry : critical_path) {
    // Make a note of the sums.
    auto& tally = op_to_sum[entry.node->op()];
    tally.first += entry.node_delay_ps;
    tally.second += 1;
  }

  int64_t total_delay = critical_path.front().path_delay_ps;
  std::cout << absl::StrFormat("Contribution by op (total %dps):\n",
                               total_delay);
  std::vector<Op> ops = AllOps();
  std::sort(ops.begin(), ops.end(), [&](Op lhs, Op rhs) {
    return op_to_sum[lhs].first > op_to_sum[rhs].first;
  });
  for (Op op : ops) {
    if (op_to_sum[op].second == 0) {
      continue;
    }
    std::cout << absl::StreamFormat(
        " %20s: %4d (%5.2f%%, %4d nodes, %5.1f avg)\n", OpToString(op),
        op_to_sum[op].first,
        static_cast<double>(op_to_sum[op].first) / total_delay * 100.0,
        op_to_sum[op].second,
        static_cast<double>(op_to_sum[op].first) / op_to_sum[op].second);
  }
  return absl::OkStatus();
}

// Returns the critical-path delay through each pipeline stage.
absl::StatusOr<std::vector<int64_t>> GetDelayPerStageInPs(
    Function* f, const PipelineSchedule& schedule,
    const DelayEstimator& delay_estimator) {
  std::vector<int64_t> delay_per_stage(schedule.length() + 1);
  // The delay from the beginning of the stage at which each node completes.
  absl::flat_hash_map<Node*, int64_t> completion_time;
  for (Node* node : TopoSort(f)) {
    int64_t start_time = 0;
    for (Node* operand : node->operands()) {
      if (schedule.cycle(operand) == schedule.cycle(node)) {
        start_time = std::max(start_time, completion_time[operand]);
      }
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    completion_time[node] = start_time + node_delay;
    delay_per_stage[schedule.cycle(node)] =
        std::max(delay_per_stage[schedule.cycle(node)], completion_time[node]);
  }
  return delay_per_stage;
}

absl::StatusOr<PipelineSchedule> ScheduleAndPrintStats(
    Package* package, const DelayEstimator& delay_estimator,
    absl::optional<int64_t> clock_period_ps,
    absl::optional<int64_t> pipeline_stages,
    absl::optional<int64_t> clock_margin_percent) {
  SchedulingPassOptions options;
  options.delay_estimator = &delay_estimator;
  if (clock_period_ps.has_value()) {
    options.scheduling_options.clock_period_ps(*clock_period_ps);
  }
  if (pipeline_stages.has_value()) {
    options.scheduling_options.pipeline_stages(*pipeline_stages);
  }
  if (clock_margin_percent.has_value()) {
    options.scheduling_options.clock_margin_percent(*clock_margin_percent);
  }

  std::unique_ptr<SchedulingCompoundPass> scheduling_pipeline =
      CreateStandardSchedulingPassPipeline();
  SchedulingPassResults results;
  SchedulingUnit scheduling_unit = {package, /*schedule=*/absl::nullopt};

  absl::Time start = absl::Now();
  XLS_RETURN_IF_ERROR(
      scheduling_pipeline->Run(&scheduling_unit, options, &results).status());
  absl::Duration total_time = absl::Now() - start;
  std::cout << absl::StreamFormat("Scheduling time: %dms\n",
                                  total_time / absl::Milliseconds(1));

  return std::move(*scheduling_unit.schedule);
}

absl::Status PrintCodegenInfo(Function* f, const PipelineSchedule& schedule,
                              const BddQueryEngine& bdd_query_engine,
                              const DelayEstimator& delay_estimator,
                              absl::optional<int64_t> clock_period_ps) {
  absl::Time start = absl::Now();
  XLS_ASSIGN_OR_RETURN(
      verilog::ModuleGeneratorResult codegen_result,
      verilog::ToPipelineModuleText(schedule, f, verilog::PipelineOptions()));
  absl::Duration total_time = absl::Now() - start;
  std::cout << absl::StreamFormat("Codegen time: %dms\n",
                                  total_time / absl::Milliseconds(1));

  int64_t total_flops = 0;
  int64_t total_duplicates = 0;
  int64_t total_constants = 0;
  std::vector<int64_t> flops_per_stage(schedule.length() + 1);
  std::vector<int64_t> duplicates_per_stage(schedule.length() + 1);
  std::vector<int64_t> constants_per_stage(schedule.length() + 1);
  for (int64_t i = 0; i <= schedule.length(); ++i) {
    absl::flat_hash_map<BddNodeIndex, std::pair<Node*, int64_t>> bdd_nodes;
    for (Node* node : schedule.GetLiveOutOfCycle(i)) {
      flops_per_stage[i] += node->GetType()->GetFlatBitCount();
      if (node->GetType()->IsBits()) {
        for (int64_t bit_index = 0; bit_index < node->BitCountOrDie();
             ++bit_index) {
          BddNodeIndex bdd_node =
              bdd_query_engine.bdd_function().GetBddNode(node, bit_index);
          if (bdd_node == bdd_query_engine.bdd_function().bdd().zero() ||
              bdd_node == bdd_query_engine.bdd_function().bdd().one()) {
            XLS_VLOG(1) << absl::StreamFormat(
                "%s:%d in stage %d is known %s", node->GetName(), bit_index, i,
                bdd_node == bdd_query_engine.bdd_function().bdd().zero()
                    ? "zero"
                    : "one");
            constants_per_stage[i]++;
            total_constants++;
          } else if (bdd_nodes.contains(bdd_node)) {
            XLS_VLOG(1) << absl::StreamFormat(
                "%s:%d in stage %d duplicate of %s:%d", node->GetName(),
                bit_index, i, bdd_nodes.at(bdd_node).first->GetName(),
                bdd_nodes.at(bdd_node).second);
            duplicates_per_stage[i]++;
            total_duplicates++;
          } else {
            bdd_nodes[bdd_node] = {node, bit_index};
          }
        }
      }
    }
    total_flops += flops_per_stage[i];
  }
  XLS_ASSIGN_OR_RETURN(std::vector<int64_t> delay_per_stage,
                       GetDelayPerStageInPs(f, schedule, delay_estimator));
  std::cout << "Pipeline:\n";
  for (int64_t i = 0; i <= schedule.length(); ++i) {
    std::string stage_str = absl::StrFormat(
        "  [Stage %2d] flops: %4d (%4d dups, %4d constant)\n", i,
        flops_per_stage[i], duplicates_per_stage[i], constants_per_stage[i]);
    std::cout << stage_str;
    if (i != schedule.length()) {
      // Horizontally offset the information about the logic in the stage from
      // the information about stage registers to make it easier to scan the
      // information vertically without register and logic info intermixing.
      std::cout << std::string(stage_str.size(), ' ');
      std::cout << absl::StreamFormat("nodes: %4d, delay: %4dps\n",
                                      schedule.nodes_in_cycle(i + 1).size(),
                                      delay_per_stage[i + 1]);
    }
  }
  std::cout << absl::StreamFormat(
      "Total pipeline flops: %d (%d dups, %4d constant)\n", total_flops,
      total_duplicates, total_constants);

  if (clock_period_ps.has_value()) {
    int64_t min_slack = std::numeric_limits<int64_t>::max();
    for (int64_t stage_delay : delay_per_stage) {
      min_slack = std::min(min_slack, *clock_period_ps - stage_delay);
    }
    std::cout << absl::StreamFormat("Min stage slack: %d\n", min_slack);
  }

  // TODO(meheff): Add an estimate of total number of gates.
  std::cout << absl::StreamFormat(
      "Lines of Verilog: %d\n",
      std::vector<std::string>(
          absl::StrSplit(codegen_result.verilog_text, '\n'))
          .size());
  return absl::OkStatus();
}

absl::Status RealMain(absl::string_view path,
                      absl::optional<int64_t> clock_period_ps,
                      absl::optional<int64_t> pipeline_stages,
                      absl::optional<int64_t> clock_margin_percent) {
  XLS_VLOG(1) << "Reading contents at path: " << path;
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(path));
  std::unique_ptr<Package> package;
  if (absl::GetFlag(FLAGS_entry).empty()) {
    XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackage(contents));
  } else {
    XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackageWithEntry(
                                      contents, absl::GetFlag(FLAGS_entry)));
  }

  XLS_RETURN_IF_ERROR(RunOptimizationAndPrintStats(package.get()));
  XLS_ASSIGN_OR_RETURN(Function * f, package->EntryFunction());
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BddQueryEngine> query_engine,
                       BddQueryEngine::Run(f, BddFunction::kDefaultPathLimit));
  PrintNodeBreakdown(f);

  absl::optional<int64_t> effective_clock_period_ps;
  if (clock_period_ps.has_value()) {
    effective_clock_period_ps = *clock_period_ps;
    if (clock_margin_percent.has_value()) {
      effective_clock_period_ps =
          *effective_clock_period_ps -
          (*clock_period_ps * *clock_margin_percent + 50) / 100;
    }
  }
  const DelayEstimator* pdelay_estimator;
  if (absl::GetFlag(FLAGS_delay_model).empty()) {
    pdelay_estimator = &GetStandardDelayEstimator();
  } else {
    XLS_ASSIGN_OR_RETURN(pdelay_estimator,
                         GetDelayEstimator(absl::GetFlag(FLAGS_delay_model)));
  }
  const auto& delay_estimator = *pdelay_estimator;
  XLS_RETURN_IF_ERROR(PrintCriticalPath(f, *query_engine, delay_estimator,
                                        effective_clock_period_ps));

  if (clock_period_ps.has_value() || pipeline_stages.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        ScheduleAndPrintStats(package.get(), delay_estimator, clock_period_ps,
                              pipeline_stages, clock_margin_percent));
    XLS_RETURN_IF_ERROR(PrintCodegenInfo(f, schedule, *query_engine,
                                         delay_estimator, clock_period_ps));
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);

  if (positional_arguments.empty() || positional_arguments[0].empty()) {
    XLS_LOG(QFATAL) << "Expected path argument with IR: " << argv[0]
                    << " <ir_path>";
  }

  absl::optional<int64_t> clock_period_ps;
  if (absl::GetFlag(FLAGS_clock_period_ps) > 0) {
    clock_period_ps = absl::GetFlag(FLAGS_clock_period_ps);
  }
  absl::optional<int64_t> pipeline_stages;
  if (absl::GetFlag(FLAGS_pipeline_stages) > 0) {
    pipeline_stages = absl::GetFlag(FLAGS_pipeline_stages);
  }
  absl::optional<int64_t> clock_margin_percent;
  if (absl::GetFlag(FLAGS_clock_margin_percent) > 0) {
    clock_margin_percent = absl::GetFlag(FLAGS_clock_margin_percent);
  }
  XLS_QCHECK_OK(xls::RealMain(positional_arguments[0], clock_period_ps,
                              pipeline_stages, clock_margin_percent));
  return EXIT_SUCCESS;
}
