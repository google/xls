// Copyright 2022 The XLS Authors
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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/analyze_critical_path.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/scheduling_options_flags.h"
#include "xls/tools/scheduling_options_flags.pb.h"

const char kUsage[] = R"(
Dump scheduling result to stdout in Graphviz's dot plain text format.
Explicitly show the pipeline stage.

Example invocation:
  sched_printer_main --clock_period_ps=500 \
       --pipeline_stages=7 \
       IR_FILE
)";

ABSL_FLAG(std::string, top, "", "Top entity to use in lieu of the default.");

namespace xls {
namespace {
// Each node in graphviz's digraph should have a unique id.
// This function will allocate id to each node. To make pipeline
// register explicit, dot nodes will be created to represent the
// pipeline register.
void AllocateDigraphNodeId(
    const PipelineSchedule& sched,
    std::vector<absl::flat_hash_map<Node*, int64_t>>* output_registers_id,
    absl::flat_hash_map<Node*, int64_t>* xls_nodes_id) {
  CHECK(output_registers_id->empty() && xls_nodes_id->empty());

  output_registers_id->resize(sched.length() - 1);

  int64_t next_id = 0;
  FunctionBase* f = sched.function_base();

  auto latest_use = [&](Node* node) {
    auto it = std::max_element(
        node->users().begin(), node->users().end(),
        [&](Node* n1, Node* n2) { return sched.cycle(n1) < sched.cycle(n2); });
    return sched.cycle(*it);
  };

  for (Node* node : TopoSort(f)) {
    (*xls_nodes_id)[node] = next_id++;
    // Supposing `node` is schedule in stage `x`, the latest usage is in stage
    // `y`, `node` should be kept in stage [x, y)'s output pipeline register.
    // So, (y - x) nodes will be created.
    if (!node->users().empty()) {
      for (int64_t stage = sched.cycle(node); stage < latest_use(node);
           ++stage) {
        output_registers_id->at(stage)[node] = next_id++;
      }
    }
  }
}

using AttributeDict = absl::flat_hash_map<std::string, std::string>;

void AddDigraphNode(std::string* out, int64_t id,
                    AttributeDict* attrs = nullptr) {
  std::string attrs_flatten;
  if (attrs != nullptr) {
    for (const auto& [k, v] : *attrs) {
      absl::StrAppend(&attrs_flatten, k, "=", v, " ");
    }
    absl::StrAppendFormat(out, "  nd_%lld [%s];\n", id, attrs_flatten);
    return;
  }
  absl::StrAppendFormat(out, "  nd_%lld;\n", id);
}

void AddDigraphEdge(std::string* out, int64_t src_id, int64_t dst_id,
                    AttributeDict* attrs = nullptr) {
  std::string attrs_flatten;
  if (attrs != nullptr) {
    for (const auto& [k, v] : *attrs) {
      absl::StrAppend(&attrs_flatten, k, "=", v, " ");
    }
    absl::StrAppendFormat(out, "  nd_%lld -> nd_%lld [%s];\n", src_id, dst_id,
                          attrs_flatten);
    return;
  }
  absl::StrAppendFormat(out, "  nd_%lld -> nd_%lld;\n", src_id, dst_id);
}

void AddGrouping(std::string* out, const PipelineSchedule& sched,
                 const absl::flat_hash_map<Node*, int64_t>& xls_nodes_id,
                 const std::vector<absl::flat_hash_map<Node*, int64_t>>&
                     output_registers_id) {
  int64_t next_id = 0;
  auto add_subgraph = [&next_id](std::string* out, std::string name,
                                 int64_t total_bits,
                                 const std::vector<int64_t>& IDs) {
    std::string cluster_content;
    // Add label.
    absl::StrAppendFormat(&cluster_content,
                          "    label=\"%s registers\\n%lld bits\"\n", name,
                          total_bits);
    // Align all the pipeline register nodes.
    absl::StrAppendFormat(
        &cluster_content, "    {rank=same %s}\n",
        absl::StrJoin(IDs, " ", [&](std::string* dst, int64_t id) {
          absl::StrAppendFormat(dst, "nd_%lld", id);
        }));

    absl::StrAppendFormat(out, "  subgraph cluster_%lld{\n%s  }\n", next_id++,
                          cluster_content);
  };

  std::vector<int64_t> ids;
  int64_t total_bits = 0;

  // Align params horizontally.
  {
    total_bits = 0;
    ids.clear();
    for (Param* param : sched.function_base()->params()) {
      ids.push_back(xls_nodes_id.at(param));
      total_bits += param->GetType()->GetFlatBitCount();
    }
  }
  add_subgraph(out, "Parameter", total_bits, ids);

  // Align return values horizontally.
  {
    total_bits = 0;
    ids.clear();
    for (Node* node : sched.function_base()->nodes()) {
      if (sched.function_base()->HasImplicitUse(node)) {
        ids.push_back(xls_nodes_id.at(node));
        total_bits += node->GetType()->GetFlatBitCount();
      }
    }
  }
  add_subgraph(out, "Return value", total_bits, ids);

  // Align pipeline registers nodes, and group them into a subgraph explicitly.
  for (int64_t stage = 0; stage < output_registers_id.size(); ++stage) {
    total_bits = 0;
    ids.clear();
    for (auto [node, id] : output_registers_id.at(stage)) {
      ids.push_back(id);
      total_bits += node->GetType()->GetFlatBitCount();
    }
    add_subgraph(out, absl::StrFormat("Stage #%lld output ", stage), total_bits,
                 ids);
  }
}

using DelayMap = absl::flat_hash_map<Node*, int64_t>;

absl::StatusOr<DelayMap> ComputeNodeDelays(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  DelayMap result;
  for (Node* node : f->nodes()) {
    XLS_ASSIGN_OR_RETURN(result[node],
                         delay_estimator.GetOperationDelayInPs(node));
  }
  return result;
}

std::string GenerateDigraphContents(
    const PipelineSchedule& sched, const DelayMap& delay_map,
    const std::vector<absl::flat_hash_map<Node*, int64_t>>& output_registers_id,
    const absl::flat_hash_map<Node*, int64_t>& xls_nodes_id,
    const absl::flat_hash_set<Node*>& nodes_on_cp) {
  absl::flat_hash_map<Node*, int64_t> topo_index;
  {
    std::vector<Node*> topo_sort = TopoSort(sched.function_base());
    for (int64_t i = 0; i < topo_sort.size(); ++i) {
      topo_index[topo_sort[i]] = i;
    }
  }

  std::string contents;

  // Add all nodes
  {
    for (Node* node : TopoSort(sched.function_base())) {
      AttributeDict xls_node_attrs;
      xls_node_attrs["shape"] = "record";
      xls_node_attrs["style"] = "rounded";
      xls_node_attrs["label"] = absl::StrFormat(
          "\"%s\\n%s(%s)\\n%s, %lld ps\"", node->GetName(),
          OpToString(node->op()),
          absl::StrJoin(node->operands(), ",",
                        [](std::string* dst, Node* operand) {
                          absl::StrAppend(dst, operand->GetName());
                        }),
          node->GetType()->ToString(), delay_map.at(node));
      xls_node_attrs["color"] = nodes_on_cp.contains(node) ? "red" : "black";

      AddDigraphNode(&contents, xls_nodes_id.at(node), &xls_node_attrs);
    }

    for (int64_t stage = 0; stage < output_registers_id.size(); ++stage) {
      for (auto [saved_node, reg_node_id] : output_registers_id.at(stage)) {
        AttributeDict reg_node_attrs;
        reg_node_attrs["shape"] = "folder";
        reg_node_attrs["label"] =
            absl::StrFormat("\"%s\"", saved_node->GetName());
        reg_node_attrs["color"] =
            nodes_on_cp.contains(saved_node) ? "red" : "black";

        AddDigraphNode(&contents, reg_node_id, &reg_node_attrs);
      }
    }
  }

  // Add all edges
  {
    for (Node* node : TopoSort(sched.function_base())) {
      for (Node* operand : node->operands()) {
        int64_t operand_id = -1;
        if (sched.cycle(operand) < sched.cycle(node)) {
          operand_id =
              output_registers_id.at(sched.cycle(node) - 1).at(operand);
        } else {
          operand_id = xls_nodes_id.at(operand);
        }

        AttributeDict edge_attrs;
        edge_attrs["color"] =
            nodes_on_cp.contains(operand) && nodes_on_cp.contains(node)
                ? "red"
                : "black";
        AddDigraphEdge(&contents, operand_id, xls_nodes_id.at(node),
                       &edge_attrs);
      }
    }

    for (int64_t stage = 0; stage < output_registers_id.size(); ++stage) {
      for (auto [saved_node, reg_node_id] : output_registers_id.at(stage)) {
        // incoming edge decl

        int64_t src_id = -1;
        if (sched.cycle(saved_node) == stage) {
          src_id = xls_nodes_id.at(saved_node);
        } else {
          src_id = output_registers_id.at(stage - 1).at(saved_node);
        }

        AttributeDict edge_attrs;
        edge_attrs["color"] =
            nodes_on_cp.contains(saved_node) ? "red" : "black";
        AddDigraphEdge(&contents, src_id, reg_node_id, &edge_attrs);
      }
    }
  }

  AddGrouping(&contents, sched, xls_nodes_id, output_registers_id);
  return contents;
}

std::string DumpScheduleResultToDot(
    const PipelineSchedule& sched, const DelayMap& delay_map,
    const absl::flat_hash_set<Node*>& nodes_on_cp) {
  std::vector<absl::flat_hash_map<Node*, int64_t>> output_registers_id;
  absl::flat_hash_map<Node*, int64_t> xls_nodes_id;
  AllocateDigraphNodeId(sched, &output_registers_id, &xls_nodes_id);

  std::string contents = GenerateDigraphContents(
      sched, delay_map, output_registers_id, xls_nodes_id, nodes_on_cp);

  return absl::StrFormat("digraph {\n%s}\n", contents);
}

absl::StatusOr<PipelineSchedule> RunSchedulingPipeline(
    FunctionBase* main, const SchedulingOptions& scheduling_options,
    const DelayEstimator* delay_estimator) {
  absl::StatusOr<PipelineSchedule> schedule_status =
      RunPipelineSchedule(main, *delay_estimator, scheduling_options);

  if (!schedule_status.ok()) {
    if (absl::IsResourceExhausted(schedule_status.status())) {
      // Resource exhausted error indicates that the schedule was
      // infeasible. Emit a meaningful error in this case.
      if (scheduling_options.pipeline_stages().has_value() &&
          scheduling_options.clock_period_ps().has_value()) {
        XLS_LOG(QFATAL) << absl::StreamFormat(
            "Design cannot be scheduled in %d stages with a %dps clock.",
            scheduling_options.pipeline_stages().value(),
            scheduling_options.clock_period_ps().value());
      }
    }
  }

  return schedule_status;
}

absl::Status RealMain(std::string_view ir_path) {
  if (ir_path == "-") {
    ir_path = "/dev/stdin";
  }

  XLS_ASSIGN_OR_RETURN(std::string ir_contents, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> p,
                       Parser::ParsePackage(ir_contents, ir_path));

  XLS_RETURN_IF_ERROR(VerifyPackage(p.get()));

  std::string top_str = absl::GetFlag(FLAGS_top);
  std::optional<std::string_view> maybe_top_str =
      top_str.empty() ? std::nullopt : std::make_optional(top_str);
  XLS_ASSIGN_OR_RETURN(FunctionBase * main, FindTop(p.get(), maybe_top_str));

  XLS_ASSIGN_OR_RETURN(
      SchedulingOptionsFlagsProto scheduling_options_flags_proto,
      GetSchedulingOptionsFlagsProto());
  XLS_ASSIGN_OR_RETURN(
      SchedulingOptions scheduling_options,
      SetUpSchedulingOptions(scheduling_options_flags_proto, p.get()));

  QCHECK(scheduling_options.pipeline_stages() != 0 ||
         scheduling_options.clock_period_ps() != 0)
      << "Must specify --pipeline_stages or --clock_period_ps (or both).";

  XLS_ASSIGN_OR_RETURN(const DelayEstimator* delay_estimator,
                       SetUpDelayEstimator(scheduling_options_flags_proto));
  XLS_ASSIGN_OR_RETURN(
      PipelineSchedule schedule,
      RunSchedulingPipeline(main, scheduling_options, delay_estimator));

  XLS_ASSIGN_OR_RETURN(
      std::vector<CriticalPathEntry> cp,
      AnalyzeCriticalPath(main, scheduling_options.clock_period_ps(),
                          *delay_estimator));

  absl::flat_hash_set<Node*> nodes_on_cp;
  for (const CriticalPathEntry& entry : cp) {
    nodes_on_cp.insert(entry.node);
  }

  XLS_ASSIGN_OR_RETURN(DelayMap delay_map,
                       ComputeNodeDelays(main, *delay_estimator));

  std::cout << DumpScheduleResultToDot(schedule, delay_map, nodes_on_cp);

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty() || positional_arguments[0].empty()) {
    XLS_LOG(QFATAL) << "Expected path argument with IR: " << argv[0]
                    << " <ir_path>";
  }
  CHECK_OK(xls::RealMain(positional_arguments[0]));
  return EXIT_SUCCESS;
}
