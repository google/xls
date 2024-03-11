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

#include "xls/fdo/extract_nodes.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {

absl::StatusOr<std::optional<std::string>> ExtractNodesAndGetVerilog(
    const absl::flat_hash_set<Node*>& nodes,
    std::string_view top_module_name,
    bool flop_inputs_outputs, bool return_all_liveouts) {

  XLS_RET_CHECK(!nodes.empty());
  FunctionBase* f = (*nodes.begin())->function_base();
  XLS_RET_CHECK(std::all_of(nodes.begin(), nodes.end(), [&](Node* node) {
    return node->function_base() == f;
  }));

  std::vector<Node*> topo_sorted_nodes;
  absl::flat_hash_set<Node*> filtered_nodes;
  topo_sorted_nodes.reserve(nodes.size());

  for (Node* node : TopoSort(f)) {
    if (nodes.contains(node)) {
      Type* ntype = node->GetType();
      if (ntype->GetFlatBitCount() > 0) {
        // Unexpected case (it seems tuples w/ tokens have been dissolved).
        // It probably *would* work correctly, but if this is encountered,
        // verify that handling is correct (that data paths are timed correctly)
        XLS_RET_CHECK(!TypeHasToken(ntype)) <<
            "Unexpected node with a type that contains a token" <<
            node->ToString();

        topo_sorted_nodes.emplace_back(node);
        filtered_nodes.insert(node);
      } else {
        // Currently support FDO only post-opt;
        // only tokens should have zero bitwidth
        XLS_RET_CHECK(ntype->IsToken()) <<
            "Not expecting non-token type with zero bits" <<
            node->ToString();
        // skip (don't include token-producing nodes)
      }
    }
  }

  // If the list is empty now, that's because the fragment had
  //   nothing except token ops; just return no value
  if (topo_sorted_nodes.empty()) {
    std::optional<std::string> no_value;
    return no_value;
  }

  // Here, we create a temporary package for holding the temporary function. The
  // rationale is, in many cases, we want to run this method concurrently for
  // multiple different set of nodes. As a result, working on the same package
  // will cause data racing that will crash the compiler very soon.
  auto tmp_package = std::make_unique<xls::Package>(top_module_name);
  auto tmp_f = std::make_unique<Function>(top_module_name, tmp_package.get());

  absl::flat_hash_map<Node*, Node*> node_map;
  std::vector<Node*> live_out;
  for (Node* node : topo_sorted_nodes) {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      if (node_map.contains(operand)) {
        new_operands.push_back(node_map.at(operand));
      } else {
        // As the temporary package doesn't own any types from the original
        // package, we need this one more step to map the original type to a new
        // one owned by our temporary package.
        XLS_ASSIGN_OR_RETURN(
            Type * operand_type,
            tmp_package->MapTypeFromOtherPackage(operand->GetType()));
        Node* new_param = tmp_f->AddNode(std::make_unique<Param>(
            operand->loc(), operand->GetName(), operand_type, tmp_f.get()));
        node_map[operand] = new_param;
        new_operands.push_back(new_param);
      }
    }
    // Hack to support viewing procs as functions.
    Node* new_node;
    if (node->Is<Send>() || node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(
          Type * node_type,
          tmp_package->MapTypeFromOtherPackage(node->GetType()));
      new_node = tmp_f->AddNode(std::make_unique<xls::Param>(
          node->loc(), node->GetName(), node_type, tmp_f.get()));
    } else {
      XLS_ASSIGN_OR_RETURN(new_node,
                           node->CloneInNewFunction(new_operands, tmp_f.get()));
    }
    // Collect the live-out of the set of nodes.
    node_map[node] = new_node;
    if (tmp_f->HasImplicitUse(node)) {
      live_out.push_back(new_node);
    } else if (return_all_liveouts) {
      if (std::any_of(node->users().begin(), node->users().end(), [&](Node* u) {
            return !filtered_nodes.contains(u) || u->Is<Send>();
          })) {
        live_out.push_back(new_node);
      }
    } else {
      // Return live-outs that only have external users if return_all_liveouts
      // is not set.
      if (std::all_of(node->users().begin(), node->users().end(), [&](Node* u) {
            return !filtered_nodes.contains(u) || u->Is<Send>();
          })) {
        live_out.push_back(new_node);
      }
    }
  }

  // If the set of nodes don't include the function output, create a final tuple
  // which gathers all live-outs. The tuple will be the return value of the new
  // function. Otherwise, just use the mapped function output.
  auto src_function = static_cast<Function*>(f);
  if (node_map.contains(src_function->return_value())) {
    XLS_RETURN_IF_ERROR(
        tmp_f->set_return_value(node_map[src_function->return_value()]));
  } else {
    if (live_out.size() == 1) {
      XLS_RETURN_IF_ERROR(tmp_f->set_return_value(live_out.front()));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * return_tuple,
                           tmp_f->MakeNode<Tuple>(SourceInfo(), live_out));
      XLS_RETURN_IF_ERROR(tmp_f->set_return_value(return_tuple));
    }
  }

  // With the temporary function, we convert it to a combinational block. If
  // flop_inputs_outputs is set, we insert registers to the inputs and outputs.
  Block* tmp_block;
  if (!flop_inputs_outputs) {
    verilog::CodegenOptions options;
    options.entry(top_module_name);
    XLS_ASSIGN_OR_RETURN(
        verilog::CodegenPassUnit unit,
        verilog::FunctionToCombinationalBlock(tmp_f.get(), options));
    XLS_RET_CHECK_NE(unit.top_block, nullptr);
    tmp_block = unit.top_block;
  } else {
    ScheduleCycleMap cycle_map;
    for (Node* node : tmp_f->nodes()) {
      cycle_map.emplace(node, 0);
    }

    // Generate block with flopped inputs and outputs. We always use verilog
    // instead of system verilog. We always split the tuple outputs into
    // individuals.
    verilog::CodegenOptions options;
    options.entry(top_module_name)
        .clock_name("clk")
        .use_system_verilog(false)
        .flop_inputs(true)
        .flop_outputs(true);

    PipelineSchedule schedule(tmp_f.get(), cycle_map, 1);
    XLS_ASSIGN_OR_RETURN(
        verilog::CodegenPassUnit unit,
        verilog::FunctionToPipelinedBlock(schedule, options, tmp_f.get()));
    XLS_RET_CHECK_NE(unit.top_block, nullptr);
    tmp_block = unit.top_block;
  }

  verilog::CodegenOptions options;
  XLS_ASSIGN_OR_RETURN(
      std::string verilog_text,
      GenerateVerilog(tmp_block, options.use_system_verilog(false)));
  return verilog_text;
}

}  // namespace xls
