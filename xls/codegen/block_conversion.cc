// Copyright 2021 The XLS Authors
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

#include "xls/codegen/block_conversion.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/codegen/vast.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace verilog {

absl::StatusOr<Block*> FunctionToBlock(Function* f,
                                       absl::string_view block_name) {
  Block* block = f->package()->AddBlock(
      absl::make_unique<Block>(block_name, f->package()));

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  // Emit the parameters first to ensure the their order is preserved in the
  // block.
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        node_map[param],
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));
  }

  for (Node* node : TopoSort(f)) {
    if (node->Is<Param>()) {
      continue;
    }

    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(Node * block_node,
                         node->CloneInNewFunction(new_operands, block));
    node_map[node] = block_node;
  }

  // TODO(meheff): 2021-03-01 Allow port names other than "out".
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort("out", node_map.at(f->return_value())).status());

  return block;
}

// Returns pipeline-stage prefixed signal name for the given node. For
// example: p3_foo.
static std::string PipelineSignalName(Node* node, int64_t stage) {
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(node->GetName()));
}

absl::StatusOr<Block*> FunctionToPipelinedBlock(
    const PipelineSchedule& schedule, Function* f,
    absl::string_view block_name) {
  Block* block = f->package()->AddBlock(
      absl::make_unique<Block>(block_name, f->package()));

  XLS_RETURN_IF_ERROR(block->AddClockPort("clk"));

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  // Emit the parameters first to ensure the their order is preserved in the
  // block.
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        node_map[param],
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));
  }

  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    for (Node* function_node : schedule.nodes_in_cycle(stage)) {
      if (function_node->Is<Param>()) {
        continue;
      }
      std::vector<Node*> new_operands;
      for (Node* operand : function_node->operands()) {
        new_operands.push_back(node_map.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(
          Node * node, function_node->CloneInNewFunction(new_operands, block));
      node_map[function_node] = node;
    }

    // Add pipeline registers. A register is needed for each node which is
    // scheduled at or before this cycle and has a use after this cycle.
    for (Node* function_node : f->nodes()) {
      if (schedule.cycle(function_node) > stage) {
        continue;
      }
      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == schedule.length() - 1) {
          return false;
        }
        if (n == f->return_value()) {
          return true;
        }
        for (Node* user : n->users()) {
          if (schedule.cycle(user) > stage) {
            return true;
          }
        }
        return false;
      };

      Node* node = node_map.at(function_node);
      if (is_live_out_of_stage(function_node)) {
        XLS_ASSIGN_OR_RETURN(Register * reg,
                             block->AddRegister(PipelineSignalName(node, stage),
                                                node->GetType()));
        XLS_RETURN_IF_ERROR(
            block
                ->MakeNode<RegisterWrite>(node->loc(), node,
                                          /*load_enable=*/absl::nullopt,
                                          /*reset=*/absl::nullopt, reg->name())
                .status());

        XLS_ASSIGN_OR_RETURN(
            node_map[function_node],
            block->MakeNode<RegisterRead>(node->loc(), reg->name()));
      }
    }
  }

  // TODO(https://github.com/google/xls/issues/448): 2021-03-01 Allow port names
  // other than "out".
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort("out", node_map.at(f->return_value())).status());

  return block;
}

}  // namespace verilog
}  // namespace xls
