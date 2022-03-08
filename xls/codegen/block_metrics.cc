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

#include "xls/codegen/block_metrics.h"

#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/node_iterator.h"

namespace xls::verilog {
namespace {

int64_t GenerateFlopCount(Block* block) {
  int64_t count = 0;

  for (Register* reg : block->GetRegisters()) {
    Type* reg_type = reg->type();
    count += reg_type->GetFlatBitCount();
  }

  return count;
}

// Returns true if there is a combinational feedthrough path from an input port
// to an output port.
bool HasFeedthroughPass(Block* block) {
  // Nodes which have a combinational path from an input port.
  absl::flat_hash_set<Node*> input_path_nodes;
  for (Node* node : TopoSort(block)) {
    if (node->Is<InputPort>()) {
      input_path_nodes.insert(node);
      continue;
    }
    for (Node* operand : node->operands()) {
      if (operand->GetType()->GetFlatBitCount() > 0 &&
          input_path_nodes.contains(operand)) {
        input_path_nodes.insert(node);
        break;
      }
    }
    if (node->Is<OutputPort>() && input_path_nodes.contains(node)) {
      return true;
    }
  }
  return false;
}

// Sets the delay fields of `proto` based on analysis of `block`.
absl::Status SetDelayFields(Block* block, const DelayEstimator& delay_estimator,
                            BlockMetricsProto* proto) {
  // Maximum delay from input to each node.
  absl::flat_hash_map<Node*, int64_t> input_delay_map;
  // Maximum delay from a register read to each node.
  absl::flat_hash_map<Node*, int64_t> reg_delay_map;

  // Delay metrics to set on the proto.
  absl::optional<int64_t> max_reg_to_reg_delay;
  absl::optional<int64_t> max_input_to_reg_delay;
  absl::optional<int64_t> max_reg_to_output_delay;
  absl::optional<int64_t> max_feedthrough_path_delay;

  for (Node* node : TopoSort(block)) {
    if (node->Is<InputPort>()) {
      input_delay_map[node] = 0;
      continue;
    }

    auto optional_max = [](int64_t value, absl::optional<int64_t> opt_value) {
      if (opt_value.has_value()) {
        return std::max(value, opt_value.value());
      }
      return value;
    };
    absl::StatusOr<int64_t> node_delay_or =
        delay_estimator.GetOperationDelayInPs(node);
    int64_t node_delay = node_delay_or.ok() ? node_delay_or.value() : 0;

    absl::optional<int64_t> input_delay;
    absl::optional<int64_t> reg_delay;
    for (Node* operand : node->operands()) {
      if (operand->GetType()->GetFlatBitCount() > 0) {
        if (input_delay_map.contains(operand)) {
          input_delay = optional_max(input_delay_map.at(operand) + node_delay,
                                     input_delay);
        }
        if (reg_delay_map.contains(operand)) {
          reg_delay =
              optional_max(reg_delay_map.at(operand) + node_delay, reg_delay);
        }
      }
    }
    if (input_delay.has_value()) {
      input_delay_map[node] = input_delay.value();
    }
    if (reg_delay.has_value()) {
      reg_delay_map[node] = reg_delay.value();
    }

    if (node->Is<OutputPort>()) {
      Node* data = node->operand(0);
      if (input_delay_map.contains(data)) {
        max_feedthrough_path_delay =
            optional_max(input_delay_map.at(data), max_feedthrough_path_delay);
      }
      if (reg_delay_map.contains(data)) {
        max_reg_to_output_delay =
            optional_max(reg_delay_map.at(data), max_reg_to_output_delay);
      }
      continue;
    }
    if (node->Is<RegisterRead>()) {
      // The delay of a RegisterRead is the clk-to-q delay.
      reg_delay_map[node] = node_delay;
      continue;
    }
    if (node->Is<RegisterWrite>()) {
      // The relevant register write operands for the purposes of to-register
      // paths is the data operand and the (optional) load enable. Reset is not
      // considered.
      std::vector<Node*> operands = {node->As<RegisterWrite>()->data()};
      if (node->As<RegisterWrite>()->load_enable().has_value()) {
        operands.push_back(node->As<RegisterWrite>()->load_enable().value());
      }
      for (Node* operand : operands) {
        if (input_delay_map.contains(operand)) {
          max_input_to_reg_delay =
              optional_max(input_delay_map.at(operand), max_input_to_reg_delay);
        }
        if (reg_delay_map.contains(operand)) {
          max_reg_to_reg_delay =
              optional_max(reg_delay_map.at(operand), max_reg_to_reg_delay);
        }
      }
      continue;
    }
  }

  if (max_reg_to_reg_delay.has_value()) {
    proto->set_max_reg_to_reg_delay_ps(max_reg_to_reg_delay.value());
  }
  if (max_input_to_reg_delay.has_value()) {
    proto->set_max_input_to_reg_delay_ps(max_input_to_reg_delay.value());
  }
  if (max_reg_to_output_delay.has_value()) {
    proto->set_max_reg_to_output_delay_ps(max_reg_to_output_delay.value());
  }
  if (max_feedthrough_path_delay.has_value()) {
    proto->set_max_feedthrough_path_delay_ps(
        max_feedthrough_path_delay.value());
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<BlockMetricsProto> GenerateBlockMetrics(
    Block* block, std::optional<const DelayEstimator*> delay_estimator) {
  BlockMetricsProto proto;
  proto.set_flop_count(GenerateFlopCount(block));
  proto.set_feedthrough_path_exists(HasFeedthroughPass(block));

  if (delay_estimator.has_value()) {
    XLS_RETURN_IF_ERROR(
        SetDelayFields(block, *delay_estimator.value(), &proto));
  }
  return proto;
}

}  // namespace xls::verilog
