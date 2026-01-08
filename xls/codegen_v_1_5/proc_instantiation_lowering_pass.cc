// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/proc_instantiation_lowering_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_utils.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

// Replaces a signal in a proc instantiation with its driving node.
absl::Status ReplaceSignal(Node* signal, Node* value) {
  if (signal->Is<OutputPort>()) {
    return signal->ReplaceOperandNumber(OutputPort::kOperandOperand, value);
  }
  XLS_RET_CHECK(signal->Is<InstantiationInput>());
  return signal->ReplaceOperandNumber(InstantiationInput::kDataOperand, value);
}

// Lowers the proc instantiations in `proc` to block instantiations in `block`,
// using `proc_to_block` to determine the block corresponding to each
// instantiation target proc. Does not remove anything from `proc`.
absl::StatusOr<bool> LowerProcInstantiations(
    ScheduledBlock* block, Proc* proc,
    const absl::flat_hash_map<Proc*, ScheduledBlock*>& proc_to_block) {
  bool changed = false;
  for (const std::unique_ptr<ProcInstantiation>& proc_instantiation :
       proc->proc_instantiations()) {
    const auto instantiated_block_it =
        proc_to_block.find(proc_instantiation->proc());
    XLS_RET_CHECK(instantiated_block_it != proc_to_block.end());
    Block* instantiated_block = instantiated_block_it->second;
    absl::flat_hash_map<std::string, Node*> inputs;
    absl::flat_hash_map<std::string, Node*> outputs;

    XLS_RET_CHECK_EQ(proc_instantiation->channel_args().size(),
                     proc_instantiation->proc()->interface().size());
    for (int64_t i = 0; i < proc_instantiation->channel_args().size(); ++i) {
      ChannelInterface* caller_interface =
          proc_instantiation->channel_args()[i];
      ChannelInterface* callee_interface =
          proc_instantiation->proc()->interface()[i];
      XLS_ASSIGN_OR_RETURN(
          ChannelPortMetadata callee_port_metadata,
          instantiated_block->GetChannelPortMetadata(
              callee_interface->name(), callee_interface->direction()));

      Node* caller_ready = nullptr;
      Node* caller_data = nullptr;
      Node* caller_valid = nullptr;
      if (proc->IsOnProcInterface(caller_interface)) {
        XLS_ASSIGN_OR_RETURN(
            ChannelPortMetadata caller_port_metadata,
            block->GetChannelPortMetadata(caller_interface->name(),
                                          caller_interface->direction()));
        if (caller_port_metadata.ready_port.has_value()) {
          XLS_ASSIGN_OR_RETURN(
              caller_ready,
              block->GetPortNode(*caller_port_metadata.ready_port));
        }
        if (caller_port_metadata.data_port.has_value()) {
          XLS_ASSIGN_OR_RETURN(
              caller_data, block->GetPortNode(*caller_port_metadata.data_port));
        }
        if (caller_port_metadata.valid_port.has_value()) {
          XLS_ASSIGN_OR_RETURN(
              caller_valid,
              block->GetPortNode(*caller_port_metadata.valid_port));
        }
      } else {
        XLS_ASSIGN_OR_RETURN(
            std::optional<InstantiationConnection*> caller_ready_connection,
            block->GetReadyInstantiationConnectionForChannel(
                caller_interface->name(), caller_interface->direction()));
        if (caller_ready_connection.has_value()) {
          caller_ready = *caller_ready_connection;
        }
        XLS_ASSIGN_OR_RETURN(
            std::optional<InstantiationConnection*> caller_data_connection,
            block->GetDataInstantiationConnectionForChannel(
                caller_interface->name(), caller_interface->direction()));
        if (caller_data_connection.has_value()) {
          caller_data = *caller_data_connection;
        }
        XLS_ASSIGN_OR_RETURN(
            std::optional<InstantiationConnection*> caller_valid_connection,
            block->GetValidInstantiationConnectionForChannel(
                caller_interface->name(), caller_interface->direction()));
        if (caller_valid_connection.has_value()) {
          caller_valid = *caller_valid_connection;
        }
      }

      XLS_RET_CHECK(callee_port_metadata.data_port.has_value());
      XLS_RET_CHECK(caller_data != nullptr);
      if (callee_interface->direction() == ChannelDirection::kReceive) {
        inputs[*callee_port_metadata.data_port] = caller_data;
        if (callee_port_metadata.valid_port.has_value()) {
          XLS_RET_CHECK(caller_valid != nullptr);
          inputs[*callee_port_metadata.valid_port] = caller_valid;
        }
        if (callee_port_metadata.ready_port.has_value()) {
          XLS_RET_CHECK(caller_ready != nullptr);
          outputs[*callee_port_metadata.ready_port] = caller_ready;
        }
      }

      if (callee_interface->direction() == ChannelDirection::kSend) {
        outputs[*callee_port_metadata.data_port] = caller_data;
        if (callee_port_metadata.ready_port.has_value()) {
          XLS_RET_CHECK(caller_ready != nullptr);
          inputs[*callee_port_metadata.ready_port] = caller_ready;
        }
        if (callee_port_metadata.valid_port.has_value()) {
          XLS_RET_CHECK(caller_valid != nullptr);
          outputs[*callee_port_metadata.valid_port] = caller_valid;
        }
      }
    }

    XLS_ASSIGN_OR_RETURN(
        Block::BlockInstantiationAndConnections block_instantiation,
        block->AddAndConnectBlockInstantiation(proc_instantiation->name(),
                                               instantiated_block, inputs));

    for (const auto& [output_name, output_signal] : outputs) {
      XLS_RETURN_IF_ERROR(ReplaceSignal(
          output_signal, block_instantiation.outputs.at(output_name)));
    }

    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ProcInstantiationLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  std::vector<std::pair<ScheduledBlock*, Proc*>> blocks =
      GetScheduledBlocksWithProcSources(package, /*new_style_only=*/true);
  absl::flat_hash_map<Proc*, ScheduledBlock*> proc_to_block;
  for (const auto& [block, proc] : blocks) {
    XLS_RET_CHECK(proc_to_block.emplace(proc, block).second);
  }

  for (const auto& [block, proc] : blocks) {
    XLS_ASSIGN_OR_RETURN(bool proc_changed,
                         LowerProcInstantiations(block, proc, proc_to_block));
    changed |= proc_changed;
  }

  // Once we have wired up instantiations, there should be no more need for
  // channels or proc instantiations in later passes.
  for (const auto& [_, proc] : blocks) {
    if (!proc->channels().empty() || !proc->interface().empty()) {
      XLS_RETURN_IF_ERROR(proc->RemoveAllProcInstantiations().status());
      XLS_RETURN_IF_ERROR(proc->RemoveAllChannelsAndInterfaces());
      changed = true;
    }
  }

  return changed;
}

}  // namespace xls::codegen
