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

#include "xls/codegen/codegen_pass.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"

namespace xls::verilog {

std::optional<Node*> StreamingInput::GetDataPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_
        ->GetDataPortForChannel(channel_name_, ChannelDirection::kReceive)
        .value();
  }
  return block_
      ->GetDataInstantiationConnectionForChannel(channel_name_,
                                                 ChannelDirection::kReceive)
      .value();
}

Node* StreamingInput::GetValidPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_
        ->GetValidPortForChannel(channel_name_, ChannelDirection::kReceive)
        .value()
        .value();
  }
  return block_
      ->GetValidInstantiationConnectionForChannel(channel_name_,
                                                  ChannelDirection::kReceive)
      .value()
      .value();
}

Node* StreamingInput::GetReadyPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_
        ->GetReadyPortForChannel(channel_name_, ChannelDirection::kReceive)
        .value()
        .value();
  }
  return block_
      ->GetReadyInstantiationConnectionForChannel(channel_name_,
                                                  ChannelDirection::kReceive)
      .value()
      .value();
}

ChannelRef StreamingInput::GetChannel() const {
  if (block_->package()->ChannelsAreProcScoped()) {
    const std::optional<BlockProvenance>& provenance = block_->GetProvenance();
    CHECK(provenance.has_value());
    CHECK(provenance->IsFromProc());
    Proc* proc = block_->package()->GetProc(provenance->name).value();
    return proc
        ->GetChannelInterface(GetChannelName(), ChannelDirection::kReceive)
        .value();
  }
  return block_->package()->GetChannel(GetChannelName()).value();
}

std::optional<Node*> StreamingOutput::GetDataPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_->GetDataPortForChannel(channel_name_, ChannelDirection::kSend)
        .value();
  }
  return block_
      ->GetDataInstantiationConnectionForChannel(channel_name_,
                                                 ChannelDirection::kSend)
      .value();
}

Node* StreamingOutput::GetValidPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_
        ->GetValidPortForChannel(channel_name_, ChannelDirection::kSend)
        .value()
        .value();
  }
  return block_
      ->GetValidInstantiationConnectionForChannel(channel_name_,
                                                  ChannelDirection::kSend)
      .value()
      .value();
}

Node* StreamingOutput::GetReadyPort() const {
  if (kind_ == ConnectionKind::kExternal) {
    return block_
        ->GetReadyPortForChannel(channel_name_, ChannelDirection::kSend)
        .value()
        .value();
  }
  return block_
      ->GetReadyInstantiationConnectionForChannel(channel_name_,
                                                  ChannelDirection::kSend)
      .value()
      .value();
}

ChannelRef StreamingOutput::GetChannel() const {
  if (block_->package()->ChannelsAreProcScoped()) {
    const std::optional<BlockProvenance>& provenance = block_->GetProvenance();
    CHECK(provenance.has_value());
    CHECK(provenance->IsFromProc());
    Proc* proc = block_->package()->GetProc(provenance->name).value();
    return proc->GetChannelInterface(GetChannelName(), ChannelDirection::kSend)
        .value();
  }
  return block_->package()->GetChannel(GetChannelName()).value();
}

void CodegenContext::GcMetadata() {
  absl::flat_hash_set<Node*> nodes;
  for (auto& [this_block, block_metadata] : metadata_) {
    nodes.clear();
    nodes.insert(this_block->nodes().begin(), this_block->nodes().end());
    absl::erase_if(
        block_metadata.streaming_io_and_pipeline.node_to_stage_map,
        [&nodes](const auto& kv) { return !nodes.contains(kv.first); });

    for (std::vector<StreamingInput>& inputs :
         block_metadata.streaming_io_and_pipeline.inputs) {
      for (StreamingInput& input : inputs) {
        if (input.GetSignalData().has_value() &&
            !nodes.contains(*input.GetSignalData())) {
          input.SetSignalData(std::nullopt);
        }
        if (input.GetSignalValid().has_value() &&
            !nodes.contains(*input.GetSignalValid())) {
          input.SetSignalValid(std ::nullopt);
        }
        if (input.GetPredicate().has_value() &&
            !nodes.contains(*input.GetPredicate())) {
          input.SetPredicate(std::nullopt);
        }
      }
    }
    for (std::vector<StreamingOutput>& outputs :
         block_metadata.streaming_io_and_pipeline.outputs) {
      for (StreamingOutput& output : outputs) {
        if (output.GetPredicate().has_value() &&
            !nodes.contains(*output.GetPredicate())) {
          output.SetPredicate(std::nullopt);
        }
      }
    }
    std::erase_if(block_metadata.streaming_io_and_pipeline.single_value_inputs,
                  [&nodes](const SingleValueInput& input) {
                    return !nodes.contains(input.GetDataPort());
                  });
    std::erase_if(block_metadata.streaming_io_and_pipeline.single_value_outputs,
                  [&nodes](const SingleValueOutput& output) {
                    return !nodes.contains(output.GetDataPort());
                  });
    for (std::optional<Node*>& valid :
         block_metadata.streaming_io_and_pipeline.pipeline_valid) {
      if (valid.has_value() && !nodes.contains(*valid)) {
        valid.reset();
      }
    }
    for (std::optional<Node*>& stage_done :
         block_metadata.streaming_io_and_pipeline.stage_done) {
      if (stage_done.has_value() && !nodes.contains(*stage_done)) {
        stage_done.reset();
      }
    }
    for (std::optional<Node*>& stage_valid :
         block_metadata.streaming_io_and_pipeline.stage_valid) {
      if (stage_valid.has_value() && !nodes.contains(*stage_valid)) {
        stage_valid.reset();
      }
    }
  }
}

}  // namespace xls::verilog
