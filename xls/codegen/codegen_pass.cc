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

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"

namespace xls::verilog {

std::string CodegenPassUnit::DumpIr() const {
  // Dump the Package and metadata. The metadata is commented out ('//') so the
  // output is parsable.
  std::string out =
      absl::StrFormat("// Generating code for proc: %s\n\n", name());
  absl::StrAppend(&out, package_->DumpIr());
  for (const auto& [_, block_metadata] : metadata_) {
    if (block_metadata.signature.has_value()) {
      for (auto line :
           absl::StrSplit(block_metadata.signature->ToString(), '\n')) {
        absl::StrAppend(&out, "// ", line, "\n");
      }
    }
  }
  return out;
}
int64_t CodegenPassUnit::GetNodeCount() const {
  return package_->GetNodeCount();
}

void CodegenPassUnit::GcMetadata() {
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
        if (input.GetDataPort().has_value() &&
            !nodes.contains(*input.GetDataPort())) {
          input.ClearDataPort();
        }
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
        if (output.GetDataPort().has_value() &&
            !nodes.contains(*output.GetDataPort())) {
          output.SetDataPort(std::nullopt);
        }
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

/* static */ ChannelMap ChannelMap::Create(const CodegenPassUnit& unit) {
  ChannelMap::StreamingInputMap channel_to_streaming_input;
  ChannelMap::StreamingOutputMap channel_to_streaming_output;
  ChannelMap::SingleValueInputMap channel_to_single_value_input;
  ChannelMap::SingleValueOutputMap channel_to_single_value_output;

  for (auto& [block, metadata] : unit.metadata()) {
    for (const std::vector<StreamingInput>& inputs :
         metadata.streaming_io_and_pipeline.inputs) {
      for (const StreamingInput& input : inputs) {
        VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                      ChannelRefName(input.GetChannel()));
        if (!input.IsExternal()) {
          VLOG(5) << absl::StreamFormat("Skipping internal input %s",
                                        ChannelRefName(input.GetChannel()));
          continue;
        }
        channel_to_streaming_input[std::get<Channel*>(input.GetChannel())] =
            &input;
      }
    }
    for (const SingleValueInput& input :
         metadata.streaming_io_and_pipeline.single_value_inputs) {
      VLOG(5) << absl::StreamFormat("Input found on %v for %s", *block,
                                    ChannelRefName(input.GetChannel()));
      channel_to_single_value_input[std::get<Channel*>(input.GetChannel())] =
          &input;
    }
    for (const std::vector<StreamingOutput>& outputs :
         metadata.streaming_io_and_pipeline.outputs) {
      for (const StreamingOutput& output : outputs) {
        VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                      ChannelRefName(output.GetChannel()));
        if (!output.IsExternal()) {
          VLOG(5) << absl::StreamFormat("Skipping internal output %s",
                                        ChannelRefName(output.GetChannel()));
          continue;
        }
        channel_to_streaming_output[std::get<Channel*>(output.GetChannel())] =
            &output;
      }
    }
    for (const SingleValueOutput& output :
         metadata.streaming_io_and_pipeline.single_value_outputs) {
      VLOG(5) << absl::StreamFormat("Output found on %v for %s.", *block,
                                    ChannelRefName(output.GetChannel()));
      channel_to_single_value_output[std::get<Channel*>(output.GetChannel())] =
          &output;
    }
  }
  return ChannelMap(std::move(channel_to_streaming_input),
                    std::move(channel_to_streaming_output),
                    std::move(channel_to_single_value_input),
                    std::move(channel_to_single_value_output));
}

}  // namespace xls::verilog
