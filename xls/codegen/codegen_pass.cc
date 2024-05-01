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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/codegen/module_signature.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"

namespace xls::verilog {

std::string CodegenPassUnit::DumpIr() const {
  // Dump the Package and metadata. The metadata is commented out ('//') so the
  // output is parsable.
  std::string out =
      absl::StrFormat("// Generating code for proc: %s\n\n", name());
  absl::StrAppend(&out, package->DumpIr());
  for (const auto& [_, block_metadata] : metadata) {
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
  return package->GetNodeCount();
}

void CodegenPassUnit::GcMetadata() {
  absl::flat_hash_set<Node*> nodes;
  for (auto& [this_block, block_metadata] : metadata) {
    nodes.clear();
    nodes.insert(this_block->nodes().begin(), this_block->nodes().end());
    absl::erase_if(
        block_metadata.streaming_io_and_pipeline.node_to_stage_map,
        [&nodes](const auto& kv) { return !nodes.contains(kv.first); });

    for (std::vector<StreamingInput>& inputs :
         block_metadata.streaming_io_and_pipeline.inputs) {
      for (StreamingInput& input : inputs) {
        if (input.port.has_value() && !nodes.contains(*input.port)) {
          input.port.reset();
        }
        if (input.signal_data.has_value() &&
            !nodes.contains(*input.signal_data)) {
          input.signal_data.reset();
        }
        if (input.signal_valid.has_value() &&
            !nodes.contains(*input.signal_valid)) {
          input.signal_valid.reset();
        }
        if (input.predicate.has_value() && !nodes.contains(*input.predicate)) {
          input.predicate.reset();
        }
      }
    }
    for (std::vector<StreamingOutput>& outputs :
         block_metadata.streaming_io_and_pipeline.outputs) {
      for (StreamingOutput& output : outputs) {
        if (output.port.has_value() && !nodes.contains(*output.port)) {
          output.port.reset();
        }
        if (output.predicate.has_value() &&
            !nodes.contains(*output.predicate)) {
          output.predicate.reset();
        }
      }
    }
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
