// Copyright 2025 The XLS Authors
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

#include "xls/codegen/passes_ng/metadata_conversion_pass.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

using NodeToStageMap = absl::flat_hash_map<Node*, int64_t>;

// Creates a NodeToStageMap that is sufficient for signature generation.
//
// The NodeToStageMap is populated with top-level nodes associated with
// InstantiationInputs/Outputs and the stage number is assigned sequentially
// with respect to the order of instantiation and not with respect to the
// actual pipeline schedule.
absl::StatusOr<NodeToStageMap> CreateArbitraryNodeToStageMap(
    Block* top_block, BlockConversionMetadata& block_conversion_metadata) {
  NodeToStageMap node_to_stage_map;

  int64_t stage_number = 0;

  // Find all instantiations of blocks associated with a stage.
  for (Instantiation* inst : top_block->GetInstantiations()) {
    if (inst->kind() != InstantiationKind::kBlock) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(BlockInstantiation * block_inst,
                         inst->AsBlockInstantiation());
    Block* stage_block = block_inst->instantiated_block();

    if (!block_conversion_metadata.HasMetadataForBlock(stage_block)) {
      continue;
    }

    for (InstantiationInput* input : top_block->GetInstantiationInputs(inst)) {
      node_to_stage_map[input] = stage_number;
    }

    for (InstantiationOutput* output :
         top_block->GetInstantiationOutputs(inst)) {
      node_to_stage_map[output] = stage_number;
    }

    ++stage_number;
  }

  return node_to_stage_map;
}

absl::StatusOr<bool> MetadataConversionPass::RunInternal(
    Package* package, const CodegenPassOptions& options, PassResults* results,
    CodegenContext& context) const {
  // TODO(tedhong): 2025-06-03 - Remove once procs are supported.
  if (!context.HasTopBlock()) {
    return false;
  }

  VLOG(3) << "Generating metadata for:";
  XLS_VLOG_LINES(3, package->DumpIr());

  // Create dummy metadata context.
  //
  // TODO(tedhong): 2025-06-03 - Actualy convert the necessary metadata for
  // downstream passes.
  CodegenMetadata metadata;

  // Create stage map metadata sufficient for signature generation
  // to understand the latency of the pipeline.
  XLS_ASSIGN_OR_RETURN(
      metadata.streaming_io_and_pipeline.node_to_stage_map,
      CreateArbitraryNodeToStageMap(context.top_block(),
                                    context.block_conversion_metadata()));

  context.SetMetadataForBlock(context.top_block(), metadata);

  return true;
}

}  // namespace xls::verilog
