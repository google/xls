// Copyright 2024 The XLS Authors
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

#include "xls/codegen/register_combining_pass.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/register_chaining_analysis.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"

namespace xls::verilog {

namespace {
absl::Status CombineRegisters(absl::Span<const RegisterData> mutex_group,
                              Block* block, CodegenMetadata& metadata) {
  XLS_RET_CHECK_GE(mutex_group.size(), 2)
      << "Attempting to combine a single register is not meaningful. Single "
         "element mutex groups should have been filtered out.";
  // Registers are listed so that 'last' one is at the end.
  // The register with a loop-back write (write from a later stage) is always
  // at the front, if one exists.
  // Merge from the front back.
  const RegisterData& first = mutex_group.front();
  std::vector<Node*> cleanup_nodes;
  absl::flat_hash_set<Register*> cleanup_regs;

  // No need to change load-enable bits, we're merging into the top which has
  // the right bits already.
  VLOG(2) << "Collapsing " << mutex_group.size() << " registers into "
          << mutex_group.front().reg->ToString();
  for (const RegisterData& merge : mutex_group.subspan(1)) {
    XLS_RETURN_IF_ERROR(merge.read->ReplaceUsesWith(first.read));
    cleanup_regs.insert(merge.reg);
    cleanup_nodes.push_back(merge.read);
    cleanup_nodes.push_back(merge.write);
  }

  // Do cleanup.
  for (auto& stage : metadata.streaming_io_and_pipeline.pipeline_registers) {
    std::erase_if(stage, [&](const PipelineRegister& pr) {
      return cleanup_regs.contains(pr.reg);
    });
  }
  for (auto& state_reg : metadata.streaming_io_and_pipeline.state_registers) {
    CHECK(!state_reg || !cleanup_regs.contains(state_reg->reg))
        << "Removed a state register: " << state_reg->reg->ToString();
  }
  for (Node* n : cleanup_nodes) {
    XLS_RETURN_IF_ERROR(block->RemoveNode(n)) << "can't remove " << n;
  }
  for (Register* r : cleanup_regs) {
    XLS_RETURN_IF_ERROR(block->RemoveRegister(r));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> RunOnBlock(Block* block, CodegenMetadata& metadata,
                                const CodegenPassOptions& options) {
  if (options.codegen_options.register_merge_strategy() ==
      CodegenOptions::RegisterMergeStrategy::kDontMerge) {
    VLOG(2) << "Not merging any registers due to manual disabling of pass.";
    return false;
  }
  if (!metadata.concurrent_stages) {
    return false;
  }
  std::vector<RegisterData> candidate_registers;
  candidate_registers.reserve(block->GetRegisters().size());
  // State registers (but not their valid/reset regs) are candidates for
  // merging.
  VLOG(2) << block->DumpIr();
  for (const auto& maybe_reg :
       metadata.streaming_io_and_pipeline.state_registers) {
    if (maybe_reg) {
      CHECK(!maybe_reg->next_values.empty());
      auto write_stage =
          absl::c_min_element(maybe_reg->next_values, [](const auto& l,
                                                         const auto& r) {
            return l.stage < r.stage;
          })->stage;
      if (maybe_reg->read_stage == write_stage) {
        // Immediate back edge.
        continue;
      }
      candidate_registers.push_back({.reg = maybe_reg->reg,
                                     .read = maybe_reg->reg_read,
                                     .read_stage = maybe_reg->read_stage,
                                     .write = maybe_reg->reg_write,
                                     .write_stage = write_stage});
    }
  }
  // pipeline registers (but not their valid/reset regs) are candidates for
  // merging.
  for (const auto& stg_regs :
       metadata.streaming_io_and_pipeline.pipeline_registers) {
    for (const auto& reg : stg_regs) {
      CHECK(metadata.streaming_io_and_pipeline.node_to_stage_map.contains(
          reg.reg_read))
          << reg.reg_read;
      CHECK(metadata.streaming_io_and_pipeline.node_to_stage_map.contains(
          reg.reg_write))
          << reg.reg_write;
      Stage read_stage =
          metadata.streaming_io_and_pipeline.node_to_stage_map.at(reg.reg_read);
      Stage write_stage =
          metadata.streaming_io_and_pipeline.node_to_stage_map.at(
              reg.reg_write);
      CHECK_EQ(write_stage + 1, read_stage)
          << "pipeline register skipping stage? " << reg.reg->ToString()
          << "\nread: " << reg.reg_read << "\nwrite: " << reg.reg_write;
      candidate_registers.push_back({
          .reg = reg.reg,
          .read = reg.reg_read,
          .read_stage = read_stage,
          .write = reg.reg_write,
          .write_stage = write_stage,
      });
    }
  }
  // chains of registers which are possibly combinable.
  RegisterChains reg_groups;

  for (const RegisterData& rd : candidate_registers) {
    reg_groups.InsertAndReduce(rd);
  }
  XLS_ASSIGN_OR_RETURN(std::vector<std::vector<RegisterData>> mutex_chains,
                       reg_groups.SplitBetweenMutexRegions(
                           *metadata.concurrent_stages, options));
  bool changed = !mutex_chains.empty();

  for (const std::vector<RegisterData>& group : mutex_chains) {
    XLS_RETURN_IF_ERROR(CombineRegisters(group, block, metadata));
  }

  return changed;
}
}  // namespace

absl::StatusOr<bool> RegisterCombiningPass::RunInternal(
    CodegenPassUnit* unit, const CodegenPassOptions& options,
    CodegenPassResults* results) const {
  bool changed = false;
  for (auto& [block, metadata] : unit->metadata) {
    XLS_ASSIGN_OR_RETURN(bool block_changed,
                         RunOnBlock(block, metadata, options));
    changed = changed || block_changed;
  }
  if (changed) {
    unit->GcMetadata();
  }

  return changed;
}

}  // namespace xls::verilog
