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

#include "xls/codegen_v_1_5/block_finalization_pass.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::StatusOr<Block*> BlockFinalizationPass::RemoveStages(
    ScheduledBlock* block, Package* package,
    const BlockConversionPassOptions& options, PassResults* results) const {
  // Pick a unique (temporary) name for the unscheduled block.
  std::vector<std::string> block_names;
  block_names.reserve(package->blocks().size());
  for (const std::unique_ptr<Block>& b : package->blocks()) {
    block_names.push_back(b->name());
  }
  NameUniquer uniquer(/*separator=*/"__", /*reserved_names=*/block_names);
  std::string temp_name = uniquer.GetSanitizedUniqueName(
      absl::StrCat(block->name(), "__unscheduled"));
  XLS_ASSIGN_OR_RETURN(Block * unscheduled_block,
                       block->CloneWithoutSchedule(temp_name));

  // Replace all references to the original block with the temporary block.
  if (package->IsTop(block)) {
    XLS_RETURN_IF_ERROR(package->SetTop(unscheduled_block));
  }
  for (const std::unique_ptr<Block>& b : package->blocks()) {
    for (Instantiation* inst : b->GetInstantiations()) {
      if (inst->kind() != InstantiationKind::kBlock) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(BlockInstantiation * block_instantiation,
                           inst->AsBlockInstantiation());
      if (block_instantiation->instantiated_block() != block) {
        continue;
      }
      XLS_RETURN_IF_ERROR(block_instantiation->ReplaceBlock(unscheduled_block));
    }
  }

  // Finish by removing the original block & taking over its name.
  std::string name = block->name();
  XLS_RETURN_IF_ERROR(package->RemoveBlock(block));
  unscheduled_block->SetName(name);
  return unscheduled_block;
}

absl::StatusOr<bool> BlockFinalizationPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  std::vector<Block*> blocks_to_process;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->IsScheduled()) {
      blocks_to_process.push_back(block.get());
    }
  }
  for (Block* block : blocks_to_process) {
    XLS_RETURN_IF_ERROR(RemoveStages(down_cast<ScheduledBlock*>(block), package,
                                     options, results)
                            .status());
  }
  return !blocks_to_process.empty();
}

}  // namespace xls::codegen
