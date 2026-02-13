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

#include "xls/codegen_v_1_5/block_conversion_utils.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"

namespace xls::codegen {

std::vector<std::pair<ScheduledBlock*, Proc*>>
GetScheduledBlocksWithProcSources(Package* p, bool new_style_only) {
  std::vector<std::pair<ScheduledBlock*, Proc*>> result;
  for (const std::unique_ptr<Block>& block : p->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }

    auto* scheduled_block = absl::down_cast<ScheduledBlock*>(block.get());
    if (scheduled_block->source() == nullptr ||
        !scheduled_block->source()->IsProc()) {
      continue;
    }

    Proc* proc = scheduled_block->source()->AsProcOrDie();
    if (new_style_only && !proc->is_new_style_proc()) {
      continue;
    }
    result.emplace_back(scheduled_block, proc);
  }
  return result;
}

absl::Status UpdateProcInstantiationsAndRemoveOldProcs(
    const absl::flat_hash_map<Proc*, Proc*>& old_to_new) {
  for (const auto& [_, new_proc] : old_to_new) {
    for (std::unique_ptr<ProcInstantiation>& instantiation :
         new_proc->proc_instantiations()) {
      instantiation->set_proc(old_to_new.at(instantiation->proc()));
    }
  }

  for (const auto& [old_proc, _] : old_to_new) {
    XLS_RETURN_IF_ERROR(old_proc->package()->RemoveFunctionBase(old_proc));
  }

  return absl::OkStatus();
}

absl::flat_hash_map<Block*, BlockInstantiation*> GetInstantiatedBlocks(
    Package* package) {
  absl::flat_hash_map<Block*, BlockInstantiation*> map;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    for (Instantiation* instantiation : block->GetInstantiations()) {
      if (instantiation->kind() == InstantiationKind::kBlock) {
        auto* block_instantiation =
            absl::down_cast<BlockInstantiation*>(instantiation);
        CHECK(map.emplace(block_instantiation->instantiated_block(),
                          block_instantiation)
                  .second);
      }
    }
  }
  return map;
}

}  // namespace xls::codegen
