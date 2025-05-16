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

#include "xls/codegen/codegen_util.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

std::vector<Block*> GetBlocksWithProcProvenance(Package* package) {
  std::vector<Block*> blocks;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->GetProvenance().has_value() &&
        block->GetProvenance()->kind == BlockProvenanceKind::kProc) {
      blocks.push_back(block.get());
    }
  }
  return blocks;
}

std::vector<Block*> GetBlocksWithFunctionProvenance(Package* package) {
  std::vector<Block*> blocks;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->GetProvenance().has_value() &&
        block->GetProvenance()->kind == BlockProvenanceKind::kFunction) {
      blocks.push_back(block.get());
    }
  }
  return blocks;
}

absl::StatusOr<absl::flat_hash_map<FunctionBase*, Block*>>
GetBlockProvenanceMap(Package* package) {
  absl::flat_hash_map<FunctionBase*, Block*> block_map;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (block->GetProvenance().has_value()) {
      FunctionBase* fb;
      if (block->GetProvenance()->kind == BlockProvenanceKind::kFunction) {
        XLS_ASSIGN_OR_RETURN(
            fb, package->GetFunction(block->GetProvenance()->name));
      } else {
        XLS_ASSIGN_OR_RETURN(fb,
                             package->GetProc(block->GetProvenance()->name));
      }
      block_map[fb] = block.get();
    }
  }
  return block_map;
}

}  // namespace xls
