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

#ifndef XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_UTILS_H_
#define XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_UTILS_H_

#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"

namespace xls::codegen {

// Extracts the scheduled blocks from `p` that have proc sources, and returns
// each block-proc pair, in the order they appear in the package. If
// `new_style_only` is true then procs using global channels are filtered out.
std::vector<std::pair<ScheduledBlock*, Proc*>>
GetScheduledBlocksWithProcSources(Package* p, bool new_style_only = false);

// Repoints any proc instantiations in the new procs in `old_to_new`, so that
// they point to new procs rather than old procs. Then removes all the old procs
// in `old_to_new` from their package.
absl::Status UpdateProcInstantiationsAndRemoveOldProcs(
    const absl::flat_hash_map<Proc*, Proc*>& old_to_new);

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_UTILS_H_
