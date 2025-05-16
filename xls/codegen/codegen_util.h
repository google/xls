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

#ifndef XLS_CODEGEN_CODEGEN_UTIL_H_
#define XLS_CODEGEN_CODEGEN_UTIL_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/block.h"
#include "xls/ir/package.h"

namespace xls {

// Return the blocks in the package which were generated from lowering
// Procs/Functions.
std::vector<Block*> GetBlocksWithProcProvenance(Package* package);
std::vector<Block*> GetBlocksWithFunctionProvenance(Package* package);

// Return a map containing Blocks which were generated from lowering
// Procs/Functions.
absl::StatusOr<absl::flat_hash_map<FunctionBase*, Block*>>
GetBlockProvenanceMap(Package* package);

}  // namespace xls

#endif  // XLS_CODEGEN_CODEGEN_UTIL_H_
