// Copyright 2022 The XLS Authors
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

#ifndef XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_
#define XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

using TokenProvenance = absl::flat_hash_map<Node*, LeafTypeTree<Node*>>;

// Compute, for each token-type in the given `FunctionBase*`, what
// side-effecting node produced that token. If a leaf type in one of the
// `LeafTypeTree`s is not a token, the corresponding `Node*` will be `nullptr`.
absl::StatusOr<TokenProvenance> TokenProvenanceAnalysis(FunctionBase* f);

std::string ToString(const TokenProvenance& provenance);

}  // namespace xls

#endif  // XLS_PASSES_TOKEN_PROVENANCE_ANALYSIS_H_
