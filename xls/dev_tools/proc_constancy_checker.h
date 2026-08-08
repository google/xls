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

#ifndef XLS_DEV_TOOLS_PROC_CONSTANCY_CHECKER_H_
#define XLS_DEV_TOOLS_PROC_CONSTANCY_CHECKER_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"
#include "xls/solvers/z3_ir_translator.h"
#include "z3/src/api/z3_api.h"

namespace xls {

// Map from original proc node to its cloned Node* instances across activations.
using NodeActivationMap = absl::flat_hash_map<Node*, std::vector<Node*>>;

// Strips non-synthesizable nodes (assert, trace, cover) and any intermediate
// nodes only consumed by them from the given proc/package using
// NonSynthRemovalPass.
absl::Status StripNonSynthNodes(Package* package, Proc* proc);

// Returns the list of non-constant, synthesizable candidate nodes in the proc
// to check for constancy.
//
// This name is a mouthful, but the idea is that when we test nodes for
// constancy, we don't want to waste time or spuriously report nodes we don't
// care about. Thus, this function enumerates nodes in the proc and filters out:
// 1. Non-synthesizable nodes (assert, trace, cover): it's OK and even expected
//    for these to be constant.
// 2. Trivial constant nodes (literals, or ops only consuming trivial constants-
//    typically aggregates of literals): these are the nodes that are supposed
//    to be constant.
// 3. Token nodes: tokens carry no value and are purely for sequencing.
absl::StatusOr<std::vector<Node*>> GetNodesFilteringNonSynthAndTrivialConstants(
    Proc* proc);

// Unrolls the proc `activation_count` times into a function using
// `proc_testutils::UnrollProc` and returns the unrolled function along with
// the mapping from original nodes to unrolled clones across activations.
absl::StatusOr<std::pair<Function*, NodeActivationMap>> UnrollProcForConstancy(
    Proc* proc, int64_t activation_count);

// Flattens a Z3 AST representing a Bits or Tuple/Array type into individual bit
// ASTs.
std::vector<Z3_ast> FlattenBitsOnly(Z3_context ctx,
                                    solvers::z3::IrTranslator* translator,
                                    Type* type, Z3_ast value);

}  // namespace xls

#endif  // XLS_DEV_TOOLS_PROC_CONSTANCY_CHECKER_H_
