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

#ifndef XLS_PASSES_UNROLL_PROC_H_
#define XLS_PASSES_UNROLL_PROC_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/node.h"
#include "xls/ir/proc.h"

namespace xls {

struct ProcUnrollInfo {
  // For any given node in the unrolled version of the proc, this map gives
  // the node in the original proc that this node was unrolled from.
  absl::flat_hash_map<Node*, Node*> original_node_map;

  // For any given node in the unrolled version of the proc, this map gives
  // the iteration of the unroll that this node belongs to.
  absl::flat_hash_map<Node*, int64_t> iteration_map;
};

// Unroll a proc to the given number of iterations.
// Returns info about the unroll.
//
// For example, if you unrolled the following proc to 3 iterations:
//
// proc example(__token: token, __state: bits[10], init={0}) {
//   literal.1: bits[10] = literal(value=1)
//   add.2: bits[10] = add(literal.1, __state)
//   next (__token, add.2)
// }
//
// Then you'd get the following proc:
//
// proc example(__token: token, __state: bits[10], init={0}) {
//   literal.1: bits[10] = literal(value=1)
//   add.2: bits[10] = add(literal.1, __state)
//   literal.3: bits[10] = literal(value=1)
//   add.4: bits[10] = add(literal.3, add.2)
//   literal.5: bits[10] = literal(value=1)
//   add.6: bits[10] = add(literal.5, add.6)
//   next (__token, add.8)
// }
//
// And the `original_node_map` would look like:
//
//   literal.1 -> literal.1
//   add.2 -> add.2
//   literal.3 -> literal.1
//   add.4 -> add.2
//   literal.5 -> literal.1
//   add.6 -> add.2
//
// While the `iteration_map` would look like:
//
//   literal.1 -> 0
//   add.2 -> 0
//   literal.3 -> 1
//   add.4 -> 1
//   literal.5 -> 2
//   add.6 -> 2
absl::StatusOr<ProcUnrollInfo> UnrollProc(Proc* proc, int64_t num_iterations);

}  // namespace xls

#endif  // XLS_PASSES_UNROLL_PROC_H_
