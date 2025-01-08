// Copyright 2020 The XLS Authors
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

#ifndef XLS_IR_TOPO_SORT_H_
#define XLS_IR_TOPO_SORT_H_

#include <optional>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"

namespace xls {

// Convenience function for concise use in foreach constructs; e.g.:
//
//  for (Node* n : TopoSort(f)) {
//    ...
//  }
//
// Yields nodes in a topological traversal order (dependency ordering is
// satisfied). If no `randomizer` is provided, the order is stable; i.e., if
// there is no dependence ordering between A and B, and A is earlier in
// f.nodes() than B, then A will be listed before B. If `randomizer` is
// provided, ties will be broken at random.
//
// Note that the ordering for all nodes is computed up front, *not*
// incrementally as iteration proceeds.
std::vector<Node*> TopoSort(
    FunctionBase* f, std::optional<absl::BitGenRef> randomizer = std::nullopt);

// As above, but returns a reverse topo order.
std::vector<Node*> ReverseTopoSort(
    FunctionBase* f, std::optional<absl::BitGenRef> randomizer = std::nullopt);

}  // namespace xls

#endif  // XLS_IR_TOPO_SORT_H_
