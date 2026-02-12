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

#include "xls/ir/ir_annotator.h"

#include <optional>
#include <vector>

#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"

namespace xls {

std::optional<std::vector<Node*>> TopoSortAnnotator::NodeOrder(
    FunctionBase* fb) const {
  if (fb->IsScheduled() || !topo_sort_) {
    return std::nullopt;
  }
  return TopoSort(fb);
}

}  // namespace xls
