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

#include "xls/passes/map_inlining_pass.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

MapInliningPass::MapInliningPass()
    : OptimizationFunctionBasePass(MapInliningPass::kName,
                                   "Inline map operations") {}

absl::StatusOr<bool> MapInliningPass::RunOnFunctionBaseInternal(
    FunctionBase* function, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  bool changed = false;
  std::vector<Node*> map_nodes;
  for (Node* node : function->nodes()) {
    if (node->Is<Map>()) {
      map_nodes.push_back(node);
      changed = true;
    }
  }

  for (Node* node : map_nodes) {
    XLS_RETURN_IF_ERROR(ReplaceMap(node->As<Map>()));
  }

  return changed;
}

absl::Status MapInliningPass::InlineOneMap(Map* map) {
  MapInliningPass pass;
  return pass.ReplaceMap(map);
}

absl::Status MapInliningPass::ReplaceMap(Map* map) const {
  FunctionBase* function = map->function_base();

  int map_inputs_size = map->operand(0)->GetType()->AsArrayOrDie()->size();
  std::vector<Node*> invocations;
  invocations.reserve(map_inputs_size);
  for (int i = 0; i < map_inputs_size; i++) {
    Value index_value =
        Value(UBits(i, Bits::MinBitCountUnsigned(map_inputs_size)));
    XLS_ASSIGN_OR_RETURN(Node * index,
                         function->MakeNode<Literal>(map->loc(), index_value));
    XLS_ASSIGN_OR_RETURN(Node * array_index, function->MakeNode<ArrayIndex>(
                                                 map->loc(), map->operand(0),
                                                 std::vector<Node*>({index})));
    XLS_ASSIGN_OR_RETURN(
        Node * node,
        function->MakeNode<Invoke>(map->loc(), absl::MakeSpan(&array_index, 1),
                                   map->to_apply()));
    invocations.push_back(node);
  }

  Type* output_element_type = map->GetType()->AsArrayOrDie()->element_type();
  XLS_RETURN_IF_ERROR(
      map->ReplaceUsesWithNew<Array>(invocations, output_element_type)
          .status());
  return function->RemoveNode(map);
}

}  // namespace xls
