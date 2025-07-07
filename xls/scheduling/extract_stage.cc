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

#include "xls/scheduling/extract_stage.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/schedule_graph.h"

namespace xls {

absl::StatusOr<Function*> ExtractStage(FunctionBase* src,
                                       const PipelineSchedule& schedule,
                                       int stage) {
  // Create a new function in the package which only contains the nodes at the
  // given stage (cycle).
  Package* package = src->package();
  auto new_f = std::make_unique<Function>(
      absl::StrFormat("%s_stage_%d", src->name(), stage), package);
  absl::flat_hash_map<Node*, Node*> node_map;
  std::vector<Node*> live_out;
  for (Node* node : TopoSort(src)) {
    if (IsUntimed(node)) {
      continue;
    }
    if (schedule.cycle(node) == stage) {
      std::vector<Node*> new_operands;
      for (Node* operand : node->operands()) {
        if (node_map.contains(operand)) {
          new_operands.push_back(node_map.at(operand));
        } else {
          Node* new_param = new_f->AddNode(
              std::make_unique<Param>(operand->loc(), operand->GetType(),
                                      operand->GetName(), new_f.get()));
          node_map[operand] = new_param;
          new_operands.push_back(new_param);
        }
      }
      // hack to support viewing procs as functions
      Node* new_node;
      if (node->Is<StateRead>() || node->Is<Next>() || node->Is<Send>() ||
          node->Is<Receive>()) {
        // NB The fact that data-dependencies is dropped is fine since prior to
        // this anything used by the Send or Next node is marked for return
        // below.
        new_node = new_f->AddNode(std::make_unique<xls::Param>(
            node->loc(), node->GetType(), node->GetName(), new_f.get()));
      } else {
        XLS_ASSIGN_OR_RETURN(
            new_node, node->CloneInNewFunction(new_operands, new_f.get()));
      }
      node_map[node] = new_node;
      // NB This checks whether the value is *USED* by a send or next and
      // returns it.
      if (std::any_of(node->users().begin(), node->users().end(), [&](Node* u) {
            return schedule.cycle(u) > stage || u->Is<Send>() ||
                   u->Is<Next>() || new_f->HasImplicitUse(node);
          })) {
        live_out.push_back(new_node);
      }
    }
  }

  // If this stage doesn't include the function output, create a final tuple
  // which gathers all nodes scheduled in the stage that are live out.
  // The tuple will be the return value of the new function.
  // Otherwise, just use the mapped function output.
  if (src->IsFunction() &&
      node_map.contains(src->AsFunctionOrDie()->return_value())) {
    XLS_RETURN_IF_ERROR(new_f->set_return_value(
        node_map.at(src->AsFunctionOrDie()->return_value())));
  } else {
    if (live_out.size() == 1) {
      XLS_RETURN_IF_ERROR(new_f->set_return_value(live_out.front()));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * return_tuple,
                           new_f->MakeNode<Tuple>(SourceInfo(), live_out));
      XLS_RETURN_IF_ERROR(new_f->set_return_value(return_tuple));
    }
  }
  return package->AddFunction(std::move(new_f));
}
}  // namespace xls
