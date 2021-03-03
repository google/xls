// Copyright 2021 The XLS Authors
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

#include "xls/codegen/function_to_proc.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace verilog {

absl::StatusOr<Proc*> FunctionToProc(Function* f, absl::string_view proc_name) {
  TokenlessProcBuilder pb(proc_name, Value::Tuple({}), "tkn", "st",
                          f->package());
  // A map from the nodes in 'f' to their correpsonding node in (potentially a
  // clone) in 'proc'.
  absl::flat_hash_map<Node*, Node*> node_map;

  // Skip zero-width inputs/output. Verilog does not support zero-width ports.
  for (Param* param : f->params()) {
    if (param->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        PortChannel * ch,
        f->package()->CreatePortChannel(
            param->GetName(), ChannelOps::kReceiveOnly, param->GetType()));
    BValue rcv = pb.Receive(ch, param->loc(), param->GetName());
    node_map[param] = rcv.node();
  }

  // Construct the proc initially using a dummy return value consisting of a
  // literal of zero value. This enables us to build the skeleton of the proc
  // using TokenlessProcBuilder, then fill in the logic by cloning (which can't
  // be done using the proc builder) the interior nodes of the function into the
  // proc.
  Node* dummy_return_value = nullptr;
  if (f->return_value()->GetType()->GetFlatBitCount() != 0) {
    // TODO(meheff): 2021-03-01 Allow port names other than "out".
    XLS_ASSIGN_OR_RETURN(
        PortChannel * out_ch,
        f->package()->CreatePortChannel("out", ChannelOps::kSendOnly,
                                        f->return_value()->GetType()));
    BValue dummy_return = pb.Literal(ZeroOfType(f->return_value()->GetType()));
    pb.Send(out_ch, dummy_return);
    dummy_return_value = dummy_return.node();
  }

  // Build the proc. Initially this consists of only the input port receives and
  // optional output port send.
  XLS_ASSIGN_OR_RETURN(Proc * proc,
                       pb.Build(/*next_state=*/pb.GetStateParam()));

  // Clone in the nodes from the function into the proc.
  for (Node* node : TopoSort(f)) {
    if (node->Is<Param>()) {
      // Parameters become receive nodes in the proc and are added above.
      continue;
    }
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(Node * proc_node,
                         node->CloneInNewFunction(new_operands, proc));
    node_map[node] = proc_node;
  }

  // Replace the (optional) dummy return node literal with the actual return
  // value to send on the module port.
  if (dummy_return_value != nullptr) {
    XLS_RETURN_IF_ERROR(
        dummy_return_value->ReplaceUsesWith(node_map[f->return_value()]));
    XLS_RETURN_IF_ERROR(proc->RemoveNode(dummy_return_value));
  }

  return proc;
}

}  // namespace verilog
}  // namespace xls
