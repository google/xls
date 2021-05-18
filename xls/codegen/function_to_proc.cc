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
#include "xls/codegen/vast.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace verilog {

// Adds the function parameters as input ports on the proc (receives on a port
// channel). For each function parameter adds a key/value to node map where the
// key is the parameter and the value is the received data from the
// port.
static absl::StatusOr<std::vector<Node*>> AddInputPorts(
    Function* f, Proc* proc, absl::flat_hash_map<Node*, Node*>* node_map) {
  std::vector<Node*> tokens;

  int64_t port_position = 0;

  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        PortChannel * ch,
        f->package()->CreatePortChannel(
            param->GetName(), ChannelOps::kReceiveOnly, param->GetType()));
    ch->SetPosition(port_position++);

    XLS_ASSIGN_OR_RETURN(
        Node * rcv,
        proc->MakeNode<Receive>(param->loc(), proc->TokenParam(), ch->id()));
    XLS_ASSIGN_OR_RETURN(Node * rcv_token, proc->MakeNode<TupleIndex>(
                                               param->loc(), rcv, /*index=*/0));
    XLS_ASSIGN_OR_RETURN(Node * rcv_data, proc->MakeNode<TupleIndex>(
                                              param->loc(), rcv, /*index=*/1));
    tokens.push_back(rcv_token);
    (*node_map)[param] = rcv_data;
  }
  return tokens.empty() ? std::vector<Node*>({proc->TokenParam()}) : tokens;
}

// Add an output port to the given proc which sends the return value of the
// given function.  The send operation uses the given token. Returns the token
// from the send operation.
static absl::StatusOr<Node*> AddOutputPort(
    Function* f, Proc* proc, const absl::flat_hash_map<Node*, Node*>& node_map,
    Node* token) {
  Node* output = node_map.at(f->return_value());

  // TODO(meheff): 2021-03-01 Allow port names other than "out".
  XLS_ASSIGN_OR_RETURN(PortChannel * out_ch,
                       f->package()->CreatePortChannel(
                           "out", ChannelOps::kSendOnly, output->GetType()));
  out_ch->SetPosition(f->params().size());
  XLS_ASSIGN_OR_RETURN(Node * send, proc->MakeNode<Send>(output->loc(), token,
                                                         output, out_ch->id()));

  return send;
}

// Creates and returns a stateless proc (state is an empty tuple).
static Proc* CreateStatelessProc(Package* p, absl::string_view proc_name) {
  return p->AddProc(
      absl::make_unique<Proc>(proc_name, Value::Tuple({}), "tkn", "state", p));
}

absl::StatusOr<Proc*> FunctionToProc(Function* f, absl::string_view proc_name) {
  Proc* proc = CreateStatelessProc(f->package(), proc_name);

  // A map from the nodes in 'f' to their corresponding node in the proc.
  absl::flat_hash_map<Node*, Node*> node_map;

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> input_tokens,
                       AddInputPorts(f, proc, &node_map));

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

  XLS_ASSIGN_OR_RETURN(
      Node * merged_input_token,
      proc->MakeNode<AfterAll>(/*loc=*/absl::nullopt, input_tokens));
  XLS_ASSIGN_OR_RETURN(Node * output_token,
                       AddOutputPort(f, proc, node_map, merged_input_token));

  XLS_RETURN_IF_ERROR(proc->SetNextToken(output_token));
  XLS_RETURN_IF_ERROR(proc->SetNextState(proc->StateParam()));

  return proc;
}

// Returns pipeline-stage prefixed signal name for the given node. For
// example: p3_foo.
static std::string PipelineSignalName(Node* node, int64_t stage) {
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(node->GetName()));
}

absl::StatusOr<Proc*> FunctionToPipelinedProc(const PipelineSchedule& schedule,
                                              Function* f,
                                              absl::string_view proc_name) {
  Proc* proc = CreateStatelessProc(f->package(), proc_name);

  // A map from the nodes in 'f' to their corresponding node in the proc.
  absl::flat_hash_map<Node*, Node*> node_map;

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> input_tokens,
                       AddInputPorts(f, proc, &node_map));

  XLS_ASSIGN_OR_RETURN(
      Node * previous_token,
      proc->MakeNode<AfterAll>(/*loc=*/absl::nullopt, input_tokens));
  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    std::vector<Node*> output_tokens;
    for (Node* function_node : schedule.nodes_in_cycle(stage)) {
      if (function_node->Is<Param>()) {
        // Parameters become receive nodes in the proc and are added above.
        continue;
      }
      std::vector<Node*> new_operands;
      for (Node* operand : function_node->operands()) {
        new_operands.push_back(node_map.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(
          Node * node, function_node->CloneInNewFunction(new_operands, proc));
      node_map[function_node] = node;

      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == schedule.length() - 1) {
          return false;
        }
        if (n == f->return_value()) {
          return true;
        }
        for (Node* user : n->users()) {
          if (schedule.cycle(user) > stage) {
            return true;
          }
        }
        return false;
      };

      if (is_live_out_of_stage(function_node)) {
        // A register is represented as a Send node (register "D" port) followed
        // by a Receive node (register "Q" port). These are connected via a
        // token. A receive operation produces a tuple of (token, data) so this
        // construct also includes a tuple-index to extract the data.
        XLS_ASSIGN_OR_RETURN(
            RegisterChannel * reg_ch,
            f->package()->CreateRegisterChannel(PipelineSignalName(node, stage),
                                                node->GetType()));
        XLS_ASSIGN_OR_RETURN(
            Send * send, proc->MakeNode<Send>(node->loc(), previous_token, node,
                                              reg_ch->id()));
        XLS_ASSIGN_OR_RETURN(
            Receive * receive,
            proc->MakeNode<Receive>(node->loc(), /*token=*/send, reg_ch->id()));
        // Receives produce a tuple of (token, data).
        XLS_ASSIGN_OR_RETURN(
            Node * receive_token,
            proc->MakeNode<TupleIndex>(node->loc(), receive, /*index=*/0));
        XLS_ASSIGN_OR_RETURN(
            Node * receive_data,
            proc->MakeNode<TupleIndex>(node->loc(), receive, /*index=*/1));
        output_tokens.push_back(receive_token);
        node_map[node] = receive_data;
      }
    }

    if (!output_tokens.empty()) {
      XLS_ASSIGN_OR_RETURN(
          previous_token,
          proc->MakeNode<AfterAll>(/*loc=*/absl::nullopt, output_tokens));
    }
  }

  XLS_ASSIGN_OR_RETURN(Node * output_token,
                       AddOutputPort(f, proc, node_map, previous_token));

  XLS_RETURN_IF_ERROR(proc->SetNextToken(output_token));
  XLS_RETURN_IF_ERROR(proc->SetNextState(proc->StateParam()));

  return proc;
}

}  // namespace verilog
}  // namespace xls
