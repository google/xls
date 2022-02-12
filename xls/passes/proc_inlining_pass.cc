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

#include "xls/passes/proc_inlining_pass.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

// Returns the unique top-level proc or an error if no such proc can be
// identified.
absl::StatusOr<Proc*> GetTopLevelProc(Package* p) {
  // Find the unique proc with sendonly and receiveonly channels.
  std::optional<Proc*> top;
  for (const std::unique_ptr<Proc>& proc : p->procs()) {
    for (Node* node : proc->nodes()) {
      if (IsChannelNode(node)) {
        Channel* ch = GetChannelUsedByNode(node).value();
        if (ch->supported_ops() == ChannelOps::kSendOnly ||
            ch->supported_ops() == ChannelOps::kReceiveOnly) {
          if (top.has_value()) {
            // Proc inlining requires a unique top-level proc.
            return absl::UnimplementedError(
                "Proc inlining requires a single top-level proc (a proc with "
                "external channels)");
          }
          top = proc.get();
          break;
        }
      }
    }
  }

  if (!top.has_value()) {
    return absl::InvalidArgumentError(
        "Unable to identify top-level proc for proc inlining. No procs have "
        "external channels.");
  }
  return top.value();
}

// Returns the set of procs to inline. Effectively this is all of the procs in
// the package except the top-level proc.
std::vector<Proc*> GetProcsToInline(Proc* top) {
  std::vector<Proc*> procs;
  for (const std::unique_ptr<Proc>& proc : top->package()->procs()) {
    if (proc.get() == top) {
      continue;
    }
    procs.push_back(proc.get());
  }
  return procs;
}

struct TokenNode {
  Node* node;
  std::vector<Node*> sources;
};

// Returns a graph representation of the network of tokens threaded through
// send, receive, and parameter nodes in the proc. This network defines the flow
// of the activation bit through the proc thread. If the proc includes
// side-effecting nodes which are not send, receive or parameter then an error
// is returned. Currently, only a linear chain of tokens is supported (no
// fan-out or fan-in).
absl::StatusOr<std::vector<TokenNode>> GetTokenNetwork(Proc* proc) {
  std::vector<TokenNode> token_nodes;
  absl::flat_hash_map<Node*, std::vector<Node*>> token_sources;
  for (Node* node : TopoSort(proc)) {
    std::vector<Node*> operand_sources;
    for (Node* operand : node->operands()) {
      if (TypeHasToken(operand->GetType())) {
        for (Node* token_source : token_sources.at(operand)) {
          if (std::find(operand_sources.begin(), operand_sources.end(),
                        token_source) == operand_sources.end()) {
            operand_sources.push_back(token_source);
          }
        }
      }
    }
    // No fan-in (or fan-out) is supported.
    if (operand_sources.size() > 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc inlining does not support ops with multiple token operands: %s",
          node->GetName()));
    }
    if (OpIsSideEffecting(node->op()) && TypeHasToken(node->GetType())) {
      if (!node->Is<Param>() && !node->Is<Send>() && !node->Is<Receive>()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Proc inlining does not support %s ops", OpToString(node->op())));
      }

      // Only a linear chain of tokens is supported.
      if (operand_sources.size() == 1 &&
          operand_sources.front() != token_nodes.back().node) {
        return absl::UnimplementedError(
            "For proc inlining, tokens must form a linear chain.");
      }

      token_nodes.push_back(TokenNode{node, std::move(operand_sources)});
      token_sources[node] = {node};
    } else {
      token_sources[node] = std::move(operand_sources);
    }
  }
  XLS_VLOG(3) << absl::StreamFormat("Token network for proc %s:", proc->name());
  for (const TokenNode& token_node : token_nodes) {
    XLS_VLOG(3) << absl::StreamFormat(
        "%s (%s)", token_node.node->GetName(),
        absl::StrJoin(token_node.sources, ", ", NodeFormatter));
  }
  return token_nodes;
}

// Makes and returns a node computing Not(node).
absl::StatusOr<Node*> Not(
    Node* node, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kNot, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node, Op::kNot);
}

// Makes and returns a node computing Identity(node).
absl::StatusOr<Node*> Identity(
    Node* node, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kIdentity, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node,
                                               Op::kIdentity);
}

// Makes and returns a node computing And(a, b).
absl::StatusOr<Node*> And(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kAnd, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kAnd);
}

// Makes and returns a node computing Or(a, b).
absl::StatusOr<Node*> Or(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kOr, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kOr);
}

// Makes and returns a node computing And(a, !b).
absl::StatusOr<Node*> AndNot(
    Node* a, Node* b, std::optional<absl::string_view> name = absl::nullopt) {
  XLS_ASSIGN_OR_RETURN(
      Node * not_b, a->function_base()->MakeNode<UnOp>(b->loc(), b, Op::kNot));
  return And(a, not_b, name);
}

// Abstraction representing a send operation on a proc thread. A virtual send
// is created by decomposing a Op::kSend node into logic operations which manage
// the data and its flow-control. In a virtual send, data is "sent" over a
// dependency edge in the proc graph rather than over a channel. This data
// structure gathers together the inputs and outputs of this logic.
struct VirtualSend {
  // The channel which the originating send communicated over.
  Channel* channel;

  // The data to be sent.
  Node* data;

  // Whether `data` is valid.
  Node* data_valid;

  // A placeholder node representing the incoming activation bit for this send
  // operation. It will eventually be replaced by the activation passed from the
  // previous operation in the activation chain on the proc thread.
  Node* dummy_activation_in;

  // The outgoing activation bit. This will be wired to the next operation
  // in the activation chain on the proc thread.
  Node* activation_out;
};

// Creates a virtual send corresponding to sending the given data on the given
// channel with an optional predicate. The nodes composing the virtual
// send are constructed in `top`.
absl::StatusOr<VirtualSend> CreateVirtualSend(
    Channel* channel, Node* data, absl::optional<Node*> send_predicate,
    Proc* top) {
  if (channel->kind() != ChannelKind::kStreaming) {
    // TODO(meheff): 2022/02/11 Add support for single-value channels.
    return absl::UnimplementedError(
        "Only streaming channels are supported in proc inlinling.");
  }

  VirtualSend virtual_send;
  virtual_send.channel = channel;
  virtual_send.data = data;

  // Create a dummy activation in. Later this will be replaced with
  // signal passed from the upstream operation in the activation chain.
  XLS_ASSIGN_OR_RETURN(
      virtual_send.dummy_activation_in,
      top->MakeNodeWithName<Literal>(
          absl::nullopt, Value(UBits(1, 1)),
          absl::StrFormat("%s_send_dummy_activation_in", channel->name())));

  // A send never holds on to the activation. The assumption is that the receive
  // is always ready to receive the data so the send never blocks.
  XLS_ASSIGN_OR_RETURN(
      virtual_send.activation_out,
      Identity(virtual_send.dummy_activation_in,
               absl::StrFormat("%s_send_activation_out", channel->name())));

  if (send_predicate.has_value()) {
    // The send is conditional. Valid logic:
    //
    //   data_valid = activation_in && cond
    XLS_ASSIGN_OR_RETURN(
        virtual_send.data_valid,
        And(virtual_send.dummy_activation_in, send_predicate.value(),
            absl::StrFormat("%s_data_valid", channel->name())));
  } else {
    // The send is unconditional. The channel has data if the send is activated.
    XLS_ASSIGN_OR_RETURN(
        virtual_send.data_valid,
        Identity(virtual_send.dummy_activation_in,
                 absl::StrFormat("%s_data_valid", channel->name())));
  }

  return virtual_send;
}

// Abstraction representing a receive operation on a proc thread. A virtual
// receive is created by decomposing a Op::kReceive node into logic operations
// which manage the data and its flow-control. In a virtual receive, data is
// "sent" over a dependency edge in the proc graph rather than over a channel.
// This data structure gathers together the inputs and outputs of this logic.
struct VirtualReceive {
  // The channel which the originating receive communicated over.
  Channel* channel;

  // A placeholder node representing the data. It will
  // eventually be replaced with the data node from the corresponding virtual
  // send.
  Node* dummy_data;

  // A placeholder node representing whether the data is valid. It will
  // eventually be replaced with the data valid node from the corresponding
  // virtual send.
  Node* dummy_data_valid;

  // A placeholder node representing the incoming activation bit for this
  // receive operation. It will eventually be replaced by the activation passed
  // from the previous operation in the activation chain on the proc thread.
  Node* dummy_activation_in;

  // A placeholder node representing whether this receive is currently holding
  // the activation bit of the proc thread. A receive might hold the activation
  // bit for multiple proc ticks while blocked waiting for data valid.
  Node* dummy_holds_activation;

  // The value of the `holds_activation` value for the receive in the next proc
  // tick.
  Node* next_holds_activation;

  // The outgoing activation bit. This will be wired to the next operation
  // in the activation chain on the proc thread.
  Node* activation_out;
};

// Creates a virtual receive corresponding to receiving the given data on the
// given channel with an optional predicate. The nodes composing the virtual
// receive are constructed in `top`.
absl::StatusOr<VirtualReceive> CreateVirtualReceive(
    Channel* channel, absl::optional<Node*> receive_predicate, Proc* top) {
  if (channel->kind() != ChannelKind::kStreaming) {
    // TODO(meheff): 2022/02/11 Add support for single-value channels.
    return absl::UnimplementedError(
        "Only streaming channels are supported in proc inlinling.");
  }

  VirtualReceive virtual_receive;
  virtual_receive.channel = channel;
  absl::optional<SourceLocation> loc;

  // Create a dummy channel-has-data node. Later this will be replaced with
  // signal generated from the corresponding send.
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.dummy_data_valid,
      top->MakeNodeWithName<Literal>(
          loc, Value(UBits(0, 1)),
          absl::StrFormat("%s_dummy_data_valid", channel->name())));

  // Create a dummy channel data node. Later this will be replaced with
  // the data sent from the the corresponding send node.
  XLS_ASSIGN_OR_RETURN(virtual_receive.dummy_data,
                       top->MakeNodeWithName<Literal>(
                           loc, ZeroOfType(channel->type()),
                           absl::StrFormat("%s_dummy_data", channel->name())));

  // Create a dummy activation in. Later this will be replaced with
  // signal passed from the upstream send/receive in the token chain..
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.dummy_activation_in,
      top->MakeNodeWithName<Literal>(
          loc, Value(UBits(1, 1)),
          absl::StrFormat("%s_rcv_dummy_activation_in", channel->name())));

  // Create a dummy holds-activation node. Later this will be replaced with a
  // bit from the proc state.
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.dummy_holds_activation,
      top->MakeNodeWithName<Literal>(
          loc, Value(UBits(0, 1)),
          absl::StrFormat("%s_rcv_dummy_holds_activation", channel->name())));

  // Logic indicating whether the activation will be passed along (if the
  // receive has the activation):
  //
  //   pass_activation_along = (!predicate || data_valid)
  Node* pass_activation_along;
  if (receive_predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * not_predicate, Not(receive_predicate.value()));
    XLS_ASSIGN_OR_RETURN(
        pass_activation_along,
        Or(not_predicate, virtual_receive.dummy_data_valid,
           absl::StrFormat("%s_rcv_pass_activation", channel->name())));
  } else {
    // Receive is unconditional. The activation is passed along iff the channel
    // has data.
    XLS_ASSIGN_OR_RETURN(
        pass_activation_along,
        Identity(virtual_receive.dummy_data_valid,
                 absl::StrFormat("%s_rcv_pass_activation", channel->name())));
  }

  // Logic about whether to the receive currently has the activation bit:
  //
  //   has_activation = activation_in || holds_activation
  XLS_ASSIGN_OR_RETURN(
      Node * has_activation,
      Or(virtual_receive.dummy_activation_in,
         virtual_receive.dummy_holds_activation,
         absl::StrFormat("%s_rcv_has_activation", channel->name())));

  // Logic for whether or not the receive holds the activation until the next
  // proc tick.
  //
  //  next_holds_activation = has_activation && !pass_activation_along
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.next_holds_activation,
      AndNot(has_activation, pass_activation_along,
             absl::StrFormat("%s_rcv_next_holds_activation", channel->name())));

  // Logic for whether to set the activation out:
  //
  //   activation_out = has_activation && pass_activation_along
  XLS_ASSIGN_OR_RETURN(
      virtual_receive.activation_out,
      And(has_activation, pass_activation_along,
          absl::StrFormat("%s_rcv_activation_out", channel->name())));

  // TODO(meheff): 2022/02/11 For conditional receives, add a select between the
  // data value and zero to preserve the zero-vale semantics when the predicate
  // is false.

  // TODO(meheff): 2022/02/11 Add assert to catch
  // instances where data is dropped on the floor (valid is true but the receive
  // does not fire).

  return virtual_receive;
}

// Abstraction representing a proc thread. A proc thread virtually evaluates a
// particular proc within the top-level proc. An activation bit is threaded
// through the proc's virtual send/receive operations and keeps track of
// progress through the proc thread.
struct ProcThread {
  // The proc which this proc thread evaluates.
  Proc* proc;

  // The token connectivity graph of the original proc.
  std::vector<TokenNode> token_network;

  // A placeholder node representing the start of the activation chain through
  // the proc thread. This will eventually replaced by a bit from the top-level
  // proc state.
  Node* dummy_activation;

  // The node indicating that the activation bit has flowed through the entire
  // proc thread. If true, then this proc thread has completed a virtual tick.
  Node* activation_out;

  // A placeholder node for this proc's state. It will later be replaced by an
  // element from the actual top-level proc state.
  Node* dummy_state;

  // The computed next state for the proc thread. This is only committed to the
  // state when the activation bit has flowed through the entire proc thread
  // (`activation_out` is true).
  Node* computed_state;

  // The state of this proc actually committed with each tick of the top-level
  // proc. If `activation_out` is true this is equal `computed_state`. Otherwise
  // it is the unmodified state. That is, state only commits when
  // `activation_out` is true.
  Node* next_state;
};

// Converts each send and receive operation over kSendReceive channels into
// virtual send/receive which execute on the proc thread via the activation
// chain. This does not wire the activation through the virtual send/receive
// operations.
absl::StatusOr<ProcThread> CreateVirtualSendReceives(
    Proc* proc, absl::flat_hash_map<Channel*, VirtualSend>& virtual_sends,
    absl::flat_hash_map<Channel*, VirtualReceive>& virtual_receives) {
  ProcThread proc_thread;
  proc_thread.proc = proc;

  // Before transforming the proc at all gather the original token connectivity
  // graph.
  XLS_ASSIGN_OR_RETURN(proc_thread.token_network, GetTokenNetwork(proc));

  // Create a dummy state node and replace all uses of the existing state param
  // with it. Later the state will be augmented to include the state of the
  // inlined procs and necessary state bits for each proc thread.
  XLS_ASSIGN_OR_RETURN(proc_thread.dummy_state,
                       proc->MakeNodeWithName<Literal>(
                           /*loc=*/absl::nullopt, ZeroOfType(proc->StateType()),
                           absl::StrFormat("%s_dummy_state", proc->name())));
  XLS_RETURN_IF_ERROR(
      proc->StateParam()->ReplaceUsesWith(proc_thread.dummy_state));

  proc_thread.computed_state = proc->NextState();

  XLS_RETURN_IF_ERROR(
      proc->StateParam()->ReplaceUsesWith(proc_thread.dummy_state));

  for (Node* node : TopoSort(proc)) {
    if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
      if (ch->supported_ops() != ChannelOps::kSendReceive) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          VirtualSend virtual_send,
          CreateVirtualSend(ch, send->data(), send->predicate(), proc));
      XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(send->token()));
      virtual_sends[ch] = std::move(virtual_send);
    } else if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
      if (ch->supported_ops() != ChannelOps::kSendReceive) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          VirtualReceive virtual_receive,
          CreateVirtualReceive(ch, receive->predicate(), proc));

      // The output of a receive operation is a tuple of (token, data).
      XLS_ASSIGN_OR_RETURN(
          Node * receive_output,
          proc->MakeNodeWithName<Tuple>(
              receive->loc(),
              std::vector<Node*>{receive->token(), virtual_receive.dummy_data},
              absl::StrFormat("%s_rcv", ch->name())));

      XLS_RETURN_IF_ERROR(receive->ReplaceUsesWith(receive_output));
      virtual_receives[ch] = std::move(virtual_receive);
    }
  }

  return std::move(proc_thread);
}

// Inlines the given proc into `top`. Sends and receives in `proc` are replaced
// with virtual sends/receives which execute via the activation
// chain. Newly created virtual send/receieves are inserted into the
// `virtual_send` and `virtual_receive` maps.
absl::StatusOr<ProcThread> InlineProcIntoTop(
    Proc* proc, Proc* top,
    absl::flat_hash_map<Channel*, VirtualSend>& virtual_sends,
    absl::flat_hash_map<Channel*, VirtualReceive>& virtual_receives) {
  ProcThread proc_thread;
  proc_thread.proc = proc;

  absl::flat_hash_map<Node*, Node*> node_map;

  // Create a dummy state node. Later this will be replaced with a subset of the
  // top proc state.
  XLS_ASSIGN_OR_RETURN(proc_thread.dummy_state,
                       top->MakeNodeWithName<Literal>(
                           /*loc=*/absl::nullopt, ZeroOfType(proc->StateType()),
                           absl::StrFormat("%s_state", proc->name())));
  for (Node* node : TopoSort(proc)) {
    if (node->Is<Param>()) {
      if (node == proc->StateParam()) {
        // The dummy state value will later be replaced with an element from the
        // top-level proc state.
        node_map[node] = proc_thread.dummy_state;
      } else {
        // Connect the inlined token network from `proc` to the token parameter
        // of `top`.
        XLS_RET_CHECK_EQ(node, proc->TokenParam());
        node_map[node] = top->TokenParam();
      }
      continue;
    }
    if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      absl::optional<Node*> predicate;
      if (send->predicate().has_value()) {
        predicate = node_map.at(send->predicate().value());
      }
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
      XLS_ASSIGN_OR_RETURN(
          VirtualSend virtual_send,
          CreateVirtualSend(ch, node_map.at(send->data()), predicate, top));

      // The output of the send operation itself is token-typed. Just use the
      // token operand of the send.
      node_map[node] = node_map.at(send->token());
      virtual_sends[ch] = std::move(virtual_send);
      continue;
    }
    if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      absl::optional<Node*> predicate;
      if (receive->predicate().has_value()) {
        predicate = node_map.at(receive->predicate().value());
      }
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
      XLS_ASSIGN_OR_RETURN(VirtualReceive virtual_receive,
                           CreateVirtualReceive(ch, predicate, top));

      // The output of a receive operation is a tuple of (token, data).
      XLS_ASSIGN_OR_RETURN(Node * receive_output,
                           top->MakeNodeWithName<Tuple>(
                               receive->loc(),
                               std::vector<Node*>{node_map.at(receive->token()),
                                                  virtual_receive.dummy_data},
                               absl::StrFormat("%s_rcv", ch->name())));
      node_map[node] = receive_output;
      virtual_receives[ch] = std::move(virtual_receive);
      continue;
    }

    // Normal operations are just cloned into `top`.
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(node_map[node],
                         node->CloneInNewFunction(new_operands, top));
  }

  proc_thread.computed_state = node_map.at(proc->NextState());

  // Wire in the next-token value from the inlined proc into the next-token of
  // the top proc. Use an afterall to join the tokens. If the top next-token is
  // already an afterall, replace it with an afterall with the proc's next-token
  // added in.
  Node* proc_next_token = node_map.at(proc->NextToken());
  if (top->NextToken()->Is<AfterAll>()) {
    std::vector<Node*> operands(top->NextToken()->operands().begin(),
                                top->NextToken()->operands().end());
    operands.push_back(proc_next_token);
    Node* old_after_all = top->NextToken();
    XLS_RETURN_IF_ERROR(
        top->NextToken()->ReplaceUsesWithNew<AfterAll>(operands).status());
    XLS_RETURN_IF_ERROR(top->RemoveNode(old_after_all));
  } else {
    XLS_RETURN_IF_ERROR(top->NextToken()
                            ->ReplaceUsesWithNew<AfterAll>(std::vector<Node*>(
                                {top->NextToken(), proc_next_token}))
                            .status());
  }

  XLS_ASSIGN_OR_RETURN(proc_thread.token_network, GetTokenNetwork(proc));

  return std::move(proc_thread);
}

// Adds the given predicate to the send. If the send already has a predicate
// then the new predicate will be `old_predicate` AND `predicate`, otherwise the
// send is replaced with a new send with `predicate` as the predicate.
absl::Status AddSendPredicate(Send* send, Node* predicate) {
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(send->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(send->ReplaceOperandNumber(2, new_predicate));
  } else {
    XLS_RETURN_IF_ERROR(send->ReplaceUsesWithNew<Send>(send->token(),
                                                       send->data(), predicate,
                                                       send->channel_id())
                            .status());
    XLS_RETURN_IF_ERROR(send->function_base()->RemoveNode(send));
  }
  return absl::OkStatus();
}

// Adds the given predicate to the receive. If the receive already has a
// predicate then the new predicate will be `old_predicate` AND `predicate`,
// otherwise the receive is replaced with a new receive with `predicate` as the
// predicate.
absl::Status AddReceivePredicate(Receive* receive, Node* predicate) {
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(receive->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(receive->ReplaceOperandNumber(1, new_predicate));
  } else {
    XLS_RETURN_IF_ERROR(receive
                            ->ReplaceUsesWithNew<Receive>(receive->token(),
                                                          predicate,
                                                          receive->channel_id())
                            .status());
    XLS_RETURN_IF_ERROR(receive->function_base()->RemoveNode(receive));
  }
  return absl::OkStatus();
}

// Threads an activation bit through the virtual sends/receive nodes created
// from send/receive nodes in `proc`. The topology of this activation network
// mirrors the topology of the token network in the original proc.
absl::Status ConnectActivationChain(
    ProcThread& proc_thread, Proc* top,
    const absl::flat_hash_map<Channel*, VirtualSend>& virtual_sends,
    const absl::flat_hash_map<Channel*, VirtualReceive>& virtual_receives) {
  // Create a dummy activation node which is the head of the activation
  // chain. Later this will be replaced a bit on the proc state.
  XLS_ASSIGN_OR_RETURN(
      proc_thread.dummy_activation,
      top->MakeNodeWithName<Literal>(
          /*loc=*/absl::nullopt, Value(UBits(1, 1)),
          absl::StrFormat("%s_dummy_activation", proc_thread.proc->name())));
  Node* activation = proc_thread.dummy_activation;
  for (const TokenNode& token_node : proc_thread.token_network) {
    if (token_node.node->Is<Param>()) {
      continue;
    }
    XLS_RET_CHECK_EQ(token_node.sources.size(), 1);
    if (token_node.node->Is<Send>()) {
      Send* send = token_node.node->As<Send>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        const VirtualSend& virtual_send = virtual_sends.at(ch);
        XLS_RETURN_IF_ERROR(
            virtual_send.dummy_activation_in->ReplaceUsesWith(activation));
        activation = virtual_send.activation_out;
      } else {
        // A receive from a SendOnly channel. Add an additional condition so
        // the operation only fires it has the activation bit.
        XLS_RET_CHECK(send->function_base() == top);
        if (send->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                               And(activation, send->predicate().value()));
          XLS_RETURN_IF_ERROR(AddSendPredicate(send, new_predicate));
        } else {
          XLS_RETURN_IF_ERROR(AddSendPredicate(send, activation));
        }
        // Activation bit is automatically pass along to next node in the
        // activation change. `activation` variable does not need to be updated.
      }
      continue;
    }
    if (token_node.node->Is<Receive>()) {
      Receive* receive = token_node.node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));
      if (ch->supported_ops() == ChannelOps::kSendReceive) {
        const VirtualReceive& virtual_receive = virtual_receives.at(ch);
        XLS_RETURN_IF_ERROR(
            virtual_receive.dummy_activation_in->ReplaceUsesWith(activation));
        activation = virtual_receive.activation_out;
      } else {
        // A receive from a ReceiveOnly channel. Add an additional condition so
        // the operation only fires it has the activation bit.
        XLS_RET_CHECK(receive->function_base() == top);
        if (receive->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                               And(activation, receive->predicate().value()));
          XLS_RETURN_IF_ERROR(AddReceivePredicate(receive, new_predicate));
        } else {
          XLS_RETURN_IF_ERROR(AddReceivePredicate(receive, activation));
        }
        // Activation bit is automatically pass along to next node in the
        // activation change. `activation` variable does not need to be updated.
      }
      continue;
    }
  }
  proc_thread.activation_out = activation;

  // Add selector to commit state if the activation token is present.
  XLS_ASSIGN_OR_RETURN(
      proc_thread.next_state,
      top->MakeNodeWithName<Select>(
          absl::nullopt, /*selector=*/activation, /*cases=*/
          std::vector<Node*>{proc_thread.dummy_state,
                             proc_thread.computed_state},
          /*default_case=*/absl::nullopt,
          absl::StrFormat("%s_next_state", proc_thread.proc->name())));

  return absl::OkStatus();
}

// An abstraction defining a element of the proc state. These elements are
// gathered together in a tuple to define the state.
struct StateElement {
  // Name of the element.
  std::string name;

  // Initial value of the element.
  Value initial_value;

  // Dummy placeholder for the state element. When the element is added to the
  // proc state all uses of this dummy node will be replaced by the state
  // element.
  Node* dummy;

  // The next value of the state for the next proc tick.
  Node* next;
};

// Replace the state of `proc` with a tuple defined by the given StateElements.
absl::Status ReplaceProcState(Proc* proc,
                              absl::Span<const StateElement> elements) {
  std::vector<Value> initial_values;
  std::vector<Type*> types;
  std::vector<Node*> nexts;
  for (const StateElement& element : elements) {
    initial_values.push_back(element.initial_value);
    types.push_back(element.dummy->GetType());
    nexts.push_back(element.next);
  }
  Value initial_value = Value::Tuple(initial_values);
  XLS_ASSIGN_OR_RETURN(
      Node * next,
      proc->MakeNodeWithName<Tuple>(
          absl::nullopt, nexts,
          absl::StrFormat("%s_next", proc->StateParam()->GetName())));
  XLS_RETURN_IF_ERROR(
      proc->ReplaceState(proc->StateParam()->name(), next, initial_value));

  for (int64_t i = 0; i < elements.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * state_element,
        proc->MakeNodeWithName<TupleIndex>(absl::nullopt, proc->StateParam(), i,
                                           elements[i].name));
    XLS_RETURN_IF_ERROR(elements[i].dummy->ReplaceUsesWith(state_element));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ProcInliningPass::RunInternal(Package* p,
                                                   const PassOptions& options,
                                                   PassResults* results) const {
  if (p->procs().size() <= 1) {
    return false;
  }
  XLS_ASSIGN_OR_RETURN(Proc * top, GetTopLevelProc(p));
  std::vector<Proc*> procs_to_inline = GetProcsToInline(top);
  if (procs_to_inline.empty()) {
    return false;
  }

  std::vector<ProcThread> proc_threads;
  absl::flat_hash_map<Channel*, VirtualSend> virtual_sends;
  absl::flat_hash_map<Channel*, VirtualReceive> virtual_receives;

  // Convert send/receives in top proc which communicate with other procs into
  // virtual send/receives.
  XLS_ASSIGN_OR_RETURN(
      ProcThread top_info,
      CreateVirtualSendReceives(top, virtual_sends, virtual_receives));
  proc_threads.push_back(top_info);

  // Inline each proc into `top`. Sends/receives are converted to virtual
  // send/receives.
  // TODO(meheff): 2022/02/11 Add analysis which determines whether inlining is
  // a legal transformation.
  for (Proc* proc : procs_to_inline) {
    XLS_ASSIGN_OR_RETURN(
        ProcThread proc_thread,
        InlineProcIntoTop(proc, top, virtual_sends, virtual_receives));
    proc_threads.push_back(std::move(proc_thread));
  }

  XLS_VLOG(3) << "After inlining procs:\n" << p->DumpIr();

  // Thread the activation bit through the virtual operations in each proc
  // thread. Each proc thread gets its own activation bit.
  for (ProcThread& proc_thread : proc_threads) {
    XLS_RETURN_IF_ERROR(ConnectActivationChain(proc_thread, top, virtual_sends,
                                               virtual_receives));
  }

  XLS_VLOG(3) << "After connecting activations:\n" << p->DumpIr();

  // Connect data and data_valid signals between the virtual send and the
  // respective virtual receive nodes.
  for (Channel* ch : p->channels()) {
    if (ch->supported_ops() != ChannelOps::kSendReceive) {
      continue;
    }
    const VirtualSend& virtual_send = virtual_sends.at(ch);
    const VirtualReceive& virtual_receive = virtual_receives.at(ch);

    XLS_VLOG(3) << absl::StreamFormat(
        "Connecting channel %s data: replacing %s with %s", ch->name(),
        virtual_receive.dummy_data->GetName(), virtual_send.data->GetName());
    XLS_RETURN_IF_ERROR(
        virtual_receive.dummy_data->ReplaceUsesWith(virtual_send.data));

    XLS_VLOG(3) << absl::StreamFormat(
        "Connecting channel %s data_valid: replacing %s with %s", ch->name(),
        virtual_receive.dummy_data_valid->GetName(),
        virtual_send.data_valid->GetName());
    XLS_RETURN_IF_ERROR(virtual_receive.dummy_data_valid->ReplaceUsesWith(
        virtual_send.data_valid));
  }

  XLS_VLOG(3) << "After connecting data channels:\n" << p->DumpIr();

  // Gather all inlined proc state and proc thread book-keeping bits and add to
  // the top-level proc state.
  std::vector<StateElement> state_elements;

  // Add the inlined (and top) proc state and activation bits.
  for (const ProcThread& proc_thread : proc_threads) {
    state_elements.push_back(StateElement{
        .name = absl::StrFormat("%s_state", proc_thread.proc->name()),
        .initial_value = proc_thread.proc->InitValue(),
        .dummy = proc_thread.dummy_state,
        .next = proc_thread.next_state});
    state_elements.push_back(StateElement{
        .name = absl::StrFormat("%s_activation", proc_thread.proc->name()),
        .initial_value = Value(UBits(1, 1)),
        .dummy = proc_thread.dummy_activation,
        .next = proc_thread.activation_out});
  }

  // Add the bits for stalled receive operations which may hold the activation
  // bit for more than one tick.
  for (Channel* ch : p->channels()) {
    if (ch->supported_ops() != ChannelOps::kSendReceive) {
      continue;
    }
    state_elements.push_back(StateElement{
        .name = absl::StrFormat("%s_rcv_holds_activation", ch->name()),
        .initial_value = Value(UBits(0, 1)),
        .dummy = virtual_receives.at(ch).dummy_holds_activation,
        .next = virtual_receives.at(ch).next_holds_activation});
  }
  XLS_RETURN_IF_ERROR(ReplaceProcState(top, state_elements));

  XLS_VLOG(3) << "After transforming proc state:\n" << p->DumpIr();

  // Delete inlined procs.
  for (Proc* proc : procs_to_inline) {
    XLS_RETURN_IF_ERROR(p->RemoveProc(proc));
  }

  // Delete send and receive nodes in top which were used for communicating with
  // the inlined procs.
  std::vector<Node*> to_remove;
  for (Node* node : top->nodes()) {
    if (!IsChannelNode(node)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(node));
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      to_remove.push_back(node);
    }
  }
  for (Node* node : to_remove) {
    XLS_RETURN_IF_ERROR(top->RemoveNode(node));
  }

  // Delete channels used for communicating with the inlined procs.
  std::vector<Channel*> channels(p->channels().begin(), p->channels().end());
  for (Channel* ch : channels) {
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      XLS_RETURN_IF_ERROR(p->RemoveChannel(ch));
    }
  }

  XLS_VLOG(3) << "After deleting inlined procs:\n" << p->DumpIr();

  return true;
}

}  // namespace xls
