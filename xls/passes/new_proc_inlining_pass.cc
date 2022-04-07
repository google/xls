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

#include "xls/passes/new_proc_inlining_pass.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

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
  Node* not_b;
  if (name.has_value()) {
    XLS_ASSIGN_OR_RETURN(not_b, a->function_base()->MakeNodeWithName<UnOp>(
                                    b->loc(), b, Op::kNot,
                                    absl::StrFormat("not_%s", b->GetName())));
  } else {
    XLS_ASSIGN_OR_RETURN(
        not_b, a->function_base()->MakeNode<UnOp>(b->loc(), b, Op::kNot));
  }
  return And(a, not_b, name);
}

// Adds the given predicate to the send. If the send already has a predicate
// then the new predicate will be `old_predicate` AND `predicate`, otherwise the
// send is replaced with a new send with `predicate` as the predicate.
absl::StatusOr<Send*> AddSendPredicate(Send* send, Node* predicate) {
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(send->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(send->ReplaceOperandNumber(2, new_predicate));
    return send;
  }
  XLS_ASSIGN_OR_RETURN(Send * new_send, send->ReplaceUsesWithNew<Send>(
                                            send->token(), send->data(),
                                            predicate, send->channel_id()));
  return new_send;
}

// Adds the given predicate to the receive. If the receive already has a
// predicate then the new predicate will be `old_predicate` AND `predicate`,
// otherwise the receive is replaced with a new receive with `predicate` as the
// predicate.
absl::StatusOr<Receive*> AddReceivePredicate(Receive* receive,
                                             Node* predicate) {
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(receive->predicate().value(), predicate));
    XLS_RETURN_IF_ERROR(receive->ReplaceOperandNumber(1, new_predicate));
    return receive;
  }
  XLS_ASSIGN_OR_RETURN(Receive * new_receive,
                       receive->ReplaceUsesWithNew<Receive>(
                           receive->token(), predicate, receive->channel_id()));
  return new_receive;
}

// An abstraction defining a element of the proc state. These elements are
// gathered together in a tuple to define the state of the proc.
class StateElement {
 public:
  static absl::StatusOr<StateElement> Create(absl::string_view name,
                                             Value initial_value, Proc* proc) {
    StateElement element;
    element.name_ = name;
    element.initial_value_ = initial_value;

    // Create placeholder state node. Later this will be replaced with an
    // element from the top-level proc state.
    XLS_ASSIGN_OR_RETURN(element.state_,
                         proc->MakeNodeWithName<Literal>(
                             /*loc=*/absl::nullopt, initial_value,
                             absl::StrFormat("%s_state", name)));

    // Create placeholder next-tick value for the state element. The caller must
    // call SetNext to replace it with a real state element.
    XLS_ASSIGN_OR_RETURN(element.next_,
                         proc->MakeNodeWithName<Literal>(
                             /*loc=*/absl::nullopt, initial_value,
                             absl::StrFormat("%s_next", name)));
    element.next_is_set_ = false;
    return std::move(element);
  }

  absl::string_view GetName() const { return name_; }
  const Value& GetInitialValue() const { return initial_value_; }

  // Returns the node holding the element from the proc state.
  Node* GetState() const { return state_; }

  // Returns the node corresponding to the value of the state eleemnt on the
  // next proc tick.
  Node* GetNext() const { return next_; }

  // Set the value of the state eleemnt on the next proc state. This must be
  // called at least once to replace the placeholder next value.
  absl::Status SetNext(Node* node) {
    XLS_RET_CHECK_EQ(node->GetType(), next_->GetType());
    XLS_RETURN_IF_ERROR(next_->ReplaceUsesWith(node));
    next_ = node;
    next_is_set_ = true;
    return absl::OkStatus();
  }

  // Returns true iff SetNext has been called.
  bool IsNextSet() const { return next_is_set_; }

 private:
  // Name of the element.
  std::string name_;

  // Initial value of the element.
  Value initial_value_;

  // Placeholder for the state element. When the element is added to the
  // proc state all uses of this node will be replaced by the state
  // element.
  Node* state_;

  // The next value of the state for the next proc tick.
  Node* next_;

  // Whether SetNext has been called.
  bool next_is_set_;
};

// Abstraction representing a channel which has been inlined. These channels are
// necessarily SendReceive channels. Data on the channel is saved to the
// containing proc state along with a valid bit. This state behaves as a
// single-element FIFO. The data is held in the state until the end of the proc
// thread tick in which the assocated receive fired.
//
// The virtual channel logic has the following inputs:
//
//   send_fired: whether the send associated with this channel fired (activation
//               bit and predicate both true).
//
//   data_in: the data value to be sent over the channel.
//
//   proc_tick_complete: asserted when the proc containing the associated
//                       virtual receive has completed a tick.
//
// and the following outputs:
//
//   valid: whether the virtual channel contains data.
//
//   data_out: the data value to be received over the channel
//
// and the following state elements:
//
//   data : the data sent to the virtual channel.
//
//   valid : whether the channel holds data
//
//   receive_fired_this_tick : whether the associated virtual receive has fired
//                             in this tick of the proc thread.
class VirtualChannel {
 public:
  static absl::StatusOr<VirtualChannel> Create(Channel* channel, Proc* proc) {
    XLS_RET_CHECK_EQ(channel->supported_ops(), ChannelOps::kSendReceive);
    VirtualChannel vc;
    vc.channel_ = channel;

    // Construct placeholder inputs. These will be replaced later.
    XLS_ASSIGN_OR_RETURN(vc.data_in_,
                         proc->MakeNodeWithName<Literal>(
                             /*loc=*/absl::nullopt, ZeroOfType(channel->type()),
                             absl::StrFormat("%s_data_in", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        vc.send_fired_in_,
        proc->MakeNodeWithName<Literal>(
            /*loc=*/absl::nullopt, Value(UBits(0, 1)),
            absl::StrFormat("%s_send_fired_in", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        vc.receive_fired_in_,
        proc->MakeNodeWithName<Literal>(
            /*loc=*/absl::nullopt, Value(UBits(0, 1)),
            absl::StrFormat("%s_receive_fired_in", channel->name())));

    XLS_ASSIGN_OR_RETURN(
        vc.proc_tick_complete_in_,
        proc->MakeNodeWithName<Literal>(
            /*loc=*/absl::nullopt, Value(UBits(0, 1)),
            absl::StrFormat("%s_proc_tick_complete_in", channel->name())));

    // Construct the data state element and logic controlling the data:
    //
    //   data_out = send_fired_in ? data_in : data_state
    //   data_state_next = data_out
    XLS_ASSIGN_OR_RETURN(vc.data_state_,
                         vc.AllocateStateElement(
                             absl::StrFormat("%s_data_state", channel->name()),
                             ZeroOfType(channel->type()), proc));
    XLS_ASSIGN_OR_RETURN(
        vc.data_out_,
        proc->MakeNodeWithName<Select>(
            absl::nullopt,
            /*selector=*/vc.send_fired_in_, /*cases=*/
            std::vector<Node*>{vc.data_state_->GetState(), vc.data_in_},
            /*default_case=*/absl::nullopt,
            absl::StrFormat("%s_data_out", channel->name())));
    XLS_RETURN_IF_ERROR(vc.data_state_->SetNext(vc.data_out_));

    // Construct the logic and state keeping track of whether the virtual
    // receive has fired this proc thread tick.
    //
    //   receive_fired = receive_fired_in || receive_fired_state
    //   receive_fired_next = receive_fired && !proc_tick_complete_in
    XLS_ASSIGN_OR_RETURN(
        vc.receive_fired_state_,
        vc.AllocateStateElement(
            absl::StrFormat("%s_receive_fired_state", channel->name()),
            Value(UBits(0, 1)), proc));
    XLS_ASSIGN_OR_RETURN(
        vc.receive_fired_this_tick_,
        Or(vc.receive_fired_in_, vc.receive_fired_state_->GetState(),
           absl::StrFormat("%s_receive_fired_this_tick", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        Node * receive_fired_next,
        AndNot(vc.receive_fired_this_tick_, vc.proc_tick_complete_in_,
               absl::StrFormat("%s_receive_fired_next", channel->name())));
    XLS_RETURN_IF_ERROR(vc.receive_fired_state_->SetNext(receive_fired_next));

    // Construct the logic and state for the valid bit. For single-value
    // channels this is:
    //
    //   valid_out = send_fired || valid_state
    //   valid_next = valid_out
    //
    // For streaming channels the virtual receive (if it fires) clears the valid
    // bit at the end of the proc thread tick so the logic is:
    //
    //   valid_out = send_fired || valid_state
    //   clear_valid = proc_tick_complete_in && receive_fired_this_tick
    //   valid_next = valid_out && !clear_valid
    XLS_ASSIGN_OR_RETURN(vc.valid_state_,
                         vc.AllocateStateElement(
                             absl::StrFormat("%s_valid_state", channel->name()),
                             Value(UBits(0, 1)), proc));
    XLS_ASSIGN_OR_RETURN(vc.valid_out_,
                         Or(vc.send_fired_in_, vc.valid_state_->GetState(),
                            absl::StrFormat("%s_valid", channel->name())));

    Node* valid_next;
    if (channel->kind() == ChannelKind::kStreaming) {
      XLS_ASSIGN_OR_RETURN(
          Node * clear_valid,
          And(vc.proc_tick_complete_in_, vc.receive_fired_this_tick_,
              absl::StrFormat("%s_clear_valid", channel->name())));
      XLS_ASSIGN_OR_RETURN(
          valid_next,
          AndNot(vc.valid_out_, clear_valid,
                 absl::StrFormat("%s_valid_next", channel->name())));
    } else {
      XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);
      XLS_ASSIGN_OR_RETURN(
          valid_next,
          Identity(vc.valid_out_,
                   absl::StrFormat("%s_valid_next", channel->name())));
    }
    XLS_RETURN_IF_ERROR(vc.valid_state_->SetNext(valid_next));

    return std::move(vc);
  }

  Channel* GetChannel() const { return channel_; }

  // Set various inputs to the virtual channel to the given node.
  absl::Status SetDataIn(Node* node) {
    XLS_RETURN_IF_ERROR(data_in_->ReplaceUsesWith(node));
    data_in_ = node;
    return absl::OkStatus();
  }
  absl::Status SetSendFiredIn(Node* node) {
    XLS_RETURN_IF_ERROR(send_fired_in_->ReplaceUsesWith(node));
    send_fired_in_ = node;
    return absl::OkStatus();
  }
  absl::Status SetReceiveFiredIn(Node* node) {
    XLS_RETURN_IF_ERROR(receive_fired_in_->ReplaceUsesWith(node));
    receive_fired_in_ = node;
    return absl::OkStatus();
  }
  absl::Status SetProcTickCompleteIn(Node* node) {
    XLS_RETURN_IF_ERROR(proc_tick_complete_in_->ReplaceUsesWith(node));
    proc_tick_complete_in_ = node;
    return absl::OkStatus();
  }

  // Returns whether the data state is valid.
  Node* GetValidState() const { return valid_state_->GetState(); }

  // Returns various outputs from the virtual channel.
  Node* GetDataOut() const { return data_out_; }
  Node* GetValidOut() const { return valid_out_; }

  // Returns whether the associated virtual receive has fired in this tick of
  // the proc thread containing the virtual receive.
  Node* GetReceiveFiredThisTick() const { return receive_fired_this_tick_; }

  // Returns all state elements of this virtual channel.
  const std::list<StateElement>& GetStateElements() const {
    return state_elements_;
  }

 private:
  Channel* channel_;

  absl::StatusOr<StateElement*> AllocateStateElement(absl::string_view name,
                                                     Value initial_value,
                                                     Proc* proc) {
    XLS_ASSIGN_OR_RETURN(StateElement element,
                         StateElement::Create(name, initial_value, proc));
    state_elements_.push_back(std::move(element));
    return &state_elements_.back();
  }

  // Inputs to the virtual channel.
  Node* data_in_;
  Node* send_fired_in_;
  Node* receive_fired_in_;
  Node* proc_tick_complete_in_;

  // Outputs of the virtual channel.
  Node* data_out_;
  Node* valid_out_;

  // Whether the virtual recieive fired in the current tick of the proc thread.
  Node* receive_fired_this_tick_;

  // State elemnts refer to elements on the state_elements_ list.
  StateElement* data_state_;
  StateElement* valid_state_;
  StateElement* receive_fired_state_;
  std::list<StateElement> state_elements_;
};

// Abstraction representing a node in the activation network of a proc
// thread. An activation bit is passed along through the network to execute the
// operations corresponding to the activation node.
struct ActivationNode {
  std::string name;

  // The activation bit(s) passed from the predecessor nodes(s) in the
  // activation network on the proc thread.
  std::vector<Node*> activations_in;

  // The outgoing activation bit. This will be wired to the successor
  // operation(s) in the activation network of the proc thread.
  Node* activation_out;

  // The state element which holds the input activation bit(s). Only present for
  // operations which can stall and hold the activation bit for more than one
  // tick of the container proc. An activation node can fire only iff it has the
  // activation bits of all of its predeccessors.
  std::vector<StateElement*> activations_in_state;

  // An optional placeholder node which represents the condition under which
  // this activation node will *not* fire even if it has the activation bit.
  std::optional<Node*> stall_condition;
};

// Abstraction representing a proc thread. A proc thread contains the logic
// required to virtually evaluate a proc (the "inlined proc") within another
// proc (the "container proc"). An activation bit is threaded through the proc's
// virtual send/receive operations and keeps track of progress through the proc
// thread.
class ProcThread {
 public:
  // Creates and returns a proc thread which executes the given proc. `top` is
  // the FunctionBase which will contain the proc thread.
  static absl::StatusOr<ProcThread> Create(Proc* inlined_proc,
                                           Proc* container_proc) {
    ProcThread proc_thread;
    proc_thread.inlined_proc_ = inlined_proc;
    proc_thread.container_proc_ = container_proc;

    // Create the state element to hold the state of the inlined proc.
    XLS_ASSIGN_OR_RETURN(proc_thread.proc_state_,
                         proc_thread.AllocateState(
                             absl::StrFormat("%s_state", inlined_proc->name()),
                             inlined_proc->InitValue()));

    // Create the state element for the activation bit of the proc thread.
    XLS_ASSIGN_OR_RETURN(
        proc_thread.activation_state_,
        proc_thread.AllocateState(
            absl::StrFormat("%s_activation", inlined_proc->name()),
            Value(UBits(1, 1))));

    // Create a dummy placeholder for the node indicating the that proc thread
    // has completed a tick. This is replaced with an actual value in
    // Finalize. This placeholder is created here so logic (such as virtual
    // channels) can depend on the signal.
    XLS_ASSIGN_OR_RETURN(
        proc_thread.proc_tick_complete_,
        container_proc->MakeNodeWithName<Literal>(
            /*loc=*/absl::nullopt, Value(UBits(0, 1)),
            absl::StrFormat("%s_proc_tick_complete", inlined_proc->name())));

    return std::move(proc_thread);
  }

  // Allocate and return a state element. The state will later be added to the
  // container proc state.
  absl::StatusOr<StateElement*> AllocateState(absl::string_view name,
                                              Value initial_value);

  // Allocates an activation node on the proc threads activation chain.
  // `activations_in` are the activation bits of the predecessors of this node
  // in the activation network. `stallable` indicates whether this activation
  // node can stall for a reason _other_ than not having the activation
  // bit. That is, the node can hold the activation bit for more than one tick.
  absl::StatusOr<ActivationNode*> AllocateActivationNode(
      absl::string_view name, absl::Span<Node* const> activations_in,
      bool stallable);

  // Completes construction of the proc thread. `next_state` is the node holding
  // the value of the proc thread state in the next tick. `proc_tick_complete`
  // indicates the the current tick of the proc thread is complete.
  absl::Status Finalize(Node* next_state, Node* proc_tick_complete) {
    XLS_VLOG(3) << "Finalize proc thread for: " << inlined_proc_->name();

    // Add selector to commit state if the proc thread tick is complete.
    XLS_ASSIGN_OR_RETURN(
        Node * state_next,
        container_proc_->MakeNodeWithName<Select>(
            absl::nullopt, /*selector=*/proc_tick_complete, /*cases=*/
            std::vector<Node*>{GetDummyState(), next_state},
            /*default_case=*/absl::nullopt,
            absl::StrFormat("%s_next_state", inlined_proc_->name())));
    XLS_RETURN_IF_ERROR(proc_state_->SetNext(state_next));
    XLS_RETURN_IF_ERROR(activation_state_->SetNext(proc_tick_complete));
    XLS_RETURN_IF_ERROR(
        proc_tick_complete_->ReplaceUsesWith(proc_tick_complete));
    return absl::OkStatus();
  }

  // Returns the node holding the activation bit associated with the given node.
  Node* GetActivationBit(Node* node) const {
    if (node->Is<Param>()) {
      return activation_state_->GetState();
    }
    return activation_map_.at(node);
  }

  // Adds the given node to the activation network (if it has a token). The
  // activation network is a network of nodes representing side-effecting
  // operations with single bit values which flow between them. The activation
  // network mirrors the network of tokens in the original proc. The activation
  // bits indicating when side-effecting operation can fire.
  absl::Status AddToActivationNetwork(Node* node) {
    XLS_VLOG(3) << "AddToActivationNetwork: " << node->GetName();
    if (node->Is<Param>()) {
      return absl::OkStatus();
    }
    absl::optional<Node*> token_operand;
    for (Node* operand : node->operands()) {
      if (TypeHasToken(operand->GetType())) {
        if (token_operand.has_value()) {
          return absl::UnimplementedError(absl::StrFormat(
              "Node %s has multiple operands with tokens", node->GetName()));
        }
        token_operand = operand;
      }
    }
    if (!token_operand.has_value()) {
      return absl::OkStatus();
    }
    activation_map_[node] = GetActivationBit(token_operand.value());
    XLS_VLOG(3) << "  activation bit: " << activation_map_.at(node)->GetName();
    return absl::OkStatus();
  }

  absl::StatusOr<Node*> ConvertSend(
      Send* send, Node* activation_in,
      absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
    XLS_RET_CHECK_EQ(send->function_base(), container_proc_);
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
    XLS_VLOG(3) << absl::StreamFormat("Converting send %s on channel %s",
                                      send->GetName(), ch->name());

    XLS_ASSIGN_OR_RETURN(
        ActivationNode * activation_node,
        AllocateActivationNode(absl::StrFormat("%s_send", ch->name()),
                               {activation_in}, /*stallable=*/false));

    Node* result;
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      // Non-external send. Convert send into a virtual send.
      XLS_ASSIGN_OR_RETURN(result, CreateVirtualSend(send, activation_node,
                                                     virtual_channels.at(ch)));
    } else if (ch->kind() == ChannelKind::kStreaming) {
      // A streaming external channel needs to be added to the activation
      // chain. The send can fire only when it is activated.
      XLS_ASSIGN_OR_RETURN(result,
                           ConvertToActivatedSend(send, activation_node));
    } else {
      // Nothing to do for external single value channels.
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
      result = send;
    }
    activation_map_[result] = activation_node->activation_out;
    return result;
  }

  absl::StatusOr<Node*> ConvertAfterAll(
      AfterAll* after_all, absl::Span<Node* const> activations_in) {
    XLS_ASSIGN_OR_RETURN(
        ActivationNode * activation_node,
        AllocateActivationNode(after_all->GetName(), activations_in,
                               /*stallable=*/true));
    activation_map_[after_all] = activation_node->activation_out;
    return after_all;
  }

  absl::StatusOr<Node*> ConvertReceive(
      Receive* receive, Node* activation_in,
      absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
    XLS_RET_CHECK_EQ(receive->function_base(), container_proc_);
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));

    XLS_VLOG(3) << absl::StreamFormat("Converting receive %s on channel %s",
                                      receive->GetName(), ch->name());

    // Only virtual receives on streaming channels are stallable (they stall on
    // data not ready).
    XLS_ASSIGN_OR_RETURN(
        ActivationNode * activation_node,
        AllocateActivationNode(
            absl::StrFormat("%s_receive", ch->name()), {activation_in},
            /*stallable=*/ch->supported_ops() == ChannelOps::kSendReceive));

    Node* result;
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      // Non-external receive. Convert receive into a virtual receive.
      XLS_ASSIGN_OR_RETURN(result,
                           CreateVirtualReceive(receive, activation_node,
                                                virtual_channels.at(ch)));
    } else if (ch->kind() == ChannelKind::kStreaming) {
      // A streaming external channel needs to be added to the activation
      // chain. The receive can fire only when it is activated. `stallable` is
      // not set on this activation node because receive operation already has
      // blocking semantics in a proc so no need for additional stalling logic
      // (unlike the virtual receive in which there is no actual Op::kReceive
      // operation).
      XLS_ASSIGN_OR_RETURN(result,
                           ConvertToActivatedReceive(receive, activation_node));
    } else {
      // Nothing to do for external single value channels.
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
      result = receive;
    }
    activation_map_[result] = activation_node->activation_out;

    XLS_VLOG(3) << absl::StreamFormat("Receive %s converted to %s",
                                      receive->GetName(), result->GetName());
    return result;
  }

  // Creates a virtual receive corresponding to receiving the given data on the
  // given channel with an optional predicate.
  absl::StatusOr<Node*> CreateVirtualReceive(Receive* receive,
                                             ActivationNode* activation_node,
                                             VirtualChannel& virtual_channel);

  // Creates a virtual send corresponding to sending the given data on the given
  // channel with an optional predicate.
  absl::StatusOr<Node*> CreateVirtualSend(Send* send,
                                          ActivationNode* activation_node,
                                          VirtualChannel& virtual_channel);

  // Converts `send` to a send operation which is connected to the proc thread's
  // activation network. The activation bit is included in the predicate of the
  // new send. Returns the new send.
  absl::StatusOr<Send*> ConvertToActivatedSend(Send* send,
                                               ActivationNode* activation_node);

  // Converts `receive` to a receive operation which is connected to the proc
  // thread's activation network. The activation bit is included in the
  // predicate of the new receive. Returns the new receive.
  absl::StatusOr<Node*> ConvertToActivatedReceive(
      Receive* receive, ActivationNode* activation_node);

  const std::list<StateElement>& GetStateElements() const {
    return state_elements_;
  }

  // Returns the dummy node representing the proc state in the proc thread.
  Node* GetDummyState() const { return proc_state_->GetState(); }

 private:
  // The proc whose logic which this proc thread evaluates.
  Proc* inlined_proc_;

  // The actual proc in which this proc thread evaluates. The container proc may
  // simultaneously evaluate multiple proc threads.
  Proc* container_proc_;

  // A map from nodes which have tokens in their type to the respective
  // "activation bit" representing the token in the proc thread. Each
  // side-effecting operation only fires when it holds it's activation bit.
  absl::flat_hash_map<Node*, Node*> activation_map_;

  // Node indicating that the tick of this proc thread is complete.
  Node* proc_tick_complete_ = nullptr;

  // The state elements required to by this proc thread. These elements are
  // later added to the container proc state.
  std::list<StateElement> state_elements_;

  // The state element representing the state of the proc. This points to an
  // element in `state_elements_`.
  StateElement* proc_state_ = nullptr;

  // The single bit of state representing the activation bit. This points to an
  // element in `state_elements_`.
  StateElement* activation_state_ = nullptr;

  // The activation nodes representing side-effecting operations including
  // virtual send/receives through which the activation bit will be
  // threaded. Stored as a list for pointer stability.
  std::list<ActivationNode> activation_nodes_;
};

absl::StatusOr<StateElement*> ProcThread::AllocateState(absl::string_view name,
                                                        Value initial_value) {
  XLS_VLOG(3) << absl::StreamFormat("AllocateState: %s, initial value %s", name,
                                    initial_value.ToString());
  XLS_ASSIGN_OR_RETURN(
      StateElement element,
      StateElement::Create(name, initial_value, container_proc_));
  state_elements_.push_back(std::move(element));
  return &state_elements_.back();
}

absl::StatusOr<ActivationNode*> ProcThread::AllocateActivationNode(
    absl::string_view name, absl::Span<Node* const> activations_in,
    bool stallable) {
  XLS_VLOG(3) << absl::StreamFormat(
      "AllocateActivationNode: %s, inputs (%s), stallable %d", name,
      absl::StrJoin(activations_in, ", ", NodeFormatter), stallable);

  ActivationNode activation_node;
  activation_node.name = name;
  activation_node.activations_in =
      std::vector(activations_in.begin(), activations_in.end());

  XLS_RET_CHECK(!activations_in.empty());
  if (activations_in.size() == 1 && !stallable) {
    XLS_ASSIGN_OR_RETURN(activation_node.activation_out,
                         Identity(activations_in.front(),
                                  absl::StrFormat("%s_activation_out", name)));
  } else {
    XLS_RET_CHECK(stallable) << absl::StreamFormat(
        "An AllocationNode with multiple activation inputs must be stallable: "
        "%s",
        name);
    XLS_ASSIGN_OR_RETURN(activation_node.stall_condition,
                         container_proc_->MakeNodeWithName<Literal>(
                             absl::nullopt, Value(UBits(0, 1)),
                             absl::StrFormat("%s_stall_condition", name)));

    XLS_ASSIGN_OR_RETURN(Node * not_stalled,
                         Not(activation_node.stall_condition.value(),
                             absl::StrFormat("%s_not_stalled", name)));
    std::vector<Node*> activation_conditions = {not_stalled};
    std::vector<Node*> has_activations;
    for (int64_t i = 0; i < activations_in.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          StateElement * activation_in_state,
          AllocateState(absl::StrFormat("%s_holds_activation_%d",
                                        activation_node.name, i),
                        Value(UBits(0, 1))));
      activation_node.activations_in_state.push_back(activation_in_state);

      XLS_ASSIGN_OR_RETURN(
          Node * has_activation,
          Or(activations_in[i], activation_in_state->GetState(),
             absl::StrFormat("%s_has_activation_%d", name, i)));

      activation_conditions.push_back(has_activation);
      has_activations.push_back(has_activation);
    }
    XLS_ASSIGN_OR_RETURN(activation_node.activation_out,
                         container_proc_->MakeNodeWithName<NaryOp>(
                             absl::nullopt, activation_conditions, Op::kAnd,
                             absl::StrFormat("%s_is_activated", name)));

    // Each activation input is held until activation out is asserted.
    for (int64_t i = 0; i < activations_in.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * holds_activation,
          AndNot(has_activations[i], activation_node.activation_out,
                 absl::StrFormat("%s_holds_activation_%d_next",
                                 activation_node.name, i)));
      XLS_RETURN_IF_ERROR(
          activation_node.activations_in_state[i]->SetNext(holds_activation));
    }
  }

  activation_nodes_.push_back(activation_node);
  return &activation_nodes_.back();
}

absl::StatusOr<Node*> ProcThread::CreateVirtualSend(
    Send* send, ActivationNode* activation_node,
    VirtualChannel& virtual_channel) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(send));
  XLS_RET_CHECK_EQ(channel, virtual_channel.GetChannel());
  XLS_VLOG(3) << absl::StreamFormat("Creating virtual send on channel %s",
                                    channel->name());

  XLS_RET_CHECK_EQ(activation_node->activations_in.size(), 1);
  Node* activation_in = activation_node->activations_in.front();

  // The send fires if it has the activation bit and the predicate (if it
  // exists) is true.
  //
  //   send_fired = activation_in && predicate
  Node* send_fired;
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        send_fired, And(activation_in, send->predicate().value(),
                        absl::StrFormat("%s_send_fired", channel->name())));
  } else {
    XLS_ASSIGN_OR_RETURN(
        send_fired, Identity(activation_in, absl::StrFormat("%s_send_fired",
                                                            channel->name())));
  }
  XLS_RETURN_IF_ERROR(virtual_channel.SetDataIn(send->data()));
  XLS_RETURN_IF_ERROR(virtual_channel.SetSendFiredIn(send_fired));

  Node* token;
  if (channel->kind() == ChannelKind::kStreaming) {
    // For streaming channels, add an assert which fires if there is already
    // data saved on the channel and the send fires. This indicates data loss.
    XLS_ASSIGN_OR_RETURN(Node * data_loss,
                         And(virtual_channel.GetValidState(), send_fired,
                             absl::StrFormat("%s_data_loss", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        Node * no_data_loss,
        Not(data_loss, absl::StrFormat("%s_no_data_loss", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        Node * asrt,
        container_proc_->MakeNode<Assert>(
            absl::nullopt, send->token(), no_data_loss,
            /*message=*/
            absl::StrFormat("Channel %s lost data", channel->name()),
            /*label=*/
            absl::StrFormat("%s_data_loss_assert", channel->name())));
    token = asrt;
  } else {
    // Single-value channels can't lose data as each send just overwrites the
    // previous send.
    token = send->token();
  }
  XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(token));
  return token;
}

absl::StatusOr<Node*> ProcThread::CreateVirtualReceive(
    Receive* receive, ActivationNode* activation_node,
    VirtualChannel& virtual_channel) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(receive));
  XLS_RET_CHECK_EQ(channel, virtual_channel.GetChannel());
  if (channel->kind() == ChannelKind::kSingleValue &&
      receive->predicate().has_value()) {
    return absl::UnimplementedError(
        "Conditional receives on single-value channels are not supported");
  }

  XLS_VLOG(3) << absl::StreamFormat("Creating virtual receive on channel %s",
                                    channel->name());
  // Logic indicating whether the receive will stall:
  //
  //   stall = predicate && !data_valid
  //
  // Logic indicating whether the receive fired (i.e, read data from the
  // channel):
  //
  //   receive_fired = predicate && activation_out
  Node* stall;
  Node* receive_fired;
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        stall,
        AndNot(receive->predicate().value(), virtual_channel.GetValidOut(),
               absl::StrFormat("%s_receive_stall", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        receive_fired,
        And(activation_node->activation_out, receive->predicate().value(),
            absl::StrFormat("%s_receive_fired", channel->name())));
  } else {
    // Receive is unconditional. The activation is stalled iff the
    // !data_valid.
    XLS_ASSIGN_OR_RETURN(
        stall, Not(virtual_channel.GetValidOut(),
                   absl::StrFormat("%s_receive_stall", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        receive_fired,
        Identity(activation_node->activation_out,
                 absl::StrFormat("%s_receive_fired", channel->name())));
  }

  XLS_RETURN_IF_ERROR(
      activation_node->stall_condition.value()->ReplaceUsesWith(stall));
  XLS_RETURN_IF_ERROR(virtual_channel.SetReceiveFiredIn(receive_fired));
  XLS_RETURN_IF_ERROR(
      virtual_channel.SetProcTickCompleteIn(proc_tick_complete_));

  // If the receive has not fired, the data value presented to uses of the
  // receive should be a literal zero. This follows the semantics of a
  // conditional receive.
  XLS_ASSIGN_OR_RETURN(Node * zero,
                       container_proc_->MakeNodeWithName<Literal>(
                           /*loc=*/absl::nullopt, ZeroOfType(channel->type()),
                           absl::StrFormat("%s_zero", channel->name())));
  XLS_ASSIGN_OR_RETURN(
      Node * data,
      container_proc_->MakeNodeWithName<Select>(
          absl::nullopt,
          /*selector=*/virtual_channel.GetReceiveFiredThisTick(), /*cases=*/
          std::vector<Node*>{zero, virtual_channel.GetDataOut()},
          /*default_case=*/absl::nullopt,
          absl::StrFormat("%s_receive_data", channel->name())));

  // The output of a receive operation is a tuple of (token, data).
  XLS_ASSIGN_OR_RETURN(
      Node * result,
      container_proc_->MakeNodeWithName<Tuple>(
          absl::nullopt, std::vector<Node*>{receive->token(), data},
          absl::StrFormat("%s_receive", channel->name())));
  XLS_RETURN_IF_ERROR(receive->ReplaceUsesWith(result));
  return result;
}

absl::StatusOr<Send*> ProcThread::ConvertToActivatedSend(
    Send* send, ActivationNode* activation_node) {
  XLS_VLOG(3) << absl::StreamFormat("Converting send %s to activated send",
                                    send->GetName());
  XLS_RET_CHECK_EQ(activation_node->activations_in.size(), 1);
  Node* activation_in = activation_node->activations_in.front();
  Send* activated_send;
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(activation_in, send->predicate().value()));
    XLS_ASSIGN_OR_RETURN(activated_send, AddSendPredicate(send, new_predicate));
  } else {
    XLS_ASSIGN_OR_RETURN(activated_send, AddSendPredicate(send, activation_in));
  }
  XLS_VLOG(3) << absl::StreamFormat("Activated send: %s",
                                    activated_send->GetName());
  return activated_send;
}

absl::StatusOr<Node*> ProcThread::ConvertToActivatedReceive(
    Receive* receive, ActivationNode* activation_node) {
  XLS_VLOG(3) << absl::StreamFormat(
      "Converting receive %s to activated receive", receive->GetName());

  XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(receive));
  XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
  XLS_RET_CHECK_EQ(activation_node->activations_in.size(), 1);
  Node* activation_in = activation_node->activations_in.front();
  Receive* activated_receive;
  if (receive->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(activation_in, receive->predicate().value()));
    XLS_ASSIGN_OR_RETURN(activated_receive,
                         AddReceivePredicate(receive, new_predicate));
  } else {
    XLS_ASSIGN_OR_RETURN(activated_receive,
                         AddReceivePredicate(receive, activation_in));
  }
  XLS_VLOG(3) << absl::StreamFormat("Activated receive: %s",
                                    activated_receive->GetName());

  // Save the current uses of the receive.
  std::vector<Node*> old_users(activated_receive->users().begin(),
                               activated_receive->users().end());

  // The receive data will be used in the same virtual tick of the proc thread,
  // but that may correspond to multiple actual ticks of the containing proc so
  // we need to save the received data in the actual proc state.
  XLS_ASSIGN_OR_RETURN(
      StateElement * data_state,
      AllocateState(absl::StrFormat("%s_data_state", channel->name()),
                    ZeroOfType(channel->type())));
  XLS_ASSIGN_OR_RETURN(Node * data_in,
                       container_proc_->MakeNodeWithName<TupleIndex>(
                           absl::nullopt, activated_receive, 1,
                           absl::StrFormat("%s_data_in", channel->name())));
  XLS_ASSIGN_OR_RETURN(
      Node * data,
      container_proc_->MakeNodeWithName<Select>(
          absl::nullopt,
          /*selector=*/activated_receive->predicate().value(), /*cases=*/
          std::vector<Node*>{data_state->GetState(), data_in},
          /*default_case=*/absl::nullopt,
          absl::StrFormat("%s_data", channel->name())));

  XLS_RETURN_IF_ERROR(data_state->SetNext(data));

  XLS_ASSIGN_OR_RETURN(Node * token, container_proc_->MakeNode<TupleIndex>(
                                         absl::nullopt, activated_receive, 0));
  XLS_ASSIGN_OR_RETURN(
      Node * saved_receive,
      container_proc_->MakeNodeWithName<Tuple>(
          absl::nullopt, std::vector<Node*>{token, data},
          absl::StrFormat("%s_saved_receive", channel->name())));

  // Replace uses of the original receive with the newly created receive.
  for (Node* old_user : old_users) {
    old_user->ReplaceOperand(activated_receive, saved_receive);
  }

  return saved_receive;
}

// Converts `proc` into a proc thread which executes within the proc
// itself. That is, `proc` becomes the container proc which executes a proc
// thread representing the original logic of `proc`.  Newly created virtual
// send/receieves are inserted into the `virtual_send` and `virtual_receive`
// maps.
absl::StatusOr<ProcThread> ConvertToProcThread(
    Proc* proc,
    absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
  XLS_ASSIGN_OR_RETURN(ProcThread proc_thread, ProcThread::Create(proc, proc));

  XLS_RETURN_IF_ERROR(
      proc->StateParam()->ReplaceUsesWith(proc_thread.GetDummyState()));

  for (Node* node : TopoSort(proc)) {
    if (node->Is<Send>()) {
      XLS_RETURN_IF_ERROR(proc_thread
                              .ConvertSend(node->As<Send>(),
                                           proc_thread.GetActivationBit(
                                               node->As<Send>()->token()),
                                           virtual_channels)
                              .status());
    } else if (node->Is<Receive>()) {
      XLS_RETURN_IF_ERROR(proc_thread
                              .ConvertReceive(node->As<Receive>(),
                                              proc_thread.GetActivationBit(
                                                  node->As<Receive>()->token()),
                                              virtual_channels)
                              .status());
    } else if (node->Is<AfterAll>()) {
      std::vector<Node*> activation_bits;
      for (Node* operand : node->operands()) {
        activation_bits.push_back(proc_thread.GetActivationBit(operand));
      }
      XLS_RETURN_IF_ERROR(
          proc_thread.ConvertAfterAll(node->As<AfterAll>(), activation_bits)
              .status());
    } else {
      XLS_RETURN_IF_ERROR(proc_thread.AddToActivationNetwork(node));
    }
  }

  XLS_RETURN_IF_ERROR(proc_thread.Finalize(
      /*next_state=*/proc->NextState(),
      /*proc_tick_complete=*/proc_thread.GetActivationBit(proc->NextToken())));

  return std::move(proc_thread);
}

// Inlines the given proc into `container_proc` as a proc thread. Sends and
// receives in `proc` are replaced with virtual sends/receives which execute via
// the activation chain. Newly created virtual send/receieves are inserted into
// the `virtual_send` and `virtual_receive` maps.
absl::StatusOr<ProcThread> InlineProcAsProcThread(
    Proc* proc_to_inline, Proc* container_proc,
    absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
  XLS_ASSIGN_OR_RETURN(ProcThread proc_thread,
                       ProcThread::Create(proc_to_inline, container_proc));
  absl::flat_hash_map<Node*, Node*> node_map;

  auto clone_node = [&](Node* node) -> absl::StatusOr<Node*> {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    return node->CloneInNewFunction(new_operands, container_proc);
  };

  for (Node* node : TopoSort(proc_to_inline)) {
    XLS_VLOG(3) << absl::StreamFormat("Inlining node %s", node->GetName());
    if (node->Is<Param>()) {
      if (node == proc_to_inline->StateParam()) {
        // The dummy state value will later be replaced with an element from the
        // container proc state.
        node_map[node] = proc_thread.GetDummyState();
      } else {
        // Connect the inlined token network from `proc` to the token parameter
        // of `container_proc`.
        XLS_RET_CHECK_EQ(node, proc_to_inline->TokenParam());
        node_map[node] = container_proc->TokenParam();
      }
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * cloned_node, clone_node(node));

    if (node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(
          Node * converted_send,
          proc_thread.ConvertSend(
              cloned_node->As<Send>(),
              proc_thread.GetActivationBit(cloned_node->As<Send>()->token()),
              virtual_channels));
      node_map[node] = converted_send;
    } else if (node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(
          Node * converted_receive,
          proc_thread.ConvertReceive(
              cloned_node->As<Receive>(),
              proc_thread.GetActivationBit(cloned_node->As<Receive>()->token()),
              virtual_channels));
      node_map[node] = converted_receive;
    } else if (node->Is<AfterAll>()) {
      std::vector<Node*> activation_bits;
      for (Node* operand : node->operands()) {
        activation_bits.push_back(
            proc_thread.GetActivationBit(node_map.at(operand)));
      }
      XLS_ASSIGN_OR_RETURN(Node * converted_after_all,
                           proc_thread.ConvertAfterAll(
                               cloned_node->As<AfterAll>(), activation_bits));
      node_map[node] = converted_after_all;
    } else {
      XLS_RETURN_IF_ERROR(proc_thread.AddToActivationNetwork(cloned_node));
      node_map[node] = cloned_node;
    }
  }

  XLS_RETURN_IF_ERROR(proc_thread.Finalize(
      /*next_state=*/node_map.at(proc_to_inline->NextState()),
      /*proc_tick_complete=*/proc_thread.GetActivationBit(
          node_map.at(proc_to_inline->NextToken()))));

  // Wire in the next-token value from the inlined proc into the next-token of
  // the container proc. Use an afterall to join the tokens. If the container
  // proc next-token is already an afterall, replace it with an afterall with
  // the inlined proc's cloned next-token added in.
  Node* proc_next_token = node_map.at(proc_to_inline->NextToken());
  if (container_proc->NextToken()->Is<AfterAll>()) {
    std::vector<Node*> operands(container_proc->NextToken()->operands().begin(),
                                container_proc->NextToken()->operands().end());
    operands.push_back(proc_next_token);
    Node* old_after_all = container_proc->NextToken();
    XLS_RETURN_IF_ERROR(
        old_after_all->ReplaceUsesWithNew<AfterAll>(operands).status());
    XLS_RETURN_IF_ERROR(container_proc->RemoveNode(old_after_all));
  } else {
    XLS_RETURN_IF_ERROR(container_proc->NextToken()
                            ->ReplaceUsesWithNew<AfterAll>(std::vector<Node*>(
                                {container_proc->NextToken(), proc_next_token}))
                            .status());
  }

  return std::move(proc_thread);
}

// Replace the state of `proc` with a tuple defined by the given StateElements.
absl::Status ReplaceProcState(Proc* proc,
                              absl::Span<const StateElement> elements) {
  std::vector<Value> initial_values;
  std::vector<Type*> types;
  std::vector<Node*> nexts;
  for (const StateElement& element : elements) {
    XLS_RET_CHECK(element.IsNextSet()) << element.GetName();
    initial_values.push_back(element.GetInitialValue());
    types.push_back(element.GetState()->GetType());
    nexts.push_back(element.GetNext());
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
                                           elements[i].GetName()));
    XLS_RETURN_IF_ERROR(elements[i].GetState()->ReplaceUsesWith(state_element));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> NewProcInliningPass::RunInternal(
    Package* p, const PassOptions& options, PassResults* results) const {
  if (!options.inline_procs || p->procs().empty()) {
    return false;
  }

  if (!p->HasTop()) {
    return absl::InvalidArgumentError(
        "Must specify top-level proc name when running proc inlining");
  }

  FunctionBase* top_func_base = p->GetTop().value();

  if (!top_func_base->IsProc()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top level %s should be a proc when running proc inlining",
        top_func_base->name()));
  }

  Proc* top = top_func_base->AsProcOrDie();

  absl::flat_hash_map<Channel*, VirtualChannel> virtual_channels;
  for (Channel* ch : p->channels()) {
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      XLS_ASSIGN_OR_RETURN(virtual_channels[ch],
                           VirtualChannel::Create(ch, top));
    }
  }

  std::vector<Proc*> procs_to_inline = GetProcsToInline(top);
  if (procs_to_inline.empty()) {
    return false;
  }

  std::vector<ProcThread> proc_threads;

  // Convert send/receives in top proc which communicate with other procs into
  // virtual send/receives.
  XLS_ASSIGN_OR_RETURN(ProcThread top_thread,
                       ConvertToProcThread(top, virtual_channels));
  proc_threads.push_back(std::move(top_thread));

  // Inline each proc into `top`. Sends/receives are converted to virtual
  // send/receives.
  // TODO(meheff): 2022/02/11 Add analysis which determines whether inlining is
  // a legal transformation.
  for (Proc* proc : procs_to_inline) {
    XLS_ASSIGN_OR_RETURN(ProcThread proc_thread,
                         InlineProcAsProcThread(proc, top, virtual_channels));
    proc_threads.push_back(std::move(proc_thread));
  }

  XLS_VLOG(3) << "After inlining procs:\n" << p->DumpIr();

  // Gather all inlined proc state and proc thread book-keeping bits and add to
  // the top-level proc state.
  std::vector<StateElement> state_elements;

  // Add the inlined (and top) proc state and activation bits.
  for (const ProcThread& proc_thread : proc_threads) {
    for (const StateElement& state_element : proc_thread.GetStateElements()) {
      state_elements.push_back(state_element);
    }
  }

  // Add virtual channel state.
  for (Channel* ch : p->channels()) {
    auto it = virtual_channels.find(ch);
    if (it != virtual_channels.end()) {
      state_elements.insert(state_elements.end(),
                            it->second.GetStateElements().begin(),
                            it->second.GetStateElements().end());
    }
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
    if (ch->supported_ops() == ChannelOps::kSendReceive || node->IsDead()) {
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
