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

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/token_provenance_analysis.h"

namespace xls {
namespace {

// Makes and returns a node computing Not(node).
absl::StatusOr<Node*> Not(Node* node,
                          std::optional<std::string_view> name = std::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kNot, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node, Op::kNot);
}

// Makes and returns a node computing Identity(node).
absl::StatusOr<Node*> Identity(
    Node* node, std::optional<std::string_view> name = std::nullopt) {
  if (name.has_value()) {
    return node->function_base()->MakeNodeWithName<UnOp>(
        node->loc(), node, Op::kIdentity, name.value());
  }
  return node->function_base()->MakeNode<UnOp>(node->loc(), node,
                                               Op::kIdentity);
}

// Makes and returns a node computing And(a, b).
absl::StatusOr<Node*> And(Node* a, Node* b,
                          std::optional<std::string_view> name = std::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kAnd, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kAnd);
}

// Makes and returns a node computing Or(a, b).
absl::StatusOr<Node*> Or(Node* a, Node* b,
                         std::optional<std::string_view> name = std::nullopt) {
  if (name.has_value()) {
    return a->function_base()->MakeNodeWithName<NaryOp>(
        a->loc(), std::vector<Node*>{a, b}, Op::kOr, name.value());
  }
  return a->function_base()->MakeNode<NaryOp>(
      a->loc(), std::vector<Node*>{a, b}, Op::kOr);
}

// Makes and returns a node computing And(a, !b).
absl::StatusOr<Node*> AndNot(
    Node* a, Node* b, std::optional<std::string_view> name = std::nullopt) {
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

absl::StatusOr<Node*> Implies(
    Node* antecedant, Node* consequence,
    std::optional<std::string_view> name = std::nullopt) {
  // p->q === !p V q
  XLS_ASSIGN_OR_RETURN(Node * not_antecedant, Not(antecedant));
  XLS_ASSIGN_OR_RETURN(Node * implication, Or(not_antecedant, consequence));
  if (name.has_value()) {
    implication->SetName(name.value());
  }
  return implication;
}

// Replace uses of `node` with `node || pred`.
absl::StatusOr<Node*> OrWithPredicate(Node* node, Node* pred) {
  std::vector<Node*> orig_users(node->users().begin(), node->users().end());
  XLS_ASSIGN_OR_RETURN(
      Node * new_or, node->function_base()->MakeNode<NaryOp>(
                         node->loc(), std::vector<Node*>{node, pred}, Op::kOr));
  for (Node* user : orig_users) {
    user->ReplaceOperand(node, new_or);
  }
  return new_or;
}

// Adds the given predicate to the send. If the send already has a predicate
// then the new predicate will be `old_predicate` AND `predicate`, otherwise the
// send is replaced with a new send with `predicate` as the predicate.
absl::StatusOr<Send*> AddSendPredicate(Send* send, Node* predicate) {
  if (send->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                         And(send->predicate().value(), predicate));
    XLS_ASSIGN_OR_RETURN(int64_t predicate_op,
                         send->predicate_operand_number());
    XLS_RETURN_IF_ERROR(
        send->ReplaceOperandNumber(predicate_op, new_predicate));
    return send;
  }
  XLS_ASSIGN_OR_RETURN(Send * new_send, send->ReplaceUsesWithNew<Send>(
                                            send->token(), send->data(),
                                            predicate, send->channel_name()));
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
    XLS_ASSIGN_OR_RETURN(int64_t predicate_op,
                         receive->predicate_operand_number());
    XLS_RETURN_IF_ERROR(
        receive->ReplaceOperandNumber(predicate_op, new_predicate));
    return receive;
  }
  XLS_ASSIGN_OR_RETURN(Receive * new_receive,
                       receive->ReplaceUsesWithNew<Receive>(
                           receive->token(), predicate, receive->channel_name(),
                           receive->is_blocking()));
  return new_receive;
}

// An abstraction defining a element of the proc state. These elements are
// gathered together in a tuple to define the state of the proc.
class StateElement {
 public:
  static absl::StatusOr<StateElement> Create(std::string_view name,
                                             Value initial_value, Proc* proc) {
    StateElement element;
    element.name_ = name;
    element.initial_value_ = initial_value;

    // Create placeholder state node. Later this will be replaced with an
    // element from the top-level proc state.
    XLS_ASSIGN_OR_RETURN(
        element.state_,
        proc->MakeNodeWithName<Literal>(SourceInfo(), initial_value,
                                        absl::StrFormat("%s_state", name)));

    // Create placeholder next-tick value for the state element. The caller must
    // call SetNext to replace it with a real state element.
    XLS_ASSIGN_OR_RETURN(element.next_, proc->MakeNodeWithName<Literal>(
                                            SourceInfo(), initial_value,
                                            absl::StrFormat("%s_next", name)));
    element.next_is_set_ = false;
    return std::move(element);
  }

  std::string_view GetName() const { return name_; }
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
// necessarily SendReceive channels.
//
// The virtual channel logic has the following inputs:
//
//   send_fired: whether the send associated with this channel fired (activation
//               bit and predicate both true).
//
//   data_in: the data value to be sent over the channel.
//
// and the following outputs:
//
//   valid: whether the virtual channel contains data.
//
//   data_out: the data value to be received over the channel
//

class VirtualChannel {
 public:
  static absl::StatusOr<VirtualChannel> Create(Channel* channel, Proc* proc) {
    XLS_RET_CHECK_EQ(channel->supported_ops(), ChannelOps::kSendReceive);
    VirtualChannel vc;
    vc.channel_ = channel;

    // Construct placeholder inputs. These will be replaced later.
    XLS_ASSIGN_OR_RETURN(vc.data_in_,
                         proc->MakeNodeWithName<Literal>(
                             SourceInfo(), ZeroOfType(channel->type()),
                             absl::StrFormat("%s_data_in", channel->name())));
    vc.is_send_attached_ = false;
    XLS_ASSIGN_OR_RETURN(
        vc.any_send_fired_,
        proc->MakeNodeWithName<Literal>(
            SourceInfo(), Value(UBits(0, 1)),
            absl::StrFormat("%s_send_fired", channel->name())));
    XLS_ASSIGN_OR_RETURN(
        vc.any_receive_fired_,
        proc->MakeNodeWithName<Literal>(
            SourceInfo(), Value(UBits(0, 1)),
            absl::StrFormat("%s_any_receive_fired_in", channel->name())));

    if (channel->kind() == ChannelKind::kStreaming) {
      StreamingChannel* streaming_channel =
          down_cast<StreamingChannel*>(channel);
      if (!streaming_channel->GetFifoDepth().has_value() ||
          streaming_channel->GetFifoDepth().value() > 1) {
        return absl::UnimplementedError(absl::StrFormat(
            "Only streaming channels of FIFO depth one or less are supported "
            "in proc inlining. Channel `%s` has depth: %s",
            streaming_channel->name(),
            streaming_channel->GetFifoDepth().has_value()
                ? absl::StrCat(streaming_channel->GetFifoDepth().value())
                : "(none)"));
      }
      if (streaming_channel->GetFifoDepth().value() == 1) {
        // For FIFO depth 1, create a state element which holds the data and one
        // which indicates whether the data is valid.
        XLS_ASSIGN_OR_RETURN(
            vc.saved_data_,
            StateElement::Create(absl::StrFormat("%s_data", channel->name()),
                                 ZeroOfType(channel->type()), proc));
        XLS_ASSIGN_OR_RETURN(
            vc.saved_data_valid_,
            StateElement::Create(
                absl::StrFormat("%s_data_valid", channel->name()),
                Value(UBits(0, 1)), proc));

        // This data state element is written when `send_fired` is asserted.
        XLS_ASSIGN_OR_RETURN(
            Node * data_next,
            proc->MakeNodeWithName<Select>(
                SourceInfo(), /*selector=*/vc.any_send_fired_, /*cases=*/
                std::vector<Node*>{vc.saved_data_->GetState(), vc.data_in_},
                /*default_case=*/std::nullopt,
                absl::StrFormat("%s_data_next", channel->name())));
        XLS_RETURN_IF_ERROR(vc.saved_data_->SetNext(data_next));

        // The data out of the virtual channel is:
        //
        // data_out = data_valid ? data_in : data_state
        XLS_ASSIGN_OR_RETURN(
            vc.data_out_,
            proc->MakeNodeWithName<Select>(
                SourceInfo(),
                /*selector=*/vc.saved_data_valid_->GetState(), /*cases=*/
                std::vector<Node*>{vc.data_in_, vc.saved_data_->GetState()},
                /*default_case=*/std::nullopt,
                absl::StrFormat("%s_data_out", channel->name())));

        // Logic for the data valid bit state:
        //
        //  data_valid_next = send && data_valid  ||
        //                    (send || data_valid) && !receive
        XLS_ASSIGN_OR_RETURN(
            Node * send_and_valid,
            And(vc.GetSendFiredInput(), vc.saved_data_valid_->GetState()));
        XLS_ASSIGN_OR_RETURN(
            Node * send_or_valid,
            Or(vc.GetSendFiredInput(), vc.saved_data_valid_->GetState()));
        XLS_ASSIGN_OR_RETURN(Node * tmp,
                             AndNot(send_or_valid, vc.GetReceiveFiredInput()));
        XLS_ASSIGN_OR_RETURN(
            Node * valid_next,
            Or(send_and_valid, tmp,
               absl::StrFormat("%s_data_valid_next", channel->name())));
        XLS_RETURN_IF_ERROR(vc.saved_data_valid_->SetNext(valid_next));

        XLS_ASSIGN_OR_RETURN(
            vc.valid_out_,
            Or(vc.any_send_fired_, vc.saved_data_valid_->GetState(),
               absl::StrFormat("%s_valid", channel->name())));
      } else {
        // For FIFO depth 0, just pass data through from `data_in` to
        // `data_out`.
        XLS_ASSIGN_OR_RETURN(
            vc.data_out_,
            Identity(vc.data_in_,
                     absl::StrFormat("%s_data_out", channel->name())));
        XLS_ASSIGN_OR_RETURN(
            vc.valid_out_,
            Identity(vc.any_send_fired_,
                     absl::StrFormat("%s_valid", channel->name())));
      }
    } else {
      XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kSingleValue);
      XLS_ASSIGN_OR_RETURN(vc.valid_out_,
                           proc->MakeNodeWithName<Literal>(
                               SourceInfo(), Value(UBits(1, 1)),
                               absl::StrFormat("%s_valid", channel->name())));
      XLS_ASSIGN_OR_RETURN(
          vc.data_out_,
          Identity(vc.data_in_,
                   absl::StrFormat("%s_data_out", channel->name())));
    }

    return std::move(vc);
  }

  Channel* GetChannel() const { return channel_; }

  // Attaches a send operation to this virtual channel. `data` is the data to be
  // sent on the channel, and `send_fired` is the signal that the send occured
  // (activated and predicate (if any) is true).
  absl::Status AttachSend(Node* data, Node* send_fired) {
    if (!is_send_attached_) {
      // Create a identity node for data_in. Then when later sends are attached,
      // they can call ReplaceUsesWith on data_in_ without perturbing the rest
      // of the graph.
      XLS_ASSIGN_OR_RETURN(Node * data_clone, Identity(data));

      // No send has been attached to the virtual channel. Connect the send data
      // to the data input.
      XLS_RETURN_IF_ERROR(data_in_->ReplaceUsesWith(data_clone));
      data_in_ = data_clone;

      XLS_RETURN_IF_ERROR(any_send_fired_->ReplaceUsesWith(send_fired));
      any_send_fired_ = send_fired;
    } else {
      // One or more sends have already been attached, add a select operation to
      // conditionally select this send's data as the channel data.
      XLS_ASSIGN_OR_RETURN(Node * select,
                           data->function_base()->MakeNode<Select>(
                               data->loc(), /*selector=*/send_fired, /*cases=*/
                               std::vector<Node*>{data_in_, data},
                               /*default_case=*/std::nullopt));
      XLS_RETURN_IF_ERROR(data_in_->ReplaceUsesWith(select));
      data_in_ = select;

      XLS_ASSIGN_OR_RETURN(any_send_fired_,
                           OrWithPredicate(any_send_fired_, send_fired));
    }
    is_send_attached_ = true;
    return absl::OkStatus();
  }

  // Attaches a receive operation to this virtual channel. `receive_fired` is
  // the signal that the receive occured (activated and predicate (if any) is
  // true).
  absl::Status AttachReceive(Node* receive_fired) {
    XLS_ASSIGN_OR_RETURN(any_receive_fired_,
                         OrWithPredicate(any_receive_fired_, receive_fired));
    return absl::OkStatus();
  }

  // Add a kAssert operation which fires if data is lost.
  absl::StatusOr<Node*> AddDataLossAssertion(Node* token) {
    if (channel_->kind() != ChannelKind::kStreaming) {
      return token;
    }
    // For streaming channels which add an assert which fires if data is
    // dropped. There are two different cases:
    //
    //  Stateless channel (fifo-depth 0): data loss if send fired but receive
    //    did not.
    //
    //  Stateful channel (fifo-depth 1): data loss if send fired and channel has
    //    data but receive did not fire.
    StreamingChannel* streaming_channel =
        down_cast<StreamingChannel*>(channel_);
    XLS_RET_CHECK(streaming_channel->GetFifoDepth().has_value());
    XLS_RET_CHECK_LE(streaming_channel->GetFifoDepth().value(), 1);
    Node* data_loss;
    if (streaming_channel->GetFifoDepth().value() == 0) {
      XLS_ASSIGN_OR_RETURN(
          data_loss, AndNot(GetValidOut(), GetReceiveFiredInput(),
                            absl::StrFormat("%s_data_loss", channel_->name())));
    } else {
      // FIFO depth 1.
      XLS_ASSIGN_OR_RETURN(
          Node * send_fired_and_channel_has_data,
          And(GetSendFiredInput(), ChannelValidState()->GetState()));
      XLS_ASSIGN_OR_RETURN(
          data_loss,
          AndNot(send_fired_and_channel_has_data, GetReceiveFiredInput(),
                 absl::StrFormat("%s_data_loss", channel_->name())));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * no_data_loss,
        Not(data_loss, absl::StrFormat("%s_no_data_loss", channel_->name())));
    return data_in_->function_base()->MakeNode<Assert>(
        SourceInfo(), token, no_data_loss,
        /*message=*/
        absl::StrFormat("Channel %s lost data, send fired but receive did not",
                        channel_->name()),
        /*label=*/
        absl::StrFormat("%s_data_loss_assert", channel_->name()),
        /*original_label=*/std::nullopt);
  }

  // Returns the signal indicating whether any send/receive fired this tick.
  Node* GetSendFiredInput() const { return any_send_fired_; }
  Node* GetReceiveFiredInput() const { return any_receive_fired_; }

  // Returns various outputs from the virtual channel.
  Node* GetDataOut() const { return data_out_; }
  Node* GetValidOut() const { return valid_out_; }

  const std::optional<StateElement>& ChannelDataState() const {
    return saved_data_;
  }
  const std::optional<StateElement>& ChannelValidState() const {
    return saved_data_valid_;
  }

 private:
  Channel* channel_;

  // Data input to the virtual channel.
  Node* data_in_;

  // Whether any of the send/receive operations attached to this channel have
  // fired.
  Node* any_send_fired_;
  Node* any_receive_fired_;

  // Whether a send operation has been attached to the virtual channel.
  bool is_send_attached_;

  // Outputs of the virtual channel.
  Node* data_out_;
  Node* valid_out_;

  // For FIFO depth 1 channels, these state elements holds the channel data and
  // whether it is valid.
  std::optional<StateElement> saved_data_;
  std::optional<StateElement> saved_data_valid_;
};

// Abstraction representing a node in the activation network of a proc
// thread. An activation bit is passed along through the network to execute the
// operations corresponding to the activation node.
struct ActivationNode {
  std::string name;

  // The side-effecting node in the inlined proc for which this activation node
  // is generated. May be nullopt for the dummy sink node of the activation
  // network.
  std::optional<Node*> original_node;

  // The activation bit(s) passed from the predecessor nodes(s) in the
  // activation network on the proc thread.
  std::vector<Node*> activations_in;

  // The outgoing activation bit. This will be wired to the successor
  // operation(s) in the activation network of the proc thread.
  Node* activation_out;

  // An optional placeholder node which represents the condition under which
  // this activation node will *not* fire even if it has the activation bit.
  std::optional<Node*> stall_condition;

  // If `original_node` is present, this is the inlined value (in the container
  // proc) corresponding to `original_node`.
  std::optional<Node*> data_out;
};

// Verifies that any data-dependency from a receive node to any other
// side-effecting node also includes a token path. This is required by proc
// inlining to ensure the receive executes before the dependent side-effecting
// node in the proc thread after inlining.
absl::Status VerifyTokenDependencies(Proc* proc) {
  // For each node in the proc, this is the set of receives which are
  // predecessors of the node.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Receive*>> predecessors;
  // For each node in the proc, this is the set of receives which are
  // predecessors of the node along paths with tokens.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Receive*>> token_predecessors;

  XLS_VLOG(4) << "VerifyTokenDependencies:";
  for (Node* node : TopoSort(proc)) {
    predecessors[node] = {};
    token_predecessors[node] = {};
    for (Node* operand : node->operands()) {
      predecessors[node].insert(predecessors.at(operand).begin(),
                                predecessors.at(operand).end());
      if (TypeHasToken(operand->GetType())) {
        token_predecessors[node].insert(token_predecessors.at(operand).begin(),
                                        token_predecessors.at(operand).end());
      }
    }

    XLS_VLOG(4) << absl::StrFormat("Receive predecessors of %s      : {%s}",
                                   node->GetName(),
                                   absl::StrJoin(predecessors.at(node), ", "));
    XLS_VLOG(4) << absl::StrFormat(
        "Receive token predecessors of %s: {%s}", node->GetName(),
        absl::StrJoin(token_predecessors.at(node), ", "));

    // For side-effecting operations, the receives which are data predecessors
    // of `node` must be a subset of the token predecessors of `node`.
    if (TypeHasToken(node->GetType()) && OpIsSideEffecting(node->op())) {
      for (Receive* predecessor : predecessors.at(node)) {
        if (!token_predecessors.at(node).contains(predecessor)) {
          return absl::UnimplementedError(
              absl::StrFormat("Node %s is data-dependent on receive %s but no "
                              "token path exists between them",
                              node->GetName(), predecessor->GetName()));
        }
      }
    }

    if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      predecessors[node].insert(receive);
      token_predecessors[node].insert(receive);
      continue;
    }
  }
  return absl::OkStatus();
}

// Visitor which computes the data dependencies of `Receive`s. Specifically, for
// each element of each node the visitor computes which `Receive`s the element
// is data dependent upon. This is stored as a std::vector<Receive*> for each
// element of each node. For example, given:
//
// foobar = ...
// rcv0 = Receive(...)
// rcv1 = Receive(...)
// rcv0_data = TupleIndex(rcv0, 1)
// rcv1_data = TupleIndex(rcv0, 1)
// x = Tuple(rcv0_data, rcv0_data + rcv1_data, foobar)
//
// The leaf type tree for `x` might be:
//
//   ({rcv0}, {rcv0, rcv1}, {})
class ReceiveDependencyVisitor : public DataFlowVisitor<std::vector<Receive*>> {
 public:
  absl::Status DefaultHandler(Node* node) override {
    // Token-typed nodes have no receive data sources.
    if (node->GetType()->IsToken()) {
      return SetValue(node,
                      LeafTypeTree<std::vector<Receive*>>(node->GetType()));
    }
    // Otherwise, conservatively assume that the data source(s) of
    // an element of the output might come from any operand data source.
    LeafTypeTree<std::vector<Receive*>> ltt(node->GetType(),
                                            FlattenOperandVectors(node));
    return SetValue(node, ltt);
  }

  absl::Status HandleReceive(Receive* receive) override {
    // A Receive produces a two element tuple: (token, data). The source of
    // `data` is only this receive (obviously).
    LeafTypeTree<std::vector<Receive*>> ltt(receive->GetType(),
                                            {receive->As<Receive>()});
    // The token element is not considered a data dependency.
    ltt.Get({0}).clear();
    return SetValue(receive, ltt);
  }

  std::vector<Receive*> FlattenOperandVectors(Node* node) {
    absl::flat_hash_set<Receive*> elements;
    for (Node* operand : node->operands()) {
      for (const std::vector<Receive*>& receives :
           GetValue(operand).elements()) {
        elements.insert(receives.begin(), receives.end());
      }
    }
    std::vector<Receive*> element_vec = SetToSortedVector(elements);
    return element_vec;
  }
};

// Returns a map containing each receive and the nodes which may be dependent on
// the data value of that receive. Data paths are tracked precisely through
// Tuple and TupleIndex instructions but other node are handled conservatively
// (i.e., if an operand of node X is data dependent upon a receive then so is
// X).
//
// For example, for the following snippet:
//
//   a = param()
//   rcv0 = Receive(...)
//   rcv1 = Receive(...)
//   rcv2 = Receive(...)
//   b = a + rcv0
//   c = rcv0 + rcv1
//   d = a + 42
//
// The following map is returned:
//
//   {rcv0: [b, c], rcv1: [c], rcv2: []}
absl::StatusOr<absl::flat_hash_map<Receive*, std::vector<Node*>>>
GetReceiveDataDependencies(Proc* inlined_proc) {
  ReceiveDependencyVisitor visitor;
  XLS_RETURN_IF_ERROR(inlined_proc->Accept(&visitor));

  if (XLS_VLOG_IS_ON(3)) {
    XLS_VLOG(3) << absl::StrFormat("Receive data dependencies for proc %s:",
                                   inlined_proc->name());
    for (Node* node : TopoSort(inlined_proc)) {
      XLS_VLOG(3) << absl::StreamFormat(
          "  %s : %s", node->GetName(),
          visitor.GetValue(node).ToString([](const std::vector<Receive*>& t) {
            return absl::StrFormat("{%s}", absl::StrJoin(t, ", "));
          }));
    }
  }

  absl::flat_hash_map<Receive*, std::vector<Node*>> result;

  for (Node* node : inlined_proc->nodes()) {
    // Ensure every receive operation has an entry in `result` even if it has no
    // data dependent operations.
    if (node->Is<Receive>()) {
      result[node->As<Receive>()];
    }
  }

  // Invert `data_sources` map to Receive* -> std::vector<Node*>.
  for (Node* node : inlined_proc->nodes()) {
    for (Receive* receive : visitor.FlattenOperandVectors(node)) {
      result.at(receive).push_back(node);
    }
  }

  return std::move(result);
}

// Abstraction representing a proc thread. A proc thread contains the logic
// required to virtually evaluate a proc (the "inlined proc") within another
// proc (the "container proc"). An activation bit is threaded through the proc's
// virtual send/receive operations and keeps track of progress through the proc
// thread.
class ProcThread {
 public:
  // Creates and returns a proc thread which executes the given proc.
  // `container_proc` is the FunctionBase which will contain the proc thread.
  static absl::StatusOr<ProcThread> Create(Proc* inlined_proc,
                                           Proc* container_proc) {
    ProcThread proc_thread;
    proc_thread.inlined_proc_ = inlined_proc;
    proc_thread.container_proc_ = container_proc;

    XLS_RETURN_IF_ERROR(VerifyTokenDependencies(inlined_proc));

    // Create the state element to hold the state of the inlined proc.
    for (int64_t i = 0; i < inlined_proc->GetStateElementCount(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          StateElement * element,
          proc_thread.AllocateState(
              absl::StrFormat("%s_state", inlined_proc->name()),
              inlined_proc->GetInitValueElement(i)));
      proc_thread.proc_state_.push_back(element);
    }

    XLS_RETURN_IF_ERROR(proc_thread.CreateActivationNetwork());

    return std::move(proc_thread);
  }

  // Sets the condition under which the activation node will hold on to the
  // activation bit. This involves adding a bit to the proc state to indicate
  // that this activation node is stalled (holding on to the activation bit).
  absl::Status SetStallCondition(ActivationNode* activation_node, Node* stall) {
    XLS_RET_CHECK(!activation_node->stall_condition.has_value());
    XLS_RET_CHECK_EQ(activation_node->activations_in.size(), 1);
    Node* activation_in = activation_node->activations_in.front();

    XLS_ASSIGN_OR_RETURN(
        Node * not_stalled,
        Not(stall, absl::StrFormat("%s_not_stalled", activation_node->name)));
    XLS_ASSIGN_OR_RETURN(StateElement * activation_in_state,
                         AllocateState(absl::StrFormat("%s_holds_activation",
                                                       activation_node->name),
                                       Value(UBits(0, 1))));
    XLS_ASSIGN_OR_RETURN(
        Node * has_activation,
        Or(activation_in, activation_in_state->GetState(),
           absl::StrFormat("%s_has_activation", activation_node->name)));

    XLS_ASSIGN_OR_RETURN(Node * new_activation_out,
                         And(not_stalled, has_activation));

    XLS_RETURN_IF_ERROR(
        activation_node->activation_out->ReplaceUsesWith(new_activation_out));
    activation_node->activation_out = new_activation_out;

    XLS_ASSIGN_OR_RETURN(Node * holds_activation,
                         AndNot(has_activation, activation_node->activation_out,
                                absl::StrFormat("%s_holds_activation_next",
                                                activation_node->name)));
    XLS_RETURN_IF_ERROR(activation_in_state->SetNext(holds_activation));
    return absl::OkStatus();
  }

  ActivationNode* GetActivationNode(Node* node) {
    return original_node_to_activation_node_.at(node);
  }

  absl::StatusOr<std::string> GetActivationNodeName(Node* node) {
    if (node->Is<Send>() || node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      return absl::StrFormat("%s_%s", channel->name(),
                             node->Is<Send>() ? "send" : "receive");
    }
    return node->GetName();
  }

  // Construct the network of ActivationNodes for this proc. The topology of the
  // activation network mirrors the token connectivity of the side-effecting
  // nodes in the inlined proc. The activation network includes a single dummy
  // source and sink nodes as well.
  absl::Status CreateActivationNetwork() {
    XLS_VLOG(3) << "CreateActivationNetwork " << inlined_proc_->name();
    XLS_ASSIGN_OR_RETURN(std::vector<NodeAndPredecessors> token_graph,
                         ComputeTopoSortedTokenDAG(inlined_proc_));
    // Create the state element for the activation bit of the proc thread.
    XLS_ASSIGN_OR_RETURN(
        activation_state_,
        AllocateState(absl::StrFormat("%s_activation", inlined_proc_->name()),
                      Value(UBits(1, 1))));

    for (const NodeAndPredecessors& token_node : token_graph) {
      ActivationNode* activation_node;
      if (token_node.node->Is<Param>()) {
        XLS_RET_CHECK_EQ(token_node.node, inlined_proc_->TokenParam());
        XLS_ASSIGN_OR_RETURN(
            activation_node,
            AllocateActivationNode(
                absl::StrFormat("%s_activation_source", inlined_proc_->name()),
                /*activations_in=*/{activation_state_->GetState()},
                token_node.node));
        source_activation_node_ = activation_node;
      } else if (token_node.node->Is<Gate>()) {
        // Gate ops are weird, they're side-effecting to prevent optimizations
        // from removing them, but they have no token. They can be data
        // dependent on a receive without a token, so use the proc activation as
        // the activation.
        XLS_RET_CHECK(token_node.predecessors.empty());
        XLS_ASSIGN_OR_RETURN(std::string name,
                             GetActivationNodeName(token_node.node));
        XLS_ASSIGN_OR_RETURN(
            activation_node,
            AllocateActivationNode(name,
                                   {original_node_to_activation_node_
                                        .at(inlined_proc_->TokenParam())
                                        ->activation_out},
                                   token_node.node));
      } else {
        XLS_ASSIGN_OR_RETURN(std::string name,
                             GetActivationNodeName(token_node.node));
        std::vector<Node*> activation_preds;
        activation_preds.reserve(token_node.predecessors.size());
        for (Node* token_pred : token_node.predecessors) {
          activation_preds.push_back(
              original_node_to_activation_node_.at(token_pred)->activation_out);
        }
        XLS_ASSIGN_OR_RETURN(
            activation_node,
            AllocateActivationNode(name, activation_preds, token_node.node));
      }
      original_node_to_activation_node_[token_node.node] = activation_node;
    }

    ActivationNode* final_activation_node =
        original_node_to_activation_node_.at(token_graph.back().node);
    XLS_ASSIGN_OR_RETURN(
        sink_activation_node_,
        AllocateActivationNode(
            absl::StrFormat("%s_activation_sink", inlined_proc_->name()),
            /*activations_in=*/{final_activation_node->activation_out},
            std::nullopt));
    XLS_RETURN_IF_ERROR(
        activation_state_->SetNext(sink_activation_node_->activation_out));

    return absl::OkStatus();
  }

  // Allocate and return a state element. The state will later be added to the
  // container proc state.
  absl::StatusOr<StateElement*> AllocateState(std::string_view name,
                                              const Value& initial_value);

  // Allocates an activation node for the proc thread's activation network.
  // `activations_in` are the activation bits of the predecessors of this node
  // in the activation network. `original_node` is the (optional) node in the
  // inlined proc for which this activation node was created.
  absl::StatusOr<ActivationNode*> AllocateActivationNode(
      std::string_view name, absl::Span<Node* const> activations_in,
      std::optional<Node*> original_node);

  // Sets the next state of the proc thread to the given value. This value is
  // committed to the corresponding element of container proc when the proc
  // thread tick is complete.
  absl::Status SetNextState(absl::Span<Node* const> next_state) {
    XLS_VLOG(3) << "SetNextState proc thread for: " << inlined_proc_->name();

    XLS_RET_CHECK_EQ(next_state.size(), inlined_proc_->GetStateElementCount());

    for (int64_t i = 0; i < inlined_proc_->GetStateElementCount(); ++i) {
      // Add selector to commit state if the proc thread tick is complete.
      XLS_ASSIGN_OR_RETURN(
          Node * state_next,
          container_proc_->MakeNodeWithName<Select>(
              SourceInfo(), /*selector=*/GetProcTickComplete(), /*cases=*/
              std::vector<Node*>{GetDummyState(i), next_state.at(i)},
              /*default_case=*/std::nullopt,
              absl::StrFormat("%s_%s_next_state", inlined_proc_->name(),
                              inlined_proc_->GetStateParam(i)->GetName())));
      XLS_RETURN_IF_ERROR(proc_state_.at(i)->SetNext(state_next));
    }
    return absl::OkStatus();
  }

  // For each receive, returns the set of activation nodes which are data
  // dependent upon the receive. For example, if a receive R feeds a send S
  // (potentially indirectly) then the activation node of S will be in the
  // returned map entry for R. If the receive feeds a next-state node for the
  // proc, then the `sink_activation_node` will be in the respective vector of
  // activation nodes.
  absl::StatusOr<absl::flat_hash_map<Receive*, std::vector<ActivationNode*>>>
  GetDataDependentActivationNodes() {
    absl::flat_hash_map<Receive*, std::vector<Node*>> receive_data_deps;
    XLS_ASSIGN_OR_RETURN(receive_data_deps,
                         GetReceiveDataDependencies(inlined_proc_));

    absl::flat_hash_set<Node*> next_state_nodes(
        inlined_proc_->NextState().begin(), inlined_proc_->NextState().end());

    std::vector<Receive*> receives;
    for (auto [receive, nodes] : receive_data_deps) {
      receives.push_back(receive);
    }
    SortByNodeId(&receives);

    absl::flat_hash_map<Receive*, std::vector<ActivationNode*>> result;

    XLS_VLOG(3) << "Activation node users of receives:";
    for (Receive* receive : receives) {
      result[receive];
      for (Node* node : receive_data_deps.at(receive)) {
        if (original_node_to_activation_node_.contains(node)) {
          // `node` has an associated activation node which necessarily means it
          // is side-effecting.
          ActivationNode* activation_node =
              original_node_to_activation_node_.at(node);
          result[receive].push_back(activation_node);
        }
        if (next_state_nodes.contains(node)) {
          // `node` is one of the next-state nodes. This means that
          result[receive].push_back(sink_activation_node_);
        }
      }
      Channel* channel = GetChannelUsedByNode(receive).value();
      XLS_VLOG(3) << absl::StreamFormat(
          "  %s receive (%s) : %s", channel->name(), receive->GetName(),
          absl::StrJoin(result.at(receive), ", ",
                        [](std::string* out, ActivationNode* n) {
                          absl::StrAppend(out, n->name);
                        }));
    }
    return result;
  }

  // Add state as necessary to the proc to save the results of receives in the
  // proc thread. Received data must be saved to the proc state if the
  // side-effecting uses of the data may not necessarily be activated in the
  // same tick of the container proc state.
  absl::Status MaybeSaveReceivedData(const BddQueryEngine& query_engine) {
    // Determine which receive nodes (including virtual receive nodes) must save
    // their result as state. A receive must save its result as state if any
    // side-effecting users of the receive might not be activated in the same
    // tick as the receive. First identify the side-effecting users (as
    // activation nodes) of the receive.
    absl::flat_hash_map<Receive*, std::vector<ActivationNode*>> receive_deps;
    XLS_ASSIGN_OR_RETURN(receive_deps, GetDataDependentActivationNodes());

    // Use a BDD to determine whether the activation of a particular receive
    // necessarily implies the activation of every side-effecting user of the
    // receive. If not, then insert state to save the received data.
    for (const ActivationNode& anode : activation_nodes_) {
      if (!anode.original_node.has_value() ||
          !anode.original_node.value()->Is<Receive>()) {
        // Skip non-receive nodes.
        continue;
      }
      Receive* original_receive = anode.original_node.value()->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * channel,
                           GetChannelUsedByNode(anode.original_node.value()));
      XLS_VLOG(3) << absl::StreamFormat(
          "  Receive on channel %s (%s, %s, %d bits), users: %s",
          channel->name(), ChannelKindToString(channel->kind()),
          ChannelOpsToString(channel->supported_ops()),
          channel->type()->GetFlatBitCount(),
          absl::StrJoin(receive_deps.at(original_receive), ", ",
                        [](std::string* out, ActivationNode* n) {
                          absl::StrAppend(out, n->name);
                        }));

      if (channel->kind() == ChannelKind::kSingleValue) {
        // Data from single-value channels need not be saved. There are two
        // cases:
        //
        // (1) External single-value channels: values from external single-value
        //     channels are considered to be invariant (changing them during
        //     execution has undefined behavior) so no need to save them.
        //
        // (2) Internal single-value channels: we assume that the proc network
        //     is not racy. That is, order of the excution of the procs does not
        //     affect the visible output. Under this condition, all
        //     side-effecting uses of received data must fire before the
        //     corresponding send can execute again and potentially change the
        //     value on the channel. The source of the sent data must be one of
        //     three cases:
        //
        //       (a) data is invariant, so no need to save it.
        //
        //       (b) data is received from another channel. If the channel is
        //           single-value, then the same logic here applies to *that*
        //           channel.If the channel is streaming, then the data will be
        //           saved to proc state when considering that streaming channel
        //           receive in this function.
        //
        //       (c) data is from the proc state. In this case, the state will
        //           not be updated until before users all fire because the proc
        //           network is not racy.
        //
        // TODO(https://github.com/google/xls/issues/614): We should move away
        // from inlining single-value channels to avoid relying on these subtle
        // and potentially erroneous arguments. A potential alternative might be
        // "direct" channels mentioned in the github issue.
        XLS_VLOG(3)
            << "    No need to save data because channel is single-value.";
        continue;
      }

      bool save_data = false;
      for (ActivationNode* user_anode : receive_deps.at(original_receive)) {
        if (user_anode->original_node.has_value() &&
            user_anode->original_node.value()->Is<Send>()) {
          XLS_ASSIGN_OR_RETURN(
              Channel * send_channel,
              GetChannelUsedByNode(user_anode->original_node.value()));
          if (send_channel->kind() == ChannelKind::kSingleValue) {
            XLS_VLOG(3) << absl::StrFormat(
                "    Must save data because activation node %s is send on "
                "single-value channel %s",
                user_anode->name, send_channel->name());
            save_data = true;
            break;
          }
        }

        bool user_always_ready = query_engine.Implies(
            TreeBitLocation{anode.activation_out, 0},
            TreeBitLocation{user_anode->activation_out, 0});
        if (!user_always_ready) {
          XLS_VLOG(3) << absl::StrFormat(
              "    Must save data because activation node %s is not "
              "necessarily active in same tick as receive on channel %s",
              user_anode->name, channel->name());
          save_data = true;
          break;
        }
      }

      if (!save_data) {
        XLS_VLOG(3) << absl::StreamFormat(
            "    No need to save data. All users activated.");
        continue;
      }

      // Add state which saves the data value of the receive.
      XLS_ASSIGN_OR_RETURN(
          StateElement * data_state,
          AllocateState(absl::StrFormat("%s_data_state", channel->name()),
                        ZeroOfType(channel->type())));
      Node* receive_out = anode.data_out.value();
      std::vector<Node*> old_users(receive_out->users().begin(),
                                   receive_out->users().end());
      XLS_ASSIGN_OR_RETURN(Node * receive_data,
                           container_proc_->MakeNodeWithName<TupleIndex>(
                               SourceInfo(), receive_out, 1,
                               absl::StrFormat("%s_data", channel->name())));
      XLS_ASSIGN_OR_RETURN(
          Node * maybe_saved_data,
          container_proc_->MakeNodeWithName<Select>(
              SourceInfo(),
              /*selector=*/anode.activation_out, /*cases=*/
              std::vector<Node*>{data_state->GetState(), receive_data},
              /*default_case=*/std::nullopt,
              absl::StrFormat("%s_data", channel->name())));
      XLS_RETURN_IF_ERROR(data_state->SetNext(maybe_saved_data));

      XLS_ASSIGN_OR_RETURN(Node * receive_token,
                           container_proc_->MakeNodeWithName<TupleIndex>(
                               SourceInfo(), receive_out, 0,
                               absl::StrFormat("%s_token", channel->name())));
      std::vector<Node*> saved_receive_elements{receive_token,
                                                maybe_saved_data};
      if (!original_receive->is_blocking()) {
        XLS_ASSIGN_OR_RETURN(
            StateElement * valid_state,
            AllocateState(
                absl::StrFormat("%s_valid_state", channel->name()),
                ZeroOfType(container_proc_->package()->GetBitsType(1))));
        XLS_ASSIGN_OR_RETURN(Node * receive_valid,
                             container_proc_->MakeNodeWithName<TupleIndex>(
                                 SourceInfo(), receive_out, 2,
                                 absl::StrFormat("%s_valid", channel->name())));
        XLS_ASSIGN_OR_RETURN(
            Node * maybe_saved_valid,
            container_proc_->MakeNodeWithName<Select>(
                SourceInfo(),
                /*selector=*/anode.activation_out, /*cases=*/
                std::vector<Node*>{valid_state->GetState(), receive_valid},
                /*default_case=*/std::nullopt,
                absl::StrFormat("%s_valid", channel->name())));
        XLS_RETURN_IF_ERROR(valid_state->SetNext(maybe_saved_valid));
        saved_receive_elements.push_back(maybe_saved_valid);
      }
      XLS_CHECK_EQ(receive_out->operands().size(),
                   saved_receive_elements.size());
      XLS_ASSIGN_OR_RETURN(
          Node * saved_receive_out,
          container_proc_->MakeNodeWithName<Tuple>(
              SourceInfo(), saved_receive_elements,
              absl::StrFormat("%s_receive_out", channel->name())));

      for (Node* old_user : old_users) {
        old_user->ReplaceOperand(receive_out, saved_receive_out);
      }
    }
    return absl::OkStatus();
  }

  absl::StatusOr<Node*> ConvertSend(
      Send* send, ActivationNode* activation_node,
      absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
    XLS_RET_CHECK_EQ(send->function_base(), container_proc_);
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(send));
    XLS_VLOG(3) << absl::StreamFormat("Converting send %s on channel %s",
                                      send->GetName(), ch->name());

    Node* result;
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      // Non-external send. Convert send into a virtual send.
      XLS_ASSIGN_OR_RETURN(result, CreateVirtualSend(send, activation_node,
                                                     virtual_channels.at(ch)));
    } else if (ch->kind() == ChannelKind::kStreaming) {
      XLS_ASSIGN_OR_RETURN(result,
                           ConvertToActivatedSend(send, activation_node));
    } else {
      // Nothing to do for external single value channels.
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
      result = send;
    }

    activation_node->data_out = result;
    return result;
  }

  absl::StatusOr<Node*> ConvertReceive(
      Receive* receive, ActivationNode* activation_node,
      absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
    XLS_RET_CHECK_EQ(receive->function_base(), container_proc_);
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(receive));

    XLS_VLOG(3) << absl::StreamFormat("Converting receive %s on channel %s",
                                      receive->GetName(), ch->name());

    Node* result;
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      // Non-external receive. Convert receive into a virtual receive.
      XLS_ASSIGN_OR_RETURN(result,
                           CreateVirtualReceive(receive, activation_node,
                                                virtual_channels.at(ch)));
    } else if (ch->kind() == ChannelKind::kStreaming) {
      // TODO: fix
      // A streaming external channel needs to be added to the activation
      // chain. The receive can fire only when it is activated. `stallable` is
      // not set on this activation node because receive operation already has
      // blocking semantics in a proc so no need for additional stalling logic
      // (unlike the virtual receive in which there is no actual Op::kReceive
      // operation).
      XLS_ASSIGN_OR_RETURN(result,
                           ConvertToActivatedReceive(receive, activation_node));
    } else {
      XLS_RET_CHECK_EQ(ch->kind(), ChannelKind::kSingleValue);
      result = receive;
    }
    XLS_VLOG(3) << absl::StreamFormat("Receive %s converted to %s",
                                      receive->GetName(), result->GetName());

    activation_node->data_out = result;
    return result;
  }

  absl::StatusOr<Node*> ConvertNonChannelSideEffectingNode(
      Node* node, ActivationNode* activation_node) {
    XLS_RET_CHECK_EQ(node->function_base(), container_proc_);
    std::string op_name = OpToString(node->op());
    Node* activation_in = activation_node->activations_in.front();
    int64_t condition_operand_index;
    Node* new_condition;
    switch (node->op()) {
      case Op::kAssert: {
        condition_operand_index = Assert::kConditionOperand;
        // Assertion should only be checked when active (true otherwise), so
        // check that activation_in -> condition
        XLS_ASSIGN_OR_RETURN(
            new_condition,
            Implies(activation_in, node->As<Assert>()->condition()));
        break;
      }
      case Op::kCover: {
        condition_operand_index = Cover::kConditionOperand;
        XLS_ASSIGN_OR_RETURN(
            new_condition, And(node->As<Cover>()->condition(), activation_in));
        break;
      }
      case Op::kGate: {
        // Gate ops are weird, they're side-effecting to prevent optimizations
        // from removing them, but they have no token. They can be data
        // dependent on a receive without a token. Ideally, we'd set the
        // condition to activation_in & condition, but we don't have a token to
        // correctly build the activation. Instead, leave the condition alone
        // and just return the op.
        return node;
      }
      case Op::kTrace: {
        condition_operand_index = Trace::kConditionOperand;
        XLS_ASSIGN_OR_RETURN(
            new_condition, And(node->As<Trace>()->condition(), activation_in));
        break;
      }
      case Op::kSend:
      case Op::kReceive:
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unexpected channel operation %s.", node->GetName()));
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Op %s is not a side-effecting node supported by "
                            "proc inlining (%s).",
                            node->GetName(), op_name));
    }
    XLS_VLOG(3) << absl::StreamFormat("Converting %s %s to activated %s",
                                      op_name, node->GetName(), op_name);
    XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(
        condition_operand_index, new_condition, /*type_must_match=*/true));
    XLS_VLOG(3) << absl::StreamFormat("Converted %s %s condition to %s",
                                      op_name, node->GetName(),
                                      new_condition->GetName());
    activation_node->data_out = node;
    return node;
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

  // Returns the dummy node representing the proc state element in the proc
  // thread.
  Node* GetDummyState(int64_t state_index) const {
    return proc_state_.at(state_index)->GetState();
  }

  Proc* GetInlinedProc() const { return inlined_proc_; }

  // Returns the signal indicating that the tick of the proc thread is complete.
  Node* GetProcTickComplete() const {
    return sink_activation_node_->activation_out;
  }

 private:
  // The proc whose logic which this proc thread evaluates.
  Proc* inlined_proc_;

  // The actual proc in which this proc thread evaluates. The container proc may
  // simultaneously evaluate multiple proc threads.
  Proc* container_proc_;

  // The state elements required to by this proc thread. These elements are
  // later added to the container proc state.
  std::list<StateElement> state_elements_;

  // The state elements representing the state of the proc. These point to an
  // element in `state_elements_`.
  std::vector<StateElement*> proc_state_;

  // The single bit of state representing the activation bit. This points to an
  // element in `state_elements_`.
  StateElement* activation_state_ = nullptr;

  // The nodes of the activation network for the proc thread.
  std::list<ActivationNode> activation_nodes_;

  // The source and sink nodes of the activation network.
  ActivationNode* source_activation_node_;
  ActivationNode* sink_activation_node_;

  // A map from the nodes in the inlined proc to the respective activation
  // node. Only side-effecting ops will have an activation node.
  absl::flat_hash_map<Node*, ActivationNode*> original_node_to_activation_node_;
};

absl::StatusOr<StateElement*> ProcThread::AllocateState(
    std::string_view name, const Value& initial_value) {
  XLS_VLOG(3) << absl::StreamFormat(
      "AllocateState: %s, size: %d, initial value %s", name,
      initial_value.GetFlatBitCount(), initial_value.ToString());
  XLS_ASSIGN_OR_RETURN(
      StateElement element,
      StateElement::Create(name, initial_value, container_proc_));
  state_elements_.push_back(std::move(element));
  return &state_elements_.back();
}

absl::StatusOr<ActivationNode*> ProcThread::AllocateActivationNode(
    std::string_view name, absl::Span<Node* const> activations_in,
    std::optional<Node*> original_node) {
  XLS_VLOG(3) << absl::StreamFormat("AllocateActivationNode: %s, inputs (%s)",
                                    name, absl::StrJoin(activations_in, ", "));

  ActivationNode activation_node;
  activation_node.name = name;
  activation_node.original_node = original_node;
  XLS_RET_CHECK(!activations_in.empty());
  for (int64_t i = 0; i < activations_in.size(); ++i) {
    XLS_RET_CHECK(activations_in[i]->function_base() == container_proc_);
    XLS_ASSIGN_OR_RETURN(
        Node * activation_in,
        Identity(activations_in[i],
                 absl::StrFormat("%s_activation_in_%d", name, i)));
    activation_node.activations_in.push_back(activation_in);
  }

  if (activations_in.size() == 1) {
    XLS_RET_CHECK(activations_in.front()->function_base() == container_proc_);
    XLS_ASSIGN_OR_RETURN(activation_node.activation_out,
                         Identity(activations_in.front(),
                                  absl::StrFormat("%s_activation_out", name)));
  } else {
    XLS_ASSIGN_OR_RETURN(activation_node.stall_condition,
                         container_proc_->MakeNodeWithName<Literal>(
                             SourceInfo(), Value(UBits(0, 1)),
                             absl::StrFormat("%s_stall_condition", name)));

    XLS_ASSIGN_OR_RETURN(Node * not_stalled,
                         Not(activation_node.stall_condition.value(),
                             absl::StrFormat("%s_not_stalled", name)));
    std::vector<Node*> activation_conditions = {not_stalled};
    std::vector<Node*> has_activations;
    std::vector<StateElement*> activations_in_state;
    for (int64_t i = 0; i < activations_in.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          StateElement * activation_in_state,
          AllocateState(absl::StrFormat("%s_holds_activation_%d",
                                        activation_node.name, i),
                        Value(UBits(0, 1))));
      activations_in_state.push_back(activation_in_state);

      XLS_ASSIGN_OR_RETURN(
          Node * has_activation,
          Or(activations_in[i], activation_in_state->GetState(),
             absl::StrFormat("%s_has_activation_%d", name, i)));

      activation_conditions.push_back(has_activation);
      has_activations.push_back(has_activation);
    }
    XLS_ASSIGN_OR_RETURN(activation_node.activation_out,
                         container_proc_->MakeNodeWithName<NaryOp>(
                             SourceInfo(), activation_conditions, Op::kAnd,
                             absl::StrFormat("%s_is_activated", name)));

    // Each activation input is held until activation out is asserted.
    for (int64_t i = 0; i < activations_in.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * holds_activation,
          AndNot(has_activations[i], activation_node.activation_out,
                 absl::StrFormat("%s_holds_activation_%d_next",
                                 activation_node.name, i)));
      XLS_RETURN_IF_ERROR(activations_in_state[i]->SetNext(holds_activation));
    }
  }

  activation_nodes_.push_back(activation_node);
  return &activation_nodes_.back();
}

absl::StatusOr<Node*> ProcThread::CreateVirtualSend(
    Send* send, ActivationNode* activation_node,
    VirtualChannel& virtual_channel) {
  XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(send));
  if (channel->kind() == ChannelKind::kSingleValue &&
      send->predicate().has_value()) {
    return absl::UnimplementedError(
        "Conditional send on single-value channels are not supported");
  }

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
  XLS_RETURN_IF_ERROR(virtual_channel.AttachSend(send->data(), send_fired));

  XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(send->token()));
  return send->token();
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
  // Only used for non-blocking receives.
  Node* receive_valid = nullptr;
  std::string stall_name = absl::StrFormat("%s_receive_stall", channel->name());
  std::string receive_fired_name =
      absl::StrFormat("%s_receive_fired", channel->name());
  std::string receive_valid_name =
      absl::StrFormat("%s_receive_valid", channel->name());
  if (receive->predicate().has_value()) {
    if (receive->is_blocking()) {
      XLS_ASSIGN_OR_RETURN(stall,
                           AndNot(receive->predicate().value(),
                                  virtual_channel.GetValidOut(), stall_name));
    } else {
      XLS_ASSIGN_OR_RETURN(stall,
                           container_proc_->MakeNodeWithName<Literal>(
                               SourceInfo(), Value(UBits(0, 1)), stall_name));
      XLS_ASSIGN_OR_RETURN(receive_valid, And(receive->predicate().value(),
                                              virtual_channel.GetValidOut(),
                                              receive_valid_name));
    }
    XLS_ASSIGN_OR_RETURN(receive_fired,
                         And(activation_node->activation_out,
                             receive->predicate().value(), receive_fired_name));
  } else {
    if (receive->is_blocking()) {
      // Receive is unconditional. The activation is stalled iff the
      // !data_valid.
      XLS_ASSIGN_OR_RETURN(stall,
                           Not(virtual_channel.GetValidOut(), stall_name));
    } else {
      // Receive is unconditional. The activation is never stalled for
      // non-blocking.
      XLS_ASSIGN_OR_RETURN(stall,
                           container_proc_->MakeNodeWithName<Literal>(
                               SourceInfo(), Value(UBits(0, 1)), stall_name));
      XLS_ASSIGN_OR_RETURN(
          receive_valid,
          Identity(virtual_channel.GetValidOut(), receive_valid_name));
    }
    XLS_ASSIGN_OR_RETURN(
        receive_fired,
        Identity(activation_node->activation_out, receive_fired_name));
  }
  XLS_RETURN_IF_ERROR(virtual_channel.AttachReceive(receive_fired));
  XLS_RETURN_IF_ERROR(SetStallCondition(activation_node, stall));

  // If the receive has not fired, the data value presented to uses of the
  // receive should be a literal zero. This follows the semantics of a
  // conditional receive.
  // TODO(meheff): 2022/05/11 Consider whether this select is only needed if
  // original receive is conditional.
  XLS_ASSIGN_OR_RETURN(Node * zero,
                       container_proc_->MakeNodeWithName<Literal>(
                           SourceInfo(), ZeroOfType(channel->type()),
                           absl::StrFormat("%s_zero", channel->name())));
  XLS_ASSIGN_OR_RETURN(
      Node * data, container_proc_->MakeNodeWithName<Select>(
                       SourceInfo(),
                       /*selector=*/receive_fired, /*cases=*/
                       std::vector<Node*>{zero, virtual_channel.GetDataOut()},
                       /*default_case=*/std::nullopt,
                       absl::StrFormat("%s_receive_data", channel->name())));

  std::vector<Node*> result_tuple{receive->token(), data};
  if (!receive->is_blocking()) {
    // Add the valid signal as the third entry in a non-blocking receive.
    XLS_CHECK_NE(receive_valid, nullptr);
    result_tuple.push_back(receive_valid);
  }

  // The output of a receive operation is a tuple of (token, data).
  XLS_ASSIGN_OR_RETURN(Node * result,
                       container_proc_->MakeNodeWithName<Tuple>(
                           SourceInfo(), result_tuple,
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

  XLS_ASSIGN_OR_RETURN(Node * data,
                       container_proc_->MakeNodeWithName<TupleIndex>(
                           SourceInfo(), activated_receive, 1,
                           absl::StrFormat("%s_data_in", channel->name())));

  XLS_ASSIGN_OR_RETURN(Node * token, container_proc_->MakeNode<TupleIndex>(
                                         SourceInfo(), activated_receive, 0));
  std::vector<Node*> saved_receive_operands{token, data};
  if (!receive->is_blocking()) {
    XLS_ASSIGN_OR_RETURN(Node * valid,
                         container_proc_->MakeNodeWithName<TupleIndex>(
                             SourceInfo(), activated_receive, 2,
                             absl::StrFormat("%s_valid_in", channel->name())));
    saved_receive_operands.push_back(valid);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * saved_receive,
      container_proc_->MakeNodeWithName<Tuple>(
          SourceInfo(), saved_receive_operands,
          absl::StrFormat("%s_saved_receive", channel->name())));

  // Replace uses of the original receive with the newly created receive.
  for (Node* old_user : old_users) {
    old_user->ReplaceOperand(activated_receive, saved_receive);
  }

  return saved_receive;
}

// Inlines the given proc into `container_proc` as a proc thread. Sends and
// receives in `proc` are replaced with virtual sends/receives which execute via
// the activation chain. Newly created virtual send/receieves are inserted into
// the `virtual_send` and `virtual_receive` maps.
absl::StatusOr<ProcThread> InlineProcAsProcThread(
    Proc* proc_to_inline, Proc* container_proc,
    absl::flat_hash_map<Channel*, VirtualChannel>& virtual_channels) {
  auto topo_sort = TopoSort(proc_to_inline);
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

  for (Node* node : topo_sort) {
    XLS_VLOG(3) << absl::StreamFormat("Inlining node %s", node->GetName());
    if (node->Is<Param>()) {
      if (node == proc_to_inline->TokenParam()) {
        // Connect the inlined token network from `proc` to the token parameter
        // of `container_proc`.
        XLS_RET_CHECK_EQ(node, proc_to_inline->TokenParam());
        node_map[node] = container_proc->TokenParam();
      } else {
        // The dummy state value will later be replaced with an element from the
        // container proc state.
        XLS_ASSIGN_OR_RETURN(
            int64_t state_index,
            proc_to_inline->GetStateParamIndex(node->As<Param>()));
        node_map[node] = proc_thread.GetDummyState(state_index);
      }
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * cloned_node, clone_node(node));

    if (node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(
          Node * converted_send,
          proc_thread.ConvertSend(cloned_node->As<Send>(),
                                  proc_thread.GetActivationNode(node),
                                  virtual_channels));
      node_map[node] = converted_send;
    } else if (node->Is<Receive>()) {
      XLS_ASSIGN_OR_RETURN(
          Node * converted_receive,
          proc_thread.ConvertReceive(cloned_node->As<Receive>(),
                                     proc_thread.GetActivationNode(node),
                                     virtual_channels));
      node_map[node] = converted_receive;
    } else if (OpIsSideEffecting(node->op())) {
      XLS_ASSIGN_OR_RETURN(
          Node * converted_node,
          proc_thread.ConvertNonChannelSideEffectingNode(
              cloned_node, proc_thread.GetActivationNode(node)));
      node_map[node] = converted_node;
    } else {
      node_map[node] = cloned_node;
    }
  }

  std::vector<Node*> next_state;
  for (int64_t i = 0; i < proc_to_inline->GetStateElementCount(); ++i) {
    next_state.push_back(node_map.at(proc_to_inline->GetNextStateElement(i)));
  }
  XLS_RETURN_IF_ERROR(proc_thread.SetNextState(next_state));

  // Wire in the next-token value from the inlined proc into the next-token of
  // the container proc.
  XLS_RETURN_IF_ERROR(container_proc->JoinNextTokenWith(
      {node_map.at(proc_to_inline->NextToken())}));
  return std::move(proc_thread);
}

// Sets the state of `proc` to the given state elements.
absl::Status SetProcState(Proc* proc, absl::Span<const StateElement> elements) {
  std::vector<std::string> names;
  std::vector<Value> initial_values;
  std::vector<Node*> nexts;
  for (const StateElement& element : elements) {
    XLS_RET_CHECK(element.IsNextSet()) << element.GetName();
    initial_values.push_back(element.GetInitialValue());
    names.push_back(std::string{element.GetName()});
    nexts.push_back(element.GetNext());
  }
  XLS_RETURN_IF_ERROR(proc->ReplaceState(names, initial_values, nexts));

  for (int64_t i = 0; i < elements.size(); ++i) {
    XLS_RETURN_IF_ERROR(
        elements[i].GetState()->ReplaceUsesWith(proc->GetStateParam(i)));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ProcInliningPass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  if (!options.inline_procs || p->procs().size() <= 1) {
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

  std::vector<Proc*> procs_to_inline;
  procs_to_inline.reserve(p->procs().size());
  for (const std::unique_ptr<Proc>& proc : p->procs()) {
    procs_to_inline.push_back(proc.get());
  }

  {
    int64_t top_ii = 1;
    if (top_func_base->GetInitiationInterval().has_value()) {
      top_ii = top_func_base->GetInitiationInterval().value();
    }
    for (Proc* proc : procs_to_inline) {
      int64_t inner_ii = 1;
      if (proc->GetInitiationInterval().has_value()) {
        inner_ii = proc->GetInitiationInterval().value();
      }
      XLS_RET_CHECK_EQ(inner_ii, top_ii)
          << "Proc " << proc->name() << " had a different initiation interval "
          << "when compared to the top proc (" << top_func_base->name() << ")";
      if (!proc->next_values().empty()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Proc %s uses next_value nodes, which are not currently "
            "supported by proc inlining.",
            proc->name()));
      }
    }
  }

  Proc* container_proc =
      p->AddProc(std::make_unique<Proc>("__container", "tkn", p));

  // Gather all inlined proc state and proc thread book-keeping bits and add to
  // the top-level proc state.
  std::vector<StateElement> state_elements;

  absl::flat_hash_map<Channel*, VirtualChannel> virtual_channels;
  for (Channel* ch : p->channels()) {
    if (ch->supported_ops() == ChannelOps::kSendReceive) {
      XLS_ASSIGN_OR_RETURN(virtual_channels[ch],
                           VirtualChannel::Create(ch, container_proc));
      if (virtual_channels[ch].ChannelDataState().has_value()) {
        state_elements.push_back(
            virtual_channels[ch].ChannelDataState().value());
        state_elements.push_back(
            virtual_channels[ch].ChannelValidState().value());
      }
    }
  }

  std::vector<ProcThread> proc_threads;

  // Inline each proc into `container_proc`. Sends/receives are converted to
  // virtual send/receives.
  // TODO(meheff): 2022/02/11 Add analysis which determines whether inlining is
  // a legal transformation.
  for (Proc* proc : procs_to_inline) {
    XLS_ASSIGN_OR_RETURN(
        ProcThread proc_thread,
        InlineProcAsProcThread(proc, container_proc, virtual_channels));
    proc_threads.push_back(std::move(proc_thread));
  }

  XLS_VLOG(3) << "After inlining procs:\n" << p->DumpIr();

  // Changes were made to IsCheapForBdds which breaks proc inlining. Replicate
  // the old IsCheapForBdds function here. This doesn't matter because proc
  // inling is to be deleted soon.
  auto should_compute_with_bdds = [](const Node* node) {
    if (std::all_of(node->operands().begin(), node->operands().end(),
                    IsSingleBitType) &&
        IsSingleBitType(node)) {
      return true;
    }
    return (node->Is<NaryOp>() || node->Is<UnOp>() || node->Is<BitSlice>() ||
            node->Is<ExtendOp>() || node->Is<Concat>() ||
            node->Is<BitwiseReductionOp>() || node->Is<Literal>());
  };
  BddQueryEngine query_engine(static_cast<int64_t>(16 * 1024),
                              should_compute_with_bdds);
  XLS_RETURN_IF_ERROR(query_engine.Populate(container_proc).status());

  for (ProcThread& proc_thread : proc_threads) {
    XLS_RETURN_IF_ERROR(proc_thread.MaybeSaveReceivedData(query_engine));
  }

  // Add the inlined (and top) proc state and activation bits.
  for (const ProcThread& proc_thread : proc_threads) {
    for (const StateElement& state_element : proc_thread.GetStateElements()) {
      state_elements.push_back(state_element);
    }
  }

  // For each virtual channel, add an assertion which fires if data is dropped.
  std::vector<Node*> assertion_tokens;
  for (Channel* ch : p->channels()) {
    if (virtual_channels.contains(ch)) {
      XLS_ASSIGN_OR_RETURN(Node * assertion_token,
                           virtual_channels.at(ch).AddDataLossAssertion(
                               container_proc->TokenParam()));
      assertion_tokens.push_back(assertion_token);
    }
  }
  if (!assertion_tokens.empty()) {
    XLS_RETURN_IF_ERROR(container_proc->JoinNextTokenWith(assertion_tokens));
  }

  XLS_RETURN_IF_ERROR(SetProcState(container_proc, state_elements));

  XLS_VLOG(3) << "After transforming proc state:\n" << p->DumpIr();

  // Delete inlined procs.
  XLS_RETURN_IF_ERROR(p->SetTop(container_proc));
  std::string top_proc_name = top_func_base->AsProcOrDie()->name();
  for (Proc* proc : procs_to_inline) {
    XLS_RETURN_IF_ERROR(p->RemoveProc(proc));
  }
  container_proc->SetName(top_proc_name);

  // Delete send and receive nodes in top which were used for communicating with
  // the inlined procs.
  std::vector<Node*> to_remove;
  for (Node* node : container_proc->nodes()) {
    if (!IsChannelNode(node)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Channel * ch, GetChannelUsedByNode(node));
    if (ch->supported_ops() == ChannelOps::kSendReceive || node->IsDead()) {
      to_remove.push_back(node);
    }
  }
  for (Node* node : to_remove) {
    XLS_RETURN_IF_ERROR(container_proc->RemoveNode(node));
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

REGISTER_OPT_PASS(ProcInliningPass);

}  // namespace xls
