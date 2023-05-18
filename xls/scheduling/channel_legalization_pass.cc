// Copyright 2023 The XLS Authors
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

#include "xls/scheduling/channel_legalization_pass.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/compare.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/proc_inlining_pass.h"
#include "xls/passes/token_provenance_analysis.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

namespace {
struct MultipleChannelOps {
  absl::flat_hash_map<int64_t, absl::flat_hash_set<Send*>> multiple_sends;
  absl::flat_hash_map<int64_t, absl::flat_hash_set<Receive*>> multiple_receives;
};

// Find instances of multiple sends/recvs on a channel.
MultipleChannelOps FindMultipleChannelOps(Package* p) {
  MultipleChannelOps result;
  for (FunctionBase* fb : p->GetFunctionBases()) {
    for (Node* node : fb->nodes()) {
      if (node->Is<Send>()) {
        Send* send = node->As<Send>();
        absl::StatusOr<Channel*> channel = p->GetChannel(send->channel_id());
        XLS_VLOG(4) << "Found send " << send->ToString();
        result.multiple_sends[send->channel_id()].insert(send);
      }
      if (node->Is<Receive>()) {
        Receive* recv = node->As<Receive>();
        absl::StatusOr<Channel*> channel = p->GetChannel(recv->channel_id());
        result.multiple_receives[recv->channel_id()].insert(recv);
        XLS_VLOG(4) << "Found recv " << recv->ToString();
      }
    }
  }

  // Erase cases where there's only one send or receive.
  absl::erase_if(result.multiple_sends,
                 [](const std::pair<int64_t, absl::flat_hash_set<Send*>>& elt) {
                   return elt.second.size() < 2;
                 });
  absl::erase_if(
      result.multiple_receives,
      [](const std::pair<int64_t, absl::flat_hash_set<Receive*>>& elt) {
        return elt.second.size() < 2;
      });

  XLS_VLOG(4) << "After erasing single accesses, found "
              << result.multiple_sends.size() << " multiple send channels and "
              << result.multiple_receives.size()
              << " multiple receive channels.";

  return result;
}

// Comparator for FunctionBases based on name. Used to ensure the topo sort of
// the token DAG is stable.
struct FunctionBaseNameLess {
  absl::strong_ordering operator()(FunctionBase* a, FunctionBase* b) const {
    if (a == b) {
      return absl::strong_ordering::equivalent;
    }
    int cmp = a->name().compare(b->name());
    if (cmp < 0) {
      return absl::strong_ordering::less;
    }
    if (cmp > 0) {
      return absl::strong_ordering::greater;
    }
    return absl::strong_ordering::equal;
  }
};

// Get stable topo-sorted list of operations of type T. Each entry of the list
// is a node together with the set of predecessor nodes. Predecessors are
// resolved to the nearest node of type T. For example, a token chain that goes
// through send0 -> recv -> after_all -> send1 would return [send0: {}, send1:
// {send0}] when invoked with T=Send, and [recv: {}] when invoked with
// T=Receive.
// Note that token params are not included from predecessor lists.
template <typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
absl::StatusOr<std::vector<NodeAndPredecessors>> GetProjectedTokenDAG(
    const absl::flat_hash_set<T*>& operations,
    MultipleChannelOpsLegalizationStrictness strictness) {
  std::vector<NodeAndPredecessors> result;
  result.reserve(operations.size());

  // Use btree set that sorts FunctionBases by name to ensure stable order of
  // iteration through procs.
  absl::btree_set<FunctionBase*, FunctionBaseNameLess> fbs;
  for (const T* operation : operations) {
    fbs.insert(operation->function_base());
  }
  for (FunctionBase* fb : fbs) {
    XLS_ASSIGN_OR_RETURN(std::vector<NodeAndPredecessors> fbs_result,
                         ComputeTopoSortedTokenDAG(fb));
    // Resolve side-effecting nodes and after_alls.
    absl::flat_hash_map<Node* const, absl::flat_hash_set<Node*>> resolved_ops =
        // Initialize with proc token param having no predecessors.
        {{fb->AsProcOrDie()->TokenParam(), {}}};
    // Keep track of prev_node which will be used if we choose a stricter order.
    std::optional<Node*> prev_node = std::nullopt;
    for (NodeAndPredecessors& fb_result : fbs_result) {
      absl::flat_hash_set<Node*> resolved_predecessors;
      for (Node* predecessor : fb_result.predecessors) {
        // If a predecessor is not type T, resolve its predecessors of type T.
        if (!predecessor->Is<T>() ||
            !operations.contains(predecessor->As<T>())) {
          absl::flat_hash_set<Node*>& resolved = resolved_ops.at(predecessor);
          resolved_predecessors.insert(resolved.begin(), resolved.end());
          continue;
        }
        resolved_predecessors.insert(predecessor);
      }
      // If this entry in the DAG is not type T, save its resolved predecessors
      // for future resolution.
      if (!fb_result.node->Is<T>() ||
          !operations.contains(fb_result.node->As<T>())) {
        resolved_ops[fb_result.node] = std::move(resolved_predecessors);
        continue;
      }
      // If we choose an arbitrary static order, clear the predecessors and
      // chose the previous value. We're already iterating through in topo
      // sorted order, so this only strengthens the dependency relationship.
      if (strictness ==
          MultipleChannelOpsLegalizationStrictness::kArbitraryStaticOrder) {
        resolved_predecessors.clear();
        if (prev_node.has_value()) {
          if (!prev_node.value()->Is<T>() ||
              !operations.contains(prev_node.value()->As<T>())) {
            resolved_predecessors.insert(resolved_ops.at(*prev_node).begin(),
                                         resolved_ops.at(*prev_node).end());
          } else {
            resolved_predecessors.insert(prev_node.value());
          }
        }
      }
      result.push_back(NodeAndPredecessors{
          .node = fb_result.node,
          .predecessors = std::move(resolved_predecessors)});
      prev_node = fb_result.node;
    }
  }
  return result;
}

absl::Status CheckIsBlocking(Node* n) {
  if (n->Is<Receive>()) {
    XLS_RET_CHECK(n->As<Receive>()->is_blocking()) << absl::StreamFormat(
        "Channel legalization cannot legalize %s because it is "
        "non-blocking.",
        n->GetName());
  }
  return absl::OkStatus();
}

struct PredicateInfo {
  StreamingChannel* channel;
  Send* send;
};

// Add a new channel to communicate a channel operation's predicate to the
// adapter proc.
// The adapter proc needs to know which operations are firing, so we add a 1-bit
// internal channel for each operation. This function:
// 1) Adds the new predicate channel for an operation.
// 2) Adds a send of the predicate on this channel at the token level of the
//    original channel operation.
// 3) Updates the token of the original channel operation to come after the new
//    predicate send.
absl::StatusOr<PredicateInfo> MakePredicateChannel(Node* operation) {
  Package* package = operation->package();

  XLS_ASSIGN_OR_RETURN(Channel * operation_channel,
                       GetChannelUsedByNode(operation));
  StreamingChannel* channel = down_cast<StreamingChannel*>(operation_channel);
  Proc* proc = operation->function_base()->AsProcOrDie();
  XLS_ASSIGN_OR_RETURN(
      StreamingChannel * pred_channel,
      package->CreateStreamingChannel(
          absl::StrFormat("%s__pred__%d", channel->name(), operation->id()),
          // This is an internal channel, so override to kSendReceive
          ChannelOps::kSendReceive,
          // This channel is used to forward the predicate to the adapter, which
          // takes 1 bit.
          package->GetBitsType(1),
          /*initial_values=*/{},
          // This is an internal channel that may be inlined during proc
          // inlining, set FIFO depth to 1.
          /*fifo_depth=*/std::optional(std::optional(1))));

  XLS_RETURN_IF_ERROR(CheckIsBlocking(operation));
  XLS_ASSIGN_OR_RETURN(std::optional<Node*> predicate,
                       GetPredicateUsedByNode(operation));
  // Send the predicate before performing the channel operation operation.
  // If predicate nullopt, that means it's an unconditional send/receive. Make a
  // true literal to send to the adapter.
  if (!predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        predicate,
        proc->MakeNodeWithName<Literal>(
            SourceInfo(), Value(UBits(1, /*bit_count=*/1)),
            absl::StrFormat("true_predicate_for_chan_%s", channel->name())));
  }
  XLS_ASSIGN_OR_RETURN(
      Send * send_pred,
      proc->MakeNodeWithName<Send>(
          SourceInfo(), proc->TokenParam(), predicate.value(), std::nullopt,
          pred_channel->id(),
          absl::StrFormat("send_predicate_for_chan_%s", channel->name())));
  return PredicateInfo{
      .channel = pred_channel,
      .send = send_pred,
  };
}

// Add a new channel to communicate a channel operation's completion to the
// adapter proc.
// The procs with the original channel operations need to know when their
// operation completes, so we add an empty-tuple-typed interal channel for each
// operation. This function:
// 1) Adds the new completion channel for an operation.
// 2) Adds a receive of the completion on this channel at the token level of the
//    original channel operation.
// 3) Updates the token of the original channel operation to come after the new
//    completion receive.
absl::StatusOr<StreamingChannel*> MakeCompletionChannel(Node* operation) {
  Package* package = operation->package();

  XLS_ASSIGN_OR_RETURN(Channel * operation_channel,
                       GetChannelUsedByNode(operation));
  StreamingChannel* channel = down_cast<StreamingChannel*>(operation_channel);
  Proc* proc = operation->function_base()->AsProcOrDie();
  XLS_ASSIGN_OR_RETURN(
      StreamingChannel * completion_channel,
      package->CreateStreamingChannel(
          absl::StrFormat("%s__completion__%d", channel->name(),
                          operation->id()),
          // This is an internal channel, so override to kSendReceive
          ChannelOps::kSendReceive,
          // This channel is used to mark the completion of the requested
          // operation and doesn't carry any data. Use an empty tuple type.
          package->GetTupleType({}),
          /*initial_values=*/{},
          // This is an internal channel that may be inlined during proc
          // inlining, set FIFO depth to 1.
          /*fifo_depth=*/std::optional(std::optional(1))));

  XLS_RETURN_IF_ERROR(CheckIsBlocking(operation));
  XLS_ASSIGN_OR_RETURN(std::optional<Node*> predicate,
                       GetPredicateUsedByNode(operation));
  // Send the predicate before performing the channel operation operation.
  // If predicate nullopt, that means it's an unconditional send/receive. Make a
  // true literal to send to the adapter.
  if (!predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        predicate,
        proc->MakeNodeWithName<Literal>(
            SourceInfo(), Value(UBits(1, /*bit_count=*/1)),
            absl::StrFormat("true_predicate_for_chan_%s", channel->name())));
  }
  XLS_ASSIGN_OR_RETURN(
      Receive * recv_completion,
      proc->MakeNodeWithName<Receive>(
          SourceInfo(), proc->TokenParam(), predicate.value(),
          completion_channel->id(), /*is_blocking=*/true,
          absl::StrFormat("recv_completion_for_chan_%s", channel->name())));
  XLS_ASSIGN_OR_RETURN(Node * recv_completion_token,
                       proc->MakeNodeWithName<TupleIndex>(
                           SourceInfo(), recv_completion, 0,
                           absl::StrFormat("recv_completion_token_for_chan_%s",
                                           channel->name())));
  // Replace usages of the token from the send/recv operation with the token
  // from the completion recv.
  switch (operation->op()) {
    case Op::kSend: {
      XLS_RETURN_IF_ERROR(operation->ReplaceUsesWith(recv_completion_token));
      // After replacing the send, we need to replace the recv_completion's
      // token with the original send.
      XLS_RETURN_IF_ERROR(recv_completion->ReplaceOperandNumber(
          Send::kTokenOperand, operation));
      break;
    }
    case Op::kReceive: {
      absl::flat_hash_map<int64_t, Node*> replacements{
          {Receive::kTokenOperand, recv_completion_token}};
      XLS_RETURN_IF_ERROR(
          ReplaceTupleElementsWith(operation, replacements).status());
      // After replacing the original receive's token, we need to replace the
      // recv_completion's token with the original.
      XLS_ASSIGN_OR_RETURN(
          Node * operation_token,
          operation->function_base()->MakeNode<TupleIndex>(
              SourceInfo(), operation, Receive::kTokenOperand));
      XLS_RETURN_IF_ERROR(recv_completion->ReplaceOperandNumber(
          Receive::kTokenOperand, operation_token));
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected %s to be a send or receive.", operation->GetName()));
  }
  return completion_channel;
}

struct ClonedChannelWithPredicate {
  StreamingChannel* cloned_channel;
  StreamingChannel* predicate_channel;
};

// Check that the token DAG is compatible with the requested strictness.
absl::Status CheckTokenDAG(
    absl::Span<NodeAndPredecessors const> topo_sorted_dag,
    MultipleChannelOpsLegalizationStrictness strictness) {
  if (strictness !=
      MultipleChannelOpsLegalizationStrictness::kProvenMutuallyExclusive) {
    XLS_RET_CHECK_GT(topo_sorted_dag.size(), 1)
        << "Legalization expected multiple channel ops.";
  }
  switch (strictness) {
    case MultipleChannelOpsLegalizationStrictness::kProvenMutuallyExclusive: {
      if (topo_sorted_dag.empty()) {
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not prove channel operations (%s) were mutually exclusive.",
          absl::StrJoin(topo_sorted_dag, ", ")));
    }
    case MultipleChannelOpsLegalizationStrictness::kTotalOrder: {
      // In topo sorted order, every node must have a single precedent (the
      // previous node in the topo sort) OR the node must be in a different,
      // not-yet-seen FunctionBase.
      absl::flat_hash_set<FunctionBase*> fbs_seen;
      fbs_seen.reserve(topo_sorted_dag[0].node->package()->procs().size());
      std::optional<Node*> previous_node;
      std::optional<FunctionBase*> previous_fb;
      for (const NodeAndPredecessors& node_and_predecessors : topo_sorted_dag) {
        if (previous_fb.has_value() &&
            previous_fb.value() !=
                node_and_predecessors.node->function_base()) {
          XLS_RET_CHECK(
              !fbs_seen.contains(node_and_predecessors.node->function_base()))
              << absl::StreamFormat(
                     "Saw %s twice in topo sorted token DAG",
                     node_and_predecessors.node->function_base()->name());
          fbs_seen.insert(node_and_predecessors.node->function_base());
          previous_node = std::nullopt;
        }
        if (node_and_predecessors.predecessors.empty()) {
          XLS_RET_CHECK(!previous_fb.has_value() ||
                        previous_fb.value() !=
                            node_and_predecessors.node->function_base())
              << absl::StreamFormat(
                     "%v is not totally ordered, multiple nodes have no "
                     "predecessors.",
                     *node_and_predecessors.node);
        } else {
          XLS_RET_CHECK_EQ(node_and_predecessors.predecessors.size(), 1)
              << absl::StreamFormat(
                     "%v is not totally ordered, has multiple predecessors "
                     "[%s].",
                     *node_and_predecessors.node,
                     absl::StrJoin(node_and_predecessors.predecessors, ", "));
          if (previous_node.has_value()) {
            XLS_RET_CHECK_EQ(previous_node.value(),
                             *node_and_predecessors.predecessors.begin())
                << absl::StreamFormat(
                       "%v is not totally ordered, should come after %v, but "
                       "comes after %v.",
                       *node_and_predecessors.node, *previous_node.value(),
                       **node_and_predecessors.predecessors.begin());
          } else {
            XLS_RET_CHECK(
                (*node_and_predecessors.predecessors.begin())->Is<Param>())
                << absl::StreamFormat(
                       "First operation in total order must only depend on "
                       "token param.");
          }
        }
        previous_node = node_and_predecessors.node;
        previous_fb = node_and_predecessors.node->function_base();
      }
      return absl::OkStatus();
    }
    case MultipleChannelOpsLegalizationStrictness::kRuntimeMutuallyExclusive:
    case MultipleChannelOpsLegalizationStrictness::kRuntimeOrdered:
    case MultipleChannelOpsLegalizationStrictness::kArbitraryStaticOrder: {
      // Arbitrary DAG OK. The runtime variants check things with assertions,
      // and the arbitrary static order variant is correct by construction.
      return absl::OkStatus();
    }
  }
}

// The adapter orders multiple channel operations with state that tracks if
// there is an outstanding channel operation waiting to complete.
struct ActivationState {
  // Index in the proc state
  int64_t state_idx;
  // The state in the current tick.
  BValue state;
  // The computed next state.
  BValue next_state;
};

// For each channel operation, an ActivationNode has the necessary information
// to determine when the operation should fire.
struct ActivationNode {
  // If nullopt, this ActivationNode's corresponding operation needs no state to
  // determine activation. This usually means the operation has no precedents
  // and can always fire when a true predicate is sent.
  std::optional<ActivationState> activation_state;
  // A 1-bit value indicating the corresponding operation can fire.
  BValue activate;
  // The token from the predicate's receive, this ActivationNode's corresponding
  // operation should depend on it.
  BValue pred_recv_token;
};

struct NodeIdCompare {
  bool operator()(Node* const lhs, Node* const rhs) const {
    return lhs->id() < rhs->id();
  }
};

struct ActivationNetwork {
  BValue network_active;
  // Use btree_map to make iteration order stable.
  absl::btree_map<Node* const, ActivationNode, NodeIdCompare> activations;
};

// Makes activation network and adds asserts.
absl::StatusOr<ActivationNetwork> MakeActivationNetwork(
    ProcBuilder& pb, absl::Span<NodeAndPredecessors const> token_dag,
    MultipleChannelOpsLegalizationStrictness strictness) {
  ActivationNetwork activation_network;

  // This is the predicate on every recv on a predicate channel.
  BValue do_pred_recvs;
  {
    std::vector<BValue> pred_state;
    pred_state.reserve(token_dag.size());
    // First, make state. We need to know the state in order to know the
    // predicate on the predicate receive.
    for (const auto& [node, predecessors] : token_dag) {
      ActivationNode activation;
      bool needs_state =
          (strictness != MultipleChannelOpsLegalizationStrictness::
                             kRuntimeMutuallyExclusive) &&
          std::any_of(predecessors.begin(), predecessors.end(), [](Node* n) {
            // Proc token param is always "active".
            return !n->Is<Param>();
          });
      if (needs_state) {
        activation.activation_state.emplace();
        activation.activation_state->state_idx = pred_state.size();
        activation.activation_state->state =
            pb.StateElement(absl::StrFormat("pred_%d", pred_state.size()),
                            Value(UBits(0, /*bit_count=*/1)));
        pred_state.push_back(activation.activation_state->state);
      }
      activation_network.activations.insert({node, activation});
    }
    if (pred_state.empty()) {
      do_pred_recvs = pb.Literal(UBits(1, /*bit_count=*/1), SourceInfo(),
                                 absl::StrFormat("do_pred_recvs"));
    } else {
      do_pred_recvs =
          pb.Not(pb.Or(pred_state, SourceInfo(), "has_predicate_waiting"),
                 SourceInfo(), "do_pred_recvs");
    }
  }

  // For each proc, keep a list of token nodes. Channel operations will be
  // updated to come after all the predicate sends.
  absl::flat_hash_map<FunctionBase*, std::vector<Node*>> pred_send_tokens;
  pred_send_tokens.reserve(token_dag.size());

  // Map of all predecessors and their transitive predecessors. We need
  // transitive predecessors to compute activations.
  absl::flat_hash_map<Node*, absl::btree_set<Node*, NodeIdCompare>>
      resolved_predecessors;

  for (const auto& [node, predecessors] : token_dag) {
    XLS_ASSIGN_OR_RETURN(PredicateInfo pred_info, MakePredicateChannel(node));
    auto& [pred_channel, pred_send] = pred_info;
    pred_send_tokens[node->function_base()].push_back(pred_send);
    BValue recv = pb.ReceiveIf(
        pred_channel, pb.GetTokenParam(), do_pred_recvs, SourceInfo(),
        absl::StrFormat("recv_pred_on_chan_%d", pred_channel->id()));
    BValue pred_recv_token = pb.TupleIndex(
        recv, 0, SourceInfo(),
        absl::StrFormat("recv_pred_on_chan_%d_token", pred_channel->id()));
    BValue pred_recv_predicate = pb.TupleIndex(
        recv, 1, SourceInfo(),
        absl::StrFormat("recv_pred_on_chan_%d_data", pred_channel->id()));

    // Resolve all predecessors and their predecessors.
    auto [itr, _] = resolved_predecessors.insert({node, {}});
    for (Node* const predecessor : predecessors) {
      const absl::btree_set<Node*, NodeIdCompare>& grand_predecessors =
          resolved_predecessors.at(predecessor);
      itr->second.insert(predecessor);
      itr->second.insert(grand_predecessors.begin(), grand_predecessors.end());
    }

    ActivationNode& activation = activation_network.activations.at(node);
    {
      std::vector<BValue> pred_tokens{pred_recv_token};
      pred_tokens.reserve(predecessors.size() + 1);
      for (const auto& predecessor : itr->second) {
        pred_tokens.push_back(
            activation_network.activations.at(predecessor).pred_recv_token);
      }
      activation.pred_recv_token = pb.AfterAll(
          pred_tokens, SourceInfo(),
          absl::StrFormat("after_recv_pred_on_chan_%d", pred_channel->id()));
    }

    std::vector<BValue> dependent_activations;
    dependent_activations.reserve(token_dag.size());

    if (activation.activation_state.has_value()) {
      for (const auto& predecessor : itr->second) {
        dependent_activations.push_back(
            activation_network.activations.at(predecessor).activate);
      }

      BValue has_active_predecessor;
      if (dependent_activations.size() > 1) {
        has_active_predecessor =
            pb.Or(dependent_activations, SourceInfo(),
                  absl::StrFormat("%v_has_active_predecessor", *node));
      } else if (dependent_activations.size() == 1) {
        has_active_predecessor = dependent_activations[0];
      } else {
        return absl::InternalError(
            absl::StrFormat("Node %v has state, but no predecessors.", *node));
      }
      BValue has_pred =
          pb.Or(pred_recv_predicate, activation.activation_state->state,
                SourceInfo(), absl::StrFormat("%v_has_pred", *node));
      BValue no_active_predecessors =
          pb.Not(has_active_predecessor, SourceInfo(),
                 absl::StrFormat("%v_no_active_predecessors", *node));
      activation.activate =
          pb.And(has_pred, no_active_predecessors, SourceInfo(),
                 absl::StrFormat("%v_activate", *node));
      activation.activation_state->next_state = pb.And(
          pb.Not(activation.activate),
          pb.Or(activation.activation_state->state, pred_recv_predicate));
    } else {
      activation.activate = pred_recv_predicate;
    }
  }

  // Build the network active bool. The network is active if there's a waiting
  // predicate or if there's a true predicate we've received this tick. We OR
  // all these signals together.
  {
    std::vector<BValue> network_active;
    network_active.reserve(token_dag.size());
    for (const auto& [node, _] : token_dag) {
      network_active.push_back(
          activation_network.activations.at(node).activate);
    }
    activation_network.network_active =
        pb.Or(network_active, SourceInfo(), "do_external_op");
  }

  // Update the original operations to come after all of the predicate sends.
  for (const auto& [node, _] : token_dag) {
    std::vector<Node*>& tokens = pred_send_tokens.at(node->function_base());
    XLS_CHECK(!tokens.empty());
    // Replace lists of multiple tokens with a single after-all, which we will
    // use potentially multiple times.
    if (tokens.size() > 1) {
      XLS_ASSIGN_OR_RETURN(Node * after_all,
                           node->function_base()->MakeNodeWithName<AfterAll>(
                               SourceInfo(), tokens, "after_send_pred"));
      tokens.clear();
      tokens.push_back(after_all);
    }

    XLS_ASSIGN_OR_RETURN(
        Node * after_pred_and_original_token,
        node->function_base()->MakeNodeWithName<AfterAll>(
            SourceInfo(), std::vector<Node*>{tokens[0], node->operand(0)},
            absl::StrFormat("after_pred_and_original_token_for_%v", *node)));
    XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(
        0, after_pred_and_original_token, /*type_must_match=*/true));
  }

  for (const auto& [node, predecessors] : token_dag) {
    std::vector<BValue> mutually_exclusive_activations;
    std::vector<BValue> mutually_exclusive_activation_tokens;
    mutually_exclusive_activation_tokens.push_back(
        activation_network.activations.at(node).pred_recv_token);

    for (const auto& [mutually_exclusive_node,
                      mutually_exclusive_activation_node] :
         activation_network.activations) {
      // Not mutually exclusive if it is node or a predecessor.
      if (mutually_exclusive_node == node ||
          predecessors.contains(mutually_exclusive_node)) {
        continue;
      }
      mutually_exclusive_activations.push_back(
          mutually_exclusive_activation_node.activate);
      mutually_exclusive_activation_tokens.push_back(
          mutually_exclusive_activation_node.pred_recv_token);
    }
    if (mutually_exclusive_activations.empty()) {
      continue;
    }
    // my_activation -> !Or(mutually_exclusive_activations) ===
    // !my_activation || !Or(mutually_exclusive_activations) ===
    // !(my_activation && Or(mutually_exclusive_activations))
    BValue my_activation = activation_network.activations.at(node).activate;
    BValue has_any_predecessor_active = pb.Or(
        mutually_exclusive_activations, SourceInfo(),
        /*name=*/absl::StrFormat("%v_has_active_mutually_exclusive", *node));
    BValue node_and_predecessors_active = pb.And(
        my_activation, has_any_predecessor_active, SourceInfo(),
        /*name=*/absl::StrFormat("%v_active_with_mutually_exclusive", *node));
    BValue my_activation_mutually_exclusive =
        pb.Not(node_and_predecessors_active, SourceInfo(),
               absl::StrFormat("%v_mutually_exclusive", *node));
    BValue assertion_token = pb.AfterAll(mutually_exclusive_activation_tokens);
    BValue mutual_exclusion_assertion = pb.Assert(
        assertion_token, my_activation_mutually_exclusive,
        absl::StrFormat("Activation for node %s was not mutually exclusive.",
                        node->GetName()));
    activation_network.activations.at(node).pred_recv_token =
        mutual_exclusion_assertion;
  }

  return activation_network;
}

absl::Status AddAdapterForMultipleReceives(
    Package* p, int64_t channel_id, const absl::flat_hash_set<Receive*>& ops,
    MultipleChannelOpsLegalizationStrictness strictness) {
  XLS_RET_CHECK_GT(ops.size(), 1);

  XLS_ASSIGN_OR_RETURN(auto token_dags, GetProjectedTokenDAG(ops, strictness));
  XLS_RETURN_IF_ERROR(CheckTokenDAG(token_dags, strictness));

  XLS_ASSIGN_OR_RETURN(Channel * old_channel, p->GetChannel(channel_id));
  std::string adapter_name =
      absl::StrFormat("chan_%s_io_receive_adapter", old_channel->name());

  XLS_VLOG(4) << absl::StreamFormat("Channel %s has token dag %s.",
                                    old_channel->name(),
                                    absl::StrJoin(token_dags, ", "));

  ProcBuilder pb(adapter_name, "tok", p);
  BValue token = pb.GetTokenParam();

  XLS_ASSIGN_OR_RETURN(auto activation_network,
                       MakeActivationNetwork(pb, token_dags, strictness));

  BValue recv =
      pb.ReceiveIf(old_channel, token, activation_network.network_active,
                   SourceInfo(), "external_receive");
  BValue recv_token =
      pb.TupleIndex(recv, 0, SourceInfo(), "external_receive_token");
  BValue recv_data =
      pb.TupleIndex(recv, 1, SourceInfo(), "external_receive_data");

  BValue send_after_all;
  std::vector<BValue> next_state;
  next_state.reserve(token_dags.size());
  {
    std::vector<BValue> send_tokens{recv_token};
    send_tokens.reserve(token_dags.size() + 1);

    std::vector<BValue> completion_tokens;
    completion_tokens.reserve(token_dags.size());

    for (const auto& [node, predecessors] : token_dags) {
      const ActivationNode& activation =
          activation_network.activations.at(node);

      XLS_ASSIGN_OR_RETURN(
          Channel * new_data_channel,
          p->CloneChannel(
              old_channel,
              absl::StrCat(old_channel->name(), "_", send_tokens.size()),
              Package::CloneChannelOverrides()
                  .OverrideSupportedOps(ChannelOps::kSendReceive)
                  .OverrideFifoDepth(1)));
      XLS_RETURN_IF_ERROR(
          ReplaceChannelUsedByNode(node, new_data_channel->id()));
      BValue send_token = pb.AfterAll({activation.pred_recv_token, recv_token});
      send_tokens.push_back(pb.SendIf(new_data_channel, send_token,
                                      activation.activate, recv_data));
      if (activation.activation_state.has_value()) {
        XLS_RET_CHECK_EQ(next_state.size(),
                         activation.activation_state->state_idx);
        next_state.push_back(activation.activation_state->next_state);
      }
    }
    send_after_all = pb.AfterAll(send_tokens);
  }
  BValue completion_after_all;
  {
    std::vector<BValue> completion_tokens;
    completion_tokens.reserve(token_dags.size());
    BValue empty_tuple_literal = pb.Literal(Value::Tuple({}));

    for (const auto& [node, predecessors] : token_dags) {
      XLS_ASSIGN_OR_RETURN(StreamingChannel * completion_channel,
                           MakeCompletionChannel(node));
      BValue completion_send =
          pb.SendIf(completion_channel, send_after_all,
                    activation_network.activations.at(node).activate,
                    empty_tuple_literal);
      completion_tokens.push_back(completion_send);
    }
    completion_after_all = pb.AfterAll(completion_tokens);
  }

  return pb.Build(completion_after_all, next_state).status();
}

absl::Status AddAdapterForMultipleSends(
    Package* p, int64_t channel_id, const absl::flat_hash_set<Send*>& ops,
    MultipleChannelOpsLegalizationStrictness strictness) {
  XLS_RET_CHECK_GT(ops.size(), 1);

  XLS_ASSIGN_OR_RETURN(auto token_dags, GetProjectedTokenDAG(ops, strictness));
  XLS_RETURN_IF_ERROR(CheckTokenDAG(token_dags, strictness));

  XLS_ASSIGN_OR_RETURN(Channel * old_channel, p->GetChannel(channel_id));
  std::string adapter_name =
      absl::StrFormat("chan_%s_io_send_adapter", old_channel->name());

  XLS_VLOG(4) << absl::StreamFormat("Channel %s has token dag %s.",
                                    old_channel->name(),
                                    absl::StrJoin(token_dags, ", "));

  ProcBuilder pb(adapter_name, "tok", p);

  XLS_ASSIGN_OR_RETURN(auto activation_network,
                       MakeActivationNetwork(pb, token_dags, strictness));

  BValue recv_after_all;
  BValue recv_data;
  BValue recv_data_valid;
  std::vector<BValue> next_state;
  next_state.reserve(token_dags.size());
  {
    std::vector<BValue> recv_tokens;
    recv_tokens.reserve(token_dags.size());
    std::vector<BValue> recv_datas;
    recv_datas.reserve(token_dags.size());
    std::vector<BValue> recv_data_valids;
    recv_data_valids.reserve(token_dags.size());

    for (const auto& [node, predecessors] : token_dags) {
      const ActivationNode& activation =
          activation_network.activations.at(node);
      XLS_ASSIGN_OR_RETURN(
          Channel * new_data_channel,
          p->CloneChannel(
              old_channel,
              absl::StrCat(old_channel->name(), "_", recv_tokens.size()),
              Package::CloneChannelOverrides()
                  .OverrideSupportedOps(ChannelOps::kSendReceive)
                  .OverrideFifoDepth(1)));
      XLS_RETURN_IF_ERROR(
          ReplaceChannelUsedByNode(node, new_data_channel->id()));
      BValue recv = pb.ReceiveIf(new_data_channel, activation.pred_recv_token,
                                 activation.activate);
      recv_tokens.push_back(pb.TupleIndex(recv, 0));
      recv_datas.push_back(pb.TupleIndex(recv, 1));
      recv_data_valids.push_back(activation.activate);
      if (activation.activation_state.has_value()) {
        XLS_RET_CHECK_EQ(next_state.size(),
                         activation.activation_state->state_idx);
        next_state.push_back(activation.activation_state->next_state);
      }
    }
    recv_after_all = pb.AfterAll(recv_tokens);
    // Reverse for one hot select order.
    std::reverse(recv_data_valids.begin(), recv_data_valids.end());
    recv_data = pb.OneHotSelect(pb.Concat(recv_data_valids), recv_datas);
    recv_data_valid = pb.Or(recv_data_valids);
  }

  BValue send_token = pb.SendIf(old_channel, recv_after_all, recv_data_valid,
                                recv_data, SourceInfo(), "external_send");
  BValue completion_after_all;
  {
    std::vector<BValue> completion_tokens;
    completion_tokens.reserve(token_dags.size());
    BValue empty_tuple_literal = pb.Literal(Value::Tuple({}));

    for (const auto& [node, predecessors] : token_dags) {
      XLS_ASSIGN_OR_RETURN(StreamingChannel * completion_channel,
                           MakeCompletionChannel(node));
      BValue completion_send =
          pb.SendIf(completion_channel, send_token,
                    activation_network.activations.at(node).activate,
                    empty_tuple_literal);
      completion_tokens.push_back(completion_send);
    }
    completion_after_all = pb.AfterAll(completion_tokens);
  }

  return pb.Build(completion_after_all, next_state).status();
}
}  // namespace

absl::StatusOr<bool> ChannelLegalizationPass::RunInternal(
    SchedulingUnit<>* unit, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  XLS_VLOG(3) << "Running channel legalization pass.";
  bool changed = false;
  MultipleChannelOps multiple_ops = FindMultipleChannelOps(unit->ir);
  for (const auto& [channel_id, ops] : multiple_ops.multiple_receives) {
    for (Receive* recv : ops) {
      if (!recv->is_blocking()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Non-blocking receives must be the only receive on "
            "a channel; there are multiple receives and %s is non-blocking.",
            recv->GetName()));
      }
    }
    changed = true;
    XLS_VLOG(3) << absl::StreamFormat(
        "Making receive channel adapter for channel %d, has receives (%s).",
        channel_id, absl::StrJoin(ops, ", "));
    XLS_RETURN_IF_ERROR(AddAdapterForMultipleReceives(
        unit->ir, channel_id, ops,
        options.scheduling_options
            .multiple_channel_ops_legalization_strictness()));
  }
  for (const auto& [channel_id, ops] : multiple_ops.multiple_sends) {
    changed = true;
    XLS_VLOG(3) << absl::StreamFormat(
        "Making send channel adapter for channel %d, has sends (%s).",
        channel_id, absl::StrJoin(ops, ", "));
    XLS_RETURN_IF_ERROR(AddAdapterForMultipleSends(
        unit->ir, channel_id, ops,
        options.scheduling_options
            .multiple_channel_ops_legalization_strictness()));
  }
  if (changed) {
    // Reschedule everything if changed.
    unit->schedule = std::nullopt;
  }
  return changed;
}
}  // namespace xls
