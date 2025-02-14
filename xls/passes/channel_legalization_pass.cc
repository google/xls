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

#include "xls/passes/channel_legalization_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/compare.h"
#include "absl/types/span.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/token_provenance_analysis.h"

namespace xls {

namespace {

// Data structure describing a channel for sending data to the adapter.
struct AdapterInputChannel {
  StreamingChannel* channel;
  // ChannelRef for sending data on the channel outside the adapter.
  SendChannelRef parent_send_channel_ref;
  // ChannelRef for receiving data on the channel inside the adapter.
  ReceiveChannelRef adapter_receive_channel_ref;
};

// Data structure describing a channel for receiving data from the adapter.
struct AdapterOutputChannel {
  StreamingChannel* channel;
  // ChannelRef for receiving data on the channel outside the adapter.
  ReceiveChannelRef parent_receive_channel_ref;
  // ChannelRef for sending data on the channel inside the adapter.
  SendChannelRef adapter_send_channel_ref;
};

// Helper class for building an adapter. Abstracts away proc-scoped vs global
// channel behind a uniform interface.
class AdapterBuilder {
 public:
  static absl::StatusOr<std::unique_ptr<AdapterBuilder>>
  CreateForProcScopedChannels(ChannelReference* adapted_channel,
                              std::string_view adapter_name,
                              Proc* parent_proc) {
    XLS_RET_CHECK(parent_proc->package()->ChannelsAreProcScoped());
    std::unique_ptr<ProcBuilder> proc_builder = std::make_unique<ProcBuilder>(
        NewStyleProc(), adapter_name, parent_proc->package());

    // Add input/output for adapted channel.
    ChannelReference* adapted_channel_ref_in_adapter;
    if (adapted_channel->direction() == ChannelDirection::kSend) {
      XLS_ASSIGN_OR_RETURN(
          adapted_channel_ref_in_adapter,
          proc_builder->AddOutputChannel(
              adapted_channel->name(), adapted_channel->type(),
              ChannelKind::kStreaming, adapted_channel->strictness()));
    } else {
      XLS_ASSIGN_OR_RETURN(
          adapted_channel_ref_in_adapter,
          proc_builder->AddInputChannel(
              adapted_channel->name(), adapted_channel->type(),
              ChannelKind::kStreaming, adapted_channel->strictness()));
    }

    // Prime the name uniquer with the current channels in the proc.
    auto name_uniquer = std::make_unique<NameUniquer>("__");
    for (Channel* ch : parent_proc->channels()) {
      name_uniquer->GetSanitizedUniqueName(ch->name());
    }
    std::unique_ptr<AdapterBuilder> builder =
        absl::WrapUnique(new AdapterBuilder(
            adapted_channel, adapted_channel_ref_in_adapter,
            std::move(proc_builder), parent_proc,
            /*global_channel_name_uniquer=*/nullptr,
            /*proc_scoped_channel_name_uniquer=*/std::move(name_uniquer)));

    builder->adapter_instantiation_args_.push_back(adapted_channel);
    return builder;
  }

  static absl::StatusOr<std::unique_ptr<AdapterBuilder>>
  CreateForGlobalChannels(Channel* adapted_channel,
                          std::string_view adapter_name, Package* package,
                          NameUniquer& channel_name_uniquer) {
    XLS_RET_CHECK(!package->ChannelsAreProcScoped());
    std::unique_ptr<ProcBuilder> proc_builder =
        std::make_unique<ProcBuilder>(adapter_name, package);
    return absl::WrapUnique(new AdapterBuilder(
        adapted_channel, adapted_channel, std::move(proc_builder),
        /*parent_proc=*/nullptr,
        /*global_channel_name_uniquer=*/&channel_name_uniquer,
        /*proc_scoped_channel_name_uniquer=*/nullptr));
  }

  // Create a channel for sending to the adapter.
  absl::StatusOr<AdapterInputChannel> AddAdapterInputChannel(
      std::string_view name, Type* type, const ChannelConfig& channel_config,
      std::optional<ChannelStrictness> strictness = std::nullopt) {
    AdapterInputChannel result;
    if (ChannelsAreProcScoped()) {
      std::string uniquified_name =
          proc_scoped_channel_name_uniquer_->GetSanitizedUniqueName(name);
      XLS_ASSIGN_OR_RETURN(result.channel,
                           package()->CreateStreamingChannelInProc(
                               uniquified_name, ChannelOps::kSendReceive, type,
                               parent_proc().value(),
                               /*initial_values=*/{}, channel_config));
      XLS_ASSIGN_OR_RETURN(result.parent_send_channel_ref,
                           parent_proc().value()->GetSendChannelReference(
                               result.channel->name()));
      XLS_ASSIGN_OR_RETURN(ReceiveChannelReference * parent_receive_channel_ref,
                           parent_proc().value()->GetReceiveChannelReference(
                               result.channel->name()));
      XLS_ASSIGN_OR_RETURN(
          result.adapter_receive_channel_ref,
          adapter_builder().AddInputChannel(name, type, ChannelKind::kStreaming,
                                            strictness));
      adapter_instantiation_args_.push_back(parent_receive_channel_ref);
    } else {
      std::string uniquified_name =
          global_channel_name_uniquer_->GetSanitizedUniqueName(name);
      XLS_ASSIGN_OR_RETURN(result.channel,
                           package()->CreateStreamingChannel(
                               uniquified_name, ChannelOps::kSendReceive, type,
                               /*initial_values=*/{}, channel_config));
      result.parent_send_channel_ref = result.channel;
      result.adapter_receive_channel_ref = result.channel;
    }
    return result;
  }

  // Create a channel for receiving from the adapter.
  absl::StatusOr<AdapterOutputChannel> AddAdapterOutputChannel(
      std::string_view name, Type* type, const ChannelConfig& channel_config,
      std::optional<ChannelStrictness> strictness = std::nullopt) {
    AdapterOutputChannel result;
    if (ChannelsAreProcScoped()) {
      std::string uniquified_name =
          proc_scoped_channel_name_uniquer_->GetSanitizedUniqueName(name);
      XLS_ASSIGN_OR_RETURN(result.channel,
                           package()->CreateStreamingChannelInProc(
                               uniquified_name, ChannelOps::kSendReceive, type,
                               parent_proc().value(),
                               /*initial_values=*/{}, channel_config));
      XLS_ASSIGN_OR_RETURN(result.parent_receive_channel_ref,
                           parent_proc().value()->GetReceiveChannelReference(
                               result.channel->name()));
      XLS_ASSIGN_OR_RETURN(SendChannelReference * parent_send_channel_ref,
                           parent_proc().value()->GetSendChannelReference(
                               result.channel->name()));
      XLS_ASSIGN_OR_RETURN(
          result.adapter_send_channel_ref,
          adapter_builder().AddOutputChannel(
              name, type, ChannelKind::kStreaming, strictness));
      adapter_instantiation_args_.push_back(parent_send_channel_ref);
    } else {
      std::string uniquified_name =
          global_channel_name_uniquer_->GetSanitizedUniqueName(name);
      XLS_ASSIGN_OR_RETURN(result.channel,
                           package()->CreateStreamingChannel(
                               uniquified_name, ChannelOps::kSendReceive, type,
                               /*initial_values=*/{}, channel_config));
      result.parent_receive_channel_ref = result.channel;
      result.adapter_send_channel_ref = result.channel;
    }
    return result;
  }

  // Build and return the adapter proc.
  absl::StatusOr<Proc*> Build(absl::Span<const BValue> next_state) {
    if (ChannelsAreProcScoped()) {
      XLS_RETURN_IF_ERROR(
          parent_proc()
              .value()
              ->AddProcInstantiation(
                  absl::StrFormat("%s_adapter_inst", adapted_channel_name()),
                  adapter_instantiation_args_, adapter_proc())
              .status());
    }
    return adapter_builder().Build(next_state);
  }

  bool ChannelsAreProcScoped() { return adapter_proc()->is_new_style_proc(); }
  Package* package() { return adapter_proc()->package(); }

  // For proc-scoped channels, returns the proc which instantates and
  // communicates with the adapter. For global channels, returns nullopt.
  std::optional<Proc*> parent_proc() { return parent_proc_; }

  Proc* adapter_proc() { return adapter_builder().proc(); }
  ProcBuilder& adapter_builder() { return *adapter_builder_; }

  // Returns the ChannelRef of original (adapted) channel for sending/receiving
  // outside the adapter.
  ChannelRef adapted_channel_ref_in_parent() {
    return adapted_channel_ref_in_parent_;
  }
  // Returns the ChannelRef of original (adapted) channel for sending/receiving
  // inside the adapter.
  ChannelRef adapted_channel_ref_in_adapter() {
    return adapted_channel_ref_in_adapter_;
  }

  // Returns various properties of the original (adapted) channel.
  ChannelStrictness adapted_channel_strictness() {
    return ChannelRefStrictness(adapted_channel_ref_in_parent()).value();
  }
  std::string_view adapted_channel_name() {
    return ChannelRefName(adapted_channel_ref_in_parent());
  }
  Type* adapted_channel_type() {
    return ChannelRefType(adapted_channel_ref_in_parent());
  }

 private:
  AdapterBuilder(ChannelRef adapted_channel_ref_in_parent,
                 ChannelRef adapted_channel_ref_in_adapter,
                 std::unique_ptr<ProcBuilder> adapter_builder,
                 std::optional<Proc*> parent_proc,
                 NameUniquer* global_channel_name_uniquer,
                 std::unique_ptr<NameUniquer> proc_scoped_channel_name_uniquer)
      : adapted_channel_ref_in_parent_(adapted_channel_ref_in_parent),
        adapted_channel_ref_in_adapter_(adapted_channel_ref_in_adapter),
        adapter_builder_(std::move(adapter_builder)),
        parent_proc_(parent_proc),
        global_channel_name_uniquer_(global_channel_name_uniquer),
        proc_scoped_channel_name_uniquer_(
            std::move(proc_scoped_channel_name_uniquer)) {}

  ChannelRef adapted_channel_ref_in_parent_;
  ChannelRef adapted_channel_ref_in_adapter_;
  std::unique_ptr<ProcBuilder> adapter_builder_;
  std::optional<Proc*> parent_proc_;
  NameUniquer* global_channel_name_uniquer_;
  std::unique_ptr<NameUniquer> proc_scoped_channel_name_uniquer_;
  std::vector<ChannelReference*> adapter_instantiation_args_;
};

struct ChannelSends {
  Channel* channel;
  std::vector<Send*> sends;
};

struct ChannelReceives {
  Channel* channel;
  std::vector<Receive*> receives;
};

struct MultipleChannelOps {
  std::vector<ChannelSends> sends;
  std::vector<ChannelReceives> receives;
};

absl::StatusOr<ProcElaboration> ElaboratePackage(Package* package) {
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value() || !(*top)->IsProc() ||
      !(*top)->AsProcOrDie()->is_new_style_proc()) {
    return ProcElaboration::ElaborateOldStylePackage(package);
  }
  return ProcElaboration::Elaborate((*top)->AsProcOrDie());
}

// Find instances of multiple sends/recvs on a channel.
absl::StatusOr<MultipleChannelOps> FindMultipleChannelOps(
    const ProcElaboration& elab) {
  // Create a map from channel to the set of send/receive nodes on the channel
  // and vice-versa.
  absl::flat_hash_map<Channel*, absl::flat_hash_set<Send*>> channel_sends;
  absl::flat_hash_map<Channel*, absl::flat_hash_set<Receive*>> channel_receives;
  absl::flat_hash_map<Send*, absl::flat_hash_set<Channel*>> send_channels;
  absl::flat_hash_map<Receive*, absl::flat_hash_set<Channel*>> receive_channels;
  for (ProcInstance* proc_instance : elab.proc_instances()) {
    for (Node* node : proc_instance->proc()->nodes()) {
      if (node->Is<Send>()) {
        Send* send = node->As<Send>();
        XLS_ASSIGN_OR_RETURN(
            ChannelInstance * channel_instance,
            proc_instance->GetChannelInstance(send->channel_name()));
        channel_sends[channel_instance->channel].insert(send);
        send_channels[send].insert(channel_instance->channel);
      }
      if (node->Is<Receive>()) {
        Receive* receive = node->As<Receive>();
        XLS_ASSIGN_OR_RETURN(
            ChannelInstance * channel_instance,
            proc_instance->GetChannelInstance(receive->channel_name()));
        channel_receives[channel_instance->channel].insert(receive);
        receive_channels[receive].insert(channel_instance->channel);
      }
    }
  }

  MultipleChannelOps result;

  // Identify channels which have multiple sends/receives. Return an error if
  // there is a send/receive which can send on different channels *AND* is also
  // part of a set multiple such ops on the same channel. For now in channel
  // legalization we require sends/receives to be uniquely mapped to a single
  // channel.
  for (auto [channel, sends] : channel_sends) {
    if (sends.size() <= 1) {
      continue;
    }
    ChannelSends element;
    element.channel = channel;
    element.sends = std::vector<Send*>(sends.begin(), sends.end());
    std::sort(element.sends.begin(), element.sends.end(), NodeIdLessThan);

    for (Send* send : element.sends) {
      if (send_channels.at(send).size() > 1) {
        return absl::UnimplementedError(
            absl::StrFormat("Send `%s` can send on different channels and is "
                            "one of multiple sends on channel `%s`",
                            send->GetName(), channel->name()));
      }
    }
    result.sends.push_back(std::move(element));
  }
  std::sort(result.sends.begin(), result.sends.end(),
            [](const ChannelSends& a, const ChannelSends& b) {
              return a.channel->id() < b.channel->id();
            });

  for (auto [channel, receives] : channel_receives) {
    if (receives.size() <= 1) {
      continue;
    }
    ChannelReceives element;
    element.channel = channel;
    element.receives = std::vector<Receive*>(receives.begin(), receives.end());
    std::sort(element.receives.begin(), element.receives.end(), NodeIdLessThan);

    for (Receive* receive : element.receives) {
      if (receive_channels.at(receive).size() > 1) {
        return absl::UnimplementedError(absl::StrFormat(
            "Receive `%s` can receive on different channels and is "
            "one of multiple receives on channel `%s`",
            receive->GetName(), channel->name()));
      }
    }
    result.receives.push_back(std::move(element));
  }
  std::sort(result.receives.begin(), result.receives.end(),
            [](const ChannelReceives& a, const ChannelReceives& b) {
              return a.channel->id() < b.channel->id();
            });

  VLOG(4) << "After erasing single accesses, found " << result.receives.size()
          << " multiple send channels and " << result.receives.size()
          << " multiple receive channels.";

  return result;
}

// Check that the token DAG is compatible with the requested strictness.
absl::Status CheckTokenDAG(
    absl::Span<NodeAndPredecessors const> topo_sorted_dag,
    ChannelStrictness strictness) {
  if (strictness != ChannelStrictness::kProvenMutuallyExclusive) {
    XLS_RET_CHECK_GT(topo_sorted_dag.size(), 1)
        << "Legalization expected multiple channel ops.";
  }
  switch (strictness) {
    case ChannelStrictness::kProvenMutuallyExclusive: {
      if (topo_sorted_dag.empty()) {
        return absl::OkStatus();
      }
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not prove channel operations (%s) were mutually exclusive.",
          absl::StrJoin(topo_sorted_dag, ", ")));
    }
    case ChannelStrictness::kTotalOrder: {
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
          if (previous_node.has_value()) {
            XLS_RET_CHECK(
                node_and_predecessors.predecessors.contains(*previous_node))
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
    case ChannelStrictness::kRuntimeMutuallyExclusive:
    case ChannelStrictness::kRuntimeOrdered:
    case ChannelStrictness::kArbitraryStaticOrder: {
      // Arbitrary DAG OK. The runtime variants check things with assertions,
      // and the arbitrary static order variant is correct by construction.
      return absl::OkStatus();
    }
  }
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
//
// This also resolves transitive predecessors, so send0 -> recv -> send1 ->
// send2 would return a map with send2 having predecessors {send0, send1}.
//
// Note that token params are not included in predecessor lists.
template <typename T, typename = std::enable_if_t<std::is_base_of_v<Node, T>>>
absl::StatusOr<std::vector<NodeAndPredecessors>> GetProjectedTokenDAG(
    absl::Span<T* const> operations, ChannelStrictness strictness) {
  // We return the result_vector, but also build a result_map to track
  // transitive dependencies.
  std::vector<NodeAndPredecessors> result_vector;
  result_vector.reserve(operations.size());

  // If channel operations are mutually exclusive, ignore all predecessors.
  // Simply add each operation to the DAG with no predecessors. If we included
  // predecessors, we would unnecessarily condition activations on predecessor
  // predicates that should always be false, e.g. activation = my_predicate_true
  // && my_predicate_valid && !my_predicate_done && !any_predecessor_active.
  // Also, assertions check that no non-predecessor got a true predicate, and
  // every other operation should cause that assertion to fire.
  if (strictness == ChannelStrictness::kRuntimeMutuallyExclusive) {
    for (T* operation : operations) {
      result_vector.push_back(
          NodeAndPredecessors{.node = operation, .predecessors = {}});
    }
    std::sort(
        result_vector.begin(), result_vector.end(),
        [](const NodeAndPredecessors& lhs, const NodeAndPredecessors& rhs) {
          return Node::NodeIdLessThan()(lhs.node, rhs.node);
        });
    return result_vector;
  }

  // result_map maps nodes to a set pointer owned by result_vector (avoids extra
  // copies).
  absl::flat_hash_map<Node*, NodeAndPredecessors::PredecessorSet*> result_map;
  result_map.reserve(operations.size());

  // Use btree set that sorts FunctionBases by name to ensure stable order of
  // iteration through procs.
  absl::btree_set<FunctionBase*, FunctionBaseNameLess> fbs;
  for (const T* operation : operations) {
    fbs.insert(operation->function_base());
  }

  absl::flat_hash_set<T*> operation_set(operations.begin(), operations.end());
  for (FunctionBase* fb : fbs) {
    XLS_ASSIGN_OR_RETURN(std::vector<NodeAndPredecessors> fbs_result,
                         ComputeTopoSortedTokenDAG(fb));
    // Resolve side-effecting nodes and after_alls.
    absl::flat_hash_map<Node*, NodeAndPredecessors::PredecessorSet>
        resolved_ops;
    // Initialize with all token-typed params & literals having no predecessors.
    for (Node* node : fb->nodes()) {
      if (node->GetType()->IsToken() &&
          node->OpIn({Op::kParam, Op::kLiteral})) {
        resolved_ops[node] = {};
      }
    }
    // Keep track of prev_node which will be used if we choose a stricter order.
    std::optional<Node*> prev_node = std::nullopt;
    for (NodeAndPredecessors& fb_result : fbs_result) {
      NodeAndPredecessors::PredecessorSet resolved_predecessors;
      for (Node* predecessor : fb_result.predecessors) {
        // If a predecessor is not type T, resolve its predecessors of type T.
        if (!predecessor->Is<T>() ||
            !operation_set.contains(predecessor->As<T>())) {
          const NodeAndPredecessors::PredecessorSet& resolved =
              resolved_ops.at(predecessor);
          resolved_predecessors.insert(resolved.begin(), resolved.end());
          continue;
        }
        resolved_predecessors.insert(predecessor);
      }
      // If this entry in the DAG is not type T, save its resolved predecessors
      // for future resolution.
      if (!fb_result.node->Is<T>() ||
          !operation_set.contains(fb_result.node->As<T>())) {
        resolved_ops[fb_result.node] = std::move(resolved_predecessors);
        continue;
      }
      // If we choose an arbitrary static order, add the previous value to the
      // set of predecessors. We're already iterating through in topo sorted
      // order, so this only strengthens the dependency relationship.
      if (strictness == ChannelStrictness::kArbitraryStaticOrder) {
        if (prev_node.has_value()) {
          if (!prev_node.value()->Is<T>() ||
              !operation_set.contains(prev_node.value()->As<T>())) {
            resolved_predecessors.insert(resolved_ops.at(*prev_node).begin(),
                                         resolved_ops.at(*prev_node).end());
          } else {
            resolved_predecessors.insert(prev_node.value());
          }
        }
      }
      NodeAndPredecessors::PredecessorSet transitive_predecessors(
          resolved_predecessors.begin(), resolved_predecessors.end());
      for (Node* predecessor : resolved_predecessors) {
        NodeAndPredecessors::PredecessorSet* grand_predecessors =
            result_map.at(predecessor);
        transitive_predecessors.insert(grand_predecessors->begin(),
                                       grand_predecessors->end());
      }
      result_vector.push_back(NodeAndPredecessors{
          .node = fb_result.node,
          .predecessors = std::move(transitive_predecessors)});
      result_map.insert({fb_result.node, &result_vector.back().predecessors});
      prev_node = fb_result.node;
    }
  }
  XLS_RETURN_IF_ERROR(CheckTokenDAG(result_vector, strictness));
  return result_vector;
}

absl::Status CheckIsBlocking(ChannelNode* n) {
  if (n->Is<Receive>()) {
    XLS_RET_CHECK(n->As<Receive>()->is_blocking()) << absl::StreamFormat(
        "Channel legalization cannot legalize %s because it is "
        "non-blocking.",
        n->GetName());
  }
  return absl::OkStatus();
}

// Add a new channel to communicate a channel operation's predicate to the
// adapter proc.
// The adapter proc needs to know which operations are firing, so we add a 1-bit
// internal channel for each operation. This function:
// 1) Adds the new predicate channel for an operation.
// 2) Adds a send of the predicate on this channel at the token level of the
//    original channel operation.
// 3) Updates the token of the original channel operation to come after the new
//    predicate send.
absl::StatusOr<AdapterInputChannel> MakePredicateChannel(
    ChannelNode* operation, ChannelRef channel_ref, int64_t instance_number,
    AdapterBuilder& ab) {
  Package* package = operation->package();
  Proc* proc = operation->function_base()->AsProcOrDie();
  XLS_ASSIGN_OR_RETURN(
      AdapterInputChannel pred_input_channel,
      ab.AddAdapterInputChannel(
          absl::StrFormat("%s__pred_%d", ChannelRefName(channel_ref),
                          instance_number),
          // This channel is used to forward the predicate to the adapter, which
          // takes 1 bit.
          package->GetBitsType(1),
          // This is an internal channel that may be inlined during proc
          // inlining, set FIFO depth to 1. Break cycles by registering push
          // outputs.
          // TODO: github/xls#1509 - revisit this if we have better ways of
          // avoiding cycles in adapters.
          ChannelConfig(FifoConfig(/*depth=*/1, /*bypass=*/false,
                                   /*register_push_outputs=*/true,
                                   /*register_pop_outputs=*/false))));

  XLS_RETURN_IF_ERROR(CheckIsBlocking(operation));
  std::optional<Node*> predicate = operation->predicate();

  // Send the predicate before performing the channel operation operation.
  // If predicate nullopt, that means it's an unconditional send/receive. Make a
  // true literal to send to the adapter.
  if (!predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(predicate,
                         proc->MakeNodeWithName<Literal>(
                             SourceInfo(), Value(UBits(1, /*bit_count=*/1)),
                             absl::StrFormat("true_predicate_for_chan_%s",
                                             ChannelRefName(channel_ref))));
  }
  XLS_ASSIGN_OR_RETURN(
      Send * send_pred,
      proc->MakeNodeWithName<Send>(
          SourceInfo(), operation->token(), predicate.value(), std::nullopt,
          ChannelRefName(
              AsChannelRef(pred_input_channel.parent_send_channel_ref)),
          absl::StrFormat("send_predicate_for_chan_%s",
                          ChannelRefName(channel_ref))));
  // Replace the op's original input token with the predicate send's token.
  XLS_RETURN_IF_ERROR(operation->ReplaceToken(send_pred));

  return pred_input_channel;
}

// Add a new channel to communicate a channel operation's completion to the
// adapter proc.
// The procs with the original channel operations need to know when their
// operation completes, so we add an empty-tuple-typed internal channel for each
// operation. This function:
// 1) Adds the new completion channel for an operation.
// 2) Adds a receive of the completion on this channel at the token level of the
//    original channel operation.
// 3) Updates the token of the original channel operation to come after the new
//    completion_42` receive.
absl::StatusOr<AdapterOutputChannel> MakeCompletionChannel(
    ChannelNode* operation, int64_t instance_number, AdapterBuilder& ab) {
  Package* package = ab.package();
  Proc* proc = operation->function_base()->AsProcOrDie();

  XLS_ASSIGN_OR_RETURN(
      AdapterOutputChannel completion_channel,
      ab.AddAdapterOutputChannel(
          absl::StrFormat("%s__completion_%i", ab.adapted_channel_name(),
                          instance_number),
          // This channel is used to mark the completion of the requested
          // operation and doesn't carry any data. Use an empty tuple type.
          package->GetTupleType({}),
          // This is an internal channel that may be inlined during proc
          // inlining, set FIFO depth to 0. Completion channels seem to not
          // cause cycles when they have timing paths between push and pop.
          // TODO: github/xls#1509 - revisit this if we have better ways of
          // avoiding cycles in adapters.
          /*channel_config=*/
          ChannelConfig(FifoConfig(/*depth=*/0, /*bypass=*/true,
                                   /*register_push_outputs=*/false,
                                   /*register_pop_outputs=*/false))));

  XLS_RETURN_IF_ERROR(CheckIsBlocking(operation));
  std::optional<Node*> predicate = operation->predicate();

  // Send the predicate before performing the channel operation operation.
  // If predicate nullopt, that means it's an unconditional send/receive. Make a
  // true literal to send to the adapter.
  if (!predicate.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        predicate, proc->MakeNodeWithName<Literal>(
                       SourceInfo(), Value(UBits(1, /*bit_count=*/1)),
                       absl::StrFormat("true_predicate_for_chan_%s",
                                       completion_channel.channel->name())));
  }
  XLS_ASSIGN_OR_RETURN(Node * free_token,
                       proc->MakeNode<Literal>(SourceInfo(), Value::Token()));
  XLS_ASSIGN_OR_RETURN(
      Receive * recv_completion,
      proc->MakeNodeWithName<Receive>(
          SourceInfo(), free_token, predicate.value(),
          completion_channel.channel->name(), /*is_blocking=*/true,
          absl::StrFormat("recv_completion_for_chan_%s",
                          completion_channel.channel->name())));
  XLS_ASSIGN_OR_RETURN(
      Node * recv_completion_token,
      proc->MakeNodeWithName<TupleIndex>(
          SourceInfo(), recv_completion, 0,
          absl::StrFormat("recv_completion_token_for_chan_%s",
                          completion_channel.channel->name())));
  // Replace usages of the token from the send/recv operation with the token
  // from the completion recv.
  switch (operation->op()) {
    case Op::kSend: {
      XLS_RETURN_IF_ERROR(operation->ReplaceUsesWith(recv_completion_token));
      // After replacing the send, we need to replace the recv_completion's
      // token with the original send.
      XLS_RETURN_IF_ERROR(recv_completion->ReplaceToken(operation));
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
      XLS_RETURN_IF_ERROR(recv_completion->ReplaceToken(operation_token));
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected %s to be a send or receive.", operation->GetName()));
  }
  return completion_channel;
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

  struct CompareByStateIdx {
    bool operator()(const ActivationState& a, const ActivationState& b) const {
      return a.state_idx < b.state_idx;
    }
    bool operator()(const ActivationState* a, const ActivationState* b) const {
      return operator()(*a, *b);
    }
  };
};

// For each channel operation, an ActivationNode has the necessary information
// to determine when the operation should fire.
struct ActivationNode {
  // The predicate for the given channel operation, valid when 'valid' is set.
  BValue predicate;
  BValue valid;
  // Indicates if the corresponding operation has fired (if predicate && valid)
  // or won't fire (if !predicate && valid) in the current round of operations.
  BValue done;
  // Indicates if the corresponding operation can fire in the current proc tick.
  // Should only activate if the predicate is true and valid, it hasn't fired
  // already, and predecessors aren't also active.
  BValue activate;
  // State elements to track predicate, valid, and done across adapter proc
  // ticks.
  ActivationState predicate_state;
  ActivationState valid_state;
  ActivationState done_state;
  // The token from the predicate's receive, this ActivationNode's corresponding
  // operation should depend on it.
  BValue pred_recv_token;
};

// Use btree_map to make iteration order stable.
using ActivationNetwork =
    absl::btree_map<Node* const, ActivationNode, Node::NodeIdLessThan>;

// Computes the next state for the state elements in an activation network.
// Assumes that the activation network's state indices are in [0,
// activations.size()).
std::vector<BValue> NextState(const ActivationNetwork& activations) {
  // Next-state vector needs to be sorted by state index. First, build a list of
  // ActivationState*, sort it, and then extract the next_state BValues into the
  // vector to return.
  std::vector<const ActivationState*> states;
  states.reserve(activations.size());
  for (const auto& [_, activation] : activations) {
    states.push_back(&activation.predicate_state);
    states.push_back(&activation.valid_state);
    states.push_back(&activation.done_state);
  }
  std::sort(states.begin(), states.end(), ActivationState::CompareByStateIdx());
  std::vector<BValue> next_state_vector;
  next_state_vector.reserve(states.size());
  for (const ActivationState* state : states) {
    CHECK_EQ(state->state_idx, next_state_vector.size());
    next_state_vector.push_back(state->next_state);
  }
  return next_state_vector;
}

// Get a token that comes after each node's pred_recv_tokens.
BValue PredRecvTokensForActivations(const ActivationNetwork& activations,
                                    absl::Span<Node* const> nodes,
                                    std::string_view name, ProcBuilder& pb) {
  std::vector<BValue> pred_tokens;
  pred_tokens.reserve(nodes.size());
  for (const auto& node : nodes) {
    pred_tokens.push_back(activations.at(node).pred_recv_token);
  }
  if (pred_tokens.empty()) {
    return pb.Literal(Value::Token());
  }
  if (pred_tokens.size() == 1) {
    return pred_tokens[0];
  }
  return pb.AfterAll(pred_tokens, SourceInfo(), name);
}

BValue AllPredecessorsDone(const ActivationNetwork& activations,
                           absl::Span<Node* const> nodes, std::string_view name,
                           ProcBuilder& pb) {
  std::vector<BValue> dones;
  dones.reserve(nodes.size());
  for (Node* node : nodes) {
    const ActivationNode& activation = activations.at(node);
    dones.push_back(
        pb.Or(activation.done_state.state,
              pb.And(pb.Not(activation.predicate), activation.valid)));
  }
  BValue all_done;
  if (dones.size() > 1) {
    all_done = pb.And(dones, SourceInfo(), name);
  } else if (dones.size() == 1) {
    all_done = dones[0];
  } else {
    all_done = pb.Literal(Value(UBits(1, /*bit_count=*/1)));
  }
  return all_done;
}

// Compute next-state values for each activation.
// Returns the 'not_all_done' signal indicating if an activation is not yet
// done.
BValue NextStateAndReturnNotAllDone(
    absl::Span<NodeAndPredecessors const> token_dag,
    ActivationNetwork& activations, ProcBuilder& pb) {
  BValue not_all_done;
  std::vector<BValue> done_signals;
  done_signals.reserve(token_dag.size());
  for (const auto& [node, _] : token_dag) {
    done_signals.push_back(activations.at(node).done);
  }
  not_all_done = pb.Not(pb.And(done_signals), SourceInfo(), "not_all_done");

  // Compute next-state signals.
  for (const auto& [node, _] : token_dag) {
    ActivationNode& activation = activations.at(node);
    activation.predicate_state.next_state =
        pb.And(activation.predicate, not_all_done);
    activation.valid_state.next_state = pb.And(activation.valid, not_all_done);
    activation.done_state.next_state = pb.And(activation.done, not_all_done);
  }

  return not_all_done;
}

// Make assertions that operations w/ no token ordering are mutually exclusive.
void MakeMutualExclusionAssertions(
    absl::Span<NodeAndPredecessors const> token_dag,
    ActivationNetwork& activations, ProcBuilder& pb) {
  // Build a map of mutually exclusive nodes. Each key's predicate must be
  // mutually exclusive with every element in value's predicate. Distinct nodes
  // are mutually exclusive if neither node is a predecessor of the other.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
      mutually_exclusive_nodes;
  // Make a set of all nodes, we default-initialize every node to being mutually
  // exclusive with every other node and remove predecessors as we go through
  // token_dag.
  absl::flat_hash_set<Node*> all_nodes;
  std::transform(token_dag.begin(), token_dag.end(),
                 std::inserter(all_nodes, all_nodes.end()),
                 [](const NodeAndPredecessors& node_and_token) {
                   return node_and_token.node;
                 });
  for (const auto& [node, predecessors] : token_dag) {
    auto [itr, node_inserted] =
        mutually_exclusive_nodes.insert({node, all_nodes});
    // We iterate in topo-sorted order, so this should be the first time seeing
    // 'node' and this insertion should always succeed.
    CHECK(node_inserted);
    itr->second.erase(node);
    for (Node* const predecessor : predecessors) {
      itr->second.erase(predecessor);
      auto predecessor_itr = mutually_exclusive_nodes.find(predecessor);
      // We have already inserted predecessors in previous iterations.
      CHECK(predecessor_itr != mutually_exclusive_nodes.end());
      predecessor_itr->second.erase(node);
    }
  }

  // Make mutual exclusion assertions.
  for (const auto& [node, predecessors] : token_dag) {
    std::vector<BValue> mutually_exclusive_fired;
    std::vector<BValue> mutually_exclusive_pred_recv_tokens;
    mutually_exclusive_pred_recv_tokens.push_back(
        activations.at(node).pred_recv_token);

    for (Node* const mutually_exclusive_node :
         mutually_exclusive_nodes.at(node)) {
      const ActivationNode& mutually_exclusive_activation_node =
          activations.at(mutually_exclusive_node);
      mutually_exclusive_fired.push_back(
          pb.And(mutually_exclusive_activation_node.predicate,
                 mutually_exclusive_activation_node.valid));
      mutually_exclusive_pred_recv_tokens.push_back(
          mutually_exclusive_activation_node.pred_recv_token);
    }
    if (mutually_exclusive_fired.empty()) {
      continue;
    }
    // my_activation -> !Or(mutually_exclusive_activations) ===
    // !my_activation || !Or(mutually_exclusive_activations) ===
    // !(my_activation && Or(mutually_exclusive_activations))
    BValue my_predicate = activations.at(node).predicate;
    BValue has_any_predecessor_fire =
        pb.Or(mutually_exclusive_fired, SourceInfo(),
              /*name=*/absl::StrFormat("%v_mutually_exclusive_fired", *node));
    BValue my_activation_mutually_exclusive = pb.AddNaryOp(
        Op::kNand, {my_predicate, has_any_predecessor_fire}, SourceInfo(),
        absl::StrFormat("%v_mutually_exclusive", *node));
    BValue assertion_token = pb.AfterAll(mutually_exclusive_pred_recv_tokens);
    BValue mutual_exclusion_assertion =
        pb.Assert(assertion_token, my_activation_mutually_exclusive,
                  absl::StrFormat(
                      "Node %s predicate was not mutually exclusive with {%s}.",
                      node->GetName(),
                      absl::StrJoin(mutually_exclusive_nodes.at(node), ", ")),
                  /*label=*/"adapter_mutually_exclusive_A");
    activations.at(node).pred_recv_token = mutual_exclusion_assertion;
  }
}

// Make a trace to aid debugging. It only fires when all ops are done.
void MakeDebugTrace(BValue condition,
                    absl::Span<NodeAndPredecessors const> token_dag,
                    ActivationNetwork& activations, ProcBuilder& pb) {
  std::vector<BValue> pred_recv_tokens;
  pred_recv_tokens.reserve(token_dag.size());
  std::vector<BValue> args;
  std::string format_string = "\nAdapter proc fire:\tpredicate\tvalid\tdone\n";
  for (const auto& [node, _] : token_dag) {
    ActivationNode& activation = activations.at(node);
    args.push_back(activation.predicate);
    args.push_back(activation.valid);
    args.push_back(activation.done);
    pred_recv_tokens.push_back(activation.pred_recv_token);
    absl::StrAppend(&format_string, node->GetName(), ":\t{}\t{}\t{}\n");
  }
  BValue trace_tkn = pb.Trace(pb.AfterAll(pred_recv_tokens), condition, args,
                              format_string, /*verbosity=*/3);
  for (const auto& [node, _] : token_dag) {
    activations.at(node).pred_recv_token = trace_tkn;
  }
}

// Makes activation network and adds asserts.
absl::StatusOr<ActivationNetwork> MakeActivationNetwork(
    AdapterBuilder& ab, absl::Span<NodeAndPredecessors const> token_dag) {
  ActivationNetwork activations;

  // First, make new predicate channels. The adapter will non-blocking receive
  // on each of these channels to get each operation's predicate.
  absl::flat_hash_map<Node*, AdapterInputChannel> pred_channels;
  pred_channels.reserve(token_dag.size());

  for (int64_t instance_number = 0; instance_number < token_dag.size();
       ++instance_number) {
    ChannelNode* node = token_dag[instance_number].node->As<ChannelNode>();
    XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                         node->As<ChannelNode>()->GetChannelRef());
    XLS_ASSIGN_OR_RETURN(
        AdapterInputChannel pred_input_channel,
        MakePredicateChannel(node, channel_ref, instance_number, ab));
    pred_channels.insert({node, pred_input_channel});
  }

  // Now make state. We need to know the state in order to know the predicate on
  // the predicate receive.
  int64_t state_idx = 0;
  for (int64_t instance_number = 0; instance_number < token_dag.size();
       ++instance_number) {
    Node* node = token_dag[instance_number].node;
    ActivationNode activation;
    activation.predicate_state.state = ab.adapter_builder().StateElement(
        absl::StrFormat("pred_%d", instance_number),
        Value(UBits(0, /*bit_count=*/1)));
    activation.valid_state.state = ab.adapter_builder().StateElement(
        absl::StrFormat("pred_%d_valid", instance_number),
        Value(UBits(0, /*bit_count=*/1)));
    activation.done_state.state = ab.adapter_builder().StateElement(
        absl::StrFormat("pred_%d_done", instance_number),
        Value(UBits(0, /*bit_count=*/1)));

    activation.predicate_state.state_idx = state_idx++;
    activation.valid_state.state_idx = state_idx++;
    activation.done_state.state_idx = state_idx++;
    activations.insert({node, activation});
  }

  // Make a non-blocking receive on the predicate channel for each operation.
  // Compute predicate values and activations for this tick.
  for (int64_t instance_number = 0; instance_number < token_dag.size();
       ++instance_number) {
    Node* node = token_dag[instance_number].node;
    const NodeAndPredecessors::PredecessorSet& predecessors =
        token_dag[instance_number].predecessors;

    const AdapterInputChannel& pred_input_channel = pred_channels.at(node);
    ActivationNode& activation = activations.at(node);

    std::vector<Node*> sorted_predecessors(predecessors.begin(),
                                           predecessors.end());
    std::sort(sorted_predecessors.begin(), sorted_predecessors.end(),
              Node::NodeIdLessThan());

    // Get token following predecessors' predicate receives.
    BValue predecessors_pred_recv_token = PredRecvTokensForActivations(
        activations, sorted_predecessors,
        absl::StrFormat("chan_%d_recv_pred_predeccesors_token",
                        instance_number),
        ab.adapter_builder());

    // Do a non-blocking receive on the predicate channel.
    //
    // Each tick of the adapter will do a non-blocking receive on the predicate
    // channel until a result with vaild=1 occurs. There are two state bits set
    // here: 'predicate' stores the value of the predicate after it has been
    // successfully received and 'valid' indicates that it has been successfully
    // received. After the adapter completes all operations, 'valid' will be
    // reset to 0 and everything starts over again.
    BValue do_pred_recv =
        ab.adapter_builder().Not(activation.valid_state.state);
    BValue recv = ab.adapter_builder().ReceiveIfNonBlocking(
        pred_input_channel.adapter_receive_channel_ref,
        predecessors_pred_recv_token, do_pred_recv, SourceInfo(),
        absl::StrFormat("recv_pred_%d", instance_number));
    activation.pred_recv_token = ab.adapter_builder().TupleIndex(
        recv, 0, SourceInfo(),
        absl::StrFormat("recv_pred_%d_token", instance_number));
    BValue pred_recv_predicate = ab.adapter_builder().TupleIndex(
        recv, 1, SourceInfo(),
        absl::StrFormat("recv_pred_%d_data", instance_number));
    BValue pred_recv_valid = ab.adapter_builder().TupleIndex(
        recv, 2, SourceInfo(),
        absl::StrFormat("recv_pred_%d_valid", instance_number));
    activation.predicate = ab.adapter_builder().Or(
        pred_recv_predicate, activation.predicate_state.state, SourceInfo(),
        absl::StrFormat("pred_%d_updated", instance_number));
    activation.valid = ab.adapter_builder().Or(
        pred_recv_valid, activation.valid_state.state, SourceInfo(),
        absl::StrFormat("pred_%d_valid_updated", instance_number));

    BValue all_predecessors_done =
        AllPredecessorsDone(activations, sorted_predecessors,
                            absl::StrFormat("%v_all_predecessors_done", *node),
                            ab.adapter_builder());
    activation.activate = ab.adapter_builder().And(
        {activation.predicate, activation.valid,
         ab.adapter_builder().Not(activation.done_state.state),
         all_predecessors_done},
        SourceInfo(), absl::StrFormat("%v_activate", *node));
    activation.done = ab.adapter_builder().Or(
        {activation.activate,
         ab.adapter_builder().And(
             ab.adapter_builder().Not(activation.predicate), activation.valid),
         activation.done_state.state});
  }

  // Make assertions that operations w/ no token ordering are mutually
  // exclusive. Note that the strictnesses runtime_mutually_exclusive and
  // arbitrary_static_order are special cases here because their predecessors
  // have been altered. In the runtime_mutually_exclusive case, all predecessors
  // are empty, so every operation should be mutually exclusive. In the
  // arbitrary_static_order case, nodes are linearized and have an added token
  // relationship with every other operation, and there should be no assertions.
  MakeMutualExclusionAssertions(token_dag, activations, ab.adapter_builder());

  // Compute next-state signals.
  // Builds not_all_done signal which indicates if every operation is done. If
  // so, start the next set of operations by setting next signals for done and
  // valid to 0.
  BValue not_all_done = NextStateAndReturnNotAllDone(token_dag, activations,
                                                     ab.adapter_builder());

  // Make a trace to aid debugging. It only fires when all ops are done.
  MakeDebugTrace(ab.adapter_builder().Not(not_all_done), token_dag, activations,
                 ab.adapter_builder());

  return std::move(activations);
}

absl::Status AddAdapterForMultipleReceives(absl::Span<Receive* const> ops,
                                           AdapterBuilder& ab) {
  XLS_RET_CHECK_GT(ops.size(), 1);

  XLS_ASSIGN_OR_RETURN(
      auto token_dags,
      GetProjectedTokenDAG(ops, ab.adapted_channel_strictness()));

  VLOG(4) << absl::StreamFormat("Channel %s has token dag %s.",
                                ab.adapted_channel_name(),
                                absl::StrJoin(token_dags, ", "));

  XLS_ASSIGN_OR_RETURN(ActivationNetwork activations,
                       MakeActivationNetwork(ab, token_dags));

  BValue any_active;
  BValue external_recv_input_token;
  {
    std::vector<BValue> all_activations;
    std::vector<BValue> all_tokens;
    all_activations.reserve(token_dags.size());
    for (const auto& [node, _] : token_dags) {
      all_activations.push_back(activations.at(node).activate);
      all_tokens.push_back(activations.at(node).pred_recv_token);
    }
    any_active =
        ab.adapter_builder().Or(all_activations, SourceInfo(), "any_active");
    external_recv_input_token = ab.adapter_builder().AfterAll(all_tokens);
  }

  BValue recv = ab.adapter_builder().ReceiveIf(
      AsReceiveChannelRefOrDie(ab.adapted_channel_ref_in_adapter()),
      external_recv_input_token, any_active, SourceInfo(), "external_receive");
  BValue recv_token = ab.adapter_builder().TupleIndex(recv, 0, SourceInfo(),
                                                      "external_receive_token");
  BValue recv_data = ab.adapter_builder().TupleIndex(recv, 1, SourceInfo(),
                                                     "external_receive_data");

  for (int64_t i = 0; i < token_dags.size(); ++i) {
    Node* node = token_dags[i].node;
    const ActivationNode& activation = activations.at(node);
    XLS_ASSIGN_OR_RETURN(
        AdapterOutputChannel output_data_channel,
        ab.AddAdapterOutputChannel(
            absl::StrFormat("%s__data_%d", ab.adapted_channel_name(), i),
            ab.adapted_channel_type(),
            // This is an internal channel that may be inlined during proc
            // inlining, set FIFO depth to 1. Break cycles by registering push
            // outputs.
            // TODO: github/xls#1509 - revisit this if we have better ways of
            // avoiding cycles in adapters.
            ChannelConfig(FifoConfig(/*depth=*/1, /*bypass=*/false,
                                     /*register_push_outputs=*/true,
                                     /*register_pop_outputs=*/false))));
    XLS_RETURN_IF_ERROR(node->As<ChannelNode>()->ReplaceChannel(ChannelRefName(
        AsChannelRef(output_data_channel.parent_receive_channel_ref))));
    BValue send_token =
        ab.adapter_builder().AfterAll({activation.pred_recv_token, recv_token});
    ab.adapter_builder().SendIf(output_data_channel.adapter_send_channel_ref,
                                send_token, activation.activate, recv_data);
  }

  return ab.Build(NextState(activations)).status();
}

absl::Status AddAdapterForMultipleSends(absl::Span<Send* const> ops,
                                        AdapterBuilder& ab) {
  XLS_RET_CHECK_GT(ops.size(), 1);

  XLS_ASSIGN_OR_RETURN(
      auto token_dags,
      GetProjectedTokenDAG(ops, ab.adapted_channel_strictness()));

  VLOG(4) << absl::StreamFormat("Channel %s has token dag %s.",
                                ab.adapted_channel_name(),
                                absl::StrJoin(token_dags, ", "));

  XLS_ASSIGN_OR_RETURN(ActivationNetwork activations,
                       MakeActivationNetwork(ab, token_dags));

  absl::flat_hash_map<Node*, AdapterInputChannel> data_channels;

  BValue recv_after_all;
  BValue recv_data;
  BValue recv_data_valid;
  {
    std::vector<BValue> recv_tokens;
    recv_tokens.reserve(token_dags.size());
    std::vector<BValue> recv_datas;
    recv_datas.reserve(token_dags.size());
    std::vector<BValue> recv_data_valids;
    recv_data_valids.reserve(token_dags.size());

    for (int64_t i = 0; i < token_dags.size(); ++i) {
      Node* node = token_dags[i].node;
      const ActivationNode& activation = activations.at(node);
      XLS_ASSIGN_OR_RETURN(
          AdapterInputChannel input_data_channel,
          ab.AddAdapterInputChannel(
              absl::StrFormat("%s__data_%d", ab.adapted_channel_name(), i),
              ab.adapted_channel_type(),
              // This is an internal channel that may be inlined during proc
              // inlining, set FIFO depth to 1. Break cycles by registering push
              // outputs.
              // TODO: github/xls#1509 - revisit this if we have better ways of
              // avoiding cycles in adapters.
              ChannelConfig(FifoConfig(/*depth=*/1, /*bypass=*/false,
                                       /*register_push_outputs=*/true,
                                       /*register_pop_outputs=*/false))));
      XLS_RETURN_IF_ERROR(
          node->As<ChannelNode>()->ReplaceChannel(ChannelRefName(
              AsChannelRef(input_data_channel.parent_send_channel_ref))));
      BValue recv = ab.adapter_builder().ReceiveIf(
          input_data_channel.adapter_receive_channel_ref,
          activation.pred_recv_token, activation.activate);
      recv_tokens.push_back(ab.adapter_builder().TupleIndex(recv, 0));
      recv_datas.push_back(ab.adapter_builder().TupleIndex(recv, 1));
      recv_data_valids.push_back(activation.activate);

      data_channels[node] = input_data_channel;
    }
    recv_after_all = ab.adapter_builder().AfterAll(recv_tokens);
    // Reverse for one hot select order.
    std::reverse(recv_data_valids.begin(), recv_data_valids.end());
    recv_data = ab.adapter_builder().OneHotSelect(
        ab.adapter_builder().Concat(recv_data_valids), recv_datas);
    recv_data_valid = ab.adapter_builder().Or(recv_data_valids);
  }

  BValue send_token = ab.adapter_builder().SendIf(
      AsSendChannelRefOrDie(ab.adapted_channel_ref_in_adapter()),
      recv_after_all, recv_data_valid, recv_data, SourceInfo(),
      "external_send");
  BValue empty_tuple_literal = ab.adapter_builder().Literal(Value::Tuple({}));

  for (int64_t i = 0; i < token_dags.size(); ++i) {
    ChannelNode* node = token_dags[i].node->As<ChannelNode>();
    XLS_ASSIGN_OR_RETURN(AdapterOutputChannel completion_channel,
                         MakeCompletionChannel(node, i, ab));
    ab.adapter_builder().SendIf(completion_channel.adapter_send_channel_ref,
                                send_token, activations.at(node).activate,
                                empty_tuple_literal);
  }

  return ab.Build(NextState(activations)).status();
}

}  // namespace

absl::StatusOr<bool> ChannelLegalizationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  VLOG(3) << "Running channel legalization pass.";
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(ProcElaboration elab, ElaboratePackage(p));
  XLS_ASSIGN_OR_RETURN(MultipleChannelOps multiple_ops,
                       FindMultipleChannelOps(elab));

  if (multiple_ops.receives.empty() && multiple_ops.sends.empty()) {
    return false;
  }

  NameUniquer global_channel_name_uniquer("__");
  if (!p->ChannelsAreProcScoped()) {
    for (Channel* channel : p->channels()) {
      global_channel_name_uniquer.GetSanitizedUniqueName(channel->name());
    }
  }

  for (const auto& [channel, ops] : multiple_ops.receives) {
    for (Receive* recv : ops) {
      if (!recv->is_blocking()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Non-blocking receives must be the only receive on "
            "a channel; there are multiple receives and %s is non-blocking.",
            recv->GetName()));
      }
    }
    if (channel->kind() != ChannelKind::kStreaming) {
      // Don't make adapters for non-streaming channels.
      VLOG(4) << absl::StreamFormat(
          "Multiple receives on non-streaming channel `%s`", channel->name());
      continue;
    }
    StreamingChannel* streaming_channel = down_cast<StreamingChannel*>(channel);
    if (streaming_channel->GetStrictness() ==
        ChannelStrictness::kProvenMutuallyExclusive) {
      // Don't make adapters for channels that must be proven to be mutually
      // exclusive- they will be handled during scheduling.
      VLOG(3) << absl::StreamFormat(
          "Multiple receives on proven mutually exclusive channel `%s`",
          channel->name());
      continue;
    }

    std::unique_ptr<AdapterBuilder> adapter_builder;
    std::string adapter_name =
        absl::StrFormat("chan_%s_io_receive_adapter", channel->name());
    if (p->ChannelsAreProcScoped()) {
      // For proc-scoped channels all receives are in the same proc.
      Proc* proc = ops.front()->function_base()->AsProcOrDie();
      for (Receive* op : ops) {
        if (op->function_base()->AsProcOrDie() != proc) {
          return absl::UnimplementedError(absl::StrFormat(
              "Channel `%s` has multiple receives in different procs: `%s` and "
              "`%s`",
              channel->name(), proc->name(), op->function_base()->name()));
        }
      }
      XLS_ASSIGN_OR_RETURN(ReceiveChannelReference * channel_ref,
                           proc->GetReceiveChannelReference(channel->name()));
      XLS_ASSIGN_OR_RETURN(
          adapter_builder,
          AdapterBuilder::CreateForProcScopedChannels(channel_ref, adapter_name,
                                                      /*parent_proc=*/proc));
    } else {
      XLS_ASSIGN_OR_RETURN(
          adapter_builder,
          AdapterBuilder::CreateForGlobalChannels(channel, adapter_name, p,
                                                  global_channel_name_uniquer));
    }
    VLOG(3) << absl::StreamFormat(
        "Making receive channel adapter for channel `%s`, has receives (%s).",
        channel->name(), absl::StrJoin(ops, ", "));
    XLS_RETURN_IF_ERROR(AddAdapterForMultipleReceives(ops, *adapter_builder));
    changed = true;
  }

  for (const auto& [channel, ops] : multiple_ops.sends) {
    if (channel->kind() != ChannelKind::kStreaming) {
      // Don't make adapters for non-streaming channels.
      VLOG(4) << absl::StreamFormat(
          "Multiple receives on non-streaming channel `%s`", channel->name());
      continue;
    }
    StreamingChannel* streaming_channel = down_cast<StreamingChannel*>(channel);
    if (streaming_channel->GetStrictness() ==
        ChannelStrictness::kProvenMutuallyExclusive) {
      // Don't make adapters for channels that must be proven to be mutually
      // exclusive- they will be handled during scheduling.
      VLOG(4) << absl::StreamFormat(
          "Multiple sends on proven mutually exclusive channel `%s`",
          channel->name());
      continue;
    }

    std::unique_ptr<AdapterBuilder> adapter_builder;
    std::string adapter_name =
        absl::StrFormat("chan_%s_io_send_adapter", channel->name());
    if (p->ChannelsAreProcScoped()) {
      // For proc-scoped channels all sends are in the same proc.
      Proc* proc = ops.front()->function_base()->AsProcOrDie();
      for (Send* op : ops) {
        if (op->function_base()->AsProcOrDie() != proc) {
          return absl::UnimplementedError(absl::StrFormat(
              "Channel `%s` has multiple sends in different procs: `%s` and "
              "`%s`",
              channel->name(), proc->name(), op->function_base()->name()));
        }
      }
      XLS_ASSIGN_OR_RETURN(SendChannelReference * channel_ref,
                           proc->GetSendChannelReference(channel->name()));
      XLS_ASSIGN_OR_RETURN(
          adapter_builder,
          AdapterBuilder::CreateForProcScopedChannels(channel_ref, adapter_name,
                                                      /*parent_proc=*/proc));
    } else {
      XLS_ASSIGN_OR_RETURN(
          adapter_builder,
          AdapterBuilder::CreateForGlobalChannels(channel, adapter_name, p,
                                                  global_channel_name_uniquer));
    }

    VLOG(4) << absl::StreamFormat(
        "Making send channel adapter for channel `%s`, has sends (%s).",
        channel->name(), absl::StrJoin(ops, ", "));
    XLS_RETURN_IF_ERROR(AddAdapterForMultipleSends(ops, *adapter_builder));
    changed = true;
  }

  return changed;
}

REGISTER_OPT_PASS(ChannelLegalizationPass);

}  // namespace xls
