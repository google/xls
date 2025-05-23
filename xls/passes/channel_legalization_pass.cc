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
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/compare.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/token_provenance_analysis.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3_api.h"

namespace xls {

namespace {

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
absl::Status CheckProjectedDAG(
    absl::Span<NodeAndPredecessors const> topo_sorted_dag,
    ChannelStrictness strictness) {
  if (strictness == ChannelStrictness::kTotalOrder) {
    // In topo sorted order, every node must have a single precedent (the
    // previous node in the topo sort) OR the node must be in a different,
    // not-yet-seen FunctionBase.
    absl::flat_hash_set<FunctionBase*> fbs_seen;
    fbs_seen.reserve(topo_sorted_dag[0].node->package()->procs().size());
    std::optional<Node*> previous_node;
    std::optional<FunctionBase*> previous_fb;
    for (const NodeAndPredecessors& node_and_predecessors : topo_sorted_dag) {
      if (previous_fb.has_value() &&
          previous_fb.value() != node_and_predecessors.node->function_base()) {
        XLS_RET_CHECK(
            !fbs_seen.contains(node_and_predecessors.node->function_base()))
            << absl::StreamFormat(
                   "Saw %s twice in topo sorted token DAG",
                   node_and_predecessors.node->function_base()->name());
        fbs_seen.insert(node_and_predecessors.node->function_base());
        previous_node = std::nullopt;
      }
      if (node_and_predecessors.predecessors.empty()) {
        if (previous_fb.has_value() &&
            *previous_fb == node_and_predecessors.node->function_base()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "%v is not totally ordered, multiple nodes have no predecessors.",
              *node_and_predecessors.node));
        }
      } else {
        if (previous_node.has_value()) {
          if (!node_and_predecessors.predecessors.contains(*previous_node)) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "%v is not totally ordered, should come after %v, but comes "
                "after %v.",
                *node_and_predecessors.node, *previous_node.value(),
                **node_and_predecessors.predecessors.begin()));
          }
        } else {
          if (!(*node_and_predecessors.predecessors.begin())->Is<Param>()) {
            return absl::InvalidArgumentError(
                absl::StrFormat("%v is not totally ordered, first operation "
                                "must only depend on token param(s).",
                                *node_and_predecessors.node));
          }
        }
      }
      previous_node = node_and_predecessors.node;
      previous_fb = node_and_predecessors.node->function_base();
    }
  }
  return absl::OkStatus();
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

// Produce a stable topo-sorted list of the given operations; each entry of the
// list is a node together with the set of predecessor nodes. Predecessors are
// resolved to the nearest node in the provided list of operations.
//
// This also resolves transitive predecessors, so if the set of operations is
// {send0, send1, send2} and we have a dependency chain send0 -> recv -> send1
// -> send2, this will return a map with send2 having predecessors {send0,
// send1}.
absl::StatusOr<std::vector<NodeAndPredecessors>> GetProjectedDAG(
    absl::Span<Node* const> operations, ChannelStrictness strictness,
    OptimizationContext& context) {
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
  if (strictness == ChannelStrictness::kProvenMutuallyExclusive ||
      strictness == ChannelStrictness::kRuntimeMutuallyExclusive) {
    for (Node* operation : operations) {
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
  for (const Node* operation : operations) {
    fbs.insert(operation->function_base());
  }

  absl::flat_hash_set<Node*> operation_set(operations.begin(),
                                           operations.end());
  for (FunctionBase* fb : fbs) {
    // Resolve to predecessors in the set of operations.
    absl::flat_hash_map<Node*, NodeAndPredecessors::PredecessorSet>
        resolved_ops;
    // Initialize with all params & literals having no predecessors.
    for (Node* node : fb->nodes()) {
      if (node->OpIn({Op::kParam, Op::kLiteral})) {
        resolved_ops[node] = {};
      }
    }
    // Keep track of prev_node which will be used if we choose a stricter order.
    std::optional<Node*> prev_node = std::nullopt;
    for (Node* node : context.TopoSort(fb)) {
      NodeAndPredecessors::PredecessorSet resolved_predecessors;
      absl::flat_hash_set<Node*> unique_operands(node->operands().begin(),
                                                 node->operands().end());
      for (Node* predecessor : unique_operands) {
        // If a predecessor is not in the set of operations, resolve its
        // predecessors to the set of operations.
        if (!operation_set.contains(predecessor)) {
          const NodeAndPredecessors::PredecessorSet& resolved =
              resolved_ops.at(predecessor);
          resolved_predecessors.insert(resolved.begin(), resolved.end());
          continue;
        }
        resolved_predecessors.insert(predecessor);
      }
      // If this entry in the DAG is not in the set of operations, save its
      // resolved predecessors for future resolution.
      if (!operation_set.contains(node)) {
        resolved_ops[node] = std::move(resolved_predecessors);
        continue;
      }
      // If we choose an arbitrary static order, add the previous value to the
      // set of predecessors. We're already iterating through in topo sorted
      // order, so this only strengthens the dependency relationship.
      if (strictness == ChannelStrictness::kArbitraryStaticOrder) {
        if (prev_node.has_value()) {
          if (!operation_set.contains(*prev_node)) {
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
          .node = node, .predecessors = std::move(transitive_predecessors)});
      result_map.insert({node, &result_vector.back().predecessors});
      prev_node = node;
    }
  }
  XLS_RETURN_IF_ERROR(CheckProjectedDAG(result_vector, strictness));
  return result_vector;
}

Z3_lbool RunSolver(Z3_context c, Z3_ast asserted) {
  Z3_solver solver = solvers::z3::CreateSolver(c, 1);
  Z3_solver_assert(c, solver, asserted);
  Z3_lbool satisfiable = Z3_solver_check(c, solver);
  Z3_solver_dec_ref(c, solver);
  return satisfiable;
}

absl::Status CheckMutualExclusion(Proc* proc,
                                  absl::Span<Node* const> operations,
                                  ChannelStrictness strictness,
                                  OptimizationContext& context) {
  XLS_RET_CHECK(absl::c_all_of(
      operations, [proc](Node* node) { return node->function_base() == proc; }))
      << "Tried to check mutual exclusion for operations not all in the same "
         "proc.";

  std::unique_ptr<solvers::z3::IrTranslator> lazy_translator;
  auto get_translator = [&]() -> absl::StatusOr<solvers::z3::IrTranslator*> {
    if (!lazy_translator) {
      XLS_ASSIGN_OR_RETURN(lazy_translator,
                           solvers::z3::IrTranslator::CreateAndTranslate(
                               proc, /*allow_unsupported=*/true));
    }
    return lazy_translator.get();
  };

  XLS_ASSIGN_OR_RETURN(std::vector<NodeAndPredecessors> dag,
                       GetProjectedDAG(operations, strictness, context));
  for (int64_t i = 0; i < dag.size(); ++i) {
    const NodeAndPredecessors& n1 = dag[i];

    ChannelNode* node = n1.node->As<ChannelNode>();
    std::vector<ChannelNode*> unrelated_nodes;
    for (int64_t j = i + 1; j < dag.size(); ++j) {
      const NodeAndPredecessors& n2 = dag[j];
      if (n1.node->function_base() != n2.node->function_base()) {
        continue;
      }
      if (n1.predecessors.contains(n2.node) ||
          n2.predecessors.contains(n1.node)) {
        // These nodes are related by a dependency chain; no
        // mutual-exclusivity required.
        continue;
      }
      unrelated_nodes.push_back(n2.node->As<ChannelNode>());
    }
    if (unrelated_nodes.empty()) {
      continue;
    }

    std::vector<Node*> unrelated_predicates;
    unrelated_predicates.reserve(unrelated_nodes.size());
    Node* unpredicated_node = nullptr;
    for (ChannelNode* unrelated_node : unrelated_nodes) {
      if (!unrelated_node->predicate().has_value()) {
        unpredicated_node = unrelated_node;
        unrelated_predicates.clear();
        break;
      }
      unrelated_predicates.push_back(*unrelated_node->predicate());
    }

    if (unrelated_predicates.empty() && !node->predicate().has_value()) {
      CHECK_NE(unpredicated_node, nullptr);
      return absl::InvalidArgumentError(absl::StrFormat(
          "Proc %s has two unconditional operations on channel %s with no "
          "ordering: %v and %v",
          proc->name(), node->channel_name(), *node, *unpredicated_node));
    }

    if (strictness != ChannelStrictness::kProvenMutuallyExclusive) {
      Node* no_conflict;
      if (unrelated_predicates.empty() && !node->predicate().has_value()) {
        CHECK_NE(unpredicated_node, nullptr);
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc %s has two unconditional operations on channel %s with no "
            "ordering: %v and %v",
            proc->name(), node->channel_name(), *node, *unpredicated_node));
      } else if (unrelated_predicates.empty()) {
        XLS_ASSIGN_OR_RETURN(
            Node * inactive,
            proc->MakeNode<UnOp>(node->loc(), *node->predicate(), Op::kNot));
        no_conflict = inactive;
      } else {
        XLS_ASSIGN_OR_RETURN(Node * no_unrelated_active,
                             proc->MakeNode<NaryOp>(
                                 node->loc(), unrelated_predicates, Op::kNor));
        no_conflict = no_unrelated_active;
        if (node->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(
              Node * inactive,
              proc->MakeNode<UnOp>(node->loc(), *node->predicate(), Op::kNot));
          XLS_ASSIGN_OR_RETURN(
              no_conflict,
              proc->MakeNode<NaryOp>(
                  node->loc(),
                  absl::MakeConstSpan({inactive, no_unrelated_active}),
                  Op::kOr));
        }
      }
      XLS_RETURN_IF_ERROR(
          proc->MakeNode<Assert>(
                  node->loc(), node->token(), no_conflict,
                  absl::StrFormat("Proc %s has active operations on channel %s "
                                  "in the same activation as %v",
                                  proc->name(), node->channel_name(), *node),
                  /*label=*/
                  verilog::SanitizeVerilogIdentifier(
                      absl::StrCat(node->GetName(), "__no_conflicting_ops_A")),
                  /*original_label=*/std::nullopt)
              .status());
      return absl::OkStatus();
    }

    if (unrelated_predicates.empty()) {
      // Prove that `node` is never active.
      XLS_ASSIGN_OR_RETURN(solvers::z3::IrTranslator * translator,
                           get_translator());
      Z3_lbool can_be_active =
          RunSolver(translator->ctx(),
                    solvers::z3::BitVectorToBoolean(
                        translator->ctx(),
                        translator->GetTranslation(*node->predicate())));
      if (can_be_active == Z3_L_FALSE) {
        // Proved mutual exclusion for `node`.
        continue;
      }
      if (can_be_active == Z3_L_TRUE) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Channel %s is %s, and %v is unconditionally active; proved that "
            "%v is also sometimes active.",
            node->channel_name(), ChannelStrictnessToString(strictness),
            *unpredicated_node, *node));
      }
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel %s is %s, and %v is unconditionally active; failed to "
          "prove that %v is never active.",
          node->channel_name(), ChannelStrictnessToString(strictness),
          *unpredicated_node, *node));
    }

    XLS_ASSIGN_OR_RETURN(solvers::z3::IrTranslator * translator,
                         get_translator());
    Z3_ast unrelated_node_active =
        translator->GetTranslation(unrelated_predicates.front());
    for (Node* predicate :
         absl::MakeConstSpan(unrelated_predicates).subspan(1)) {
      unrelated_node_active =
          Z3_mk_bvor(translator->ctx(), unrelated_node_active,
                     translator->GetTranslation(predicate));
    }

    if (!node->predicate().has_value()) {
      // Prove that no other node can be active.
      XLS_ASSIGN_OR_RETURN(solvers::z3::IrTranslator * translator,
                           get_translator());
      Z3_lbool unrelated_can_be_active = RunSolver(
          translator->ctx(), solvers::z3::BitVectorToBoolean(
                                 translator->ctx(), unrelated_node_active));
      if (unrelated_can_be_active == Z3_L_FALSE) {
        // Proved mutual exclusion for `node`; no assert needed.
        continue;
      }
      if (unrelated_can_be_active == Z3_L_TRUE) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Channel %s is %s, and %v is unconditionally active; proved that "
            "another node on the same channel can be active in the same "
            "activation.",
            node->channel_name(), ChannelStrictnessToString(strictness),
            *node));
      }
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel %s is %s, and %v is unconditionally active; failed to prove "
          "that no other node on the same channel can be active in the same "
          "activation.",
          node->channel_name(), ChannelStrictnessToString(strictness), *node));
    }

    Z3_lbool not_mutually_exclusive = RunSolver(
        translator->ctx(),
        solvers::z3::BitVectorToBoolean(
            translator->ctx(),
            Z3_mk_bvand(translator->ctx(),
                        translator->GetTranslation(*node->predicate()),
                        unrelated_node_active)));
    if (not_mutually_exclusive == Z3_L_FALSE) {
      // Proved mutual exclusion for `node`; no assert needed.
      continue;
    }
    if (not_mutually_exclusive == Z3_L_TRUE) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Channel %s is %s, and %v can be active at the same time as another "
          "node on the same channel.",
          node->channel_name(), ChannelStrictnessToString(strictness), *node));
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "Channel %s is %s, and %v is unconditionally active; failed to prove "
        "that no other node on the same channel can be active in the same "
        "activation.",
        node->channel_name(), ChannelStrictnessToString(strictness), *node));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ChannelLegalizationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  VLOG(3) << "Running channel legalization pass.";
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(ProcElaboration elab, ElaboratePackage(p));
  XLS_ASSIGN_OR_RETURN(MultipleChannelOps multiple_ops,
                       FindMultipleChannelOps(elab));

  for (const auto& [channel, ops] : multiple_ops.receives) {
    if (channel->kind() != ChannelKind::kStreaming) {
      // Don't legalize non-streaming channels.
      VLOG(4) << absl::StreamFormat(
          "Multiple receives on non-streaming channel `%s`", channel->name());
      continue;
    }

    Proc* proc = ops.front()->function_base()->AsProcOrDie();

    std::optional<ChannelStrictness> strictness = ChannelRefStrictness(channel);
    CHECK(strictness.has_value());
    XLS_RETURN_IF_ERROR(
        CheckMutualExclusion(proc, std::vector<Node*>(ops.begin(), ops.end()),
                             *strictness, context));

    // Add cross-activation constraints for all operations on this channel.
    std::vector<StateRead*> self_tokens;
    self_tokens.reserve(ops.size());
    for (Receive* recv : ops) {
      if (recv->function_base()->AsProcOrDie() != proc) {
        return absl::UnimplementedError(absl::StrFormat(
            "Channel `%s` has multiple receives in different procs: `%s` and "
            "`%s`",
            channel->name(), proc->name(), recv->function_base()->name()));
      }
      if (!recv->is_blocking()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Non-blocking receives must be the only receive on "
            "a channel; there are multiple receives and %s is non-blocking.",
            recv->GetName()));
      }
      XLS_ASSIGN_OR_RETURN(StateRead * tok, proc->AppendStateElement(
                                                absl::StrCat("implicit_token__",
                                                             recv->GetName()),
                                                Value::Token()));
      self_tokens.push_back(tok);
      XLS_ASSIGN_OR_RETURN(Node * recv_tok,
                           proc->MakeNode<TupleIndex>(recv->loc(), recv, 0));
      XLS_RETURN_IF_ERROR(proc->MakeNode<Next>(recv->loc(), /*state_read=*/tok,
                                               /*value=*/recv_tok,
                                               /*predicate=*/recv->predicate())
                              .status());
    }
    for (Receive* recv : ops) {
      std::vector<Node*> incoming_tokens;
      incoming_tokens.reserve(1 + self_tokens.size());
      incoming_tokens.push_back(recv->token());
      incoming_tokens.insert(incoming_tokens.end(), self_tokens.begin(),
                             self_tokens.end());
      XLS_ASSIGN_OR_RETURN(
          Node * incoming_token,
          proc->MakeNode<AfterAll>(recv->loc(), incoming_tokens));
      XLS_RETURN_IF_ERROR(recv->ReplaceToken(incoming_token));
    }
    changed = true;
  }

  for (const auto& [channel, ops] : multiple_ops.sends) {
    if (channel->kind() != ChannelKind::kStreaming) {
      // Don't legalize non-streaming channels.
      VLOG(4) << absl::StreamFormat(
          "Multiple sends on non-streaming channel `%s`", channel->name());
      continue;
    }

    Proc* proc = ops.front()->function_base()->AsProcOrDie();

    std::optional<ChannelStrictness> strictness = ChannelRefStrictness(channel);
    CHECK(strictness.has_value());
    XLS_RETURN_IF_ERROR(
        CheckMutualExclusion(proc, std::vector<Node*>(ops.begin(), ops.end()),
                             *strictness, context));

    // Add cross-activation constraints for all operations on this channel.
    std::vector<StateRead*> self_tokens;
    self_tokens.reserve(ops.size());
    for (Send* send : ops) {
      if (send->function_base()->AsProcOrDie() != proc) {
        return absl::UnimplementedError(absl::StrFormat(
            "Channel `%s` has multiple sends in different procs: `%s` and "
            "`%s`",
            channel->name(), proc->name(), send->function_base()->name()));
      }
      XLS_ASSIGN_OR_RETURN(StateRead * tok, proc->AppendStateElement(
                                                absl::StrCat("implicit_token__",
                                                             send->GetName()),
                                                Value::Token()));
      self_tokens.push_back(tok);
      XLS_RETURN_IF_ERROR(proc->MakeNode<Next>(send->loc(), /*state_read=*/tok,
                                               /*value=*/send,
                                               /*predicate=*/send->predicate())
                              .status());
    }
    for (Send* send : ops) {
      std::vector<Node*> incoming_tokens;
      incoming_tokens.reserve(1 + self_tokens.size());
      incoming_tokens.push_back(send->token());
      incoming_tokens.insert(incoming_tokens.end(), self_tokens.begin(),
                             self_tokens.end());
      XLS_ASSIGN_OR_RETURN(
          Node * incoming_token,
          proc->MakeNode<AfterAll>(send->loc(), incoming_tokens));
      XLS_RETURN_IF_ERROR(send->ReplaceToken(incoming_token));
    }
    changed = true;
  }

  return changed;
}

REGISTER_OPT_PASS(ChannelLegalizationPass);

}  // namespace xls
