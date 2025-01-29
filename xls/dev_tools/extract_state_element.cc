// Copyright 2025 The XLS Authors
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

#include "xls/dev_tools/extract_state_element.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {
class RemoveUnusedReceivesPass : public OptimizationProcPass {
 public:
  RemoveUnusedReceivesPass()
      : OptimizationProcPass("remove_unused_receives",
                             "Remove unused receives") {}

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& opts,
                                         PassResults* results) const override {
    bool changed = false;
    for (Node* n : TopoSort(proc)) {
      if (n->Is<Receive>()) {
        if (n->users().empty()) {
          XLS_RETURN_IF_ERROR(proc->RemoveNode(n));
          changed = true;
        }
      } else if (n->Is<TupleIndex>() && n->GetType()->IsToken()) {
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<Literal>(Value::Token()).status());
        changed = true;
      }
    }

    if (changed) {
      // Get rid of unused channels.
      std::vector<Channel*> chans(proc->package()->channels().begin(),
                                  proc->package()->channels().end());
      for (Channel* chan : chans) {
        if (absl::c_all_of(proc->package()->procs(), [&](auto& p) -> bool {
              return absl::c_none_of(p->nodes(), [&](Node* n) -> bool {
                return (n->Is<Receive>() &&
                        n->As<Receive>()->channel_name() == chan->name()) ||
                       (n->Is<Send>() &&
                        n->As<Send>()->channel_name() == chan->name());
              });
            })) {
          XLS_RETURN_IF_ERROR(proc->package()->RemoveChannel(chan));
        }
      }
    }
    return changed;
  }
};

absl::Status ExtractSegmentInto(ProcBuilder& pb, Proc* original,
                                absl::Span<StateElement* const> state_elements,
                                bool send_state_values) {
  Package* new_pkg = pb.package();
  absl::flat_hash_map<Node*, Node*> old_to_new;
  old_to_new.reserve(original->node_count());
  for (Node* n : TopoSort(original)) {
    // Specific node types to handle specially.
    if (n->Is<Invoke>() || n->Is<Map>() || n->Is<CountedFor>() ||
        n->Is<DynamicCountedFor>()) {
      return absl::UnimplementedError("Subroutine calls not supported");
    }
    if (n->OpIn({Op::kRegisterRead, Op::kRegisterWrite, Op::kInputPort,
                 Op::kInstantiationInput, Op::kInstantiationOutput,
                 Op::kOutputPort})) {
      return absl::UnimplementedError("Block node found.");
    }
    if (n->Is<Send>()) {
      // Ignore sends, replace with the input token, DCE will later remove the
      // extra edges by replacing all token values with a literal token.
      old_to_new[n] = old_to_new[n->As<Send>()->token()];
    } else if (n->Is<Receive>()) {
      Receive* r = n->As<Receive>();
      Channel* chan;
      if (new_pkg->HasChannelWithName(r->channel_name())) {
        XLS_ASSIGN_OR_RETURN(chan, new_pkg->GetChannel(r->channel_name()));
      } else {
        XLS_ASSIGN_OR_RETURN(auto orig_chan_ref, r->GetChannelRef());
        Channel* orig_chan = std::get<Channel*>(orig_chan_ref);
        XLS_ASSIGN_OR_RETURN(
            Type * map_ty, new_pkg->MapTypeFromOtherPackage(orig_chan->type()));
        XLS_ASSIGN_OR_RETURN(
            chan, new_pkg->CreateStreamingChannel(
                      r->channel_name(), ChannelOps::kReceiveOnly, map_ty,
                      orig_chan->initial_values()));
        if (r->is_blocking()) {
          if (r->predicate()) {
            old_to_new[n] =
                pb.ReceiveIf(chan, BValue(old_to_new[r->token()], &pb),
                             BValue(old_to_new[*r->predicate()], &pb))
                    .node();
          } else {
            old_to_new[n] =
                pb.Receive(chan, BValue(old_to_new[r->token()], &pb)).node();
          }
        } else {
          if (r->predicate()) {
            old_to_new[n] = pb.ReceiveIfNonBlocking(
                                  chan, BValue(old_to_new[r->token()], &pb),
                                  BValue(old_to_new[*r->predicate()], &pb))
                                .node();
          } else {
            old_to_new[n] =
                pb.ReceiveNonBlocking(chan, BValue(old_to_new[r->token()], &pb))
                    .node();
          }
        }
      }
    } else if (n->Is<StateRead>()) {
      StateRead* s = n->As<StateRead>();
      XLS_ASSIGN_OR_RETURN(Type * ty, new_pkg->MapTypeFromOtherPackage(
                                          s->state_element()->type()));
      if (absl::c_contains(state_elements, s->state_element())) {
        BValue copied_state = pb.StateElement(
            s->state_element()->name(), s->state_element()->initial_value(),
            s->predicate()
                ? std::make_optional(BValue(old_to_new[*s->predicate()], &pb))
                : std::nullopt,
            n->loc());

        old_to_new[n] = copied_state.node();
        if (send_state_values) {
          XLS_ASSIGN_OR_RETURN(
              Channel * st_chan,
              new_pkg->CreateStreamingChannel(
                  absl::StrFormat("%s_value_chan", s->state_element()->name()),
                  ChannelOps::kSendOnly, ty));
          pb.Send(st_chan, pb.Literal(Value::Token()), copied_state, n->loc(),
                  n->HasAssignedName() ? n->GetName() : "");
        }
      } else {
        XLS_ASSIGN_OR_RETURN(
            Channel * st_chan,
            new_pkg->CreateStreamingChannel(
                absl::StrFormat("%s_chan", s->state_element()->name()),
                ChannelOps::kReceiveOnly, ty));
        old_to_new[n] =
            pb.TupleIndex(pb.Receive(st_chan, pb.Literal(Value::Token())), 1,
                          n->loc(), n->HasAssignedName() ? n->GetName() : "")
                .node();
      }
    } else if (n->Is<Next>()) {
      Next* nxt = n->As<Next>();
      if (absl::c_contains(
              state_elements,
              nxt->state_read()->As<StateRead>()->state_element())) {
        if (nxt->predicate()) {
          old_to_new[n] =
              pb.Next(BValue(old_to_new[nxt->state_read()], &pb),
                      BValue(old_to_new[nxt->value()], &pb),
                      BValue(old_to_new[*nxt->predicate()], &pb), nxt->loc(),
                      nxt->HasAssignedName() ? nxt->GetName() : "")
                  .node();
        } else {
          old_to_new[n] =
              pb.Next(BValue(old_to_new[nxt->state_read()], &pb),
                      BValue(old_to_new[nxt->value()], &pb), std::nullopt,
                      nxt->loc(), nxt->HasAssignedName() ? nxt->GetName() : "")
                  .node();
        }
      }
      // Non-extracted nexts can be dropped.
    } else {
      std::vector<Node*> new_ops;
      for (Node* op : n->operands()) {
        XLS_RET_CHECK(old_to_new.contains(op)) << op << " of " << n;
        new_ops.push_back(old_to_new.at(op));
      }
      XLS_ASSIGN_OR_RETURN(auto node,
                           n->CloneInNewFunction(new_ops, pb.function()));
      old_to_new[n] = node;
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<std::unique_ptr<Package>> ExtractStateElementsInNewPackage(
    Proc* proc, absl::Span<StateElement* const> state_elements,
    bool send_state_values) {
  std::unique_ptr<Package> pkg =
      std::make_unique<Package>(proc->package()->name());
  ProcBuilder pb(proc->name(), pkg.get());
  XLS_RETURN_IF_ERROR(
      ExtractSegmentInto(pb, proc, state_elements, send_state_values));
  XLS_RETURN_IF_ERROR(pb.SetAsTop());
  XLS_RETURN_IF_ERROR(pb.Build().status());
  // Use DCE to remove nodes which are not fed into the state elements we
  // actually care about. We know these are the only ones left since sends are
  // removed and other states are replaced with recvs.
  OptimizationFixedPointCompoundPass cleanup("cleanup", "Cleanup");
  cleanup.Add<DeadCodeEliminationPass>();
  cleanup.Add<CsePass>();
  cleanup.Add<DeadCodeEliminationPass>();
  cleanup.Add<RemoveUnusedReceivesPass>();
  PassResults results;
  XLS_RETURN_IF_ERROR(cleanup.Run(pkg.get(), {}, &results).status());
  return pkg;
}

}  // namespace xls
