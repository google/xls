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

#include "xls/passes/non_synth_separation_pass.h"

#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// This visitor removes non-synthesizable nodes from a function base.
class RemoveNonSynthNodesVisitor : public DfsVisitorWithDefault {
 public:
  explicit RemoveNonSynthNodesVisitor(FunctionBase* function_base)
      : function_base_(function_base) {}

  absl::Status HandleAssert(Assert* assert) override {
    // Replace the assert with its token before deleting it so that nodes using
    // it as an operand are not left with empty operands.
    XLS_RETURN_IF_ERROR(assert->ReplaceUsesWith(assert->token()));
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(assert));
    return absl::OkStatus();
  }

  absl::Status HandleTrace(Trace* trace) override {
    XLS_RETURN_IF_ERROR(trace->ReplaceUsesWith(trace->token()));
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(trace));
    return absl::OkStatus();
  }

  absl::Status HandleCover(Cover* cover) override {
    XLS_RETURN_IF_ERROR(
        cover->ReplaceUsesWithNew<Tuple>(std::vector<Node*>()).status());
    XLS_RETURN_IF_ERROR(function_base_->RemoveNode(cover));
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }

 private:
  FunctionBase* function_base_;
};

// This visitor replaces gate nodes with equivalent select nodes in a function.
class ReplaceGatesWithSelectsVisitor : public DfsVisitorWithDefault {
 public:
  explicit ReplaceGatesWithSelectsVisitor(Function* f) : f_(f) {}

  absl::Status HandleGate(Gate* gate) override {
    XLS_ASSIGN_OR_RETURN(
        Node * zero_literal,
        f_->MakeNode<Literal>(gate->loc(),
                              Value(UBits(0, gate->data()->BitCountOrDie()))));
    XLS_RETURN_IF_ERROR(
        gate->ReplaceUsesWithNew<Select>(gate->condition(),
                                         std::vector<Node*>({gate->data()}),
                                         zero_literal)
            .status());
    XLS_RETURN_IF_ERROR(f_->RemoveNode(gate));
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }

 private:
  Function* f_;
};

absl::Status InsertInvokeInProc(Proc* proc, Function* non_synth_function,
                                std::vector<Node*> non_synth_args) {
  XLS_RETURN_IF_ERROR(proc->MakeNode<Invoke>(proc->nodes().begin()->loc(),
                                             non_synth_args, non_synth_function)
                          .status());
  return absl::OkStatus();
}

absl::Status InsertInvokeInFunction(Function* f, Function* non_synth_function) {
  XLS_RETURN_IF_ERROR(
      f->MakeNode<Invoke>(
           f->nodes().begin()->loc(),
           std::vector<Node*>(f->params().begin(), f->params().end()),
           non_synth_function)
          .status());
  return absl::OkStatus();
}

absl::Status MakeFunctionReturnUseless(Function* f) {
  XLS_ASSIGN_OR_RETURN(
      Node * default_tuple,
      f->MakeNode<Tuple>(f->nodes().begin()->loc(), std::vector<Node*>()));
  XLS_RETURN_IF_ERROR(f->set_return_value(default_tuple));
  return absl::OkStatus();
}

// This visitor creates a new function from a proc with all of the same nodes
// except proc only nodes which get passed to the new function as parameters.
class CloneProcAsFunctionVisitor : public DfsVisitorWithDefault {
 public:
  explicit CloneProcAsFunctionVisitor(Proc* proc, Function* non_synth_function,
                                      Package* p)
      : proc_(proc), non_synth_function_(non_synth_function), p_(p) {}

  // Add a param to the new function that has the same type as the send node
  // return value. The send node return value is then passed to the new function
  // as an argument.
  // TODO: Reconsider not changing the token graph in the scenario that a
  // send/receive depends on an assert/trace token.
  absl::Status HandleSend(Send* send) override {
    XLS_ASSIGN_OR_RETURN(Node * param,
                         non_synth_function_->MakeNode<Param>(
                             send->loc(), send->package()->GetTokenType()));
    // Map the send node to the param node in the new function.
    node_map_[send] = param;
    // Prepare the send node as an argument used by the invoke.
    args_.push_back(send);
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* receive) override {
    // Get the param type of the receive.
    Type* channel_type;
    if (proc_->is_new_style_proc()) {
      XLS_ASSIGN_OR_RETURN(
          ChannelInterface * channel_ref,
          proc_->GetChannelInterface(receive->channel_name(),
                                     ChannelDirection::kReceive));
      channel_type = channel_ref->type();
    } else {
      XLS_ASSIGN_OR_RETURN(Channel * channel,
                           p_->GetChannel(receive->channel_name()));
      channel_type = channel->type();
    }
    Type* param_type =
        receive->is_blocking()
            ? p_->GetTupleType({p_->GetTokenType(), channel_type})
            : p_->GetTupleType(
                  {p_->GetTokenType(), channel_type, p_->GetBitsType(1)});
    XLS_ASSIGN_OR_RETURN(Node * param, non_synth_function_->MakeNode<Param>(
                                           receive->loc(), param_type));
    node_map_[receive] = param;
    args_.push_back(receive);
    return absl::OkStatus();
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    XLS_ASSIGN_OR_RETURN(
        Node * param,
        non_synth_function_->MakeNode<Param>(
            state_read->loc(), state_read->state_element()->type()));
    node_map_[state_read] = param;
    args_.push_back(state_read);
    return absl::OkStatus();
  }

  // Do not transfer the next node to the new function.
  absl::Status HandleNext(Next* next) override { return absl::OkStatus(); }

  // Clone all other nodes to the new function.
  absl::Status DefaultHandler(Node* node) override {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map_.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_node,
        node->CloneInNewFunction(new_operands, non_synth_function_));
    node_map_[node] = new_node;
    return absl::OkStatus();
  }

  Function* f() { return non_synth_function_; }
  std::vector<Node*> args() { return args_; }

 private:
  Proc* proc_;
  Function* non_synth_function_;
  Package* p_;
  absl::flat_hash_map<Node*, Node*> node_map_;
  std::vector<Node*> args_;
};

absl::StatusOr<std::pair<Function*, std::vector<Node*>>> CloneProcAsFunction(
    Proc* proc, std::string_view non_synth_function_name, Package* p) {
  Function* f =
      p->AddFunction(std::make_unique<Function>(non_synth_function_name, p));
  CloneProcAsFunctionVisitor clone_function_visitor(proc, f, p);
  XLS_RETURN_IF_ERROR(proc->Accept(&clone_function_visitor));
  return std::make_pair(clone_function_visitor.f(),
                        clone_function_visitor.args());
}

bool IsNonSynthNode(Node* node) {
  return node->Is<Assert>() || node->Is<Trace>() || node->Is<Cover>();
}

}  // namespace

absl::StatusOr<bool> NonSynthSeparationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;
  NameUniquer name_uniquer(/*separator=*/"_", p->GetFunctionNames());
  for (FunctionBase* function_base : p->GetFunctionBases()) {
    // No need to create a new function for empty function bases.
    if (absl::c_none_of(function_base->nodes(), IsNonSynthNode)) {
      continue;
    }
    changed = true;
    // Use "non_synth_" as a prefix for the new non-synthesizable function. If
    // the function name is already taken, the name uniquer will modify the
    // prefix to make the name unique.
    std::string non_synth_function_name = name_uniquer.GetSanitizedUniqueName(
        absl::StrFormat("non_synth_%s", function_base->name()));
    Function* non_synth_function = nullptr;
    if (function_base->IsFunction()) {
      Function* f = function_base->AsFunctionOrDie();
      // Clone the function to act as its non-sythesizable version.
      XLS_ASSIGN_OR_RETURN(non_synth_function,
                           f->Clone(non_synth_function_name, p));
      // Remove the return value of the non-synthesizable function.
      XLS_RETURN_IF_ERROR(MakeFunctionReturnUseless(non_synth_function));
      // Insert an invoke to the non-synthesizable function so that the
      // non-sythesizing nodes are reachable from the top function while being
      // optimized separately.
      XLS_RETURN_IF_ERROR(InsertInvokeInFunction(f, non_synth_function));
    } else if (function_base->IsProc()) {
      Proc* proc = function_base->AsProcOrDie();
      std::vector<Node*> non_synth_args;
      // Clone the proc as a non-sythesizable function. Proc only nodes are
      // removed and any node that uses the proc only nodes retrieve their
      // values from new function parameters.
      XLS_ASSIGN_OR_RETURN(
          std::tie(non_synth_function, non_synth_args),
          CloneProcAsFunction(proc, non_synth_function_name, p));
      // Add a useless return type to the function.
      XLS_RETURN_IF_ERROR(MakeFunctionReturnUseless(non_synth_function));
      // Create an invoke in the proc to the non-synthesizable function. Pass
      // the proc only nodes as arguments.
      XLS_RETURN_IF_ERROR(
          InsertInvokeInProc(proc, non_synth_function, non_synth_args));
    }
    // Replace gate nodes in the non-synthesizable function with equivalent
    // select nodes because gates are special power-optimization nodes that we
    // generally won't remove so we don't want to duplicate them.
    ReplaceGatesWithSelectsVisitor replace_gates_visitor(non_synth_function);
    for (Node* node : context.TopoSort(non_synth_function)) {
      XLS_RETURN_IF_ERROR(node->VisitSingleNode(&replace_gates_visitor));
    }
    // Remove the non-synthesizable nodes from the original function base.
    RemoveNonSynthNodesVisitor remove_visitor(function_base);
    for (Node* node : context.TopoSort(function_base)) {
      XLS_RETURN_IF_ERROR(node->VisitSingleNode(&remove_visitor));
    }
  }
  return changed;
}

}  // namespace xls
