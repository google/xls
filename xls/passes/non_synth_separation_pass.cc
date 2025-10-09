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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
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
        f_->MakeNode<Literal>(gate->loc(), ZeroOfType(gate->GetType())));
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
// except proc only nodes which get their values passed to the new function as
// parameters.
class CloneProcAsFunctionVisitor : public DfsVisitorWithDefault {
 public:
  explicit CloneProcAsFunctionVisitor(Function* non_synth_function)
      : non_synth_function_(non_synth_function) {}

  // Track the send token graph.
  //
  // TODO(allight): If we end up creating a 'try_send' as has been suggested
  // this will need to be updated to take the result of that send.
  absl::Status HandleSend(Send* send) override {
    node_map_[send] = node_map_[send->token()];
    return absl::OkStatus();
  }

  absl::Status HandleReceive(Receive* receive) override {
    // We don't need to hold the token itself.
    Type* inp_type = receive->package()->GetTupleType(
        receive->GetType()->AsTupleOrDie()->element_types().subspan(1));
    std::string param_name =
        receive->HasAssignedName()
            ? absl::StrCat(receive->GetNameView(), "_param")
            : absl::StrFormat("recv_%d_param", receive->id());
    XLS_ASSIGN_OR_RETURN(Node * param,
                         non_synth_function_->MakeNodeWithName<Param>(
                             receive->loc(), inp_type, param_name));
    // Splice in the token.
    XLS_ASSIGN_OR_RETURN(
        node_map_[receive],
        InsertIntoTuple(param, node_map_[receive->token()], {0}));
    // Make the invoke arg.
    XLS_ASSIGN_OR_RETURN(Node * sliced_recv, SliceTuple(receive, 1));
    args_.push_back(sliced_recv);
    return absl::OkStatus();
  }

  // Slice out any token.
  absl::Status HandleStateRead(StateRead* state_read) override {
    std::string param_name =
        state_read->HasAssignedName()
            ? absl::StrCat(state_read->GetNameView(), "_param")
            : absl::StrFormat("state_read_%d_param", state_read->id());
    if (!TypeHasToken(state_read->GetType())) {
      XLS_ASSIGN_OR_RETURN(
          Node * param, non_synth_function_->MakeNodeWithName<Param>(
                            state_read->loc(),
                            state_read->state_element()->type(), param_name));
      node_map_[state_read] = param;
      args_.push_back(state_read);
      return absl::OkStatus();
    }
    // Assert/trace/cover ordering is not really observable at the proc/func
    // level and the scheduler has significant freedom to reorder them during
    // codegen. This means we can simply break all cross activation token edges
    // feeding the asserts/trace/covers and just give them all a new token to
    // order themselves with.
    //
    // TODO(allight): We should consider collapsing all tokens in state elements
    // together in some cases.
    Node* inp = state_read;
    std::vector<std::vector<int64_t>> removed_tokens;
    while (true) {
      LeafTypeTree<std::monostate> ltt(inp->GetType(), std::monostate{});
      XLS_ASSIGN_OR_RETURN(
          std::optional<std::vector<int64_t>> token_indexes,
          leaf_type_tree::FindMatchingIndex<std::monostate>(
              ltt.AsView(),
              [](Type* t, auto& v, auto idx) -> absl::StatusOr<bool> {
                return t->IsToken();
              }));
      if (!token_indexes) {
        break;
      }
      XLS_ASSIGN_OR_RETURN(inp, RemoveFromTuple(inp, *token_indexes));
      removed_tokens.push_back(*std::move(token_indexes));
    }
    XLS_ASSIGN_OR_RETURN(Node * param,
                         non_synth_function_->MakeNodeWithName<Param>(
                             state_read->loc(), inp->GetType(), param_name));
    // Restore the tokens. Just use literal token for all of them.
    XLS_ASSIGN_OR_RETURN(Node * token, non_synth_function_->MakeNode<Literal>(
                                           param->loc(), Value::Token()));
    for (absl::Span<int64_t const> indices : iter::reversed(removed_tokens)) {
      XLS_ASSIGN_OR_RETURN(param, InsertIntoTuple(param, token, indices));
    }
    node_map_[state_read] = param;
    args_.push_back(inp);
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
  Function* non_synth_function_;
  absl::flat_hash_map<Node*, Node*> node_map_;
  std::vector<Node*> args_;
};

absl::StatusOr<std::pair<Function*, std::vector<Node*>>> CloneProcAsFunction(
    Proc* proc, std::string_view non_synth_function_name, Package* p,
    OptimizationContext& context) {
  Function* f =
      p->AddFunction(std::make_unique<Function>(non_synth_function_name, p));
  CloneProcAsFunctionVisitor clone_function_visitor(f);
  for (Node* n : context.TopoSort(proc)) {
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&clone_function_visitor));
  }
  return std::make_pair(clone_function_visitor.f(),
                        clone_function_visitor.args());
}

bool IsNonSynthNode(Node* node) {
  return node->OpIn({Op::kAssert, Op::kTrace, Op::kCover});
}

}  // namespace

absl::StatusOr<bool> NonSynthSeparationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;
  NameUniquer name_uniquer(/*separator=*/"_", p->GetFunctionNames());
  for (FunctionBase* function_base : p->GetFunctionBases()) {
    // No need to create a new function for function bases that don't contain
    // non-synthesizable nodes.
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
          CloneProcAsFunction(proc, non_synth_function_name, p, context));
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
