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

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// This visitor removes non-synthesizable nodes from a function.
class RemoveNonSynthNodesVisitor : public DfsVisitorWithDefault {
 public:
  explicit RemoveNonSynthNodesVisitor(FunctionBase* f) : f_(f) {}

  absl::Status HandleAssert(Assert* assert) override {
    // Replace the assert with its token before deleting it so that nodes using
    // it as an operand are not left with empty operands.
    XLS_RETURN_IF_ERROR(assert->ReplaceUsesWith(assert->token()));
    XLS_RETURN_IF_ERROR(f_->RemoveNode(assert));
    return absl::OkStatus();
  }

  absl::Status HandleTrace(Trace* trace) override {
    XLS_RETURN_IF_ERROR(trace->ReplaceUsesWith(trace->token()));
    XLS_RETURN_IF_ERROR(f_->RemoveNode(trace));
    return absl::OkStatus();
  }

  absl::Status HandleCover(Cover* cover) override {
    XLS_RETURN_IF_ERROR(
        cover->ReplaceUsesWithNew<Tuple>(std::vector<Node*>()).status());
    XLS_RETURN_IF_ERROR(f_->RemoveNode(cover));
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(Node* node) override { return absl::OkStatus(); }

 private:
  FunctionBase* f_;
};

absl::Status InsertInvokeToNonSynthVersion(
    FunctionBase* f, std::string_view non_synth_function_name, Package* p) {
  XLS_ASSIGN_OR_RETURN(Function * non_synth_function,
                       p->GetFunction(non_synth_function_name));
  XLS_RETURN_IF_ERROR(
      f->MakeNode<Invoke>(
           f->nodes().begin()->loc(),
           std::vector<Node*>(f->params().begin(), f->params().end()),
           non_synth_function)
          .status());
  return absl::OkStatus();
}

// This visitor replaces gate nodes with equivalent select nodes in a function.
class ReplaceGatesWithSelectsVisitor : public DfsVisitorWithDefault {
 public:
  explicit ReplaceGatesWithSelectsVisitor(FunctionBase* f) : f_(f) {}

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
  FunctionBase* f_;
};

absl::Status MakeFunctionReturnUseless(Function* f) {
  XLS_ASSIGN_OR_RETURN(
      Node * default_tuple,
      f->MakeNode<Tuple>(f->nodes().begin()->loc(), std::vector<Node*>()));
  XLS_RETURN_IF_ERROR(f->set_return_value(default_tuple));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> NonSynthSeparationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  NameUniquer name_uniquer(/*separator=*/"_", p->GetFunctionNames());
  for (FunctionBase* f : p->GetFunctionBases()) {
    // TODO: Add support for procs.
    if (!f->IsFunction()) {
      continue;
    }
    // Use "non_synth_" as a prefix for the new non-synthesizable function. If
    // the function name is already taken, the name uniquer will modify the
    // prefix to make the name unique.
    std::string non_synth_function_name = name_uniquer.GetSanitizedUniqueName(
        absl::StrFormat("non_synth_%s", f->name()));
    // Clone the function to act as its non sythesizable version.
    XLS_ASSIGN_OR_RETURN(
        Function * non_synth_function,
        f->AsFunctionOrDie()->Clone(non_synth_function_name, p));
    // Remove the return value of the non-synthesizable function.
    XLS_RETURN_IF_ERROR(MakeFunctionReturnUseless(non_synth_function));
    // Replace gate nodes in the non-synthesizable function with equivalent
    // select nodes because gates are special power-optimization nodes that we
    // generally won't remove so we don't want to duplicate them.
    ReplaceGatesWithSelectsVisitor replace_gates_visitor(non_synth_function);
    for (Node* node : context.TopoSort(non_synth_function)) {
      // Use of VisitSingleNode() instead of Accept() because Accept() sometimes
      // fails if nodes are deleted.
      XLS_RETURN_IF_ERROR(node->VisitSingleNode(&replace_gates_visitor));
    }
    // Insert an invoke to the non-synthesizable function so that the
    // non-sythesizing nodes are reachable from the top function while being
    // optimized separately.
    XLS_RETURN_IF_ERROR(
        InsertInvokeToNonSynthVersion(f, non_synth_function_name, p));
    // Remove the non-synthesizable nodes from the original function.
    RemoveNonSynthNodesVisitor remove_visitor(f);
    for (Node* node : context.TopoSort(f)) {
      XLS_RETURN_IF_ERROR(node->VisitSingleNode(&remove_visitor));
    }
  }
  return true;
}

}  // namespace xls
