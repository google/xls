// Copyright 2020 The XLS Authors
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

#include "xls/passes/unroll_pass.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls {
namespace {

// Finds an "effectively used" (has users or is return value) counted for in the
// function f, or returns nullptr if none is found.
CountedFor* FindCountedFor(FunctionBase* f) {
  for (Node* node : TopoSort(f)) {
    if (node->Is<CountedFor>() &&
        (f->HasImplicitUse(node) || !node->users().empty())) {
      return node->As<CountedFor>();
    }
  }
  return nullptr;
}

// Unrolls the node "loop" by replacing it with a sequence of dependent
// invocations.
absl::Status UnrollCountedFor(CountedFor* loop) {
  FunctionBase* f = loop->function_base();
  Node* loop_carry = loop->initial_value();
  int64_t ivar_bit_count = loop->body()->params()[0]->BitCountOrDie();
  for (int64_t trip = 0, iv = 0; trip < loop->trip_count();
       ++trip, iv += loop->stride()) {
    XLS_ASSIGN_OR_RETURN(
        Literal * iv_node,
        f->MakeNode<Literal>(loop->loc(), Value(UBits(iv, ivar_bit_count))));

    // Construct the args for invocation.
    std::vector<Node*> invoke_args = {iv_node, loop_carry};
    for (Node* invariant_arg : loop->invariant_args()) {
      invoke_args.push_back(invariant_arg);
    }

    XLS_ASSIGN_OR_RETURN(
        loop_carry,
        f->MakeNode<Invoke>(loop->loc(), absl::MakeSpan(invoke_args),
                            loop->body()));
  }
  XLS_RETURN_IF_ERROR(loop->ReplaceUsesWith(loop_carry));
  return f->RemoveNode(loop);
}

}  // namespace

absl::StatusOr<bool> UnrollPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  while (true) {
    CountedFor* loop = FindCountedFor(f);
    if (loop == nullptr) {
      break;
    }
    XLS_RETURN_IF_ERROR(UnrollCountedFor(loop));
    changed = true;
  }
  return changed;
}

REGISTER_OPT_PASS(UnrollPass);

}  // namespace xls
