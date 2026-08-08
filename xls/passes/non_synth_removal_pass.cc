// Copyright 2026 The XLS Authors
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

#include "xls/passes/non_synth_removal_pass.h"

#include <iterator>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/non_synth_separation_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

class NonSynthInvokeRemovalPass : public OptimizationFunctionBasePass {
 public:
  NonSynthInvokeRemovalPass()
      : OptimizationFunctionBasePass("non_synth_invoke_removal",
                                     "Remove non-synthesizable invokes") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    bool changed = false;
    auto node_it = f->nodes().begin();
    while (node_it != f->nodes().end()) {
      auto next_it = std::next(node_it);
      Node* n = *node_it;
      if (n->Is<Invoke>()) {
        Invoke* invoke = n->As<Invoke>();
        if (invoke->to_apply()->non_synth()) {
          XLS_RETURN_IF_ERROR(f->RemoveNode(n));
          changed = true;
        }
      }
      node_it = next_it;
    }
    return changed;
  }
};

}  // namespace

NonSynthRemovalPass::NonSynthRemovalPass()
    : OptimizationCompoundPass(kName,
                               "Strip non-synthesizable nodes and functions") {
  Add<NonSynthSeparationPass>();
  Add<NonSynthInvokeRemovalPass>();
  Add<DeadCodeEliminationPass>();
  Add<DeadFunctionEliminationPass>();
}

REGISTER_OPT_PASS(NonSynthRemovalPass);

}  // namespace xls
