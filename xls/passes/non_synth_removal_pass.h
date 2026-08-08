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

#ifndef XLS_PASSES_NON_SYNTH_REMOVAL_PASS_H_
#define XLS_PASSES_NON_SYNTH_REMOVAL_PASS_H_

#include <string_view>

#include "xls/passes/optimization_pass.h"

namespace xls {

// A compound optimization pass that removes non-synthesizable operations
// (assert, cover, trace) and their exclusive intermediate dependencies.
//
// Internally runs:
// 1. NonSynthSeparationPass (clones non-synthesizable operations into a
//    non-synth function invoked from the original function).
// 2. NonSynthInvokeRemovalPass (deletes invoke nodes calling non-synth
//    functions).
// 3. DeadCodeEliminationPass (DCE) to delete intermediate operations consumed
//    only by the removed non-synth invokes.
// 4. DeadFunctionEliminationPass (DFE) to remove the cloned non-synth
//    functions.
class NonSynthRemovalPass : public OptimizationCompoundPass {
 public:
  static constexpr std::string_view kName = "non_synth_removal";

  explicit NonSynthRemovalPass();
  ~NonSynthRemovalPass() override = default;
};

}  // namespace xls

#endif  // XLS_PASSES_NON_SYNTH_REMOVAL_PASS_H_
