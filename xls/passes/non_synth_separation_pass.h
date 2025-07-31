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

#ifndef XLS_PASSES_NON_SYNTH_SEPARATION_PASS_H_
#define XLS_PASSES_NON_SYNTH_SEPARATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Separates out non-synthesizable nodes like assert/cover/trace from the main
// function into a cloned function. Every function effectively has two versions,
// one with synthesizable nodes and one without. The synthesizable version
// invokes the non-synthesizable version of its function. This ensures that
// non-synthesizable uses of values do not affect the optimization of the
// synthesizable parts of the function.
class NonSynthSeparationPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "non_synth_separation";
  explicit NonSynthSeparationPass()
      : OptimizationPass(kName, "Non-Synthesizable Separation") {}
  ~NonSynthSeparationPass() override = default;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_NON_SYNTH_SEPARATION_PASS_H_
