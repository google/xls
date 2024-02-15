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

#ifndef XLS_PASSES_INLINING_PASS_H_
#define XLS_PASSES_INLINING_PASS_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

class InliningPass : public OptimizationPass {
 public:
  static constexpr std::string_view kName = "inlining";
  InliningPass() : OptimizationPass(kName, "Inlines invocations") {}

  // Inline a single invoke instruction. Provided for test and utility
  // (ir_minimizer) use.
  // Because this is only for ir-minimizer use it allows the inlined function to
  // have invokes in the function code.
  static absl::Status InlineOneInvoke(Invoke* invoke);

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_INLINING_PASS_H_
