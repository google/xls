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

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {

// Inlines a package toward the `top` function/proc.
//
// If `full` then all functions are inlined into the `top`.
//
// If `leaf` then only leaf functions are inlined into their caller. This allows
// other passes to optimize on smaller graphs.
class InliningPass : public OptimizationPass {
 public:
  enum class InlineDepth {
    kFull,
    kLeafOnly,
  };
  static constexpr std::string_view kName = "inlining";
  explicit InliningPass(InlineDepth depth = InlineDepth::kFull)
      : OptimizationPass(kName, "Inlines invocations"), depth_(depth) {}

  // Inline a single invoke instruction. Provided for test and utility
  // (ir_minimizer) use.
  // Because this is only for ir-minimizer use it allows the inlined function to
  // have invokes in the function code.
  static absl::Status InlineOneInvoke(Invoke* invoke);

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override;

 protected:
  absl::StatusOr<bool> RunInternal(Package* p,
                                   const OptimizationPassOptions& options,
                                   PassResults* results,
                                   OptimizationContext& context) const override;
  InlineDepth depth_;
};

}  // namespace xls

#endif  // XLS_PASSES_INLINING_PASS_H_
