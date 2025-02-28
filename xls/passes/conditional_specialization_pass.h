// Copyright 2021 The XLS Authors
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

#ifndef XLS_PASSES_CONDITIONAL_SPECIALIZATION_PASS_H_
#define XLS_PASSES_CONDITIONAL_SPECIALIZATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {

// Pass which specializes arms of select operations based on their selector
// value.
class ConditionalSpecializationPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "cond_spec";
  // If `use_bdd` is true, then binary decision diagrams (BDDs) are used for
  // stronger analysis at the cost of slower transformation.
  explicit ConditionalSpecializationPass(bool use_bdd)
      : OptimizationFunctionBasePass(kName, "Conditional specialization"),
        use_bdd_(use_bdd) {}
  ~ConditionalSpecializationPass() override = default;

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override;

 protected:
  bool use_bdd_;
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_CONDITIONAL_SPECIALIZATION_PASS_H_
