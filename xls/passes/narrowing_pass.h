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

#ifndef XLS_PASSES_NARROWING_PASS_H_
#define XLS_PASSES_NARROWING_PASS_H_

#include <cstdint>
#include <ostream>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"

namespace xls {

// A pass which reduces the width of operations eliminating redundant or unused
// bits.
class NarrowingPass : public OptimizationFunctionBasePass {
 public:
  enum class AnalysisType : uint8_t {
    kTernary,
    kRange,
    // Use the select context of instructions when calculating ranges.
    kRangeWithContext,
    // Use the select context controlled by the optimization options.
    kRangeWithOptionalContext,
  };
  static constexpr std::string_view kName = "narrow";
  explicit NarrowingPass(AnalysisType analysis = AnalysisType::kRange)
      : OptimizationFunctionBasePass(kName, "Narrowing"), analysis_(analysis) {}
  ~NarrowingPass() override = default;

  absl::StatusOr<PassPipelineProto::Element> ToProto() const override;

 protected:
  AnalysisType analysis_;

  AnalysisType RealAnalysis(const OptimizationPassOptions& options) const;
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};

std::ostream& operator<<(std::ostream& os, NarrowingPass::AnalysisType a);

}  // namespace xls

#endif  // XLS_PASSES_NARROWING_PASS_H_
