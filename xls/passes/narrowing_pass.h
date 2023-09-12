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
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/optimization_pass.h"

namespace xls {


// A pass which reduces the width of operations eliminating redundant or unused
// bits.
class NarrowingPass : public OptimizationFunctionBasePass {
 public:
  enum class AnalysisType : uint8_t {
    kBdd,
    kRange,
  };
  explicit NarrowingPass(AnalysisType analysis = AnalysisType::kRange,
                         int64_t opt_level = kMaxOptLevel)
      : OptimizationFunctionBasePass("narrow", "Narrowing"),
        analysis_(analysis),
        opt_level_(opt_level) {}
  ~NarrowingPass() override = default;

 protected:
  AnalysisType analysis_;
  int64_t opt_level_;
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_NARROWING_PASS_H_
