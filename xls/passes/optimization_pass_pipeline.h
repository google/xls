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

#ifndef XLS_PASSES_OPTIMIZATION_PASS_PIPELINE_H_
#define XLS_PASSES_OPTIMIZATION_PASS_PIPELINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"

namespace xls {

class SimplificationPass : public OptimizationCompoundPass {
 public:
  explicit SimplificationPass(int64_t opt_level);
};

class FixedPointSimplificationPass : public OptimizationFixedPointCompoundPass {
 public:
  explicit FixedPointSimplificationPass(int64_t opt_level);
};

// CreateOptimizationPassPipeline connects together the various optimization
// and analysis passes in the order of execution.
std::unique_ptr<OptimizationCompoundPass> CreateOptimizationPassPipeline(
    int64_t opt_level = kMaxOptLevel);

// Creates and runs the standard pipeline on the given package with default
// options.
absl::StatusOr<bool> RunOptimizationPassPipeline(
    Package* package, int64_t opt_level = kMaxOptLevel);

class OptimizationPassPipelineGenerator final
    : public OptimizationPipelineGenerator {
 public:
  OptimizationPassPipelineGenerator(std::string_view short_name,
                                    std::string_view long_name,
                                    int64_t opt_level)
      : OptimizationPipelineGenerator(short_name, long_name),
        opt_level_(opt_level) {}

  std::vector<std::string_view> GetAvailablePasses() const;
  std::string GetAvailablePassesStr() const;

 protected:
  absl::Status AddPassToPipeline(OptimizationCompoundPass* pass,
                                 std::string_view pass_name) const final;

 private:
  int64_t opt_level_;
};

inline OptimizationPassPipelineGenerator GetOptimizationPipelineGenerator(
    int64_t opt_level) {
  return OptimizationPassPipelineGenerator(
      "opt_pipeline", "optimization_pass_pipeline_generator", opt_level);
}

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_PIPELINE_H_
