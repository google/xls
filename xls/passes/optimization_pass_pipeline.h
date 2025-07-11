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
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pipeline_generator.h"

namespace xls {

// CreateOptimizationPassPipeline connects together the various optimization
// and analysis passes in the order of execution. The actual passes executed is
// defined by the OptimizationPipelineProto passed to the pass registry.
//
// By default this is found in `optimization_pass_pipeline.txtpb`.
absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
TryCreateOptimizationPassPipeline(
    bool debug_optimizations = false,
    const OptimizationPassRegistry& registry = GetOptimizationRegistry());

// CreateOptimizationPassPipeline connects together the various optimization
// and analysis passes in the order of execution. The actual passes executed is
// defined by the OptimizationPipelineProto passed to the pass registry.
//
// By default this is found in `optimization_pass_pipeline.txtpb`.
inline std::unique_ptr<OptimizationCompoundPass> CreateOptimizationPassPipeline(
    bool debug_optimizations = false,
    const OptimizationPassRegistry& registry = GetOptimizationRegistry()) {
  absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>> res =
      TryCreateOptimizationPassPipeline(debug_optimizations, registry);
  CHECK_OK(res);
  return *std::move(res);
}

// Creates and runs the standard pipeline on the given package with default
// options. The actual passes executed is defined by the
// OptimizationPipelineProto passed to the pass registry.
//
// By default this is found in `optimization_pass_pipeline.txtpb`.
absl::StatusOr<bool> RunOptimizationPassPipeline(
    Package* package, int64_t opt_level = kMaxOptLevel,
    bool debug_optimizations = false);

class OptimizationPassPipelineGenerator : public OptimizationPipelineGenerator {
 public:
  OptimizationPassPipelineGenerator(
      std::string_view short_name, std::string_view long_name,
      const OptimizationPassRegistry& registry = GetOptimizationRegistry())
      : OptimizationPipelineGenerator(short_name, long_name),
        registry_(registry) {}

  std::vector<std::string_view> GetAvailablePasses() const;
  std::string GetAvailablePassesStr() const;

  absl::StatusOr<std::unique_ptr<OptimizationPass>> FinalizeWithOptions(
      std::unique_ptr<OptimizationPass>&& cur,
      const BasicPipelineOptions& options) const override;

 protected:
  absl::Status AddPassToPipeline(
      OptimizationCompoundPass* pass, std::string_view pass_name,
      const BasicPipelineOptions& options) const final;

 private:
  const OptimizationPassRegistry& registry_;
};

inline OptimizationPassPipelineGenerator GetOptimizationPipelineGenerator(
    const OptimizationPassRegistry& registry = GetOptimizationRegistry()) {
  return OptimizationPassPipelineGenerator(
      "opt_pipeline", "optimization_pass_pipeline_generator", registry);
}

}  // namespace xls

#endif  // XLS_PASSES_OPTIMIZATION_PASS_PIPELINE_H_
