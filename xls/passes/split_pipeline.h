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

#ifndef XLS_PASSES_SPLIT_PIPELINE_H_
#define XLS_PASSES_SPLIT_PIPELINE_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls {

// Factory that enables creating segments of the XLS IR optimization pipeline
// split between specified individual optimization passes.
class SplitPipelineFactory {
 public:
  // A segment of an overall XLS IR optimization pipeline.
  // SplitPassInfo elements without a split_pass_name represent sets of passes
  // between split passes.
  struct SplitPassInfo {
    std::string default_pass_name;
    std::optional<std::string> split_pass_name;
  };

  explicit SplitPipelineFactory(
      std::vector<SplitPassInfo> split_pass_infos,
      absl::flat_hash_map<std::string, size_t> split_pass_name_to_index,
      OptimizationPassRegistry registry)
      : split_pass_infos_(std::move(split_pass_infos)),
        split_pass_name_to_index_(std::move(split_pass_name_to_index)),
        registry_(std::move(registry)) {}

  // TODO(joshuata): Move reordering of pipeline proto to within Create()
  static absl::StatusOr<std::unique_ptr<SplitPipelineFactory>> Create(
      OptimizationPassRegistry& registry,
      absl::Span<const std::string> split_passes,
      const OptimizationPipelineProto& pipeline_proto);

  // Returns a pipeline containing all passes before the given split pass.
  // Returns an empty pipeline if the split pass is the first pass in the
  // overall pipeline.
  absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
  GetPipelineBeforeSplit(std::string_view pass_name);

  // Returns a pipeline containing all passes after the given split pass.
  // Returns an empty pipeline if the split pass is the last pass in the overall
  // pipeline.
  absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
  GetPipelineAfterSplit(std::string_view pass_name);

 private:
  const std::vector<SplitPassInfo> split_pass_infos_;
  const absl::flat_hash_map<std::string, size_t> split_pass_name_to_index_;
  const OptimizationPassRegistry registry_;
};

}  // namespace xls

#endif  // XLS_PASSES_SPLIT_PIPELINE_H_
