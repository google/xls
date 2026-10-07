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

#include "xls/passes/split_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/enumerate.hpp"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls {

namespace {
constexpr int64_t kDefaultMinOptLevel = 0;
constexpr int64_t kDefaultCapOptLevel = 5;

struct PassInfo {
  std::string name;
  int64_t min_opt_level;
  int64_t cap_opt_level;
};

// Recursively walks the given compound pass and accumulates a list of leaf pass
// names along with their opt level ranges.
absl::Status GetLeafPassHelper(
    std::string_view pass_name,
    const absl::flat_hash_map<std::string,
                              OptimizationPipelineProto::CompoundPass>&
        name_to_compound_pass_map,
    std::vector<PassInfo>& leaf_passes,
    absl::flat_hash_set<std::string>& active_path,
    int64_t default_min_opt_level = kDefaultMinOptLevel,
    int64_t default_cap_opt_level = kDefaultCapOptLevel) {
  if (active_path.contains(pass_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cycle detected involving pass '", pass_name,
                     "' and active path: ", absl::StrJoin(active_path, ", ")));
  }

  auto compound_pass_it = name_to_compound_pass_map.find(pass_name);
  if (compound_pass_it == name_to_compound_pass_map.end()) {
    leaf_passes.push_back(PassInfo{
        .name = std::string(pass_name),
        .min_opt_level = default_min_opt_level,
        .cap_opt_level = default_cap_opt_level,
    });
    return absl::OkStatus();
  }
  const OptimizationPipelineProto::CompoundPass& compound_pass =
      compound_pass_it->second;
  const int64_t min_opt_level = compound_pass.options().has_min_opt_level()
                                    ? compound_pass.options().min_opt_level()
                                    : default_min_opt_level;
  const int64_t cap_opt_level = compound_pass.options().has_cap_opt_level()
                                    ? compound_pass.options().cap_opt_level()
                                    : default_cap_opt_level;
  if (compound_pass.fixedpoint()) {
    leaf_passes.push_back(PassInfo{
        .name = std::string(pass_name),
        .min_opt_level = min_opt_level,
        .cap_opt_level = cap_opt_level,
    });
    return absl::OkStatus();
  }

  active_path.insert(std::string(pass_name));
  absl::Cleanup remove_from_path = [&active_path, pass_name] {
    active_path.erase(pass_name);
  };

  for (const std::string_view pass : compound_pass.passes()) {
    XLS_RETURN_IF_ERROR(GetLeafPassHelper(pass, name_to_compound_pass_map,
                                          leaf_passes, active_path,
                                          min_opt_level, cap_opt_level));
  }
  return absl::OkStatus();
}

// Recursively walks the given optimization pipeline proto and returns a list
// of leaf optimization pass infos. Leaf passes are passes that are not
// CompoundPasses or that have fixedpoint set to true.
absl::StatusOr<std::vector<PassInfo>> GetLeafPassInfos(
    const OptimizationPipelineProto& pipeline_proto,
    const absl::flat_hash_map<std::string,
                              OptimizationPipelineProto::CompoundPass>&
        name_to_compound_pass_map) {
  std::vector<PassInfo> leaf_passes;
  absl::flat_hash_set<std::string> active_path;
  for (const std::string_view pass : pipeline_proto.default_pipeline()) {
    XLS_RETURN_IF_ERROR(GetLeafPassHelper(pass, name_to_compound_pass_map,
                                          leaf_passes, active_path));
  }
  return leaf_passes;
}

struct SplitPassResult {
  std::vector<SplitPipelineFactory::SplitPassInfo> split_pass_infos;
  std::vector<OptimizationPipelineProto::CompoundPass> synthetic_passes;
  std::vector<OptimizationPipelineProto::CompoundPass> top_level_passes;
};

// Returns an ordered list of pass information splitting the optimization
// pipeline up based on the defined split passes, using the default pipeline
// order.
//
// The pass hierarchy is rearranged into two level:
//   - Top level passes: Passes that are either split passes or groups of passes
//   between split passes.
//   - Synthetic passes: Passes created to hold individual leaf passes, to
//   maintain correct opt levels.
absl::StatusOr<SplitPassResult> GetSplitPassInfos(
    const OptimizationPipelineProto& pipeline_proto,
    const absl::Span<const std::string> split_passes) {
  VLOG(5) << "Creating split pass info for splits:";
  VLOG(5) << "  " << absl::StrJoin(split_passes, ", ");

  absl::flat_hash_map<std::string, OptimizationPipelineProto::CompoundPass>
      pass_name_to_compound_pass;
  for (const OptimizationPipelineProto::CompoundPass& compound_pass :
       pipeline_proto.compound_passes()) {
    pass_name_to_compound_pass[compound_pass.short_name()] = compound_pass;
  }

  XLS_ASSIGN_OR_RETURN(
      std::vector<PassInfo> leaf_passes,
      GetLeafPassInfos(pipeline_proto, pass_name_to_compound_pass));

  // Determine splits
  absl::flat_hash_set<std::string> seen_splits;
  std::vector<size_t> split_indices;
  absl::flat_hash_set<std::string> split_pass_names(split_passes.begin(),
                                                    split_passes.end());
  for (size_t i = 0; i < leaf_passes.size(); ++i) {
    const PassInfo& pass_info = leaf_passes[i];
    if (!split_pass_names.contains(pass_info.name)) {
      continue;
    }
    if (!seen_splits.insert(pass_info.name).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Multiple instances of split pass '", pass_info.name,
                       "' found in pipeline"));
    }
    split_indices.push_back(i);
  }

  if (seen_splits.size() < split_pass_names.size()) {
    std::vector<std::string> missing_passes;
    for (const auto& name : split_pass_names) {
      if (!seen_splits.contains(name)) {
        missing_passes.push_back(name);
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat("Requested split passes not found as valid leaf passes: ",
                     absl::StrJoin(missing_passes, ", ")));
  }

  auto get_next_top_level_pass_name = [&,
                                       counter = 0]() mutable -> std::string {
    std::string pass_name;
    do {
      pass_name = absl::StrCat("TOP_LEVEL_PASS_", counter);
      ++counter;
    } while (pass_name_to_compound_pass.contains(pass_name));
    return pass_name;
  };

  std::vector<OptimizationPipelineProto::CompoundPass> synthetic_passes;
  synthetic_passes.reserve(leaf_passes.size());
  auto add_synthetic_pass =
      [&, counter = 0](const PassInfo& pass) mutable -> std::string {
    std::string pass_name;
    do {
      pass_name = absl::StrCat("SYNTH_PASS_", counter);
      ++counter;
    } while (pass_name_to_compound_pass.contains(pass_name));
    OptimizationPipelineProto::CompoundPass& synthetic_pass =
        synthetic_passes.emplace_back();
    synthetic_pass.set_short_name(pass_name);
    synthetic_pass.add_passes(pass.name);
    synthetic_pass.mutable_options()->set_min_opt_level(pass.min_opt_level);
    synthetic_pass.mutable_options()->set_cap_opt_level(pass.cap_opt_level);
    return synthetic_pass.short_name();
  };

  std::vector<OptimizationPipelineProto::CompoundPass> top_level_passes;
  std::vector<SplitPipelineFactory::SplitPassInfo> split_pass_infos;
  std::optional<size_t> last_split_index;
  for (size_t split_index : split_indices) {
    // Build synthetic compound passes for passes between last split and this
    // split to maintain correct opt levels.
    size_t start_index =
        last_split_index.has_value() ? *last_split_index + 1 : 0;
    if (split_index > start_index) {
      OptimizationPipelineProto::CompoundPass& top_level_pass =
          top_level_passes.emplace_back();
      std::string top_level_name = get_next_top_level_pass_name();
      top_level_pass.set_short_name(top_level_name);
      for (size_t i = start_index; i < split_index; ++i) {
        std::string synthetic_pass_name = add_synthetic_pass(leaf_passes[i]);
        top_level_pass.add_passes(synthetic_pass_name);
      }
      split_pass_infos.emplace_back(SplitPipelineFactory::SplitPassInfo{
          .default_pass_name = top_level_name,
          .split_pass_name = std::nullopt});
    }

    const PassInfo& split_info = leaf_passes[split_index];
    std::string synthetic_split_pass_name = add_synthetic_pass(split_info);
    OptimizationPipelineProto::CompoundPass& split_top_level_pass =
        top_level_passes.emplace_back();
    std::string top_level_name = get_next_top_level_pass_name();
    split_top_level_pass.set_short_name(top_level_name);
    split_top_level_pass.add_passes(synthetic_split_pass_name);
    split_pass_infos.emplace_back(SplitPipelineFactory::SplitPassInfo{
        .default_pass_name = top_level_name,
        .split_pass_name = std::string(split_info.name)});

    last_split_index = split_index;
  }

  // Add any remaining passes after the last split.
  if (last_split_index.has_value() &&
      *last_split_index + 1 < leaf_passes.size()) {
    OptimizationPipelineProto::CompoundPass& top_level_pass =
        top_level_passes.emplace_back();
    std::string top_level_name = get_next_top_level_pass_name();
    top_level_pass.set_short_name(top_level_name);
    for (size_t i = *last_split_index + 1; i < leaf_passes.size(); ++i) {
      std::string synthetic_pass_name = add_synthetic_pass(leaf_passes[i]);
      top_level_pass.add_passes(synthetic_pass_name);
    }
    split_pass_infos.emplace_back(SplitPipelineFactory::SplitPassInfo{
        .default_pass_name = top_level_name, .split_pass_name = std::nullopt});
  }

  SplitPassResult result;
  result.split_pass_infos = std::move(split_pass_infos);
  result.synthetic_passes = std::move(synthetic_passes);
  result.top_level_passes = std::move(top_level_passes);

  return result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SplitPipelineFactory>>
SplitPipelineFactory::Create(OptimizationPassRegistry& registry,
                             absl::Span<const std::string> split_passes,
                             const OptimizationPipelineProto& pipeline_proto) {
  VLOG(1) << "Creating SplitPipelineFactory";

  OptimizationPassRegistry registry_clone = registry.OverridableClone();
  XLS_ASSIGN_OR_RETURN(SplitPassResult split_result,
                       GetSplitPassInfos(pipeline_proto, split_passes));

  OptimizationPipelineProto split_pipeline_proto = pipeline_proto;

  split_pipeline_proto.mutable_compound_passes()->Add(
      split_result.synthetic_passes.begin(),
      split_result.synthetic_passes.end());
  split_pipeline_proto.mutable_compound_passes()->Add(
      split_result.top_level_passes.begin(),
      split_result.top_level_passes.end());
  split_pipeline_proto.mutable_default_pipeline()->Clear();
  for (const auto& pass_info : split_result.split_pass_infos) {
    split_pipeline_proto.add_default_pipeline(pass_info.default_pass_name);
  }

  XLS_RETURN_IF_ERROR(registry_clone.RegisterPipelineProto(split_pipeline_proto,
                                                           "split_pipeline"));

  absl::flat_hash_map<std::string, size_t> split_pass_name_to_index;

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Created " << split_result.split_pass_infos.size()
            << " optimization splits";

    for (const SplitPassInfo& pass_info : split_result.split_pass_infos) {
      if (pass_info.split_pass_name.has_value()) {
        VLOG(5) << "  Default pass: " << pass_info.default_pass_name
                << ", Split pass: " << pass_info.split_pass_name.value();
      } else {
        VLOG(5) << "  Default pass: " << pass_info.default_pass_name
                << ", Split pass: null";
      }
    }
  }

  for (const auto& [i, pass_info] :
       iter::enumerate(split_result.split_pass_infos)) {
    const std::optional<std::string>& split_pass_name =
        pass_info.split_pass_name;
    if (split_pass_name.has_value()) {
      split_pass_name_to_index[*split_pass_name] = i;
    }
  }

  return std::make_unique<SplitPipelineFactory>(
      std::move(split_result.split_pass_infos),
      std::move(split_pass_name_to_index), std::move(registry_clone));
}

absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
SplitPipelineFactory::GetPipelineBeforeSplit(std::string_view pass_name) {
  auto split_it = split_pass_name_to_index_.find(pass_name);
  if (split_it == split_pass_name_to_index_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Split pass ", pass_name, " not found in split pipeline"));
  }
  size_t split_index = split_it->second;
  if (split_index == 0) {
    return GetOptimizationPipelineGenerator(registry_).GeneratePipeline("");
  }

  std::vector<std::string_view> pass_names;
  pass_names.reserve(split_index);
  for (size_t i = 0; i < split_index; ++i) {
    pass_names.push_back(split_pass_infos_[i].default_pass_name);
  }
  return GetOptimizationPipelineGenerator(registry_).GeneratePipeline(
      absl::StrJoin(pass_names, " "));
}

absl::StatusOr<std::unique_ptr<OptimizationCompoundPass>>
SplitPipelineFactory::GetPipelineAfterSplit(std::string_view pass_name) {
  auto split_it = split_pass_name_to_index_.find(pass_name);
  if (split_it == split_pass_name_to_index_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Split pass ", pass_name, " not found in split pipeline"));
  }
  size_t split_index = split_it->second;
  if (split_index >= split_pass_infos_.size() - 1) {
    return GetOptimizationPipelineGenerator(registry_).GeneratePipeline("");
  }

  std::vector<std::string_view> pass_names;
  pass_names.reserve(split_pass_infos_.size() - split_index - 1);
  for (size_t i = split_index + 1; i < split_pass_infos_.size(); ++i) {
    pass_names.push_back(split_pass_infos_[i].default_pass_name);
  }
  return GetOptimizationPipelineGenerator(registry_).GeneratePipeline(
      absl::StrJoin(pass_names, " "));
}

}  // namespace xls
