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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
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

absl::Status GetLeafPassCountsHelper(
    std::string_view compound_pass_name,
    const absl::flat_hash_map<std::string,
                              OptimizationPipelineProto::CompoundPass>&
        name_to_compound_pass_map,
    absl::flat_hash_set<std::string>& active_path,
    absl::flat_hash_map<std::string, int>& pass_counts) {
  if (active_path.contains(compound_pass_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cycle detected involving pass '", compound_pass_name,
                     "' and active path: ", absl::StrJoin(active_path, ", ")));
  }

  auto compound_pass_it = name_to_compound_pass_map.find(compound_pass_name);
  if (compound_pass_it == name_to_compound_pass_map.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not find compound pass for '", compound_pass_name, "'"));
  }

  active_path.insert(std::string(compound_pass_name));
  for (const std::string_view subpass : compound_pass_it->second.passes()) {
    if (name_to_compound_pass_map.contains(subpass)) {
      XLS_RETURN_IF_ERROR(GetLeafPassCountsHelper(
          subpass, name_to_compound_pass_map, active_path, pass_counts));
    } else {
      ++pass_counts[subpass];
    }
  }
  active_path.erase(compound_pass_name);
  return absl::OkStatus();
}

// Returns a map of pass names to their counts in the given compound pass. Only
// passes that are not themselves compound passes are tallied.
absl::StatusOr<absl::flat_hash_map<std::string, int>> GetLeafPassCounts(
    std::string_view top_level_pass,
    const absl::flat_hash_map<std::string,
                              OptimizationPipelineProto::CompoundPass>&
        name_to_compound_pass_map) {
  if (!name_to_compound_pass_map.contains(top_level_pass)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Top level pass '", top_level_pass,
                     "' not found in compound pass map."));
  }

  absl::flat_hash_set<std::string> active_path;
  absl::flat_hash_map<std::string, int> pass_counts;
  XLS_RETURN_IF_ERROR(GetLeafPassCountsHelper(
      top_level_pass, name_to_compound_pass_map, active_path, pass_counts));
  return pass_counts;
}

// Returns an ordered list of pass information splitting the optimization
// pipeline up based on the defined split passes, using the default pipeline
// order.
// It is assumed that split pass appear exactly once in the default pipeline
// and have been moved into their own top level compound passes prior to
// SplitPipelineFactory creation.
absl::StatusOr<std::vector<SplitPipelineFactory::SplitPassInfo>>
GetSplitPassInfos(const OptimizationPipelineProto& pipeline_proto,
                  const absl::Span<const std::string>& split_passes) {
  absl::flat_hash_set<std::string> seen_splits;

  VLOG(5) << "Creating split pass info for splits:";
  VLOG(5) << "  " << absl::StrJoin(split_passes, ", ");

  absl::flat_hash_map<std::string, OptimizationPipelineProto::CompoundPass>
      pass_name_to_compound_pass;
  for (const OptimizationPipelineProto::CompoundPass& compound_pass :
       pipeline_proto.compound_passes()) {
    pass_name_to_compound_pass[compound_pass.short_name()] = compound_pass;
  }

  std::vector<SplitPipelineFactory::SplitPassInfo> split_pass_infos;
  for (const std::string_view default_pass :
       pipeline_proto.default_pipeline()) {
    VLOG(5) << "Processing pass " << default_pass;
    if (!pass_name_to_compound_pass.contains(default_pass)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Pass '", default_pass, "' not found in compound passes"));
    }

    absl::flat_hash_map<std::string, int> leaf_pass_counts;
    XLS_ASSIGN_OR_RETURN(
        leaf_pass_counts,
        GetLeafPassCounts(default_pass, pass_name_to_compound_pass));

    // Split passes must be the only leaf pass in their top level compound pass.
    std::optional<std::string> found_split_pass;
    for (const auto& [leaf_pass, leaf_pass_count] : leaf_pass_counts) {
      if (!absl::c_linear_search(split_passes, leaf_pass)) {
        continue;
      }
      if (leaf_pass_counts.size() > 1 || leaf_pass_count > 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("Split pass '", leaf_pass,
                         "' must be the only leaf pass in compound pass '",
                         default_pass, "'"));
      }
      if (!seen_splits.insert(leaf_pass).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Multiple instances of split pass '", leaf_pass,
                         "' found in pipeline (at pass '", default_pass, "')"));
      }
      found_split_pass = leaf_pass;
    }

    if (found_split_pass.has_value()) {
      VLOG(5) << "Adding split " << *found_split_pass << " with default pass "
              << default_pass;
      split_pass_infos.push_back(SplitPipelineFactory::SplitPassInfo{
          .default_pass_name = std::string(default_pass),
          .split_pass_name = *found_split_pass});
      continue;
    }

    VLOG(5) << "Adding pass " << default_pass << " without split";
    split_pass_infos.push_back(SplitPipelineFactory::SplitPassInfo{
        .default_pass_name = std::string(default_pass),
        .split_pass_name = std::nullopt});
  }
  return split_pass_infos;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SplitPipelineFactory>>
SplitPipelineFactory::Create(OptimizationPassRegistry& registry,
                             absl::Span<const std::string> split_passes,
                             const OptimizationPipelineProto& pipeline_proto) {
  VLOG(1) << "Creating SplitPipelineFactory";

  OptimizationPassRegistry registry_clone = registry.OverridableClone();
  XLS_RETURN_IF_ERROR(
      registry_clone.RegisterPipelineProto(pipeline_proto, "split_pipeline"));

  XLS_ASSIGN_OR_RETURN(std::vector<SplitPassInfo> split_pass_infos,
                       GetSplitPassInfos(pipeline_proto, split_passes));

  absl::flat_hash_map<std::string, size_t> split_pass_name_to_index;

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Created " << split_pass_infos.size() << " optimization splits";

    for (const SplitPassInfo& pass_info : split_pass_infos) {
      if (pass_info.split_pass_name.has_value()) {
        VLOG(5) << "  Default pass: " << pass_info.default_pass_name
                << ", Split pass: " << pass_info.split_pass_name.value();
      } else {
        VLOG(5) << "  Default pass: " << pass_info.default_pass_name
                << ", Split pass: null";
      }
    }
  }

  for (const auto& [i, pass_info] : iter::enumerate(split_pass_infos)) {
    const std::optional<std::string>& split_pass_name =
        pass_info.split_pass_name;
    if (split_pass_name.has_value()) {
      split_pass_name_to_index[*split_pass_name] = i;
    }
  }

  return std::make_unique<SplitPipelineFactory>(
      std::move(split_pass_infos), std::move(split_pass_name_to_index),
      std::move(registry_clone));
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
  for (const SplitPassInfo& pass_info :
       absl::MakeSpan(split_pass_infos_).subspan(0, split_index)) {
    pass_names.push_back(pass_info.default_pass_name);
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
  for (const SplitPassInfo& pass_info :
       absl::MakeSpan(split_pass_infos_).subspan(split_index + 1)) {
    pass_names.push_back(pass_info.default_pass_name);
  }
  return GetOptimizationPipelineGenerator(registry_).GeneratePipeline(
      absl::StrJoin(pass_names, " "));
}

}  // namespace xls
