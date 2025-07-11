// Copyright 2024 The XLS Authors
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

#include "xls/passes/optimization_pass_registry.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.pb.h"
#include "xls/passes/pipeline_generator.h"

namespace xls {

OptimizationPassRegistry& GetOptimizationRegistry() {
  static OptimizationPassRegistry kSingleOptimizationRegistry;
  return kSingleOptimizationRegistry;
}

absl::StatusOr<std::unique_ptr<OptimizationPass>> WrapPassWithOptions(
    std::unique_ptr<OptimizationPass>&& cur,
    const BasicPipelineOptions& options) {
  // Make sure both places passes can be configured are updated.
  std::unique_ptr<OptimizationPass> base = std::move(cur);
  if (options.max_opt_level.has_value()) {
    base = std::make_unique<
        xls::internal::DynamicCapOptLevel<OptimizationWrapperPass>>(
        *options.max_opt_level, std::move(base));
  }
  if (options.min_opt_level.has_value()) {
    base = std::make_unique<
        xls::internal::DynamicIfOptLevelAtLeast<OptimizationWrapperPass>>(
        *options.min_opt_level, std::move(base));
  }
  if (options.requires_resource_sharing) {
    base = std::make_unique<
        xls::IfResourceSharingEnabled<OptimizationWrapperPass>>(
        std::move(base));
  }
  return base;
}

namespace {
class CompoundPassAdder final : public OptimizationPassGenerator {
 public:
  explicit CompoundPassAdder(OptimizationPipelineProto::CompoundPass compound)
      : compound_(std::move(compound)) {}
  absl::StatusOr<std::unique_ptr<OptimizationPass>> Generate() const final {
    // Actually construct and insert the PassClass instance
    std::unique_ptr<OptimizationCompoundPass> res;
    if (compound_.fixedpoint()) {
      res = std::make_unique<OptimizationFixedPointCompoundPass>(
          compound_.short_name(), compound_.long_name());
    } else {
      res = std::make_unique<OptimizationCompoundPass>(compound_.short_name(),
                                                       compound_.long_name());
    }
    for (const auto& pass : compound_.passes()) {
      XLS_ASSIGN_OR_RETURN(auto* generator,
                           GetOptimizationRegistry().Generator(pass));
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationPass> pass_instance,
                           generator->Generate());
      XLS_ASSIGN_OR_RETURN(
          pass_instance,
          WrapPassWithOptions(
              std::move(pass_instance),
              {
                  .max_opt_level =
                      compound_.options().has_cap_opt_level()
                          ? std::make_optional(
                                compound_.options().cap_opt_level())
                          : std::nullopt,
                  .min_opt_level =
                      compound_.options().has_min_opt_level()
                          ? std::make_optional(
                                compound_.options().min_opt_level())
                          : std::nullopt,
                  .requires_resource_sharing =
                      compound_.options().resource_sharing_required(),
              }));
      res->AddOwned(std::move(pass_instance));
    }
    return res;
  }

 private:
  // All the arguments needed to construct a PassClass instance
  OptimizationPipelineProto::CompoundPass compound_;
};
}  // namespace

OptimizationPassRegistry OptimizationPassRegistry::OverridableClone() const {
  // Just copy then call the protected overwrite function.
  OptimizationPassRegistry cpy = *this;
  cpy.set_allow_overwrite(true);
  return cpy;
}

absl::Status OptimizationPassRegistry::RegisterPipelineProto(
    const OptimizationPipelineProto& pipeline, std::string_view file) {
  for (const auto& compound : pipeline.compound_passes()) {
    XLS_RETURN_IF_ERROR(Register(compound.short_name(),
                                 std::make_unique<CompoundPassAdder>(compound)))
        << "Failed to register compound pass " << compound.short_name();
    GetOptimizationRegistry().AddRegistrationInfo(
        compound.short_name(), "OptimizationPipelineProto.CompoundPass", file);
  }
  // Now add the default pipeline.
  OptimizationPipelineProto::CompoundPass compound;
  compound.set_short_name(kDefaultPassPipelineName);
  compound.set_long_name("Default pass pipeline");
  for (const auto& pass : pipeline.default_pipeline()) {
    compound.add_passes(pass);
  }
  AddRegistrationInfo(compound.short_name(), "OptimizationPipelineProto", file);
  XLS_RETURN_IF_ERROR(GetOptimizationRegistry().Register(
      compound.short_name(), std::make_unique<CompoundPassAdder>(compound)))
      << "Failed to register compound pass " << compound.short_name();
  return absl::OkStatus();
}

absl::Status RegisterOptimizationPipelineProtoData(
    absl::Span<uint8_t const> data, std::string_view file) {
  OptimizationPipelineProto pipeline;
  if (!pipeline.ParseFromArray(data.data(), data.size())) {
    return absl::InvalidArgumentError("Failed to parse pipeline proto data");
  }
  return GetOptimizationRegistry().RegisterPipelineProto(pipeline, file);
}

}  // namespace xls
