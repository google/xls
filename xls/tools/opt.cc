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

#include "xls/tools/opt.h"

#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/area_model/area_estimators.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/verifier_checker.h"
#include "xls/tools/opt_flags.pb.h"
#include "xls/tools/opt_flags_handler_registry.h"

namespace xls::tools {

absl::StatusOr<OptFlagsProto> OptFlagsFromOptOptions(
    const OptOptions& options) {
  OptFlagsProto proto;
  proto.set_opt_level(options.opt_level);
  proto.set_top(options.top);
  if (!options.ir_dump_path.empty()) {
    proto.set_ir_dump_path(options.ir_dump_path);
  }
  if (options.ir_path.has_value()) {
    proto.set_ir_path(*options.ir_path);
  }
  for (const auto& skip_pass : options.skip_passes) {
    proto.add_skip_passes(skip_pass);
  }
  if (options.convert_array_index_to_select.has_value()) {
    proto.set_convert_array_index_to_select(
        *options.convert_array_index_to_select);
  }
  if (options.split_next_value_selects.has_value()) {
    proto.set_split_next_value_selects(*options.split_next_value_selects);
  } else {
    proto.set_split_next_value_selects(0);
  }
  for (const auto& ram_rewrite : options.ram_rewrites) {
    XLS_ASSIGN_OR_RETURN(*proto.mutable_ram_rewrites()->add_rewrites(),
                         ram_rewrite.ToProto());
  }
  proto.set_use_context_narrowing_analysis(
      options.use_context_narrowing_analysis);
  proto.set_optimize_for_best_case_throughput(
      options.optimize_for_best_case_throughput);
  proto.set_enable_resource_sharing(options.enable_resource_sharing);
  proto.set_force_resource_sharing(options.force_resource_sharing);
  proto.set_area_model(options.area_model);
  if (options.custom_registry.has_value()) {
    *proto.mutable_custom_registry() = *options.custom_registry;
  }
  if (options.pass_pipeline.has_value()) {
    *proto.mutable_pipeline() = *options.pass_pipeline;
  }
  if (options.bisect_limit.has_value()) {
    proto.set_passes_bisect_limit(*options.bisect_limit);
  }
  proto.set_debug_optimizations(options.debug_optimizations);
  if (options.delay_model.has_value()) {
    proto.set_delay_model(*options.delay_model);
  }
  return proto;
}

absl::Status OptimizeIrForTop(Package* package, const OptFlagsProto& options,
                              OptMetadata* metadata,
                              const OptModifiers& modifiers) {
  if (!options.top().empty()) {
    VLOG(3) << "OptimizeIrForEntry; top: '" << options.top()
            << "'; opt_level: " << options.opt_level();
  } else {
    VLOG(3) << "OptimizeIrForEntry; opt_level: " << options.opt_level();
  }

  if (!options.top().empty()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(options.top()));
  }
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity not set for package: %s.", package->name()));
  }
  VLOG(3) << "Top entity: '" << top.value()->name() << "'";

  DelayEstimator* delay_estimator = nullptr;
  if (options.has_delay_model()) {
    XLS_ASSIGN_OR_RETURN(delay_estimator,
                         GetDelayEstimator(options.delay_model()));
  }
  AreaEstimator* area_estimator = nullptr;
  if (!options.area_model().empty()) {
    XLS_ASSIGN_OR_RETURN(area_estimator,
                         GetAreaEstimator(options.area_model()));
  }

  std::optional<OptimizationPassRegistry> registry;
  if (options.has_custom_registry()) {
    registry.emplace(GetOptimizationRegistry().OverridableClone());
    XLS_RETURN_IF_ERROR(registry->RegisterPipelineProto(
        options.custom_registry(), "custom-registry"));
  }
  const OptimizationPassRegistry& chosen_registry =
      registry.value_or(GetOptimizationRegistry());
  std::optional<std::unique_ptr<OptimizationPassRegistry>> decorated_registry;
  if (modifiers.registry_decorator.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        decorated_registry,
        (*modifiers.registry_decorator)->Decorate(chosen_registry));
  }
  const OptimizationPassRegistry& final_registry =
      decorated_registry ? **decorated_registry : chosen_registry;

  std::unique_ptr<OptimizationPass> pipeline;
  if (options.has_pipeline()) {
    XLS_RET_CHECK(options.skip_passes().empty())
        << "Skipping/restricting passes while running a custom "
           "pipeline is probably not something you want to do.";
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<OptimizationCompoundPass> res,
                         GetOptimizationPipelineGenerator(final_registry)
                             .GeneratePipeline(options.pipeline()));
    if (options.debug_optimizations()) {
      res->AddInvariantChecker<VerifierChecker>();
      res->AddInvariantChecker<QueryEngineChecker>();
    } else {
      res->AddWeakInvariantChecker<VerifierChecker>();
    }
    pipeline = std::move(res);
  } else {
    XLS_ASSIGN_OR_RETURN(pipeline,
                         TryCreateOptimizationPassPipeline(
                             options.debug_optimizations(), final_registry));
  }

  // Since we are running during an opt the passes might be in states which the
  // estimators might not normally expect (eg having live invoke nodes etc). We
  // use wrappers to filter these nodes out of the estimator.
  std::optional<delay_adapters::FilterNonSynth> delay_estimator_wrapper;
  if (delay_estimator != nullptr) {
    delay_estimator_wrapper.emplace(*delay_estimator);
    delay_estimator = &*delay_estimator_wrapper;
  }
  std::optional<area_adapters::FilterNonSynth> area_estimator_wrapper;
  if (area_estimator != nullptr) {
    area_estimator_wrapper.emplace(*area_estimator);
    area_estimator = &*area_estimator_wrapper;
  }

  // TODO(allight): Some refactoring of how we pass around these options is
  // probably warranted. There are a lot of things here.
  OptimizationPassOptions pass_options;
  pass_options.opt_level = options.opt_level();
  pass_options.delay_estimator = delay_estimator;
  pass_options.area_estimator = area_estimator;
  pass_options.ir_dump_path = options.ir_dump_path();
  absl::c_copy(options.skip_passes(),
               std::back_inserter(pass_options.skip_passes));
  if (options.has_convert_array_index_to_select()) {
    pass_options.convert_array_index_to_select =
        options.convert_array_index_to_select();
  }
  pass_options.split_next_value_selects =
      options.split_next_value_selects() > 0
          ? std::make_optional(options.split_next_value_selects())
          : std::nullopt;
  std::vector<RamRewrite> ram_rewrites;
  ram_rewrites.reserve(options.ram_rewrites().rewrites_size());
  for (const auto& ram_rewrite : options.ram_rewrites().rewrites()) {
    XLS_ASSIGN_OR_RETURN(std::back_inserter(ram_rewrites),
                         RamRewrite::FromProto(ram_rewrite));
  }
  pass_options.ram_rewrites = ram_rewrites;
  pass_options.use_context_narrowing_analysis =
      options.use_context_narrowing_analysis();
  pass_options.optimize_for_best_case_throughput =
      options.optimize_for_best_case_throughput();
  pass_options.enable_resource_sharing = options.enable_resource_sharing();
  pass_options.force_resource_sharing = options.force_resource_sharing();
  if (options.has_passes_bisect_limit()) {
    pass_options.bisect_limit = options.passes_bisect_limit();
  }
  XLS_RETURN_IF_ERROR(OptFlagsHandlerRegistry::Process(options, pass_options));
  PassResults results;
  OptimizationContext context;
  XLS_RETURN_IF_ERROR(
      pipeline->Run(package, pass_options, &results, context).status());
  if (metadata != nullptr) {
    metadata->metrics = results.ToProto();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> OptimizeIrForTop(std::string_view ir,
                                             const OptFlagsProto& options,
                                             OptMetadata* metadata,
                                             const OptModifiers& modifiers) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Package> package,
      Parser::ParsePackage(ir, options.has_ir_path()
                                   ? std::make_optional(options.ir_path())
                                   : std::nullopt));
  XLS_RETURN_IF_ERROR(
      OptimizeIrForTop(package.get(), options, metadata, modifiers));
  return package->DumpIr();
}

}  // namespace xls::tools
