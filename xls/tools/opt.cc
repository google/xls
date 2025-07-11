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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_metrics.pb.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/query_engine_checker.h"
#include "xls/passes/verifier_checker.h"

namespace xls::tools {

absl::Status OptimizeIrForTop(Package* package, const OptOptions& options,
                              OptMetadata* metadata) {
  if (!options.top.empty()) {
    VLOG(3) << "OptimizeIrForEntry; top: '" << options.top
            << "'; opt_level: " << options.opt_level;
  } else {
    VLOG(3) << "OptimizeIrForEntry; opt_level: " << options.opt_level;
  }

  if (!options.top.empty()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(options.top));
  }
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity not set for package: %s.", package->name()));
  }
  VLOG(3) << "Top entity: '" << top.value()->name() << "'";

  std::optional<OptimizationPassRegistry> registry;
  if (options.custom_registry) {
    registry.emplace(GetOptimizationRegistry().OverridableClone());
    XLS_RETURN_IF_ERROR(registry->RegisterPipelineProto(
        *options.custom_registry, "custom-registry"));
  }
  const OptimizationPassRegistry& chosen_registry =
      registry.value_or(GetOptimizationRegistry());

  using PipelineResult = absl::StatusOr<std::unique_ptr<OptimizationPass>>;
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<OptimizationPass> pipeline,
      std::visit(
          Visitor{
              [&](std::nullopt_t) -> PipelineResult {
                return CreateOptimizationPassPipeline(
                    options.debug_optimizations, chosen_registry);
              },
              [&](std::string_view list) -> PipelineResult {
                XLS_RET_CHECK(options.skip_passes.empty())
                    << "Skipping/restricting passes while running a custom "
                       "pipeline is probably not something you want to do.";
                XLS_ASSIGN_OR_RETURN(
                    std::unique_ptr<OptimizationCompoundPass> res,
                    GetOptimizationPipelineGenerator(chosen_registry)
                        .GeneratePipeline(list));
                if (options.debug_optimizations) {
                  res->AddInvariantChecker<VerifierChecker>();
                  res->AddInvariantChecker<QueryEngineChecker>();
                } else {
                  res->AddWeakInvariantChecker<VerifierChecker>();
                }
                return res;
              },
              [&](const PassPipelineProto& list) -> PipelineResult {
                XLS_RET_CHECK(options.skip_passes.empty())
                    << "Skipping/restricting passes while running a custom "
                       "pipeline is probably not something you want to do.";
                XLS_ASSIGN_OR_RETURN(
                    std::unique_ptr<OptimizationCompoundPass> res,
                    GetOptimizationPipelineGenerator(chosen_registry)
                        .GeneratePipeline(list));
                if (options.debug_optimizations) {
                  res->AddInvariantChecker<VerifierChecker>();
                  res->AddInvariantChecker<QueryEngineChecker>();
                } else {
                  res->AddWeakInvariantChecker<VerifierChecker>();
                }
                return res;
              },
          },
          options.pass_pipeline));
  OptimizationPassOptions pass_options;
  pass_options.opt_level = options.opt_level;
  pass_options.ir_dump_path = options.ir_dump_path;
  pass_options.skip_passes = options.skip_passes;
  pass_options.convert_array_index_to_select =
      options.convert_array_index_to_select;
  pass_options.split_next_value_selects = options.split_next_value_selects;
  pass_options.ram_rewrites = options.ram_rewrites;
  pass_options.use_context_narrowing_analysis =
      options.use_context_narrowing_analysis;
  pass_options.optimize_for_best_case_throughput =
      options.optimize_for_best_case_throughput;
  pass_options.enable_resource_sharing = options.enable_resource_sharing;
  pass_options.force_resource_sharing = options.force_resource_sharing;
  pass_options.area_model = options.area_model;
  pass_options.bisect_limit = options.bisect_limit;
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
                                             const OptOptions& options,
                                             OptMetadata* metadata) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, options.ir_path));
  XLS_RETURN_IF_ERROR(OptimizeIrForTop(package.get(), options, metadata));
  return package->DumpIr();
}

}  // namespace xls::tools
