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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/verifier.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/verifier_checker.h"

namespace xls::tools {

absl::Status OptimizeIrForTop(Package* package, const OptOptions& options) {
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

  std::unique_ptr<OptimizationCompoundPass> pipeline;
  if (!options.pass_list) {
    pipeline = CreateOptimizationPassPipeline();
  } else {
    XLS_RET_CHECK(options.skip_passes.empty())
        << "Skipping/restricting passes while running a custom pipeline is "
           "probably not something you want to do.";
    XLS_ASSIGN_OR_RETURN(pipeline,
                         GetOptimizationPipelineGenerator().GeneratePipeline(
                             *options.pass_list));
    pipeline->AddInvariantChecker<VerifierChecker>();
  }
  OptimizationPassOptions pass_options;
  pass_options.opt_level = options.opt_level;
  pass_options.ir_dump_path = options.ir_dump_path;
  pass_options.skip_passes = options.skip_passes;
  pass_options.inline_procs = options.inline_procs;
  pass_options.convert_array_index_to_select =
      options.convert_array_index_to_select;
  pass_options.split_next_value_selects = options.split_next_value_selects;
  pass_options.ram_rewrites = options.ram_rewrites;
  pass_options.use_context_narrowing_analysis =
      options.use_context_narrowing_analysis;
  pass_options.bisect_limit = options.bisect_limit;
  PassResults results;
  XLS_RETURN_IF_ERROR(pipeline->Run(package, pass_options, &results).status());
  return absl::OkStatus();
}

absl::StatusOr<std::string> OptimizeIrForTop(std::string_view ir,
                                             const OptOptions& options) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, options.ir_path));
  XLS_RETURN_IF_ERROR(OptimizeIrForTop(package.get(), options));
  return package->DumpIr();
}

absl::StatusOr<std::string> OptimizeIrForTop(
    std::string_view input_path, int64_t opt_level, std::string_view top,
    std::string_view ir_dump_path, absl::Span<const std::string> skip_passes,
    int64_t convert_array_index_to_select, int64_t split_next_value_selects,
    bool inline_procs, std::string_view ram_rewrites_pb,
    bool use_context_narrowing_analysis, std::optional<std::string> pass_list,
    std::optional<int64_t> bisect_limit) {
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
  std::vector<RamRewrite> ram_rewrites;
  if (!ram_rewrites_pb.empty()) {
    RamRewritesProto ram_rewrite_proto;
    XLS_RETURN_IF_ERROR(xls::ParseTextProtoFile(
        std::filesystem::path(ram_rewrites_pb), &ram_rewrite_proto));
    XLS_ASSIGN_OR_RETURN(ram_rewrites, RamRewritesFromProto(ram_rewrite_proto));
  }
  const OptOptions options = {
      .opt_level = opt_level,
      .top = top,
      .ir_dump_path = std::string(ir_dump_path),
      .skip_passes =
          std::vector<std::string>(skip_passes.begin(), skip_passes.end()),
      .convert_array_index_to_select =
          (convert_array_index_to_select < 0)
              ? std::nullopt
              : std::make_optional(convert_array_index_to_select),
      .split_next_value_selects =
          (split_next_value_selects < 0)
              ? std::nullopt
              : std::make_optional(split_next_value_selects),
      .inline_procs = inline_procs,
      .ram_rewrites = std::move(ram_rewrites),
      .use_context_narrowing_analysis = use_context_narrowing_analysis,
      .pass_list = std::move(pass_list),
      .bisect_limit = bisect_limit,
  };
  return OptimizeIrForTop(ir, options);
}

}  // namespace xls::tools
