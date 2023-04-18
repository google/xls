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

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/verifier.h"
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"

namespace xls::tools {

absl::StatusOr<std::string> OptimizeIrForTop(std::string_view ir,
                                             const OptOptions& options) {
  if (!options.top.empty()) {
    XLS_VLOG(3) << "OptimizeIrForEntry; top: '" << options.top
                << "'; opt_level: " << options.opt_level;
  } else {
    XLS_VLOG(3) << "OptimizeIrForEntry; opt_level: " << options.opt_level;
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir, options.ir_path));
  if (!options.top.empty()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(options.top));
  }
  std::optional<FunctionBase*> top = package->GetTop();
  if (!top.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Top entity not set for package: %s.", package->name()));
  }
  XLS_VLOG(3) << "Top entity: '" << top.value()->name() << "'";

  std::unique_ptr<CompoundPass> pipeline =
      CreateStandardPassPipeline(options.opt_level);
  const PassOptions pass_options = {
      .ir_dump_path = options.ir_dump_path,
      .run_only_passes = options.run_only_passes,
      .skip_passes = options.skip_passes,
      .inline_procs = options.inline_procs,
      .convert_array_index_to_select = options.convert_array_index_to_select,
      .ram_rewrites = options.ram_rewrites,
  };
  PassResults results;
  XLS_RETURN_IF_ERROR(
      pipeline->Run(package.get(), pass_options, &results).status());
  return package->DumpIr();
}

absl::StatusOr<std::string> OptimizeIrForTop(
    std::string_view input_path, int64_t opt_level, std::string_view top,
    std::string_view ir_dump_path,
    absl::Span<const std::string> run_only_passes,
    absl::Span<const std::string> skip_passes,
    int64_t convert_array_index_to_select, bool inline_procs,
    std::string_view ram_rewrites_pb) {
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
      .run_only_passes =
          run_only_passes.empty()
              ? std::nullopt
              : std::make_optional(std::vector<std::string>(
                    run_only_passes.begin(), run_only_passes.end())),
      .skip_passes =
          std::vector<std::string>(skip_passes.begin(), skip_passes.end()),
      .convert_array_index_to_select =
          (convert_array_index_to_select < 0)
              ? std::nullopt
              : std::make_optional(convert_array_index_to_select),
      .inline_procs = inline_procs,
      .ram_rewrites = std::move(ram_rewrites),
  };
  return OptimizeIrForTop(ir, options);
}

}  // namespace xls::tools
