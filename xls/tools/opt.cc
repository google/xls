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

#include "xls/dslx/ir_converter.h"
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
  };
  PassResults results;
  XLS_RETURN_IF_ERROR(
      pipeline->Run(package.get(), pass_options, &results).status());
  // If opt returns something that obviously can't be codegenned, that's a bug
  // in opt, not codegen.
  XLS_RETURN_IF_ERROR(xls::VerifyPackage(package.get(), /*codegen=*/true));
  return package->DumpIr();
}

}  // namespace xls::tools
