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
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"

namespace xls::tools {

absl::StatusOr<std::string> OptimizeIrForEntry(absl::string_view ir,
                                               const OptOptions& options) {
  XLS_VLOG(3) << "OptimizeIrForEntry; entry: '" << options.entry
              << "'; opt_level: " << options.opt_level;
  std::unique_ptr<Package> package;
  if (options.entry.empty()) {
    XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackage(ir, options.ir_path));
  } else {
    XLS_ASSIGN_OR_RETURN(package, Parser::ParsePackageWithEntry(
                                      ir, options.entry, options.ir_path));
  }
  XLS_VLOG(3) << "Entry function: '" << package->EntryFunction().value()->name()
              << "'";
  std::unique_ptr<CompoundPass> pipeline =
      CreateStandardPassPipeline(options.opt_level);
  const PassOptions pass_options = {
      .ir_dump_path = options.ir_dump_path,
      .run_only_passes = options.run_only_passes,
      .skip_passes = options.skip_passes,
  };
  PassResults results;
  XLS_RETURN_IF_ERROR(
      pipeline->Run(package.get(), pass_options, &results).status());
  return package->DumpIr();
}

}  // namespace xls::tools
