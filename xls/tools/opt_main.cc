// Copyright 2020 The XLS Authors
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

// Takes in an IR file and produces an IR file that has been run through the
// standard optimization pipeline.

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/passes/passes.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/tools/opt.h"

ABSL_FLAG(std::string, entry, "", "Entry function name to optimize.");
ABSL_FLAG(std::string, ir_dump_path, "",
          "Dump all intermediate IR files to the given directory");
ABSL_FLAG(std::vector<std::string>, run_only_passes, {},
          "If specified, only passes in this comma-separated list of (short) "
          "pass names are be run.");
ABSL_FLAG(std::vector<std::string>, skip_passes, {},
          "If specified, passes in this comma-separated list of (short) "
          "pass names are skipped. If both --run_only_passes and --skip_passes "
          "are specified only passes which are present in --run_only_passes "
          "and not present in --skip_passes will be run.");
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel,
          absl::StrFormat("Optimization level. Ranges from 1 to %d.",
                          xls::kMaxOptLevel));

namespace xls::tools {
namespace {

absl::Status RealMain(absl::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }
  XLS_ASSIGN_OR_RETURN(std::string ir, GetFileContents(input_path));
  std::string entry = absl::GetFlag(FLAGS_entry);
  std::string ir_dump_path = absl::GetFlag(FLAGS_ir_dump_path);
  std::vector<std::string> run_only_passes =
      absl::GetFlag(FLAGS_run_only_passes);
  const OptOptions options = {
      .opt_level = absl::GetFlag(FLAGS_opt_level),
      .entry = entry,
      .ir_dump_path = ir_dump_path,
      .run_only_passes = run_only_passes.empty()
                             ? absl::nullopt
                             : absl::make_optional(std::move(run_only_passes)),
      .skip_passes = absl::GetFlag(FLAGS_skip_passes),
  };
  XLS_ASSIGN_OR_RETURN(std::string opt_ir,
                       tools::OptimizeIrForEntry(ir, options));
  std::cout << opt_ir;
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char **argv) {
  std::vector<absl::string_view> positional_arguments =
      xls::InitXls(argv[0], argc, argv);

  if (positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                          argv[0]);
  }

  XLS_QCHECK_OK(xls::tools::RealMain(positional_arguments[0]));
  return EXIT_SUCCESS;
}
