// Copyright 2025 The XLS Authors
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

// Prints summary information about an IR file to the terminal.
// Output will be added as needs warrant, so feel free to make additions!

#include <iostream>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dev_tools/pass_metrics.h"
#include "xls/passes/pass_metrics.pb.h"

ABSL_FLAG(bool, show_all_changed_passes, false,
          "If true, include a line in the hierarchical table for every pass "
          "run in the pipeline. If false, sequences of leaf passes are "
          "collapsed into a single summary line..");

namespace xls {

static absl::Status RealMain(std::string_view metrics_path) {
  XLS_ASSIGN_OR_RETURN(
      auto metrics, ParseTextProtoFile<PassPipelineMetricsProto>(metrics_path));
  std::cout << SummarizePassPipelineMetrics(
      metrics, absl::GetFlag(FLAGS_show_all_changed_passes));
  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_args =
      xls::InitXls(argv[0], argc, argv);
  QCHECK_EQ(positional_args.size(), 1);

  return xls::ExitStatus(xls::RealMain(positional_args[0]));
}
