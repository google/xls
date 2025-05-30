// Copyright 2023 The XLS Authors
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

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"

ABSL_FLAG(std::optional<std::string>, run_dir, std::nullopt,
          "The directory to run the crasher in.");
ABSL_FLAG(std::optional<std::string>, simulator, std::nullopt,
          "Verilog simulator to use. If not specified, the value specified in "
          "the crasher file. If the simulator is not specified in either "
          "location, the default simulator is used.");
ABSL_FLAG(bool, unopt_interpreter, true,
          "Should the interpreter be run on unopt-ir");

namespace xls {
namespace {

// Runs the sample in the given run directory.
absl::Status RealMain(const std::filesystem::path& crasher_path,
                      const std::filesystem::path& run_dir,
                      const std::optional<std::string>& simulator) {
  XLS_ASSIGN_OR_RETURN(std::string serialized_crasher,
                       GetFileContents(crasher_path));

  XLS_ASSIGN_OR_RETURN(Sample crasher, Sample::Deserialize(serialized_crasher));
  if (simulator.has_value()) {
    SampleOptions options = crasher.options();
    options.set_simulator(*simulator);
    crasher = Sample(crasher.input_text(), options, crasher.testvector());
  }

  LOG(INFO) << "Running crasher in directory " << run_dir;
  if (!absl::GetFlag(FLAGS_unopt_interpreter)) {
    SampleOptions options = crasher.options();
    options.set_disable_unopt_interpreter(true);
    crasher = Sample(crasher.input_text(), options, crasher.testvector());
  }
  return RunSample(crasher, run_dir).status();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  const std::string usage = absl::StrCat(
      "Invalid command-line arguments; want ", argv[0], " <crasher path>");

  std::vector<std::string_view> positional_arguments =
      xls::InitXls(usage, argc, argv);
  if (positional_arguments.size() != 1) {
    LOG(QFATAL) << usage;
    return EXIT_FAILURE;
  }

  std::filesystem::path run_dir;
  absl::StatusOr<xls::TempDirectory> temp_run_dir;
  if (absl::GetFlag(FLAGS_run_dir).has_value()) {
    run_dir = absl::GetFlag(FLAGS_run_dir).value();
  } else {
    temp_run_dir = xls::TempDirectory::Create();
    run_dir = temp_run_dir->path();
  }
  return xls::ExitStatus(xls::RealMain(
      /*crasher_path=*/positional_arguments[0], run_dir,
      absl::GetFlag(FLAGS_simulator)));
}
