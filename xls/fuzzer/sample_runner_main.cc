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
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/fuzzer/sample_runner.h"

constexpr std::string_view kUsage = R"(Sample runner program.

Runs a fuzzer code sample in the given run directory. Files are copied into the
run directory if they don't already reside there. If no run directory is
specified then a temporary directory is created.

sample_runner_main --options_file=OPT_FILE \
  --input_file=INPUT_FILE \
  --args_file=ARGS_FILE \
  [RUN_DIR]
)";

ABSL_FLAG(std::string, options_file, "",
          "File to load sample runner options from.");
ABSL_FLAG(std::string, input_file, "", "Code input file.");
ABSL_FLAG(std::optional<std::string>, args_file, std::nullopt,
          "Optional file containing arguments to use for interpretation and "
          "simulation.");
ABSL_FLAG(std::optional<std::string>, ir_channel_names_file, std::nullopt,
          "Optional file containing IR names of input channels for a proc.");

namespace xls {

namespace {

// Copies the file to the directory if it is not already in the directory;
// returns the basename of the file.
std::filesystem::path MaybeCopyFile(const std::filesystem::path& file_path,
                                    const std::filesystem::path& dir_path) {
  DCHECK(std::filesystem::is_directory(dir_path));
  std::filesystem::path basename = file_path.filename();
  if (file_path.parent_path() != dir_path) {
    std::filesystem::copy_file(file_path, dir_path / basename);
  }
  return dir_path / basename;
}

}  // namespace

// Runs the sample in the given run directory.
static absl::Status RealMain(
    const std::filesystem::path& run_dir, const std::string& options_file,
    const std::string& input_file, const std::optional<std::string>& args_file,
    const std::optional<std::string>& ir_channel_names_file) {
  SampleRunner runner(run_dir);
  std::filesystem::path input_filename = MaybeCopyFile(input_file, run_dir);
  std::filesystem::path options_filename = MaybeCopyFile(options_file, run_dir);
  std::optional<std::filesystem::path> args_filename =
      args_file.has_value()
          ? std::make_optional(MaybeCopyFile(*args_file, run_dir))
          : std::nullopt;
  std::optional<std::filesystem::path> ir_channel_names_filename =
      ir_channel_names_file.has_value()
          ? std::make_optional(MaybeCopyFile(*ir_channel_names_file, run_dir))
          : std::nullopt;
  return runner.RunFromFiles(input_filename, options_filename, args_filename,
                             ir_channel_names_filename);
}

}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  QCHECK(!absl::GetFlag(FLAGS_options_file).empty())
      << "--options_file is required.";
  QCHECK(!absl::GetFlag(FLAGS_input_file).empty())
      << "--input_file is required.";

  if (positional_arguments.size() > 1) {
    LOG(QFATAL) << "Usage:\n" << kUsage;
  }

  std::filesystem::path run_dir;
  std::optional<xls::TempDirectory> temp_dir;
  if (positional_arguments.empty()) {
    absl::StatusOr<xls::TempDirectory> t = xls::TempDirectory::Create();
    if (!t.ok()) {
      std::cerr << "Error: unable to create temp directory, " << t.status()
                << "\n";
      return EXIT_FAILURE;
    }
    temp_dir = *std::move(t);
    run_dir = temp_dir->path();
  } else {
    run_dir = std::filesystem::path(positional_arguments[0]);
    if (!std::filesystem::is_directory(run_dir)) {
      std::cerr << "Error: " << run_dir
                << " is not a directory or does not exist.\n";
      return EXIT_FAILURE;
    }
  }

  return xls::ExitStatus(xls::RealMain(
      run_dir, absl::GetFlag(FLAGS_options_file),
      absl::GetFlag(FLAGS_input_file), absl::GetFlag(FLAGS_args_file),
      absl::GetFlag(FLAGS_ir_channel_names_file)));
}
