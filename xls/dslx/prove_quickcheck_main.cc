// Copyright 2024 The XLS Authors
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
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/warning_kind.h"

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
ABSL_FLAG(bool, counterexamples_only, false,
          "Only dumps a counterexample (if the proof process finds a "
          "counterexample) or an empty string");

static constexpr std::string_view kUsage = R"(
Attempts to proves a single quickcheck property in a given module to be
infallible, or provide a counterexample.
)";

namespace xls::dslx {
namespace {

absl::StatusOr<TestResultData> RealMain(
    std::string_view entry_module_path, std::string_view quickcheck_name,
    absl::Span<const std::filesystem::path> dslx_paths, bool warnings_as_errors,
    bool counterexamples_only,
    std::optional<std::string_view> xml_output_file) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      WarningKindSetFromDisabledString(absl::GetFlag(FLAGS_disable_warnings)));
  XLS_ASSIGN_OR_RETURN(std::string program, GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(entry_module_path));

  const ParseAndProveOptions options = {
      .dslx_paths = dslx_paths,
      .warnings_as_errors = warnings_as_errors,
      .warnings = warnings,
  };

  XLS_ASSIGN_OR_RETURN(ParseAndProveResult result,
                       ParseAndProve(program, module_name, entry_module_path,
                                     quickcheck_name, options));

  if (result.counterexample.has_value()) {
    std::string counterexample_str =
        absl::StrJoin(result.counterexample.value(), ", ");
    if (counterexamples_only) {
      // In "counterexamples only" mode we either spit out a counterexample or
      // nothing (and produce an appropriate retcode from the binary).
      std::cout << counterexample_str << "\n";
    } else {
      std::cerr << "found counterexample: " << counterexample_str << " in "
                << result.test_result_data.duration() << "\n";
    }
  }

  return result.test_result_data;
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.size() != 2) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got "
                << positional_arguments.size() << ": `"
                << absl::StrJoin(positional_arguments, " ") << "`; want "
                << argv[0] << " <input-file> <quickcheck-name>";
  }

  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  // See https://bazel.build/reference/test-encyclopedia#initial-conditions
  std::optional<std::string> xml_output_file;
  if (const char* xml_output_file_env = getenv("XML_OUTPUT_FILE");
      xml_output_file_env != nullptr &&
      !std::string_view(xml_output_file_env).empty()) {
    xml_output_file = xml_output_file_env;
  }

  bool warnings_as_errors = absl::GetFlag(FLAGS_warnings_as_errors);
  bool counterexamples_only = absl::GetFlag(FLAGS_counterexamples_only);

  absl::StatusOr<xls::dslx::TestResultData> test_result = xls::dslx::RealMain(
      positional_arguments[0], positional_arguments[1], dslx_paths,
      warnings_as_errors, counterexamples_only, xml_output_file);
  if (!test_result.ok()) {
    return xls::ExitStatus(test_result.status());
  }

  if (test_result->result() != xls::dslx::TestResult::kAllPassed) {
    if (!counterexamples_only) {
      std::cerr << "Proof attempt(s) failed; terminating with error status.\n";
    }
    return EXIT_FAILURE;
  }
  if (!counterexamples_only) {
    std::cout << "Proven! elapsed: " << test_result->duration() << "\n";
  }
  return EXIT_SUCCESS;
}
