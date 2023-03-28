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

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/run_routines.h"

// LINT.IfChange
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(
    std::string, trace_format_preference, "default",
    "Preference for display of trace!() output: default|binary|hex|decimal");
ABSL_FLAG(bool, execute, true, "Execute tests within the entry module.");
ABSL_FLAG(std::string, compare, "jit",
          "Compare DSL-interpreted results with an IR execution for each"
          " function for consistency checking; options: none|jit|interpreter.");
ABSL_FLAG(
    int64_t, seed, 0,
    "Seed for quickcheck random stimulus; 0 for an nondetermistic value.");
// TODO(leary): 2021-01-19 allow filters with wildcards.
ABSL_FLAG(std::string, test_filter, "",
          "Target (currently *single*) test name to run.");
ABSL_FLAG(bool, warnings_as_errors, true,
          "Whether to fail early, as an error, if warnings are detected");
ABSL_FLAG(bool, trace_channels, false,
          "If true, values sent and received on channels are emitted as trace "
          "messages");
ABSL_FLAG(int64_t, max_ticks, 100000,
          "If non-zero, the maximum number of ticks to execute on any proc. If "
          "exceeded an error is returned.");
// LINT.ThenChange(//xls/build_rules/xls_dslx_rules.bzl)

namespace xls::dslx {
namespace {

enum class CompareFlag {
  kNone,
  kJit,
  kInterpreter,
};

const char* kUsage = R"(
Parses, typechecks, and executes all tests inside of a DSLX module.
)";

absl::Status RealMain(std::string_view entry_module_path,
                      absl::Span<const std::filesystem::path> dslx_paths,
                      std::optional<std::string> test_filter,
                      FormatPreference trace_format_preference,
                      CompareFlag compare_flag, bool execute,
                      bool warnings_as_errors, std::optional<int64_t> seed,
                      bool trace_channels, std::optional<int64_t> max_ticks,
                      bool* printed_error) {
  XLS_ASSIGN_OR_RETURN(std::string program, GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(entry_module_path));
  std::optional<RunComparator> run_comparator;
  switch (compare_flag) {
    case CompareFlag::kNone:
      break;
    case CompareFlag::kJit:
      run_comparator.emplace(CompareMode::kJit);
      break;
    case CompareFlag::kInterpreter:
      run_comparator.emplace(CompareMode::kInterpreter);
      break;
  }
  ParseAndTestOptions options = {
      .dslx_paths = dslx_paths,
      .test_filter = test_filter,
      .trace_format_preference = trace_format_preference,
      .run_comparator = run_comparator ? &run_comparator.value() : nullptr,
      .execute = execute,
      .seed = seed,
      .warnings_as_errors = warnings_as_errors,
      .trace_channels = trace_channels,
      .max_ticks = max_ticks};
  XLS_ASSIGN_OR_RETURN(
      TestResult test_result,
      ParseAndTest(program, module_name, entry_module_path, options));
  *printed_error = test_result != TestResult::kAllPassed;
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_path_strs = absl::StrSplit(dslx_path, ':');
  std::vector<std::filesystem::path> dslx_paths;
  dslx_paths.reserve(dslx_path_strs.size());
  for (const auto& path : dslx_path_strs) {
    dslx_paths.push_back(std::filesystem::path(path));
  }

  std::string compare_flag_str = absl::GetFlag(FLAGS_compare);
  bool execute = absl::GetFlag(FLAGS_execute);
  bool warnings_as_errors = absl::GetFlag(FLAGS_warnings_as_errors);
  bool trace_channels = absl::GetFlag(FLAGS_trace_channels);
  std::optional<int64_t> max_ticks =
      absl::GetFlag(FLAGS_max_ticks) == 0
          ? std::nullopt
          : std::optional<int64_t>(absl::GetFlag(FLAGS_max_ticks));

  xls::dslx::CompareFlag compare_flag;
  if (compare_flag_str == "none") {
    compare_flag = xls::dslx::CompareFlag::kNone;
  } else if (compare_flag_str == "jit") {
    compare_flag = xls::dslx::CompareFlag::kJit;
  } else if (compare_flag_str == "interpreter") {
    compare_flag = xls::dslx::CompareFlag::kInterpreter;
  } else {
    XLS_LOG(QFATAL) << "Invalid -compare flag: " << compare_flag_str
                    << "; must be one of none|jit|interpreter";
  }

  // Optional seed value.
  std::optional<int64_t> seed;
  if (int64_t seed_flag_value = absl::GetFlag(FLAGS_seed);
      seed_flag_value != 0) {
    seed = seed_flag_value;
  }

  // Optional test filter.
  std::optional<std::string> test_filter;
  if (std::string flag = absl::GetFlag(FLAGS_test_filter); !flag.empty()) {
    test_filter = std::move(flag);
  }

  absl::StatusOr<xls::FormatPreference> preference =
      xls::FormatPreferenceFromString(
          absl::GetFlag(FLAGS_trace_format_preference));
  XLS_QCHECK_OK(preference.status())
      << "-trace_format_preference accepts default|binary|hex|decimal";

  bool printed_error = false;
  absl::Status status =
      xls::dslx::RealMain(args[0], dslx_paths, test_filter, preference.value(),
                          compare_flag, execute, warnings_as_errors, seed,
                          trace_channels, max_ticks, &printed_error);
  if (printed_error) {
    return EXIT_FAILURE;
  }
  XLS_QCHECK_OK(status);
  return EXIT_SUCCESS;
}
