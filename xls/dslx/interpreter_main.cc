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
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/run_routines/run_comparator.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/run_routines/test_xml.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/format_preference.h"
#include "re2/re2.h"

// LINT.IfChange
ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(
    std::string, format_preference, "",
    "Default format preference to use when one is not specified. Used for "
    "`trace!`, `{}` format strings in `trace_fmt!`, traced channel values, "
    "`assert_*` statements and elsewhere. Valid values: binary|hex|decimal");
ABSL_FLAG(bool, execute, true, "Execute tests within the entry module.");
ABSL_FLAG(std::string, compare, "none",
          "Compare DSL-interpreted results with an IR execution for each"
          " function for consistency checking; options: none|jit|interpreter.");
ABSL_FLAG(
    int64_t, seed, 0,
    "Seed for quickcheck random stimulus; 0 for an nondetermistic value.");
ABSL_FLAG(std::string, test_filter, "",
          "Regexp that must be a full match of test name(s) to run.");

ABSL_FLAG(std::string, disable_warnings, "",
          "Comma-delimited list of warnings to disable from the default set of "
          "warnings used -- not generally "
          "recommended, but can be used in exceptional circumstances");
ABSL_FLAG(std::string, enable_warnings, "",
          "Comma-delimited list of warnings to enable that are disabled in the "
          "default set");
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

enum class CompareFlag : uint8_t {
  kNone,
  kJit,
  kInterpreter,
};

static constexpr std::string_view kUsage = R"(
Parses, typechecks, and executes all tests inside of a DSLX module.
)";

absl::StatusOr<TestResult> RealMain(
    std::string_view entry_module_path,
    absl::Span<const std::filesystem::path> dslx_paths,
    const std::optional<std::string>& test_filter,
    FormatPreference format_preference, CompareFlag compare_flag, bool execute,
    bool warnings_as_errors, std::optional<int64_t> seed, bool trace_channels,
    std::optional<int64_t> max_ticks,
    std::optional<std::string_view> xml_output_file) {
  XLS_ASSIGN_OR_RETURN(
      WarningKindSet warnings,
      WarningKindSetFromDisabledString(absl::GetFlag(FLAGS_disable_warnings)));

  XLS_ASSIGN_OR_RETURN(
      const WarningKindSet warnings_to_enable,
      WarningKindSetFromString(absl::GetFlag(FLAGS_enable_warnings)));

  warnings |= warnings_to_enable;

  XLS_ASSIGN_OR_RETURN(std::string program, GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(entry_module_path));

  std::unique_ptr<AbstractRunComparator> run_comparator;
  switch (compare_flag) {
    case CompareFlag::kNone:
      break;
    case CompareFlag::kJit:
      run_comparator = std::make_unique<RunComparator>(CompareMode::kJit);
      break;
    case CompareFlag::kInterpreter:
      run_comparator =
          std::make_unique<RunComparator>(CompareMode::kInterpreter);
      break;
  }

  std::optional<RE2> test_filter_re;
  const RE2* test_filter_re_ptr = nullptr;
  if (test_filter.has_value()) {
    test_filter_re.emplace(test_filter.value());
    if (!test_filter_re->ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Could not create regular expression from test "
                          "filter: `%s`; error: %s; error argument: `%s`",
                          test_filter.value(), test_filter_re->error(),
                          test_filter_re->error_arg()));
    }
    test_filter_re_ptr = &test_filter_re.value();
  }
  ParseAndTestOptions options = {.dslx_paths = dslx_paths,
                                 .test_filter = test_filter_re_ptr,
                                 .format_preference = format_preference,
                                 .run_comparator = run_comparator.get(),
                                 .execute = execute,
                                 .seed = seed,
                                 .warnings_as_errors = warnings_as_errors,
                                 .warnings = warnings,
                                 .trace_channels = trace_channels,
                                 .max_ticks = max_ticks};

  XLS_ASSIGN_OR_RETURN(
      TestResultData test_result,
      ParseAndTest(program, module_name, entry_module_path, options));

  if (xml_output_file.has_value()) {
    test_xml::TestSuites suites = test_result.ToXmlSuites(module_name);
    absl::TimeZone local_tz = absl::LocalTimeZone();
    std::unique_ptr<test_xml::XmlNode> root = ToXml(suites, local_tz);
    std::string contents = test_xml::XmlRootToString(*root);
    XLS_RETURN_IF_ERROR(SetFileContents(xml_output_file.value(), contents));
  }

  return test_result.result();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    LOG(QFATAL) << "Wrong number of command-line arguments; got " << args.size()
                << ": `" << absl::StrJoin(args, " ") << "`; want " << argv[0]
                << " <input-file>";
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
    LOG(QFATAL) << "Invalid -compare flag: " << compare_flag_str
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

  // See https://bazel.build/reference/test-encyclopedia
  std::string_view test_filter_env;
  if (const char* testbridge_value = getenv("TESTBRIDGE_TEST_ONLY");
      testbridge_value != nullptr) {
    test_filter_env = testbridge_value;
  }

  if (!test_filter_env.empty()) {
    if (test_filter.has_value()) {
      std::cerr << "TESTBRIDGE_TEST_ONLY environment variable set "
                   "simultaneously with --test_filter; only one is allowed.\n";
      return EXIT_FAILURE;
    }
    test_filter = test_filter_env;
  }

  // See https://bazel.build/reference/test-encyclopedia#initial-conditions
  std::optional<std::string> xml_output_file;
  if (const char* xml_output_file_env = getenv("XML_OUTPUT_FILE");
      xml_output_file_env != nullptr &&
      !std::string_view(xml_output_file_env).empty()) {
    xml_output_file = xml_output_file_env;
  }

  xls::FormatPreference preference = xls::FormatPreference::kDefault;
  if (!absl::GetFlag(FLAGS_format_preference).empty()) {
    absl::StatusOr<xls::FormatPreference> flag_preference =
        xls::FormatPreferenceFromString(absl::GetFlag(FLAGS_format_preference));
    // `default` is not a legal overriding format preference.
    QCHECK(flag_preference.ok() &&
           absl::GetFlag(FLAGS_format_preference) != "default")
        << "-format_preference accepts binary|hex|decimal";
    preference = flag_preference.value();
  }

  absl::StatusOr<xls::dslx::TestResult> test_result = xls::dslx::RealMain(
      args[0], dslx_paths, test_filter, preference, compare_flag, execute,
      warnings_as_errors, seed, trace_channels, max_ticks, xml_output_file);
  if (!test_result.ok()) {
    return xls::ExitStatus(test_result.status());
  }
  if (*test_result != xls::dslx::TestResult::kAllPassed) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
