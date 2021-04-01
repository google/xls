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

#include <time.h>
#include <unistd.h>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/run_routines.h"

ABSL_FLAG(std::string, dslx_path, "",
          "Additional paths to search for modules (colon delimited).");
ABSL_FLAG(bool, trace_all, false, "Trace every expression.");
ABSL_FLAG(bool, compare_jit, true,
          "Compare interpreted and JIT execution of each function.");
ABSL_FLAG(
    int64_t, seed, 0,
    "Seed for quickcheck random stimulus; 0 for an nondetermistic value.");
// TODO(leary): 2021-01-19 allow filters with wildcards.
ABSL_FLAG(std::string, test_filter, "",
          "Target (currently *single*) test name to run.");

namespace xls::dslx {
namespace {

const char* kUsage = R"(
Parses, typechecks, and executes all tests inside of a DSLX module.
)";

absl::Status RealMain(absl::string_view entry_module_path,
                      absl::Span<const std::string> dslx_paths,
                      absl::optional<std::string> test_filter, bool trace_all,
                      bool compare_jit, absl::optional<int64_t> seed,
                      bool* printed_error) {
  XLS_ASSIGN_OR_RETURN(std::string program, GetFileContents(entry_module_path));
  XLS_ASSIGN_OR_RETURN(std::string module_name, PathToName(entry_module_path));
  JitComparator jit_comparator;
  XLS_ASSIGN_OR_RETURN(
      *printed_error,
      ParseAndTest(program, module_name, entry_module_path, dslx_paths,
                   test_filter, trace_all,
                   compare_jit ? &jit_comparator : nullptr, seed));
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls::dslx

int main(int argc, char* argv[]) {
  std::vector<absl::string_view> args =
      xls::InitXls(xls::dslx::kUsage, argc, argv);
  if (args.empty()) {
    XLS_LOG(QFATAL) << "Wrong number of command-line arguments; got "
                    << args.size() << ": `" << absl::StrJoin(args, " ")
                    << "`; want " << argv[0] << " <input-file>";
  }
  std::string dslx_path = absl::GetFlag(FLAGS_dslx_path);
  std::vector<std::string> dslx_paths = absl::StrSplit(dslx_path, ':');

  bool trace_all = absl::GetFlag(FLAGS_trace_all);
  bool compare_jit = absl::GetFlag(FLAGS_compare_jit);

  // Optional seed value.
  absl::optional<int64_t> seed;
  if (int64_t seed_flag_value = absl::GetFlag(FLAGS_seed);
      seed_flag_value != 0) {
    seed = seed_flag_value;
  }

  // Optional test filter.
  absl::optional<std::string> test_filter;
  if (std::string flag = absl::GetFlag(FLAGS_test_filter); !flag.empty()) {
    test_filter = std::move(flag);
  }

  bool printed_error = false;
  absl::Status status =
      xls::dslx::RealMain(args[0], dslx_paths, test_filter, trace_all,
                          compare_jit, seed, &printed_error);
  if (printed_error) {
    return EXIT_FAILURE;
  }
  XLS_QCHECK_OK(status);
  return EXIT_SUCCESS;
}
