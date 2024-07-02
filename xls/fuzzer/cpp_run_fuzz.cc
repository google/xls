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

#include "xls/fuzzer/cpp_run_fuzz.h"

#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/fuzzer/sample.h"

namespace xls {
namespace {

absl::StatusOr<std::filesystem::path> GetIrMinimizerMainPath() {
  return GetXlsRunfilePath("xls/tools/ir_minimizer_main");
}

absl::StatusOr<std::filesystem::path> GetSampleRunnerMainPath() {
  return GetXlsRunfilePath("xls/fuzzer/sample_runner_main");
}

absl::StatusOr<std::filesystem::path> GetFindFailingInputMainPath() {
  return GetXlsRunfilePath("xls/fuzzer/find_failing_input_main");
}

}  // namespace

absl::StatusOr<std::optional<std::filesystem::path>> MinimizeIr(
    const Sample& smp, std::filesystem::path run_dir,
    std::optional<std::string> inject_jit_result,
    std::optional<absl::Duration> timeout) {
  VLOG(3) << "MinimizeIr; run_dir: " << run_dir;
  if (!std::filesystem::exists(run_dir / "sample.ir")) {
    VLOG(3) << "sample.ir file did not exist within: " << run_dir;
    return std::nullopt;
  }

  SampleOptions ir_minimize_options = smp.options();
  ir_minimize_options.set_input_is_dslx(false);
  std::string ir_minimize_options_str;
  google::protobuf::TextFormat::PrintToString(ir_minimize_options.proto(),
                                    &ir_minimize_options_str);
  XLS_RETURN_IF_ERROR(SetFileContents(run_dir / "ir_minimizer.options.pbtxt",
                                      ir_minimize_options_str));

  XLS_ASSIGN_OR_RETURN(std::filesystem::path sample_runner_main_path,
                       GetSampleRunnerMainPath());
  XLS_ASSIGN_OR_RETURN(std::filesystem::path ir_minimizer_main_path,
                       GetIrMinimizerMainPath());

  {
    std::vector<std::string> args = {
        std::string{sample_runner_main_path}, "--logtostderr",
        "--options_file=ir_minimizer.options.pbtxt", "--args_file=args.txt",
        "--input_file=$1"};
    const std::filesystem::path test_script = run_dir / "ir_minimizer_test.sh";
    XLS_RETURN_IF_ERROR(SetFileContents(
        test_script, absl::StrCat("#!/bin/sh\n! ", absl::StrJoin(args, " "))));
    std::filesystem::permissions(test_script,
                                 std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::add);

    std::string basename = ir_minimizer_main_path.stem();
    std::filesystem::path stderr_path =
        run_dir / absl::StrCat(basename, ".stderr");

    XLS_ASSIGN_OR_RETURN(
        SubprocessResult result,
        InvokeSubprocess(
            {ir_minimizer_main_path, "--alsologtostderr",
             absl::StrCat("--test_executable=", test_script.string()),
             "sample.ir"},
            /*cwd=*/run_dir, timeout));
    XLS_RETURN_IF_ERROR(SetFileContents(stderr_path, result.stderr_content));

    if (result.exit_status == 0) {
      std::filesystem::path minimized_ir_path = run_dir / "minimized.ir";
      XLS_RETURN_IF_ERROR(
          SetFileContents(minimized_ir_path, result.stdout_content));
      return minimized_ir_path;
    }
  }

  // If we're opting to not run the JIT, then we're done.
  if (!smp.options().use_jit()) {
    VLOG(3) << "MinimizeIr; not using JIT, so finished";
    return std::nullopt;
  }

  // Next try to minimize assuming the underlying cause was a JIT mismatch.
  // The IR minimizer binary has special machinery for reducing these kinds
  // of failures. The minimization occurs in two steps:
  // (1) Find an input that results in a JIT vs interpreter mismatch (if any)
  // (2) Run the minimization tool using this input as the test.
  std::vector<std::string> extra_args;
  if (inject_jit_result.has_value()) {
    extra_args.push_back("--test_only_inject_jit_result=" +
                         inject_jit_result.value());
  }

  std::optional<SubprocessResult> find_failing_input_result;
  {
    XLS_ASSIGN_OR_RETURN(std::filesystem::path find_failing_input_main_path,
                         GetFindFailingInputMainPath());
    std::string basename = find_failing_input_main_path.stem();
    std::filesystem::path stderr_path =
        run_dir / absl::StrCat(basename, ".stderr");

    std::vector<std::string> args = {find_failing_input_main_path,
                                     "--input_file=args.txt", "sample.ir"};
    args.insert(args.end(), extra_args.begin(), extra_args.end());
    XLS_ASSIGN_OR_RETURN(
        find_failing_input_result,
        InvokeSubprocess(args, /*cwd=*/run_dir, /*optional_timeout=*/timeout));
    XLS_RETURN_IF_ERROR(SetFileContents(
        stderr_path, find_failing_input_result->stderr_content));
    if (find_failing_input_result->timeout_expired) {
      VLOG(3) << "MinimizeIr; find_failing_input timeout expired";
      return std::nullopt;
    }
  }

  VLOG(3) << "find_failing_input_main; exit status: "
          << find_failing_input_result->exit_status;
  XLS_LOG_LINES(INFO, find_failing_input_result->stderr_content);
  if (find_failing_input_result->exit_status == 0) {
    // A failing input for JIT vs interpreter was found.
    VLOG(3) << "Failing input for JIT-vs-interpreter was found.";
    std::string failed_input = find_failing_input_result->stdout_content;
    std::string basename = ir_minimizer_main_path.stem();
    std::filesystem::path stderr_path =
        run_dir / absl::StrCat(basename, "_jit.stderr");
    std::vector<std::string> minimize_args = {
        ir_minimizer_main_path,    "--logtostderr",
        "--test_llvm_jit",         "--use_optimization_pipeline",
        "--input=" + failed_input, "sample.ir"};
    minimize_args.insert(minimize_args.end(), extra_args.begin(),
                         extra_args.end());
    XLS_ASSIGN_OR_RETURN(SubprocessResult minimize_result,
                         InvokeSubprocess(minimize_args, /*cwd=*/run_dir,
                                          /*optional_timeout=*/timeout));
    XLS_RETURN_IF_ERROR(
        SetFileContents(stderr_path, minimize_result.stderr_content));
    if (minimize_result.timeout_expired) {
      VLOG(3) << "MinimizeIr; ir_minimizer_main timeout expired";
      return std::nullopt;
    }
    if (minimize_result.exit_status == 0) {
      std::filesystem::path minimized_ir_path = run_dir / "minimized.ir";
      XLS_RETURN_IF_ERROR(
          SetFileContents(minimized_ir_path, minimize_result.stdout_content));
      return minimized_ir_path;
    }
  }

  return std::nullopt;
}

}  // namespace xls
