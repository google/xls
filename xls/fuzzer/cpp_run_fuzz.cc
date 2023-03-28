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

#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/subprocess.h"

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

// Writes the content into a file of the given name in the directory.
absl::Status WriteToFile(const std::filesystem::path& dir_path,
                         std::string_view filename, std::string_view content,
                         bool executable = false) {
  std::filesystem::path path = dir_path / filename;
  XLS_RETURN_IF_ERROR(SetFileContents(path, content));
  if (executable) {
    std::filesystem::permissions(path, std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::add);
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::optional<std::filesystem::path>> MinimizeIr(
    const Sample& smp, std::filesystem::path run_dir,
    std::optional<std::string> inject_jit_result,
    std::optional<absl::Duration> timeout) {
  XLS_VLOG(3) << "MinimizeIr; run_dir: " << run_dir;
  if (!std::filesystem::exists(run_dir / "sample.ir")) {
    XLS_VLOG(3) << "sample.ir file did not exist within: " << run_dir;
    return std::nullopt;
  }

  auto ir_minimize_options = smp.options().ReplaceInputIsDslx(false);
  XLS_RETURN_IF_ERROR(WriteToFile(run_dir, "ir_minimizer.options.json",
                                  ir_minimize_options.ToJsonText()));

  XLS_ASSIGN_OR_RETURN(std::filesystem::path sample_runner_main_path,
                       GetSampleRunnerMainPath());
  XLS_ASSIGN_OR_RETURN(std::filesystem::path ir_minimizer_main_path,
                       GetIrMinimizerMainPath());

  {
    std::vector<std::string> args = {std::string{sample_runner_main_path},
                                     "--logtostderr",
                                     "--options_file=ir_minimizer.options.json",
                                     "--args_file=args.txt", "--input_file=$1"};
    XLS_RETURN_IF_ERROR(
        WriteToFile(run_dir, "ir_minimizer_test.sh",
                    absl::StrCat("#!/bin/sh\n! ", absl::StrJoin(args, " ")),
                    /*executable=*/true));

    std::string basename = ir_minimizer_main_path.stem();
    std::filesystem::path stderr_path =
        run_dir / absl::StrCat(basename, ".stderr");

    XLS_ASSIGN_OR_RETURN(
        SubprocessResult result,
        InvokeSubprocess(
            {ir_minimizer_main_path, "--alsologtostderr",
             "--test_executable=ir_minimizer_test.sh", "sample.ir"},
            /*cwd=*/run_dir, timeout));
    XLS_RETURN_IF_ERROR(SetFileContents(stderr_path, result.stderr));

    if (result.exit_status == 0) {
      std::filesystem::path minimized_ir_path = run_dir / "minimized.ir";
      XLS_RETURN_IF_ERROR(SetFileContents(minimized_ir_path, result.stdout));
      return minimized_ir_path;
    }
  }

  // If we're opting to not run the JIT, then we're done.
  if (!smp.options().use_jit()) {
    XLS_VLOG(3) << "MinimizeIr; not using JIT, so finished";
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
        InvokeSubprocess(args, /*cwd=*/run_dir, /*timeout=*/timeout));
    XLS_RETURN_IF_ERROR(
        SetFileContents(stderr_path, find_failing_input_result->stderr));
    if (find_failing_input_result->timeout_expired) {
      XLS_VLOG(3) << "MinimizeIr; find_failing_input timeout expired";
      return std::nullopt;
    }
  }

  XLS_VLOG(3) << "find_failing_input_main; exit status: "
              << find_failing_input_result->exit_status;
  XLS_LOG_LINES(INFO, find_failing_input_result->stderr);
  if (find_failing_input_result->exit_status == 0) {
    // A failing input for JIT vs interpreter was found.
    XLS_VLOG(3) << "Failing input for JIT-vs-interpreter was found.";
    std::string failed_input = find_failing_input_result->stdout;
    std::string basename = ir_minimizer_main_path.stem();
    std::filesystem::path stderr_path =
        run_dir / absl::StrCat(basename, "_jit.stderr");
    std::vector<std::string> minimize_args = {
        ir_minimizer_main_path,    "--logtostderr",
        "--test_llvm_jit",         "--use_optimization_pipeline",
        "--input=" + failed_input, "sample.ir"};
    minimize_args.insert(minimize_args.end(), extra_args.begin(),
                         extra_args.end());
    XLS_ASSIGN_OR_RETURN(
        SubprocessResult minimize_result,
        InvokeSubprocess(minimize_args, /*cwd=*/run_dir, /*timeout=*/timeout));
    XLS_RETURN_IF_ERROR(SetFileContents(stderr_path, minimize_result.stderr));
    if (minimize_result.timeout_expired) {
      XLS_VLOG(3) << "MinimizeIr; ir_minimizer_main timeout expired";
      return std::nullopt;
    }
    if (minimize_result.exit_status == 0) {
      std::filesystem::path minimized_ir_path = run_dir / "minimized.ir";
      XLS_RETURN_IF_ERROR(
          SetFileContents(minimized_ir_path, minimize_result.stdout));
      return minimized_ir_path;
    }
  }

  return std::nullopt;
}

}  // namespace xls
