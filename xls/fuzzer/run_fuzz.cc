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

#include "xls/fuzzer/run_fuzz.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "openssl/sha.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/common/subprocess.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/cpp_run_fuzz.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample_generator.h"
#include "xls/fuzzer/sample_runner.h"
#include "xls/fuzzer/sample_summary.pb.h"

namespace xls {

namespace {

constexpr std::string_view kSampleRunnerMainPath =
    "xls/fuzzer/sample_runner_main";

constexpr std::string_view kSummarizeIrMainPath =
    "xls/fuzzer/summarize_ir_main";

std::string ArgvToCmdline(const std::vector<std::string>& argv) {
  static constexpr auto ShellDoubleQuoteFormatter = [](std::string* out,
                                                       std::string_view arg) {
    if (absl::StartsWith(arg, "\"") && absl::EndsWith(arg, "\"")) {
      // Assume the argument is already quoted appropriately.
      absl::StrAppend(out, arg);
      return;
    }

    size_t pos = arg.find_first_of(R"($'"*\)");
    if (pos == std::string_view::npos) {
      // No quotes needed!
      absl::StrAppend(out, arg);
      return;
    }

    static constexpr std::string_view kEscapedWithinQuotes = R"($"\)";

    // We'll need at least the length of the string, plus the opening and
    // ending quote.
    out->reserve(out->size() + arg.size() + 2);

    absl::StrAppend(out, "\"");
    size_t start_pos = 0;
    for (pos = arg.find_first_of(kEscapedWithinQuotes);
         pos < arg.size() && pos != std::string_view::npos;
         pos = arg.find_first_of(kEscapedWithinQuotes, pos + 1)) {
      absl::StrAppend(out, arg.substr(start_pos, pos - start_pos));
      start_pos = pos + 1;
    }
    if (start_pos < arg.size() && start_pos != std::string_view::npos) {
      absl::StrAppend(out, arg.substr(start_pos));
    }
    absl::StrAppend(out, "\"");
  };

  return absl::StrJoin(argv, " ", ShellDoubleQuoteFormatter);
}

absl::Status WriteIrSummaries(const std::filesystem::path& run_dir,
                              const fuzzer::SampleTimingProto& timing,
                              const std::filesystem::path& summary_path) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path summarize_ir_main_path,
                       GetXlsRunfilePath(kSummarizeIrMainPath));
  std::string timing_str;
  if (!google::protobuf::TextFormat::PrintToString(timing, &timing_str)) {
    LOG(ERROR) << "Failed to serialize timing: " << timing.DebugString();
  }
  std::vector<std::string> argv = {
      summarize_ir_main_path,
      "--logtostderr",
      "--minloglevel=2",
      absl::StrCat("--summary_file=", summary_path.string()),
      absl::StrCat("--timing=", timing_str),
  };

  bool has_ir = false;
  std::filesystem::path unoptimized_path = run_dir / "sample.ir";
  if (xls::FileExists(unoptimized_path).ok()) {
    argv.push_back("--unoptimized_ir=" + unoptimized_path.string());
    has_ir = true;
  }
  std::filesystem::path optimized_path = run_dir / "sample.opt.ir";
  if (xls::FileExists(optimized_path).ok()) {
    argv.push_back("--optimized_ir=" + optimized_path.string());
    has_ir = true;
  }
  if (!has_ir) {
    return absl::OkStatus();
  }

  absl::StatusOr<SubprocessResult> result =
      SubprocessErrorAsStatus(InvokeSubprocess(argv, /*cwd=*/run_dir));
  if (!result.ok()) {
    LOG(ERROR) << "Failed to write IR summaries: " << result.status();
  }
  return absl::OkStatus();
}

// Save the sample into a new directory in the crasher directory.
absl::StatusOr<std::filesystem::path> SaveCrasher(
    const std::filesystem::path& run_dir, const Sample& smp,
    const absl::Status& error, const std::filesystem::path& crasher_dir) {
  std::array<char, SHA256_DIGEST_LENGTH> digest;
  SHA256(reinterpret_cast<const uint8_t*>(smp.input_text().data()),
         smp.input_text().size(), reinterpret_cast<uint8_t*>(digest.data()));
  // Extract the first 4 bytes of the digest as 8 hex digits.
  static_assert(digest.size() >= 4);
  std::string hex_digest = absl::BytesToHexString({digest.data(), 4});

  std::filesystem::path sample_crasher_dir = crasher_dir / hex_digest;
  LOG(INFO) << "Saving crasher to " << sample_crasher_dir;
  // TODO(epastor): 2023-09-28 - Make sure this preserves permissions.
  std::filesystem::copy(run_dir, sample_crasher_dir,
                        std::filesystem::copy_options::recursive);

  XLS_RETURN_IF_ERROR(
      SetFileContents(sample_crasher_dir / "exception.txt", error.ToString()));

  XLS_RETURN_IF_ERROR(SetFileContents(
      sample_crasher_dir /
          absl::StrFormat(
              "crasher_%s_%s.x",
              absl::FormatTime("%Y-%m-%d", absl::Now(), absl::LocalTimeZone()),
              hex_digest.substr(0, 4)),
      smp.ToCrasher(error.message())));
  return sample_crasher_dir;
}

}  // namespace

absl::Status RunSample(const Sample& smp, const std::filesystem::path& run_dir,
                       const std::optional<std::filesystem::path>& summary_file,
                       std::optional<absl::Duration> generate_sample_elapsed) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path sample_runner_main_path,
                       GetXlsRunfilePath(kSampleRunnerMainPath));

  Stopwatch stopwatch;
  std::vector<std::string> argv = {
      sample_runner_main_path.string(),
      "--logtostderr",
  };

  std::filesystem::path sample_file_name = run_dir / "sample.x";
  XLS_RETURN_IF_ERROR(
      SetFileContents(run_dir / sample_file_name, smp.input_text()));
  argv.push_back("--input_file=sample.x");

  std::filesystem::path options_file_name = run_dir / "options.pbtxt";
  XLS_RETURN_IF_ERROR(
      SetFileContents(run_dir / options_file_name, smp.options().ToPbtxt()));
  argv.push_back("--options_file=options.pbtxt");

  std::filesystem::path args_file_name = run_dir / "args.txt";
  XLS_RETURN_IF_ERROR(SetFileContents(run_dir / args_file_name,
                                      ArgsBatchToText(smp.args_batch())));
  argv.push_back("--args_file=args.txt");

  std::optional<std::filesystem::path> ir_channel_names_file_name =
      std::nullopt;
  if (!smp.ir_channel_names().empty()) {
    ir_channel_names_file_name = run_dir / "ir_channel_names.txt";
    XLS_RETURN_IF_ERROR(
        SetFileContents(run_dir / *ir_channel_names_file_name,
                        IrChannelNamesToText(smp.ir_channel_names())));
    argv.push_back("--ir_channel_names_file=ir_channel_names.txt");
  }

  argv.push_back(run_dir.string());

  std::filesystem::path run_script_path = run_dir / "run.sh";
  XLS_RETURN_IF_ERROR(
      SetFileContents(run_script_path, absl::StrFormat(R"(#!/bin/sh

{ %s }
)",
                                                       ArgvToCmdline(argv))));
  std::filesystem::permissions(run_script_path,
                               std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::add);

  VLOG(1) << "Starting to run sample";
  VLOG(2) << smp.input_text();
  SampleRunner runner(run_dir);
  XLS_RETURN_IF_ERROR(runner.RunFromFiles(sample_file_name, options_file_name,
                                          args_file_name,
                                          ir_channel_names_file_name));

  fuzzer::SampleTimingProto timing = runner.timing();

  absl::Duration total_elapsed = stopwatch.GetElapsedTime();
  if (generate_sample_elapsed.has_value()) {
    timing.set_generate_sample_ns(
        absl::ToInt64Nanoseconds(*generate_sample_elapsed));

    // The sample generation time, if given, is not part of the measured total
    // time, so add it in.
    total_elapsed += *generate_sample_elapsed;
  }
  timing.set_total_ns(absl::ToInt64Nanoseconds(total_elapsed));

  VLOG(1) << "Completed running sample, elapsed: " << total_elapsed;

  if (summary_file.has_value()) {
    XLS_RETURN_IF_ERROR(WriteIrSummaries(run_dir, timing, *summary_file));
  }
  return absl::OkStatus();
}

absl::StatusOr<Sample> GenerateSampleAndRun(
    dslx::FileTable& file_table, absl::BitGenRef bit_gen,
    const dslx::AstGeneratorOptions& ast_generator_options,
    const SampleOptions& sample_options, const std::filesystem::path& run_dir,
    const std::optional<std::filesystem::path>& crasher_dir,
    const std::optional<std::filesystem::path>& summary_file,
    bool force_failure) {
  Stopwatch stopwatch;
  XLS_ASSIGN_OR_RETURN(
      Sample smp, GenerateSample(ast_generator_options, sample_options, bit_gen,
                                 file_table));
  absl::Duration generate_sample_elapsed = stopwatch.GetElapsedTime();

  absl::Status status =
      RunSample(smp, run_dir, summary_file, generate_sample_elapsed);
  if (force_failure) {
    status = absl::InternalError("Forced sample failure.");
  }
  if (status.ok()) {
    return smp;
  }

  LOG(ERROR) << "Sample failed: " << status;
  if (crasher_dir.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::filesystem::path sample_crasher_dir,
                         SaveCrasher(run_dir, smp, status, *crasher_dir));
    if (!absl::IsDeadlineExceeded(status)) {
      LOG(INFO) << "Attempting to minimize IR...";
      std::optional<absl::Duration> timeout =
          sample_options.timeout_seconds().has_value()
              ? std::optional<absl::Duration>(
                    absl::Seconds(*sample_options.timeout_seconds()))
              : std::nullopt;
      XLS_ASSIGN_OR_RETURN(
          std::optional<std::filesystem::path> minimized_path,
          MinimizeIr(smp, run_dir, /*inject_jit_result=*/std::nullopt,
                     timeout));
      if (minimized_path.has_value()) {
        LOG(INFO) << "...minimization successful; output at "
                  << *minimized_path;
        std::filesystem::copy(*minimized_path, sample_crasher_dir);
      } else {
        LOG(INFO) << "...minimization failed.";
      }
    }
  }
  return status;
}

}  // namespace xls
