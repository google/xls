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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/value_generator.h"

constexpr std::string_view kUsage = R"(Fuzzer multiprocess worker.

Runs a fuzzing instance with the appropriate worker configuration, outputting expected log lines to stdout and stderr.

run_fuzz_multiprocess_worker \
  --number=WORKER_NUMBER \
  --ast_generator_options=SERIALIZED_GENERATOR_OPTIONS \
  --sample_options=SERIALIZED_SAMPLE_OPTIONS \
  --crasher_dir=CRASHER_DIR \
  [--seed=SEED] \
  [--top_run_dir=TOP_RUN_DIR] \
  [--summary_dir=SUMMARY_DIR] \
  [--sample_count=SAMPLE_COUNT] \
  [--duration=DURATION] \
  [--force_failure]
)";

ABSL_FLAG(uint64_t, number, 0, "Worker number");
ABSL_FLAG(xls::dslx::AstGeneratorOptions, ast_generator_options, {},
          "AST generator options");
ABSL_FLAG(xls::SampleOptions, sample_options, {}, "Sample options");
ABSL_FLAG(std::string, crasher_dir, {}, "Crasher directory");
ABSL_FLAG(std::optional<int64_t>, seed, std::nullopt, "RNG seed");
ABSL_FLAG(std::optional<std::string>, top_run_dir, std::nullopt,
          "Top run directory");
ABSL_FLAG(std::optional<std::string>, summary_dir, std::nullopt,
          "Summary directory");
ABSL_FLAG(std::optional<int64_t>, sample_count, std::nullopt, "Sample count");
ABSL_FLAG(std::optional<double>, duration_seconds, std::nullopt,
          "Duration in seconds");
ABSL_FLAG(bool, force_failure, false, "Force failure");

namespace xls {
namespace {

static constexpr std::string_view kBlueText = "\033[34m";
static constexpr std::string_view kRedText = "\033[31m";
static constexpr std::string_view kDefaultColor = "\033[0m";

absl::Status RealMain(uint64_t worker_number,
                      const dslx::AstGeneratorOptions& ast_generator_options,
                      const SampleOptions& sample_options,
                      const std::filesystem::path& crasher_dir,
                      const std::optional<int64_t>& seed,
                      const std::optional<std::filesystem::path>& top_run_dir,
                      const std::optional<std::filesystem::path>& summary_dir,
                      const std::optional<int64_t>& sample_count,
                      const std::optional<absl::Duration>& duration,
                      bool force_failure) {
  int64_t crashers = 0;
  std::cout << "--- Started worker " << worker_number << "\n";
  Stopwatch stopwatch;

  std::optional<std::filesystem::path> summary_file;
  std::optional<TempFile> summary_temp_file;
  if (summary_dir.has_value()) {
    summary_file =
        *summary_dir / absl::StrCat("summary_", worker_number, ".binarypb");
    XLS_ASSIGN_OR_RETURN(summary_temp_file, TempFile::Create("_temp_summary"));
  }

  int64_t rng_seed;
  if (seed.has_value()) {
    // Set seed deterministically based on the worker number so different
    // workers generate different samples.
    rng_seed = *seed + worker_number;
  } else {
    // Choose a nondeterministic seed.
    std::random_device r;
    rng_seed = std::uniform_int_distribution<int64_t>()(r);
    std::cout << kBlueText << "--- NOTE: Worker #" << worker_number
              << " chose a nondeterministic seed for value generation: 0x"
              << absl::StreamFormat("%16X", rng_seed) << kDefaultColor << "\n";
  }
  ValueGenerator rng{std::mt19937_64(rng_seed)};

  int64_t sample = 0;
  while (true) {
    std::filesystem::path run_dir;
    std::optional<TempDirectory> temp_run_dir;
    if (top_run_dir.has_value()) {
      run_dir = *top_run_dir /
                absl::StrFormat("worker%d-sample%d", worker_number, sample);
      XLS_RETURN_IF_ERROR(RecursivelyCreateDir(run_dir));
    } else {
      XLS_ASSIGN_OR_RETURN(temp_run_dir, TempDirectory::Create());
      run_dir = temp_run_dir->path();
    }

    absl::Status sample_status =
        GenerateSampleAndRun(&rng, ast_generator_options, sample_options,
                             run_dir, crasher_dir, summary_file, force_failure)
            .status();
    if (!sample_status.ok()) {
      std::cout << kRedText
                << absl::StreamFormat(
                       "--- Worker #%d noted crasher #%d for sample number %d",
                       worker_number, crashers, sample)
                << kDefaultColor << "\n";
      crashers++;
    }

    if (summary_file.has_value() && sample % 25 == 0) {
      // Append the local temporary summary file to the actual, potentially
      // remote one, and delete the temporary file.
      XLS_ASSIGN_OR_RETURN(std::string temp_summary,
                           GetFileContents(summary_temp_file->path()));
      XLS_RETURN_IF_ERROR(AppendStringToFile(*summary_file, temp_summary));
      XLS_ASSIGN_OR_RETURN(summary_temp_file,
                           TempFile::Create("_temp_summary"));
    }

    absl::Duration elapsed = stopwatch.GetElapsedTime();
    if (sample > 0 && sample % 16 == 0) {
      std::vector<std::string> metrics;
      metrics.reserve(3);
      if (sample_count.has_value()) {
        metrics.push_back(
            absl::StrFormat("%d/%d samples", sample, *sample_count));
      } else {
        metrics.push_back(absl::StrCat(sample, " samples"));
      }
      metrics.push_back(absl::StrFormat(
          "%.2f samples/s",
          static_cast<double>(sample) / absl::ToDoubleSeconds(elapsed)));
      if (duration.has_value()) {
        metrics.push_back(absl::StrFormat("running for %s (limit %s)",
                                          absl::FormatDuration(elapsed),
                                          absl::FormatDuration(*duration)));
      } else {
        metrics.push_back(
            absl::StrFormat("running for %s", absl::FormatDuration(elapsed)));
      }
      std::cout << absl::StreamFormat("--- Worker #%d: %s\n", worker_number,
                                      absl::StrJoin(metrics, ", "));
      std::cout.flush();
    }

    if (duration.has_value() && elapsed >= *duration) {
      std::cout << absl::StreamFormat("--- Worker #%d: Ran for %s. Exiting.\n",
                                      worker_number,
                                      absl::FormatDuration(elapsed));
      break;
    }
    ++sample;
    if (sample_count.has_value() && sample >= *sample_count) {
      std::cout << absl::StreamFormat(
          "--- Worker #%d: Ran %d samples. Exiting.\n", worker_number, sample);
      break;
    }
  }

  absl::Duration elapsed = stopwatch.GetElapsedTime();
  std::cout << absl::StreamFormat(
      "--- Worker #%d finished! %d samples; %d crashers; %.2f samples/s; ran "
      "for %s\n",
      worker_number, sample, crashers,
      static_cast<double>(sample) / absl::ToDoubleSeconds(elapsed),
      absl::FormatDuration(elapsed));
  std::cout.flush();
  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (!positional_arguments.empty()) {
    XLS_LOG(QFATAL) << "Usage:\n" << kUsage;
  }

  std::optional<absl::Duration> duration;
  if (std::optional<double> duration_seconds =
          absl::GetFlag(FLAGS_duration_seconds);
      duration_seconds.has_value()) {
    duration = absl::Seconds(*duration_seconds);
  }
  return xls::ExitStatus(xls::RealMain(
      absl::GetFlag(FLAGS_number), absl::GetFlag(FLAGS_ast_generator_options),
      absl::GetFlag(FLAGS_sample_options), absl::GetFlag(FLAGS_crasher_dir),
      absl::GetFlag(FLAGS_seed), absl::GetFlag(FLAGS_top_run_dir),
      absl::GetFlag(FLAGS_summary_dir), absl::GetFlag(FLAGS_sample_count),
      duration, absl::GetFlag(FLAGS_force_failure)));
}
