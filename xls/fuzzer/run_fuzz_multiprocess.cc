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

#include "xls/fuzzer/run_fuzz_multiprocess.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/common/thread.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"

namespace xls {
namespace {

static constexpr std::string_view kBlueText = "\033[34m";
static constexpr std::string_view kRedText = "\033[31m";
static constexpr std::string_view kDefaultColor = "\033[0m";

absl::Status GenerateAndRunSamples(
    int64_t worker_number,
    const dslx::AstGeneratorOptions& ast_generator_options,
    const SampleOptions& sample_options, const std::optional<uint64_t>& seed,
    const std::optional<std::filesystem::path>& top_run_dir,
    const std::optional<std::filesystem::path>& crasher_dir,
    const std::optional<std::filesystem::path>& summary_dir,
    std::optional<int64_t> sample_count,
    const std::optional<absl::Duration>& duration, bool force_failure) {
  int64_t crashers = 0;
  LOG(INFO) << "--- Started worker " << worker_number;
  Stopwatch stopwatch;

  std::optional<std::filesystem::path> summary_file;
  if (summary_dir.has_value()) {
    summary_file =
        *summary_dir / absl::StrCat("summary_", worker_number, ".binarypb");
  }

  uint64_t rng_seed;
  if (seed.has_value()) {
    // Set seed deterministically based on the worker number so different
    // workers generate different samples.
    rng_seed = *seed + worker_number;
  } else {
    // Choose a nondeterministic seed.
    rng_seed = absl::Uniform<uint64_t>(absl::BitGen());
    LOG(INFO) << kBlueText << "--- NOTE: Worker #" << worker_number
              << " chose a nondeterministic seed for value generation: "
              << absl::StreamFormat("0x%16X", rng_seed) << kDefaultColor;
  }
  std::mt19937_64 rng{rng_seed};

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
        GenerateSampleAndRun(rng, ast_generator_options, sample_options,
                             run_dir, crasher_dir, summary_file, force_failure)
            .status();
    if (!sample_status.ok()) {
      LOG(INFO) << kRedText
                << absl::StreamFormat(
                       "--- Worker #%d noted crasher #%d for sample number %d",
                       worker_number, crashers, sample)
                << kDefaultColor;
      crashers++;
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
      LOG(INFO) << absl::StreamFormat("--- Worker #%d: %s", worker_number,
                                      absl::StrJoin(metrics, ", "));
    }

    if (duration.has_value() && elapsed >= *duration) {
      LOG(INFO) << absl::StreamFormat("--- Worker #%d: Ran for %s. Exiting.",
                                      worker_number,
                                      absl::FormatDuration(elapsed));
      break;
    }
    ++sample;
    if (sample_count.has_value() && sample >= *sample_count) {
      LOG(INFO) << absl::StreamFormat(
          "--- Worker #%d: Ran %d samples. Exiting.", worker_number, sample);
      break;
    }
  }

  absl::Duration elapsed = stopwatch.GetElapsedTime();
  LOG(INFO) << absl::StreamFormat(
      "--- Worker #%d finished! %d samples; %d crashers; %.2f samples/s; ran "
      "for %s",
      worker_number, sample, crashers,
      static_cast<double>(sample) / absl::ToDoubleSeconds(elapsed),
      absl::FormatDuration(elapsed));
  return absl::OkStatus();
}

}  // namespace

absl::Status ParallelGenerateAndRunSamples(
    int64_t worker_count,
    const dslx::AstGeneratorOptions& ast_generator_options,
    const SampleOptions& sample_options, std::optional<uint64_t> seed,
    const std::optional<std::filesystem::path>& top_run_dir,
    const std::optional<std::filesystem::path>& crasher_dir,
    const std::optional<std::filesystem::path>& summary_dir,
    std::optional<int64_t> sample_count, std::optional<absl::Duration> duration,
    bool force_failure) {
  std::vector<std::unique_ptr<Thread>> workers;
  workers.resize(worker_count);
  std::vector<absl::Status> worker_status;
  worker_status.resize(workers.size(),
                       absl::InternalError("worker did not terminate."));
  for (int64_t i = 0; i < workers.size(); ++i) {
    std::optional<int64_t> worker_sample_count =
        sample_count.has_value()
            ? std::make_optional((*sample_count + i) / worker_count)
            : std::nullopt;
    workers[i] = std::make_unique<Thread>([&, i, worker_sample_count,
                                           status = &worker_status[i]] {
      *status =
          GenerateAndRunSamples(i, ast_generator_options, sample_options, seed,
                                top_run_dir, crasher_dir, summary_dir,
                                worker_sample_count, duration, force_failure);
    });
  }
  for (int64_t i = 0; i < workers.size(); ++i) {
    LOG(INFO) << "-- Waiting on worker " << i;
    workers[i]->Join();
    if (!worker_status[i].ok()) {
      LOG(ERROR) << kRedText << "-- Worker #" << i
                 << " failed: " << worker_status[i] << kDefaultColor;
    }
  }
  return absl::OkStatus();
}

}  // namespace xls
