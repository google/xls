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
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <random>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/stopwatch.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/run_fuzz.h"
#include "xls/fuzzer/sample.h"

ABSL_FLAG(bool, use_nondeterministic_seed, false,
          "Use a non-deterministic seed for the random number generator. If "
          "specified, the flag --seed is ignored.");
ABSL_FLAG(uint64_t, seed, 0, "Seed value for generation");
ABSL_FLAG(int64_t, sample_count, 10, "Number of samples to generate");
ABSL_FLAG(std::optional<absl::Duration>, duration, std::nullopt,
          "Duration to run the fuzzer for. Examples: 60s, 30m, 5h");
ABSL_FLAG(int64_t, calls_per_sample, 128,
          "Arguments to generate per sample. The value is valid when "
          "`--generate_proc` is `false`.");
ABSL_FLAG(int64_t, proc_ticks, 128,
          "The number of ticks to execute when generating a proc. The value is "
          "valid when `--generate_proc` is `true`.");
ABSL_FLAG(bool, simulate, false, "Run Verilog simulation.");
ABSL_FLAG(std::optional<std::string>, simulator, std::nullopt,
          "Verilog simulator to use. For example: \"iverilog\".");
ABSL_FLAG(int64_t, max_width_bits_types, 64,
          "The maximum width of bits types in the generated samples.");
ABSL_FLAG(int64_t, max_width_aggregate_types, 1024,
          "The maximum width of aggregate types (tuples and arrays) in the "
          "generated samples.");
ABSL_FLAG(
    bool, force_failure, false,
    "Forces the samples to fail. Can be used to test failure code paths.");
ABSL_FLAG(bool, use_system_verilog, false,
          "Whether to generate SystemVerilog or Verilog.");
ABSL_FLAG(
    std::optional<absl::Duration>, sample_timeout, std::nullopt,
    "Maximum time to run each sample before timing out. Examples: 10s, 2m, 1h");
ABSL_FLAG(bool, generate_proc, false, "Generate a proc sample.");

// The maximum number of failures before the test aborts.
constexpr int64_t kMaxFailures = 10;

namespace xls {
namespace {

// Returns the directory in which to write crashers.
//
// Crashers are written to the undeclared outputs directory, if it is available.
// Otherwise a temporary directory is created.
absl::StatusOr<std::optional<std::filesystem::path>> GetCrasherDir() {
  char* undeclared_outputs_dir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (undeclared_outputs_dir == nullptr) {
    return std::nullopt;
  }
  std::filesystem::path crasher_dir =
      std::filesystem::path(undeclared_outputs_dir) / "crashers";
  XLS_RETURN_IF_ERROR(RecursivelyCreateDir(crasher_dir));
  return std::move(crasher_dir);
}

TEST(FuzzIntegrationTest, Fuzzing) {
  XLS_ASSERT_OK_AND_ASSIGN(std::optional<std::filesystem::path> crasher_dir,
                           GetCrasherDir());
  uint64_t seed = absl::GetFlag(FLAGS_seed);
  if (absl::GetFlag(FLAGS_use_nondeterministic_seed)) {
    seed = absl::Uniform<uint64_t>(absl::BitGen());
    LOG(INFO) << "Random seed (generated nondeterministically): " << seed;
  } else {
    LOG(INFO) << "Random seed specified via flag: " << seed;
  }
  std::mt19937_64 rng{seed};

  dslx::AstGeneratorOptions ast_generator_options{
      .max_width_bits_types = absl::GetFlag(FLAGS_max_width_bits_types),
      .max_width_aggregate_types =
          absl::GetFlag(FLAGS_max_width_aggregate_types),
      .emit_gate = !absl::GetFlag(FLAGS_simulate),
      .generate_proc = absl::GetFlag(FLAGS_generate_proc)};

  SampleOptions sample_options;
  sample_options.set_input_is_dslx(true);
  sample_options.set_ir_converter_args({"--top=main"});
  sample_options.set_convert_to_ir(true);
  if (absl::GetFlag(FLAGS_generate_proc)) {
    sample_options.set_calls_per_sample(0);
    sample_options.set_proc_ticks(absl::GetFlag(FLAGS_proc_ticks));
  } else {
    sample_options.set_calls_per_sample(absl::GetFlag(FLAGS_calls_per_sample));
    sample_options.set_proc_ticks(0);
  }
  sample_options.set_use_jit(true);
  sample_options.set_codegen(absl::GetFlag(FLAGS_simulate));
  sample_options.set_simulate(absl::GetFlag(FLAGS_simulate));
  if (absl::GetFlag(FLAGS_simulator).has_value()) {
    sample_options.set_simulator(*absl::GetFlag(FLAGS_simulator));
  }
  sample_options.set_use_system_verilog(
      absl::GetFlag(FLAGS_use_system_verilog));
  if (absl::GetFlag(FLAGS_sample_timeout).has_value()) {
    sample_options.set_timeout_seconds(
        absl::ToInt64Seconds(*absl::GetFlag(FLAGS_sample_timeout)));
  }

  int64_t crasher_count = 0;
  int64_t sample_count = 0;

  Stopwatch run_time;

  auto keep_going = [target_duration = absl::GetFlag(FLAGS_duration),
                     target_sample_count = absl::GetFlag(FLAGS_sample_count),
                     &run_time, &sample_count]() -> bool {
    if (target_duration.has_value()) {
      if (run_time.GetElapsedTime() >= *target_duration) {
        LOG(INFO) << "Ran for target duration of " << *target_duration
                  << ". Exiting.";
        return false;
      }
    } else {
      if (sample_count >= target_sample_count) {
        LOG(INFO) << "Generated target number of samples. Exiting.";
        return false;
      }
    }
    return true;
  };

  while (keep_going()) {
    LOG(INFO) << "Running sample " << sample_count++;
    XLS_ASSERT_OK_AND_ASSIGN(TempDirectory run_dir, TempDirectory::Create());
    absl::Status status =
        GenerateSampleAndRun(rng, ast_generator_options, sample_options,
                             run_dir.path(), crasher_dir,
                             /*summary_file=*/std::nullopt,
                             absl::GetFlag(FLAGS_force_failure))
            .status();
    if (!status.ok()) {
      LOG(ERROR) << "Sample failed: " << status;
      crasher_count += 1;
    }

    if (crasher_count >= kMaxFailures) {
      break;
    }
  }

  if (crasher_count >= kMaxFailures) {
    FAIL() << "Fuzzing stopped after finding " << crasher_count
           << " failures. Generated " << sample_count
           << " total samples [seed = " << seed << "]";
  }
  if (crasher_count > 0) {
    ADD_FAILURE() << "Fuzzing found " << crasher_count
                  << " failures. Generated " << sample_count
                  << " total samples [seed = " << seed << "]";
  }
}

}  // namespace
}  // namespace xls
