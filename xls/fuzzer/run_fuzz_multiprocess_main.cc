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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/thread.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/run_fuzz_multiprocess.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"

ABSL_FLAG(absl::Duration, duration, absl::InfiniteDuration(),
          "Duration to run the sample generator for.");
ABSL_FLAG(int64_t, calls_per_sample, 128, "Arguments to generate per sample.");
ABSL_FLAG(std::optional<std::string>, crash_path, std::nullopt,
          "Path at which to place crash data.");
ABSL_FLAG(bool, codegen, false, "Run code generation.");
ABSL_FLAG(bool, emit_loops, true, "Emit loops in generator.");
ABSL_FLAG(
    bool, force_failure, false,
    "Forces the samples to fail. Can be used to test failure code paths.");
ABSL_FLAG(bool, generate_proc, false, "Generate a proc sample.");
ABSL_FLAG(int64_t, max_width_aggregate_types, 1024,
          "The maximum width of aggregate types (tuples and arrays) in the "
          "generated samples.");
ABSL_FLAG(int64_t, max_width_bits_types, 64,
          "The maximum width of bits types in the generated samples.");
ABSL_FLAG(int64_t, proc_ticks, 100,
          "Number of ticks to execute the generated procs.");
ABSL_FLAG(std::optional<int64_t>, sample_count, std::nullopt,
          "Number of samples to generate.");
ABSL_FLAG(std::optional<std::string>, save_temps_path, std::nullopt,
          "Path of directory in which to save temporary files. These temporary "
          "files include DSLX, IR, and arguments. A separate numerically-named "
          "subdirectory is created for each sample.");
ABSL_FLAG(std::optional<int64_t>, seed, std::nullopt,
          "Seed value for generation. By default, a nondetermistic seed is "
          "used; if a seed is provided, it is used for determinism");
ABSL_FLAG(bool, simulate, false, "Run Verilog simulation.");
ABSL_FLAG(std::optional<std::string>, simulator, std::nullopt,
          "Verilog simulator to use.");
ABSL_FLAG(std::optional<std::string>, summary_path, std::nullopt,
          "Directory in which to write the sample summary information. This "
          "records information about each generated sample including which XLS "
          "op types and widths. Information is written in Protobuf text format "
          "with one file per worker. Files are appended to by the worker.");
ABSL_FLAG(std::optional<int64_t>, timeout_seconds, std::nullopt,
          "The timeout value in seconds for each subcommand invocation.");
ABSL_FLAG(bool, use_llvm_jit, true,
          "Use LLVM JIT to evaluate IR. The interpreter is still invoked at "
          "least once on the IR even with this option enabled, but this option "
          "can be used to disable the JIT entirely.");
ABSL_FLAG(
    bool, use_system_verilog, true,
    "If true, emit SystemVerilog during codegen; otherwise emit Verilog.");
ABSL_FLAG(std::optional<int64_t>, worker_count, std::nullopt,
          "Number of workers to use for execution; defaults to number of "
          "physical cores detected.");
ABSL_FLAG(bool, with_valid_holdoff, false,
          "If true, emit valid random holdoffs on proc input channels.");

namespace xls {
namespace {

struct Options {
  absl::Duration duration;
  int64_t calls_per_sample;
  std::optional<std::filesystem::path> crash_path;
  bool codegen;
  bool emit_loops;
  bool force_failure;
  bool generate_proc;
  int64_t max_width_aggregate_types;
  int64_t max_width_bits_types;
  int64_t proc_ticks;
  std::optional<int64_t> sample_count;
  std::optional<std::filesystem::path> save_temps_path;
  std::optional<int64_t> seed;
  bool simulate;
  std::optional<std::string> simulator;
  std::optional<std::filesystem::path> summary_path;
  std::optional<int64_t> timeout_seconds;
  bool use_llvm_jit;
  bool use_system_verilog;
  std::optional<int64_t> worker_count;
  bool with_valid_holdoff;
};

absl::Status CheckOrCreateWritableDirectory(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path) &&
      !std::filesystem::create_directory(path)) {
    return absl::InvalidArgumentError(
        absl::StrCat(path.string(), " could not be created"));
  }
  if (!std::filesystem::is_directory(path)) {
    return absl::InvalidArgumentError(
        absl::StrCat(path.string(), " is not a directory"));
  }
  XLS_RETURN_IF_ERROR(SetFileContents(path / "test", "test"));
  return absl::OkStatus();
}

absl::Status RealMain(const Options& options) {
  if (options.crash_path.has_value()) {
    XLS_RETURN_IF_ERROR(CheckOrCreateWritableDirectory(*options.crash_path));
  }
  if (options.summary_path.has_value()) {
    XLS_RETURN_IF_ERROR(CheckOrCreateWritableDirectory(*options.summary_path));
  }

  int64_t worker_count;
  if (options.worker_count.has_value()) {
    worker_count = *options.worker_count;
  } else {
    worker_count = std::max(AvailableCPUs(), 1);
  }

  dslx::AstGeneratorOptions ast_generator_options;
  ast_generator_options.emit_gate = !options.codegen;
  ast_generator_options.emit_loops = options.emit_loops;
  ast_generator_options.max_width_bits_types = options.max_width_bits_types;
  ast_generator_options.max_width_aggregate_types =
      options.max_width_aggregate_types;
  ast_generator_options.generate_proc = options.generate_proc;

  SampleOptions sample_options;
  sample_options.set_calls_per_sample(
      options.generate_proc ? 0 : options.calls_per_sample);
  sample_options.set_codegen(options.codegen);
  sample_options.set_convert_to_ir(true);
  sample_options.set_input_is_dslx(true);
  sample_options.set_ir_converter_args({"--top=main"});
  sample_options.set_optimize_ir(true);
  sample_options.set_proc_ticks(options.generate_proc ? options.proc_ticks : 0);
  sample_options.set_sample_type(options.generate_proc
                                     ? fuzzer::SAMPLE_TYPE_PROC
                                     : fuzzer::SAMPLE_TYPE_FUNCTION);
  sample_options.set_simulate(options.simulate);
  if (options.simulator.has_value()) {
    sample_options.set_simulator(*options.simulator);
  }
  if (options.timeout_seconds.has_value()) {
    sample_options.set_timeout_seconds(*options.timeout_seconds);
  }
  sample_options.set_use_jit(options.use_llvm_jit);
  sample_options.set_use_system_verilog(options.use_system_verilog);
  sample_options.set_with_valid_holdoff(options.with_valid_holdoff);

  return ParallelGenerateAndRunSamples(
      worker_count, ast_generator_options, sample_options, options.seed,
      /*top_run_dir=*/options.save_temps_path,
      /*crasher_dir=*/options.crash_path, /*summary_dir=*/options.summary_path,
      options.sample_count, options.duration, options.force_failure);
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments = xls::InitXls(
      absl::StrCat(
          "Runs multiple fuzzing threads in parallel; sample usage:\n\n\t",
          argv[0]),
      argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Unexpected positional arguments: "
                << absl::StrJoin(positional_arguments, ", ");
  }
  if (absl::GetFlag(FLAGS_simulate) && !absl::GetFlag(FLAGS_codegen)) {
    LOG(QFATAL) << "Must specify --codegen when --simulate is given.";
  }

  return xls::ExitStatus(xls::RealMain({
      .duration = absl::GetFlag(FLAGS_duration),
      .calls_per_sample = absl::GetFlag(FLAGS_calls_per_sample),
      .crash_path = absl::GetFlag(FLAGS_crash_path),
      .codegen = absl::GetFlag(FLAGS_codegen),
      .emit_loops = absl::GetFlag(FLAGS_emit_loops),
      .force_failure = absl::GetFlag(FLAGS_force_failure),
      .generate_proc = absl::GetFlag(FLAGS_generate_proc),
      .max_width_aggregate_types =
          absl::GetFlag(FLAGS_max_width_aggregate_types),
      .max_width_bits_types = absl::GetFlag(FLAGS_max_width_bits_types),
      .proc_ticks = absl::GetFlag(FLAGS_proc_ticks),
      .sample_count = absl::GetFlag(FLAGS_sample_count),
      .save_temps_path = absl::GetFlag(FLAGS_save_temps_path),
      .seed = absl::GetFlag(FLAGS_seed),
      .simulate = absl::GetFlag(FLAGS_simulate),
      .simulator = absl::GetFlag(FLAGS_simulator),
      .summary_path = absl::GetFlag(FLAGS_summary_path),
      .timeout_seconds = absl::GetFlag(FLAGS_timeout_seconds),
      .use_llvm_jit = absl::GetFlag(FLAGS_use_llvm_jit),
      .use_system_verilog = absl::GetFlag(FLAGS_use_system_verilog),
      .worker_count = absl::GetFlag(FLAGS_worker_count),
      .with_valid_holdoff = absl::GetFlag(FLAGS_with_valid_holdoff),
  }));
}
