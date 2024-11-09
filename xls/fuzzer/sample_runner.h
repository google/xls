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

#ifndef XLS_FUZZER_SAMPLE_RUNNER_H_
#define XLS_FUZZER_SAMPLE_RUNNER_H_

#include <filesystem>  // NOLINT
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample_summary.pb.h"

namespace xls {

// A class for performing various operations on a code sample.

// Code sample can be in DSLX or IR. The possible operations include:
//     * Converting DSLX to IR (DSLX input only).
//     * Interpeting the code with supplied arguments.
//     * Optimizing IR.
//     * Generating Verilog.
//     * Simulating Verilog.
//     * Comparing interpreter/simulation results for equality.
// The runner operates in a single directory supplied at construction time and
// records all state, command invocations, and outputs to that directory to
// enable easier debugging and replay.
class SampleRunner {
 public:
  struct Commands {
    // Call the particular operation with given arguments and options. Return
    // their output or failure status.
    using Callable = std::function<absl::StatusOr<std::string>(
        const std::vector<std::string>& args,
        const std::filesystem::path& run_dir, const SampleOptions& options)>;

    // Various tools that can be invoked as simple function call.
    // Functions might invoke external binaries to perform their task.
    std::optional<Callable> codegen_main;
    std::optional<Callable> eval_ir_main;
    std::optional<Callable> eval_proc_main;
    std::optional<Callable> ir_converter_main;
    std::optional<Callable> ir_opt_main;
    std::optional<Callable> simulate_module_main;
  };

  explicit SampleRunner(std::filesystem::path run_dir)
      : run_dir_(std::move(run_dir)) {}
  SampleRunner(std::filesystem::path run_dir, Commands commands)
      : run_dir_(std::move(run_dir)), commands_(std::move(commands)) {}

  // Runs the provided sample, writing out files under the SampleRunner's
  // `run_dir` as appropriate.
  absl::Status Run(const Sample& sample);

  // Runs the provided files as a sample, writing out only outputs under the
  // SampleRunner's `run_dir`.
  absl::Status RunFromFiles(
      const std::filesystem::path& input_path,
      const std::filesystem::path& options_path,
      const std::optional<std::filesystem::path>& args_path,  // deprecated
      const std::optional<std::filesystem::path>&
          ir_channel_names_path,  // same
      const std::optional<std::filesystem::path>& testvector_path);

  const fuzzer::SampleTimingProto& timing() const { return timing_; }

 private:
  // Runs a sample with a function as the top which is read from files.
  absl::Status RunFunction(
      const std::filesystem::path& input_path, const SampleOptions& options,
      const std::optional<std::filesystem::path>& args_path,
      const std::optional<std::filesystem::path>& testvector_path);

  absl::Status RunProc(
      const std::filesystem::path& input_path, const SampleOptions& options,
      const std::optional<std::filesystem::path>& args_path,
      const std::optional<std::filesystem::path>& ir_channel_names_path,
      const std::optional<std::filesystem::path>& testvector_path);

  const std::filesystem::path run_dir_;
  const Commands commands_;
  fuzzer::SampleTimingProto timing_;
};

}  // namespace xls

#endif  // XLS_FUZZER_SAMPLE_RUNNER_H_
