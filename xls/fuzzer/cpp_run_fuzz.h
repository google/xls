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

#ifndef XLS_FUZZER_CPP_RUN_FUZZ_H_
#define XLS_FUZZER_CPP_RUN_FUZZ_H_

#include <filesystem>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/fuzzer/sample.h"

namespace xls {

// Tries to minimize the IR of the given sample in the `run_dir`.
//
// Side Effects:
//  Writes a test script into the `run_dir` for testing the IR for the failure.
//  Passes this test script to ir_minimizer_main to try to create a minimal IR
//  sample.
//
// Args:
//   smp: The sample to try to minimize.
//   run_dir: The run directory the sample was run in.
//   inject_jit_result: For testing only. Value to produce as the JIT result.
//   timeout: Timeout for running the minimizer.
//
// Returns:
//   The path to the minimized IR file (created in the `run_dir`), or nullopt if
//   minimization was not possible.
absl::StatusOr<std::optional<std::filesystem::path>> MinimizeIr(
    const Sample& smp, std::filesystem::path run_dir,
    std::optional<std::string> inject_jit_result = std::nullopt,
    std::optional<absl::Duration> timeout = std::nullopt);

}  // namespace xls

#endif  // XLS_FUZZER_CPP_RUN_FUZZ_H_
