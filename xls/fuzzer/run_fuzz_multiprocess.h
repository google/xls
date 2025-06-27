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

#ifndef XLS_FUZZER_RUN_FUZZ_MULTIPROCESS_H_
#define XLS_FUZZER_RUN_FUZZ_MULTIPROCESS_H_

#include <cstdint>
#include <filesystem>
#include <optional>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"

namespace xls {

// Generate and run fuzzer samples on `worker_count` threads; runs up to
// `sample_count` samples (unbounded if unspecified) for up to `duration` time.
//
// Generates samples according to `ast_generator_options`, and runs them
// according to `sample_options`. Uses a nondeterministic seed if `seed` is not
// specified. Creates run directories in `top_run_dir` if specified; otherwise
// creates ephemeral temporary directories for each sample. Failing samples are
// written to `crasher_dir`, and summaries to `summary_dir`.
//
// If `force_failure` is true, every sample run will be considered a failure.
// This is useful for testing failure paths.
absl::Status ParallelGenerateAndRunSamples(
    int64_t worker_count,
    const dslx::AstGeneratorOptions& ast_generator_options,
    const SampleOptions& sample_options,
    std::optional<uint64_t> seed = std::nullopt,
    const std::optional<std::filesystem::path>& top_run_dir = std::nullopt,
    const std::optional<std::filesystem::path>& crasher_dir = std::nullopt,
    const std::optional<std::filesystem::path>& summary_dir = std::nullopt,
    std::optional<int64_t> sample_count = std::nullopt,
    std::optional<absl::Duration> duration = std::nullopt,
    bool force_failure = false);

}  // namespace xls

#endif  // XLS_FUZZER_RUN_FUZZ_MULTIPROCESS_H_
