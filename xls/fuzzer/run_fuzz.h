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

#ifndef XLS_FUZZER_RUN_FUZZ_H_
#define XLS_FUZZER_RUN_FUZZ_H_

#include <filesystem>  // NOLINT
#include <optional>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"

namespace xls {

// Runs the given sample in `run_dir`. If `summary_file` is given, the sample
// summary will be appended to this file; if `generate_sample_elapsed` is also
// given, it will be recorded in the timings in the sample summary.
//
// `run_dir` must be an empty directory.
absl::Status RunSample(
    const Sample& smp, const std::filesystem::path& run_dir,
    const std::optional<std::filesystem::path>& summary_file = std::nullopt,
    std::optional<absl::Duration> generate_sample_elapsed = std::nullopt);

absl::StatusOr<Sample> GenerateSampleAndRun(
    dslx::FileTable& file_table, absl::BitGenRef bit_gen,
    const dslx::AstGeneratorOptions& ast_generator_options,
    const SampleOptions& sample_options, const std::filesystem::path& run_dir,
    const std::optional<std::filesystem::path>& crasher_dir = std::nullopt,
    const std::optional<std::filesystem::path>& summary_file = std::nullopt,
    bool force_failure = false);

}  // namespace xls

#endif  // XLS_FUZZER_RUN_FUZZ_H_
