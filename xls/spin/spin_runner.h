// Copyright 2026 The XLS Authors
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

// Entry point for Promela generation + SPIN simulation + trace comparison.

#ifndef XLS_SPIN_SPIN_RUNNER_H_
#define XLS_SPIN_SPIN_RUNNER_H_

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/spin/trace_compare.h"

namespace xls::spin {

enum class SpinExecutionType { kGuided, kExhaustive };

// Options forwarded from interpreter_main to RunSpinCheck.
struct SpinRunOptions {
  SpinExecutionType exec_type = SpinExecutionType::kGuided;
  std::string dslx_stdlib_path;
  std::vector<std::filesystem::path> dslx_paths;
  std::optional<std::string> test_filter;
  bool type_inference_v2 = false;
  // Required when exec_type == kGuided; ignored for kExhaustive.
  xls::EvaluatorResultsProto* results_proto = nullptr;
};

// DSLX-level: full pipeline from source text -- picks a #[test_proc] (using
// options.test_filter if there are several), converts it, and verifies it.
absl::Status RunSpinCheck(std::string_view dslx_source,
                          std::string_view entry_module_path,
                          std::string_view module_name,
                          const SpinRunOptions& options = {});

}  // namespace xls::spin

#endif  // XLS_SPIN_SPIN_RUNNER_H_
