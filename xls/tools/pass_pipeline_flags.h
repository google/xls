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

#ifndef XLS_TOOLS_PASS_PIPELINE_FLAGS_H_
#define XLS_TOOLS_PASS_PIPELINE_FLAGS_H_

#include "absl/status/statusor.h"
#include "xls/tools/pass_pipeline_config.h"

namespace xls {

// Result of parsing the optimization pass pipeline command-line flags.
struct PassPipelineFlagsResult {
  OptimizationPipelineConfig config;
  // True if any pass-pipeline CLI flag was specified on the command line or set
  // to a non-default value.
  bool any_flags_set = false;
};

// Reads the pass-pipeline command-line flags (`--passes`, `--passes_proto`,
// `--passes_textproto`, `--pipeline_proto`, `--pipeline_textproto`,
// `--skip_passes`, `--passes_bisect_limit`, `--passes_bisect_limit_is_error`,
// `--debug_optimizations`) and returns the resolved configuration. Returns an
// error if conflicting or invalid flag values are provided.
absl::StatusOr<PassPipelineFlagsResult> GetPassPipelineFlags();

}  // namespace xls

#endif  // XLS_TOOLS_PASS_PIPELINE_FLAGS_H_
