// Copyright 2021 The XLS Authors
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

// Library that backs the `opt_main` tool's primary functionality.

#ifndef XLS_TOOLS_OPT_H_
#define XLS_TOOLS_OPT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// TODO(meheff): 2021-10-04 Remove this header.
#include "xls/passes/passes.h"

namespace xls::tools {

// Various optimizer options (that generally funnel from the `opt_main` tool to
// this consolidated library).
struct OptOptions {
  int64_t opt_level = xls::kMaxOptLevel;
  int64_t mutual_exclusion_z3_rlimit = -1;
  std::string_view top;
  std::string ir_dump_path = "";
  std::optional<std::string> ir_path = std::nullopt;
  std::optional<std::vector<std::string>> run_only_passes = std::nullopt;
  std::vector<std::string> skip_passes;
  std::optional<int64_t> convert_array_index_to_select = std::nullopt;
  bool inline_procs;
  std::vector<RamRewrite> ram_rewrites = {};
};

// Helper used in the opt_main tool, optimizes the given IR for a particular
// top-level entity (e.g., function, proc, etc) at the given opt level and
// returns the resulting optimized IR.
absl::StatusOr<std::string> OptimizeIrForTop(std::string_view ir,
                                             const OptOptions& options);

// Convenience wrapper around the above that builds an OptOptions appropriately.
// Analagous to calling opt_main with each argument being a flag.
absl::StatusOr<std::string> OptimizeIrForTop(
    std::string_view input_path, int64_t opt_level,
    int64_t mutual_exclusion_z3_rlimit, std::string_view top,
    std::string_view ir_dump_path,
    absl::Span<const std::string> run_only_passes,
    absl::Span<const std::string> skip_passes,
    int64_t convert_array_index_to_select, bool inline_procs,
    std::string_view ram_rewrites_pb);

}  // namespace xls::tools

#endif  // XLS_TOOLS_OPT_H_
