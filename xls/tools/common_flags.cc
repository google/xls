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

#include "xls/tools/common_flags.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "xls/passes/optimization_pass.h"

// LINT.IfChange
ABSL_FLAG(std::string, top, "", "Top entity of the package.");
ABSL_FLAG(std::optional<std::string>, ir_dump_path, std::nullopt,
          "Location to dump intermediate IR files.");
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel, []() -> const std::string& {
  static const std::string kDescription = absl::StrFormat(
      "Optimization level. Ranges from 0 to %d.", xls::kMaxOptLevel);
  return kDescription;
}());
// LINT.ThenChange(
//   //xls/build_rules/xls_providers.bzl,
//   //docs_src/codegen_options.md
// )
