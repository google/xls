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

#include "xls/dslx/warning_kind.h"

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

absl::StatusOr<WarningKind> WarningKindFromString(std::string_view s) {
  if (s == "constexpr_eval_rollover") {
    return WarningKind::kConstexprEvalRollover;
  }
  if (s == "single_line_tuple_trailing_comma") {
    return WarningKind::kSingleLineTupleTrailingComma;
  }
  if (s == "misleading_function_name") {
    return WarningKind::kMisleadingFunctionName;
  }
  if (s == "useless_let_binding") {
    return WarningKind::kUselessLetBinding;
  }
  if (s == "useless_struct_splat") {
    return WarningKind::kUselessStructSplat;
  }
  if (s == "empty_range_literal") {
    return WarningKind::kEmptyRangeLiteral;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown warning kind: `", s, "`"));
}

absl::StatusOr<WarningKindSet> WarningKindSetFromDisabledString(
    std::string_view disabled_string) {
  WarningKindSet enabled = kAllWarningsSet;
  if (disabled_string.empty()) {
    return enabled;
  }
  for (std::string_view s : absl::StrSplit(disabled_string, ',')) {
    XLS_ASSIGN_OR_RETURN(WarningKind k, WarningKindFromString(s));
    enabled = DisableWarning(enabled, k);
  }
  return enabled;
}

}  // namespace xls::dslx
