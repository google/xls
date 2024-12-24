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

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/status_macros.h"

namespace xls::dslx {

absl::StatusOr<std::string_view> WarningKindToString(WarningKind kind) {
  switch (kind) {
    case WarningKind::kConstexprEvalRollover:
      return "constexpr_eval_rollover";
    case WarningKind::kSingleLineTupleTrailingComma:
      return "single_line_tuple_trailing_comma";
    case WarningKind::kMisleadingFunctionName:
      return "misleading_function_name";
    case WarningKind::kUselessLetBinding:
      return "useless_let_binding";
    case WarningKind::kUselessStructSplat:
      return "useless_struct_splat";
    case WarningKind::kEmptyRangeLiteral:
      return "empty_range_literal";
    case WarningKind::kUnusedDefinition:
      return "unused_definition";
    case WarningKind::kUselessExpressionStatement:
      return "useless_expression_statement";
    case WarningKind::kTrailingTupleAfterSemi:
      return "trailing_tuple_after_semi";
    case WarningKind::kConstantNaming:
      return "constant_naming";
    case WarningKind::kMemberNaming:
      return "member_naming";
    case WarningKind::kShouldUseAssert:
      return "should_use_assert";
    case WarningKind::kAlreadyExhaustiveMatch:
      return "already_exhaustive_match";
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid warning kind: ", static_cast<int>(kind)));
}

absl::StatusOr<WarningKind> WarningKindFromString(std::string_view s) {
  for (WarningKind kind : kAllWarningKinds) {
    // Note: .value() is safe because it's a known valid WarningKind because it
    // comes directly from the kAllWarningKinds set.
    if (s == WarningKindToString(kind).value()) {
      return kind;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Unknown warning kind: `", s, "`; all warning kinds: ",
      absl::StrJoin(kAllWarningKinds, ", ",
                    [](std::string* out, WarningKind k) {
                      absl::StrAppend(out, WarningKindToString(k).value());
                    })));
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

absl::StatusOr<WarningKindSet> WarningKindSetFromString(
    std::string_view enabled_string) {
  WarningKindSet enabled = kNoWarningsSet;
  if (enabled_string.empty()) {
    return enabled;
  }
  for (std::string_view s : absl::StrSplit(enabled_string, ',')) {
    XLS_ASSIGN_OR_RETURN(WarningKind k, WarningKindFromString(s));
    enabled = EnableWarning(enabled, k);
  }
  return enabled;
}

std::string WarningKindSetToString(WarningKindSet set) {
  std::vector<std::string_view> enabled_warnings;
  for (WarningKind kind : kAllWarningKinds) {
    if (WarningIsEnabled(set, kind)) {
      enabled_warnings.push_back(WarningKindToString(kind).value());
    }
  }
  return absl::StrJoin(enabled_warnings, ",");
}

absl::StatusOr<WarningKindSet> GetWarningsSetFromFlags(
    std::string_view enable_warnings, std::string_view disable_warnings) {
  XLS_ASSIGN_OR_RETURN(WarningKindSet enabled,
                       WarningKindSetFromString(enable_warnings));
  XLS_ASSIGN_OR_RETURN(WarningKindSet disabled,
                       WarningKindSetFromString(disable_warnings));
  if ((enabled & disabled) != kNoWarningsSet) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot both enable and disable the same warning(s); enabled: %s "
        "disabled: %s",
        WarningKindSetToString(enabled), WarningKindSetToString(disabled)));
  }
  return (kDefaultWarningsSet | enabled) & Complement(disabled);
}

}  // namespace xls::dslx
