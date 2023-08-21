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

#ifndef XLS_DSLX_WARNING_KIND_H_
#define XLS_DSLX_WARNING_KIND_H_

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/common/strong_int.h"

namespace xls::dslx {

// The integer type underlying a warning -- this is used in enough places, and
// has a high enough probability that it will grow over time as the flag set
// gets larger, that we factor it out here into an alias.
using WarningKindInt = uint8_t;

enum class WarningKind : WarningKindInt {
  kConstexprEvalRollover = 1 << 0,
  kSingleLineTupleTrailingComma = 1 << 1,
  kMisleadingFunctionName = 1 << 2,
  kUselessLetBinding = 1 << 3,
  kUselessStructSplat = 1 << 4,
  kEmptyRangeLiteral = 1 << 5,
};
constexpr WarningKindInt kWarningKindCount = 6;

inline constexpr auto kAllWarningKinds = {
    WarningKind::kConstexprEvalRollover,
    WarningKind::kSingleLineTupleTrailingComma,
    WarningKind::kMisleadingFunctionName,
    WarningKind::kUselessStructSplat,
    WarningKind::kEmptyRangeLiteral,
};

// Flag set datatype.
XLS_DEFINE_STRONG_INT_TYPE(WarningKindSet, WarningKindInt);

inline constexpr WarningKindSet kAllWarningsSet =
    WarningKindSet{(WarningKindInt{1} << kWarningKindCount) - 1};

// Converts a string representation of a warnings to its corresponding enum
// value.
absl::StatusOr<WarningKind> WarningKindFromString(std::string_view s);

// Converts a comma-delimited string of warning kinds that should be *disabled*
// into an "enabled set".
absl::StatusOr<WarningKindSet> WarningKindSetFromDisabledString(
    std::string_view disabled_string);

// Disables "warning" out of "set" and returns that updated result.
inline WarningKindSet DisableWarning(WarningKindSet set, WarningKind warning) {
  return WarningKindSet{set.value() & ~static_cast<WarningKindInt>(warning)};
}

// Returns whether "warning" is enabled in "set".
inline bool WarningIsEnabled(WarningKindSet set, WarningKind warning) {
  return (set.value() & static_cast<WarningKindInt>(warning)) != 0;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_WARNING_KIND_H_
