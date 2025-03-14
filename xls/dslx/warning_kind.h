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

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/common/strong_int.h"

namespace xls::dslx {

// The integer type underlying a warning -- this is used in enough places, and
// has a high enough probability that it will grow over time as the flag set
// gets larger, that we factor it out here into an alias.
using WarningKindInt = uint16_t;

enum class WarningKind : WarningKindInt {
  kConstexprEvalRollover = 1 << 0,
  kSingleLineTupleTrailingComma = 1 << 1,
  kMisleadingFunctionName = 1 << 2,
  kUselessLetBinding = 1 << 3,
  kUselessStructSplat = 1 << 4,
  kEmptyRangeLiteral = 1 << 5,
  kUnusedDefinition = 1 << 6,
  kUselessExpressionStatement = 1 << 7,
  kTrailingTupleAfterSemi = 1 << 8,
  kConstantNaming = 1 << 9,
  kMemberNaming = 1 << 10,
  kShouldUseAssert = 1 << 11,
  kAlreadyExhaustiveMatch = 1 << 12,
};
constexpr WarningKindInt kWarningKindCount = 13;

inline constexpr std::array<WarningKind, kWarningKindCount> kAllWarningKinds = {
    WarningKind::kConstexprEvalRollover,
    WarningKind::kSingleLineTupleTrailingComma,
    WarningKind::kMisleadingFunctionName,
    WarningKind::kUselessLetBinding,
    WarningKind::kUselessStructSplat,
    WarningKind::kEmptyRangeLiteral,
    WarningKind::kUnusedDefinition,
    WarningKind::kUselessExpressionStatement,
    WarningKind::kTrailingTupleAfterSemi,
    WarningKind::kConstantNaming,
    WarningKind::kMemberNaming,
    WarningKind::kShouldUseAssert,
    WarningKind::kAlreadyExhaustiveMatch,
};

// Flag set datatype.
XLS_DEFINE_STRONG_INT_TYPE(WarningKindSet, WarningKindInt);

inline constexpr WarningKindSet kNoWarningsSet = WarningKindSet{0};

// Note: for the "default" set of warnings to use in an application, prefer
// `kDefaultWarningsSet` below.
inline constexpr WarningKindSet kAllWarningsSet =
    WarningKindSet{(WarningKindInt{1} << kWarningKindCount) - 1};

// Set intersection.
inline WarningKindSet operator&(WarningKindSet a, WarningKindSet b) {
  return WarningKindSet{a.value() & b.value()};
}

// Set union.
inline WarningKindSet operator|(WarningKindSet a, WarningKindSet b) {
  return WarningKindSet{a.value() | b.value()};
}

// Returns the complement of a warning set.
//
// Note that we define this instead of operator~ because it has an existing
// overload for STRONG_INT_TYPE.
inline WarningKindSet Complement(WarningKindSet a) {
  return WarningKindSet{~a.value() & kAllWarningsSet.value()};
}

// Disables "warning" out of "set" and returns that updated result.
constexpr WarningKindSet DisableWarning(WarningKindSet set,
                                        WarningKind warning) {
  return WarningKindSet{set.value() & ~static_cast<WarningKindInt>(warning)};
}

constexpr WarningKindSet EnableWarning(WarningKindSet set,
                                       WarningKind warning) {
  return WarningKindSet{set.value() | static_cast<WarningKindInt>(warning)};
}

// Returns whether "warning" is enabled in "set".
inline bool WarningIsEnabled(WarningKindSet set, WarningKind warning) {
  return (set.value() & static_cast<WarningKindInt>(warning)) != 0;
}

// TODO(leary): 2024-03-15 Enable "should use fail if" by default after some
// propagation time.
// TODO(cdleary): 2025-02-03 Enable "already exhaustive match" by default after
// some propagation time.
inline constexpr WarningKindSet kDefaultWarningsSet = DisableWarning(
    DisableWarning(kAllWarningsSet, WarningKind::kShouldUseAssert),
    WarningKind::kAlreadyExhaustiveMatch);

// Converts a string representation of a warnings to its corresponding enum
// value.
absl::StatusOr<WarningKind> WarningKindFromString(std::string_view s);

absl::StatusOr<std::string_view> WarningKindToString(WarningKind kind);

// Converts a comma-delimited string of warning kinds that should be *disabled*
// into an "enabled set".
absl::StatusOr<WarningKindSet> WarningKindSetFromDisabledString(
    std::string_view disabled_string);

// As above, but starts with an empty set and enables warnings as they appear in
// the string.
absl::StatusOr<WarningKindSet> WarningKindSetFromString(
    std::string_view enabled_string);

std::string WarningKindSetToString(WarningKindSet set);

// Returns the default warning set with the modifications given in flags.
//
// If flags are contradictory, returns an error.
absl::StatusOr<WarningKindSet> GetWarningsSetFromFlags(
    std::string_view enable_warnings, std::string_view disable_warnings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_WARNING_KIND_H_
