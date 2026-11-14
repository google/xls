// Copyright 2020 The XLS Authors
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

#include "xls/ir/name_uniquer.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

namespace {

bool IsAllowed(char c) {
  return (static_cast<int>(absl::ascii_isalnum(c)) != 0) || c == '_';
}

// Sanitizes and returns the name. Unallowed characters will be replaced with
// '_'. The result will match the regexp "[a-zA-Z_][a-zA-Z0-9_]*". Names which
// collide with `reserved_names` will be prefixed with an underscore.
std::string SanitizeName(
    std::string_view name,
    const absl::flat_hash_set<std::string>& reserved_names) {
  if (name.empty()) {
    return "";
  }

  std::string result(name);

  while (reserved_names.contains(result)) {
    result = "_" + result;
  }

  for (int i = 0; i < name.size(); ++i) {
    if (!IsAllowed(result[i])) {
      result[i] = '_';
    }
  }
  // If name does not begin with an alphabetic character or underscore, prefix
  // the name with an underscore.
  if (!absl::ascii_isalpha(result[0]) && result[0] != '_') {
    result = "_" + result;
  }
  return result;
}

// Strip away a numeric suffix. For example, grab "foo" from "foo__42". This
// avoids the possibility of a given prefix (e.g., prefix is "foo__42")
// colliding with a uniquified name (e.g., GetSanitizedUniqueName("foo")
// returns "foo__42")
std::optional<int64_t> ParseNumericSuffix(std::string_view name,
                                          std::string_view separator,
                                          std::string_view* prefix) {
  size_t separator_index = name.rfind(separator);
  *prefix = name;
  if (separator_index == std::string::npos) {
    return std::nullopt;
  }
  std::string_view suffix = name.substr(separator_index + separator.size());
  // Reject non-canonical integer suffixes that `absl::SimpleAtoi` accepts (such
  // as leading whitespace, `+`/`-` signs, or leading zeros like `"01"`), since
  // they do not round-trip through `absl::StrCat(root, separator, i)` (e.g.,
  // `"foo__01"` and `"foo__1"` are distinct valid identifiers).
  if (suffix.empty() || !absl::ascii_isdigit(suffix[0]) ||
      (suffix.size() > 1 && suffix[0] == '0')) {
    return std::nullopt;
  }
  int64_t i;
  if (absl::SimpleAtoi(suffix, &i)) {
    // Remove numeric suffix from root.
    *prefix = name.substr(0, separator_index);
    return i;
  }
  return std::nullopt;
}

}  // namespace

std::string NameUniquer::GetSanitizedUniqueName(std::string_view prefix) {
  std::string sanitized = SanitizeName(prefix, reserved_names_);

  std::string_view root;
  std::optional<int64_t> numeric_suffix =
      ParseNumericSuffix(sanitized, separator_, &root);

  // If the root is empty after stripping off the suffix, use a generic name.
  root = root.empty() ? "name" : root;

  // This will create  a map entry if it does not already exist.
  PrefixTracker& prefix_tracker = generated_names_[root];

  SequentialIdGenerator& generator = prefix_tracker.generator;
  if (numeric_suffix.has_value()) {
    return absl::StrCat(root, separator_,
                        generator.RegisterId(numeric_suffix.value()));
  }
  if (prefix_tracker.bare_prefix_taken) {
    // There already exists a node with the same root name (no suffix), add a
    // suffix to uniquify the name.
    return absl::StrCat(root, separator_, generator.NextId());
  }

  // Root has not been seen before and there is no suffix. Just return it.
  prefix_tracker.bare_prefix_taken = true;
  return std::string(root);
}

/* static */ bool NameUniquer::IsValidIdentifier(std::string_view str) {
  if (str.empty()) {
    return false;
  }
  if (!absl::ascii_isalpha(str[0]) && str[0] != '_') {
    return false;
  }
  for (int64_t i = 1; i < str.size(); ++i) {
    if (!absl::ascii_isalnum(str[i]) && str[i] != '_') {
      return false;
    }
  }
  return true;
}

absl::Status NameUniquer::ReserveIdentifier(std::string_view sv) {
  std::string_view root;
  std::optional<int64_t> suffix = ParseNumericSuffix(sv, separator_, &root);
  if (suffix) {
    SequentialIdGenerator& generator = generated_names_[root].generator;
    XLS_RET_CHECK(!generator.IsUsed(*suffix))
        << "Identifier '" << sv
        << "' is already in use with suffix: " << *suffix;
    XLS_RET_CHECK_EQ(generator.RegisterId(*suffix), *suffix);
  } else {
    PrefixTracker& tracker = generated_names_[sv];
    XLS_RET_CHECK(!tracker.bare_prefix_taken)
        << "Identifier '" << sv << "' is already in use.";
    tracker.bare_prefix_taken = true;
  }
  return absl::OkStatus();
}

absl::Status NameUniquer::ReleaseIdentifier(std::string_view sv) {
  if (auto it = generated_names_.find(sv);
      it != generated_names_.end() && it->second.bare_prefix_taken) {
    // this is an unadorned name.
    it->second.bare_prefix_taken = false;
    return absl::OkStatus();
  }
  std::string_view root;
  std::optional<int64_t> suffix = ParseNumericSuffix(sv, separator_, &root);
  XLS_RET_CHECK(suffix)
      << "Name '" << sv
      << "' was not a bare identifier and didn't have a numeric suffix?";
  auto it = generated_names_.find(root);
  XLS_RET_CHECK(it != generated_names_.end())
      << "Name '" << sv
      << "' was not a bare identifier and didn't have a numeric suffix?";
  return it->second.generator.Release(*suffix);
}

std::optional<std::string> NameUniquer::GetOriginalName(
    std::string_view sv) const {
  std::string_view root;
  std::optional<int64_t> suffix = ParseNumericSuffix(sv, separator_, &root);
  if (!suffix.has_value()) {
    return std::nullopt;
  }
  auto it = generated_names_.find(root);
  if (it == generated_names_.end()) {
    return std::nullopt;
  }
  std::optional<std::optional<int64_t>> orig_id =
      it->second.generator.GetOriginalId(*suffix);
  if (!orig_id.has_value() || *orig_id == *suffix) {
    return std::nullopt;
  }
  if (!orig_id->has_value()) {
    return std::string(root);
  }
  return absl::StrCat(root, separator_, **orig_id);
}

bool NameUniquer::IsNameAvailable(std::string_view sv) const {
  std::string_view root;
  std::optional<int64_t> suffix = ParseNumericSuffix(sv, separator_, &root);
  if (suffix.has_value()) {
    auto it = generated_names_.find(root);
    if (it == generated_names_.end()) {
      return true;
    }
    return !it->second.generator.IsUsed(*suffix);
  }
  auto it = generated_names_.find(sv);
  if (it == generated_names_.end()) {
    return true;
  }
  return !it->second.bare_prefix_taken;
}

absl::StatusOr<std::optional<std::string>>
NameUniquer::TryReclaimUnqualifiedName(std::string_view sv) {
  if (auto it = generated_names_.find(sv);
      it != generated_names_.end() && it->second.bare_prefix_taken) {
    // This is an unadorned name; it was not uniquified.
    return std::nullopt;
  }
  std::string_view root;
  std::optional<int64_t> suffix = ParseNumericSuffix(sv, separator_, &root);
  XLS_RET_CHECK(suffix.has_value())
      << "Name '" << sv
      << "' was not a bare identifier and didn't have a numeric suffix?";
  auto it = generated_names_.find(root);
  XLS_RET_CHECK(it != generated_names_.end())
      << "Name '" << sv
      << "' was not a bare identifier and didn't have a numeric suffix?";
  PrefixTracker& tracker = it->second;
  std::optional<std::optional<int64_t>> orig_id =
      tracker.generator.GetOriginalId(*suffix);
  XLS_RET_CHECK(orig_id.has_value()) << *suffix << " is not marked as in use";
  if (*orig_id == *suffix) {
    return std::nullopt;
  }
  if (!orig_id->has_value()) {
    if (tracker.bare_prefix_taken) {
      return std::nullopt;
    }
    XLS_RETURN_IF_ERROR(tracker.generator.Release(*suffix));
    tracker.bare_prefix_taken = true;
    return std::string(root);
  }
  int64_t target_id = **orig_id;
  if (tracker.generator.IsUsed(target_id)) {
    return std::nullopt;
  }
  XLS_RETURN_IF_ERROR(tracker.generator.Release(*suffix));
  XLS_RET_CHECK_EQ(tracker.generator.RegisterId(target_id), target_id);
  return absl::StrCat(root, separator_, target_id);
}

absl::Status NameUniquer::SequentialIdGenerator::Release(int64_t id) {
  XLS_RET_CHECK(used_.contains(id)) << id << " is not marked as in use";
  used_.erase(id);
  return absl::OkStatus();
}

}  // namespace xls
