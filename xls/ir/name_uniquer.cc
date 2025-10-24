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
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/ret_check.h"

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

absl::Status NameUniquer::ReleaseIdentifier(std::string_view sv) {
  if (auto it = generated_names_.find(sv); it != generated_names_.end()) {
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

absl::Status NameUniquer::SequentialIdGenerator::Release(int64_t id) {
  XLS_RET_CHECK(used_.contains(id)) << id << " is not marked as in use";
  used_.erase(id);
  return absl::OkStatus();
}

}  // namespace xls
