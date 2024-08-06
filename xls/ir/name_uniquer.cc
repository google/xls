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
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"

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

}  // namespace

std::string NameUniquer::GetSanitizedUniqueName(std::string_view prefix) {
  std::string root = SanitizeName(prefix, reserved_names_);

  // Strip away a numeric suffix. For example, grab "foo" from "foo__42". This
  // avoids the possibility of a given prefix (e.g., prefix is "foo__42")
  // colliding with a uniquified name (e.g., GetSanitizedUniqueName("foo")
  // returns "foo__42")
  std::optional<int64_t> numeric_suffix;
  size_t separator_index = root.rfind(separator_);
  if (separator_index != std::string::npos) {
    std::string suffix = root.substr(separator_index + separator_.size());
    int64_t i;
    if (absl::SimpleAtoi(suffix, &i)) {
      numeric_suffix = i;
      // Remove numeric suffix from root.
      root = root.substr(0, separator_index);
    }
  }

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
  return root;
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

}  // namespace xls
