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

#include "xls/dslx/type_system/parametric_env.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

bool ParametricEnv::operator==(const ParametricEnv& other) const {
  if (bindings_.size() != other.bindings_.size()) {
    return false;
  }
  for (int64_t i = 0; i < bindings_.size(); ++i) {
    if (bindings_[i] != other.bindings_[i]) {
      return false;
    }
  }
  return true;
}
bool ParametricEnv::operator!=(const ParametricEnv& other) const {
  return !(*this == other);
}

std::string ParametricEnv::ToString() const {
  return absl::StrFormat(
      "{%s}", absl::StrJoin(bindings_, ", ",
                            [](std::string* out, const ParametricEnvItem& sb) {
                              absl::StrAppendFormat(out, "%s: %s",
                                                    sb.identifier,
                                                    sb.value.ToString());
                            }));
}

absl::flat_hash_map<std::string, InterpValue> ParametricEnv::ToMap() const {
  absl::flat_hash_map<std::string, InterpValue> map;
  for (const ParametricEnvItem& binding : bindings_) {
    map.insert({binding.identifier, binding.value});
  }
  return map;
}

absl::btree_set<std::string> ParametricEnv::GetKeySet() const {
  absl::btree_set<std::string> set;
  for (const ParametricEnvItem& binding : bindings_) {
    set.insert(binding.identifier);
  }
  return set;
}

void ParametricEnv::Sort() {
  // We'll use the convention that bits types < tuple types < array types.
  // Returns true if lhs < rhs.
  std::sort(
      bindings_.begin(), bindings_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.identifier < rhs.identifier ||
               (lhs.identifier == rhs.identifier && lhs.value < rhs.value);
      });
}

std::optional<InterpValue> ParametricEnv::GetValue(
    const NameDef* binding) const {
  if (binding->parent() == nullptr ||
      binding->parent()->kind() != AstNodeKind::kParametricBinding) {
    return std::nullopt;
  }
  for (const ParametricEnvItem& item : bindings_) {
    if (item.identifier == binding->identifier()) {
      return item.value;
    }
  }
  return std::nullopt;
}

}  // namespace xls::dslx
