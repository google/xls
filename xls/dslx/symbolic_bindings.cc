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

#include "xls/dslx/symbolic_bindings.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace xls::dslx {

bool SymbolicBindings::operator==(const SymbolicBindings& other) const {
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
bool SymbolicBindings::operator!=(const SymbolicBindings& other) const {
  return !(*this == other);
}

std::string SymbolicBindings::ToString() const {
  return absl::StrFormat(
      "{%s}", absl::StrJoin(bindings_, ", ",
                            [](std::string* out, const SymbolicBinding& sb) {
                              absl::StrAppendFormat(out, "%s: %s",
                                                    sb.identifier,
                                                    sb.value.ToString());
                            }));
}

absl::flat_hash_map<std::string, InterpValue> SymbolicBindings::ToMap() const {
  absl::flat_hash_map<std::string, InterpValue> map;
  for (const SymbolicBinding& binding : bindings_) {
    map.insert({binding.identifier, binding.value});
  }
  return map;
}

absl::btree_set<std::string> SymbolicBindings::GetKeySet() const {
  absl::btree_set<std::string> set;
  for (const SymbolicBinding& binding : bindings_) {
    set.insert(binding.identifier);
  }
  return set;
}

void SymbolicBindings::Sort() {
  // We'll use the convention that bits types < tuple types < array types.
  // Returns true if lhs < rhs.
  std::sort(
      bindings_.begin(), bindings_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.identifier < rhs.identifier ||
               (lhs.identifier == rhs.identifier && lhs.value < rhs.value);
      });
}

}  // namespace xls::dslx
