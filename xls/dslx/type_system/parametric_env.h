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

#ifndef XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_ENV_H_
#define XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_ENV_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

// A single parametric environment entry (binds a parametric integral typed
// variable name to a value). For example, in:
//
//    fn id<N: u32>(x: uN[N]) -> uN[N] { x }
//    fn main() -> u32 { id(u32:0) }
//
// The parametric env item for N given id invoked in main is `{"N", 32}`.
struct ParametricEnvItem {
  std::string identifier;
  InterpValue value;

  bool operator==(const ParametricEnvItem& other) const {
    return identifier == other.identifier && value == other.value;
  }
  bool operator!=(const ParametricEnvItem& other) const {
    return !(*this == other);
  }
};

// Sequence of parametric bindings in stable order (wraps the backing vector
// storage to make it immutable, hashable, among other utility functions).
//
// Stable order is that bindings are sorted as (identifier, value) tuples.
class ParametricEnv {
 public:
  ParametricEnv() = default;

  explicit ParametricEnv(
      absl::Span<std::pair<std::string, InterpValue> const> items) {
    for (const auto& item : items) {
      bindings_.push_back(ParametricEnvItem{item.first, item.second});
    }
    Sort();
  }
  explicit ParametricEnv(
      const absl::flat_hash_map<std::string, InterpValue>& mapping) {
    for (const auto& item : mapping) {
      bindings_.push_back(ParametricEnvItem{item.first, item.second});
    }
    Sort();
  }

  template <typename H>
  friend H AbslHashValue(H h, const ParametricEnv& self) {
    for (const ParametricEnvItem& sb : self.bindings_) {
      h = H::combine(std::move(h), sb.identifier, sb.value.ToString());
    }
    return h;
  }

  bool operator==(const ParametricEnv& other) const;
  bool operator!=(const ParametricEnv& other) const;

  bool empty() const { return bindings_.empty(); }

  // Returns a string representation of the contained parametric bindings
  // suitable for debugging.
  std::string ToString() const;

  // Returns a map representation of these parametric bindings.
  absl::flat_hash_map<std::string, InterpValue> ToMap() const;

  // Returns the set of keys in these parametric bindings. A btree is given for
  // ease of range-based comparisons.
  absl::btree_set<std::string> GetKeySet() const;

  int64_t size() const { return bindings_.size(); }
  const std::vector<ParametricEnvItem>& bindings() const { return bindings_; }

  // Returns the value of the given binding, if any is mapped in this env.
  std::optional<InterpValue> GetValue(const NameDef* binding) const;

 private:
  // Sorts all of the bindings_ elements (to maintain the sorted-order invariant
  // of this data structure).
  void Sort();

  std::vector<ParametricEnvItem> bindings_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ParametricEnv& parametric_env) {
  os << parametric_env.ToString();
  return os;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_ENV_H_
