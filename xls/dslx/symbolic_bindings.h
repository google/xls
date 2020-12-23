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

#ifndef XLS_DSLX_SYMBOLIC_BINDINGS_H_
#define XLS_DSLX_SYMBOLIC_BINDINGS_H_

#include <ostream>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xls/common/integral_types.h"

namespace xls::dslx {

// A single symbolic binding entry (binds a parametric integral typed variable
// name to a value). For example, in:
//
//    fn [N: u32] id(x: uN[N]) -> uN[N] { x }
//    fn main() -> u32 { id(u32:0) }
//
// The symbolic binding for N given id invoked in main is `{"N", 32}`.
struct SymbolicBinding {
  std::string identifier;
  int64 value;

  bool operator==(const SymbolicBinding& other) const {
    return identifier == other.identifier && value == other.value;
  }
  bool operator!=(const SymbolicBinding& other) const {
    return !(*this == other);
  }
};

// Sequence of symbolic bindings in stable order (wraps the backing vector
// storage to make it immutable, hashable, among other utility functions).
//
// Stable order is that bindings are sorted as (identifier, value) tuples.
class SymbolicBindings {
 public:
  SymbolicBindings() = default;

  explicit SymbolicBindings(
      absl::Span<std::pair<std::string, int64> const> items) {
    for (const auto& item : items) {
      bindings_.push_back(SymbolicBinding{item.first, item.second});
    }
    Sort();
  }
  explicit SymbolicBindings(
      const absl::flat_hash_map<std::string, int64>& mapping) {
    for (const auto& item : mapping) {
      bindings_.push_back(SymbolicBinding{item.first, item.second});
    }
    Sort();
  }

  template <typename H>
  friend H AbslHashValue(H h, const SymbolicBindings& self) {
    for (const SymbolicBinding& sb : self.bindings_) {
      h = H::combine(std::move(h), sb.identifier, sb.value);
    }
    return h;
  }

  bool operator==(const SymbolicBindings& other) const;
  bool operator!=(const SymbolicBindings& other) const;

  bool empty() const { return bindings_.empty(); }

  // Returns a string representation of the contained symbolic bindings suitable
  // for debugging.
  std::string ToString() const;

  absl::flat_hash_map<std::string, int64> ToMap() const {
    absl::flat_hash_map<std::string, int64> map;
    for (const SymbolicBinding& binding : bindings_) {
      map.insert({binding.identifier, binding.value});
    }
    return map;
  }

  int64 size() const { return bindings_.size(); }
  const std::vector<SymbolicBinding>& bindings() const { return bindings_; }

 private:
  void Sort() {
    std::sort(
        bindings_.begin(), bindings_.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.identifier < rhs.identifier ||
                 (lhs.identifier == rhs.identifier && lhs.value < rhs.value);
        });
  }

  std::vector<SymbolicBinding> bindings_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const SymbolicBindings& symbolic_bindings) {
  os << symbolic_bindings.ToString();
  return os;
}

}  // namespace xls::dslx

#endif  // XLS_DSLX_SYMBOLIC_BINDINGS_H_
