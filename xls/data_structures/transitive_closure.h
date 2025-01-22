// Copyright 2022 The XLS Authors
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

#ifndef XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
#define XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"

namespace xls {

namespace internal {
// Compute the transitive closure of a relation.
template <typename Relation>
void TransitiveClosure(Relation relation) {
  // Warshall's algorithm; https://cs.winona.edu/lin/cs440/ch08-2.pdf
  // modified in the typical way to avoid unnecessary copies of the expanded
  // relation. It's safe to update the relation as we go, since at each stage k,
  // i relates to k via nodes < k iff i relates to k via nodes <= k, and
  // similarly for k relating to j.
  //
  // Callbacks are used to avoid having to deal with the complication of
  // enumerating the values in a consistent way with both the array and map
  // formulations.
  relation.ForEachKeyValue([&](const auto& k, const auto& from_k) {
    relation.ForEachKeyValue([&](const auto& i, auto& from_i) {
      if (i == k) {
        // Updating would be a no-op, so skip it.
        return;
      }
      if (relation.Contains(from_i, k)) {
        // i relates to k (via nodes < k), so:
        //   for any j where k relates to j (via nodes < k),
        //     i relates to j (via nodes <= k).
        relation.UnionInPlace(from_i, from_k);
      }
    });
  });
}

template <typename V>
class HashRelation {
 public:
  explicit HashRelation(
      absl::flat_hash_map<V, absl::flat_hash_set<V>>& relation)
      : relation_(relation) {}
  template <typename F>
  void ForEachKeyValue(F f) const {
    for (auto& [j, from_j] : relation_) {
      f(j, from_j);
    }
  }
  bool Contains(const absl::flat_hash_set<V>& vs, const V& v) const {
    return vs.contains(v);
  }
  void UnionInPlace(absl::flat_hash_set<V>& i,
                    const absl::flat_hash_set<V>& k) const {
    i.insert(k.begin(), k.end());
  }

 private:
  absl::flat_hash_map<V, absl::flat_hash_set<V>>& relation_;
};

class DenseIdRelation {
 public:
  explicit DenseIdRelation(absl::Span<InlineBitmap> relation)
      : relation_(relation) {}
  template <typename F>
  void ForEachKeyValue(F f) const {
    for (int64_t i = 0; i < relation_.size(); ++i) {
      f(i, relation_[i]);
    }
  }
  bool Contains(const InlineBitmap& vs, int64_t v) const { return vs.Get(v); }
  void UnionInPlace(InlineBitmap& i, const InlineBitmap& k) const {
    i.Union(k);
  }

 private:
  absl::Span<InlineBitmap> relation_;
};

}  // namespace internal

template <typename V>
using HashRelation = absl::flat_hash_map<V, absl::flat_hash_set<V>>;

// Compute the transitive closure of a relation represented as an explicit
// adjacency list.
template <typename V>
HashRelation<V> TransitiveClosure(HashRelation<V> v) {
  internal::TransitiveClosure(internal::HashRelation<V>(v));
  return v;
}

// TODO(allight): Using a more efficient bitmap format like croaring might give
// a speedup here.
using DenseIdRelation = absl::Span<InlineBitmap>;
// Compute the transitive closure of a relation represented as a boolean
// adjacency matrix.
inline DenseIdRelation TransitiveClosure(DenseIdRelation v) {
  internal::TransitiveClosure(internal::DenseIdRelation(v));
  return v;
}

// Compute the transitive closure of a relation represented as a boolean
// adjacency matrix.
inline std::vector<InlineBitmap> TransitiveClosure(
    std::vector<InlineBitmap> v) {
  TransitiveClosure(absl::MakeSpan(v));
  return v;
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_TRANSITIVE_CLOSURE_H_
