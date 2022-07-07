// Copyright 2021 The XLS Authors
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

#ifndef XLS_DATA_STRUCTURES_UNION_FIND_MAP_H_
#define XLS_DATA_STRUCTURES_UNION_FIND_MAP_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"

namespace xls {

// A union-find data structure that acts like a hashmap from unionable keys to
// values.
//
// Size is limited to roughly 2^32 elements, so that the offsets can be 32 bits
// rather than 64.
template <typename K, typename V>
class UnionFindMap {
 public:
  UnionFindMap() = default;

  // Insert the given key/value pair.
  //
  // If a value already exists at that key, the value is replaced.
  // `V` must be copyable.
  void Insert(const K& key, const V& value) {
    this->Insert(key, value,
                 [](const V& old_val, const V& new_val) { return new_val; });
  }

  // Insert the given key/value pair.
  //
  // The `merge` parameter is used to combine the inserted value with an
  // existing value, if there is one, and should have type compatible with
  // `std::function<V(const V&, const V&)>`. The first argument is the old
  // value, and the second argument is the inserted value.
  template <typename F>
  void Insert(const K& key, const V& value, F merge) {
    if (key_to_index_.contains(key)) {
      uint32_t index = FindRoot(GetIndex(key).value());
      values_.at(index) = merge(values_.at(index), value);
      return;
    }
    uint32_t index = nodes_.size();
    nodes_.push_back({index, 1});
    key_to_index_.try_emplace(key, index);
    keys_.push_back(key);
    values_.push_back(value);
  }

  // Given a key, returns the representative element in that key's equivalence
  // class, along with the associated value. Returns `absl::nullopt` if the
  // given key has never been inserted.
  std::optional<std::pair<K, V&>> Find(const K& key) {
    if (auto index = GetIndex(key)) {
      uint32_t found = FindRoot(index.value());
      return {{keys_.at(found), values_.at(found)}};
    }
    return absl::nullopt;
  }

  // Union together the equivalence classes of two keys.
  //
  // Returns true if both keys were previously inserted, and false otherwise.
  // The state of the data structure is unchanged if false was returned.
  //
  // The `merge` parameter is used to combine the associated values, and should
  // have type compatible with `std::function<V(const V&, const V&)>`.
  //
  // Note that if two keys are merged, the merged value is stored twice in this
  // data structure, so `V` must be copyable and should be small if space is a
  // concern.
  template <typename F>
  bool Union(const K& x, const K& y, F merge) {
    std::optional<uint32_t> x_index = GetIndex(x);
    if (!x_index.has_value()) {
      return false;
    }
    std::optional<uint32_t> y_index = GetIndex(y);
    if (!y_index.has_value()) {
      return false;
    }
    uint32_t x_root = FindRoot(x_index.value());
    uint32_t y_root = FindRoot(y_index.value());

    if (x_root != y_root) {
      uint32_t x_root_size = nodes_.at(x_root).size;
      uint32_t y_root_size = nodes_.at(y_root).size;
      if (x_root_size < y_root_size) {
        std::swap(x_root, y_root);
      }
      V new_value = merge(values_.at(x_root), values_.at(y_root));
      values_.at(x_root) = new_value;
      values_.at(y_root) = new_value;
      nodes_.at(y_root).parent = x_root;
      nodes_.at(x_root).size = x_root_size + y_root_size;
    }

    return true;
  }

  // Returns true if the given key was ever inserted.
  bool Contains(const K& key) const { return key_to_index_.contains(key); }

  // Returns every key ever inserted, with unspecified ordering.
  const std::vector<K>& GetKeys() const { return keys_; }

  // Returns the smallest element of every equivalence class.
  absl::flat_hash_set<K> GetRepresentatives() {
    absl::flat_hash_set<K> result;
    for (const K& key : keys_) {
      result.insert(Find(key)->first);
    }
    return result;
  }

 private:
  // The `parent` field should be a valid index into `nodes_` et al.
  // A root node will have itself as its parent.
  struct Node {
    uint32_t parent;
    uint32_t size;
  };

  std::optional<uint32_t> GetIndex(const K& key) const {
    if (!key_to_index_.contains(key)) {
      return absl::nullopt;
    }
    return key_to_index_.at(key);
  }

  // Uses the path-halving algorithm.
  uint32_t FindRoot(uint32_t index) {
    uint32_t x = index;
    while (true) {
      uint32_t p = nodes_.at(x).parent;
      uint32_t pp = nodes_.at(p).parent;
      if (p == x) {
        return x;
      }
      nodes_.at(p).parent = pp;
      x = p;
    }
  }

  // Allows mapping a key to an index. The image of the mapping should be equal
  // to the set {0, ..., n - 1} where n is the number of elements inserted.
  absl::flat_hash_map<K, uint32_t> key_to_index_;

  // Maps an index to a key. Should have size equal to the number of indices.
  std::vector<K> keys_;

  // Maps an index to a value. Should have size equal to the number of indices.
  std::vector<V> values_;

  // Maps an index to a node. Should have size equal to the number of indices.
  std::vector<Node> nodes_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_UNION_FIND_MAP_H_
