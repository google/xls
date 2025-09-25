// Copyright 2025 The XLS Authors
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

#ifndef XLS_IR_NODE_MAP_H_
#define XLS_IR_NODE_MAP_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <list>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/hash_container_defaults.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xls/common/pointer_utils.h"
#include "xls/ir/block.h"     // IWYU pragma: keep
#include "xls/ir/function.h"  // IWYU pragma: keep
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"  // IWYU pragma: keep

namespace xls {

using ForceAllowNodeMap = std::false_type;
namespace internal {
template <typename kIsLikelySlower>
struct ErrorOnSlower {
  static_assert(
      !kIsLikelySlower::value,
      "NodeMap is likely slower than absl::flat_hash_map for this Value type "
      "because it is trivially copyable. To override declare node map as "
      "NodeMap<{ValueT}, ForceAllowNodeMap>. Care should be taken to validate "
      "that this is actually a performance win however.");
};
};  // namespace internal

// `xls::NodeMap` is a map-like interface for holding mappings from `xls::Node*`
// to `ValueT`. It is designed to be a partial drop-in replacement for
// `absl::flat_hash_map<Node*, ValueT>` but with better performance in XLS
// workloads.
//
// `NodeMap` achieves this performance by storing `ValueT` logically within the
// `Node` object itself as 'user-data'. This avoids hashing `Node*` and reduces
// cache misses compared to `absl::flat_hash_map`. A read of a value requires
// only 4 pointer reads.
//
// Notable Differences from `absl::flat_hash_map`:
//
// * All operations inherently perform pointer reads on any Node* typed values
//   in key positions. This means that attempting to use deallocated Node*s as
//   keys **in any way** (including just calling contains, etc) is UB.
// * The node-map has pointer stability of its values as well as iterator
//   stability (except for iterators pointing to an entry which is removed
//   either by a call to erase or by removing the node which is the entries
//   key).
// * `reserve()` is not available as `NodeMap` does not require upfront storage
//   allocation in the same way as `absl::flat_hash_map`.
// * Iteration order is from most-recently inserted to least-recently inserted.
// * All keys in a `NodeMap` must come from the same `Package`.
// * If a `Node` is deleted from its function/package, any associated data in
//   any `NodeMap` is deallocated and it is removed from the map.
// * Each node with data in any `NodeMap` has an internal vector to hold
//   user-data for all live maps. This extra space is not cleaned up until
//   package destruction, based on the assumption that only a small number of
//   maps will be simultaneously live.
//
// WARNING: This map is not thread safe. Also destruction of a node which has a
// value mapped to it is a modification of the map and needs to be taken into
// account if using this map in a multi-threaded context.
//
// NB This does a lot of very unsafe stuff internally to store the data using
// node user-data.
template <typename ValueT,
          typename kIsLikelySlower = std::is_trivially_copyable<ValueT>>
class NodeMap : public internal::ErrorOnSlower<kIsLikelySlower> {
 private:
  // Intrusive list node to hold the actual data allowing us to iterate.
  struct DataHolder {
    template <typename... Args>
    DataHolder(Node* n, Args&&... args)
        : value(std::piecewise_construct, std::forward_as_tuple(n),
                std::forward_as_tuple(std::forward<Args>(args)...)),
          iter() {}

    ~DataHolder() {
      // Remove itself from the list on deletion.
      if (configured_list) {
        configured_list->erase(iter);
      }
    }

    std::pair<Node* const, ValueT> value;
    // Intrusive list node to allow for iteration that's somewhat fast.
    std::list<DataHolder*>::iterator iter;
    std::list<DataHolder*>* configured_list = nullptr;
  };

  class ConstIterator;
  class Iterator {
   public:
    using difference_type = ptrdiff_t;
    using value_type = std::pair<Node* const, ValueT>;
    using reference = value_type&;
    using pointer = value_type*;
    using element_type = value_type;
    using const_reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;

    Iterator() : iter_() {}
    Iterator(std::list<DataHolder*>::iterator iter) : iter_(iter) {}
    Iterator(const Iterator& other) : iter_(other.iter_) {}
    Iterator& operator=(const Iterator& other) {
      iter_ = other.iter_;
      return *this;
    }
    Iterator& operator++() {
      ++iter_;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    reference operator*() const { return (*iter_)->value; }
    pointer get() const { return &**this; }
    pointer operator->() const { return &(*iter_)->value; }
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.iter_ == b.iter_;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return !(a == b);
    }

   private:
    std::list<DataHolder*>::iterator iter_;
    friend class ConstIterator;
  };
  class ConstIterator {
   public:
    using difference_type = ptrdiff_t;
    using value_type = std::pair<Node* const, ValueT>;
    using reference = const value_type&;
    using pointer = const value_type*;
    using element_type = value_type;
    using const_reference = const value_type&;
    using iterator_category = std::forward_iterator_tag;
    ConstIterator() : iter_() {}
    ConstIterator(std::list<DataHolder*>::const_iterator iter) : iter_(iter) {}
    ConstIterator(std::list<DataHolder*>::iterator iter) : iter_(iter) {}
    ConstIterator(const ConstIterator& other) : iter_(other.iter_) {}
    ConstIterator(const Iterator& other) : iter_(other.iter_) {}
    ConstIterator& operator=(const ConstIterator& other) {
      iter_ = other.iter_;
      return *this;
    }
    ConstIterator& operator++() {
      ++iter_;
      return *this;
    }
    ConstIterator operator++(int) {
      ConstIterator tmp = *this;
      ++(*this);
      return tmp;
    }
    reference operator*() const { return (*iter_)->value; }
    pointer get() const { return &**this; }
    pointer operator->() const { return &(*iter_)->value; }
    friend bool operator==(const ConstIterator& a, const ConstIterator& b) {
      return a.iter_ == b.iter_;
    }
    friend bool operator!=(const ConstIterator& a, const ConstIterator& b) {
      return !(a == b);
    }

   private:
    std::list<DataHolder*>::const_iterator iter_;
  };

 public:
  static constexpr bool kIsNodeMap = true;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using key_equal = absl::DefaultHashContainerEq<Node*>;
  using value_type = std::pair<Node*, ValueT>;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = Iterator;
  using const_iterator = ConstIterator;

  // Creates an empty `NodeMap` associated with the given `Package`.
  explicit NodeMap(Package* pkg)
      : pkg_(pkg),
        id_(pkg->AllocateNodeUserDataId()),
        values_(std::make_unique<std::list<DataHolder*>>()) {}
  // Creates an empty `NodeMap` which will become associated with a `Package`
  // upon first insertion.
  NodeMap()
      : pkg_(nullptr),
        id_(-1),
        values_(std::make_unique<std::list<DataHolder*>>()) {}
  // Releases all data held by this map and informs the package that this
  // map's user-data ID can be reused.
  ~NodeMap() {
    // Release all the data.
    clear();
  }
  // Copy constructor.
  NodeMap(const NodeMap& other)
      : pkg_(other.pkg_),
        id_(pkg_ != nullptr ? pkg_->AllocateNodeUserDataId() : -1),
        values_(std::make_unique<std::list<DataHolder*>>()) {
    for (auto& [k, v] : other) {
      this->insert(k, v);
    }
  }
  // Copy assignment.
  NodeMap& operator=(const NodeMap& other) {
    if (HasPackage()) {
      clear();
    } else {
      pkg_ = other.pkg_;
      id_ = pkg_ != nullptr ? pkg_->AllocateNodeUserDataId() : -1;
    }
    for (auto& [k, v] : other) {
      this->insert(k, v);
    }
    return *this;
  }
  // Move constructor.
  NodeMap(NodeMap&& other) {
    pkg_ = other.pkg_;
    id_ = other.id_;
    values_ = std::move(other.values_);
    other.id_ = -1;
    other.pkg_ = nullptr;
  }
  // Move assignment.
  NodeMap& operator=(NodeMap&& other) {
    pkg_ = other.pkg_;
    id_ = other.id_;
    values_ = std::move(other.values_);
    other.pkg_ = nullptr;
    other.id_ = -1;
    return *this;
  }
  // Range constructor.
  template <typename It>
  NodeMap(It first, It last)
      : pkg_(nullptr),
        id_(-1),
        values_(std::make_unique<std::list<DataHolder*>>()) {
    for (auto it = first; it != last; ++it) {
      insert(it->first, it->second);
    }
  }

  // Range constructor with explicit package.
  template <typename It>
  NodeMap(Package* pkg, It first, It last)
      : pkg_(pkg),
        id_(pkg->AllocateNodeUserDataId()),
        values_(std::make_unique<std::list<DataHolder*>>()) {
    for (auto it = first; it != last; ++it) {
      insert(it->first, it->second);
    }
  }
  // Constructs a `NodeMap` from an `absl::flat_hash_map`.
  NodeMap(const absl::flat_hash_map<Node*, ValueT>& other)
      : pkg_(nullptr),
        id_(-1),
        values_(std::make_unique<std::list<DataHolder*>>()) {
    for (auto& [k, v] : other) {
      this->insert(k, v);
    }
  }
  // Assigns contents from an `absl::flat_hash_map`.
  NodeMap& operator=(const absl::flat_hash_map<Node*, ValueT>& other) {
    clear();
    for (auto& [k, v] : other) {
      this->insert(k, v);
    }
  }
  // Initializer list constructor.
  NodeMap(std::initializer_list<value_type> init)
      : pkg_(nullptr),
        id_(-1),
        values_(std::make_unique<std::list<DataHolder*>>()) {
    for (const auto& pair : init) {
      insert(pair.first, pair.second);
    }
  }

  // Returns true if the map has an associated package.
  bool HasPackage() const { return pkg_ != nullptr; }

  // Returns true if the map contains no elements.
  bool empty() const {
    CheckValidId();
    return values_->empty();
  }
  // Returns the number of elements in the map.
  size_t size() const {
    CheckValidId();
    return values_->size();
  }

  // Returns true if the map contains an element with key `n`.
  bool contains(Node* n) const {
    CheckValidId(n);
    if (!HasPackage()) {
      return false;
    }
    return n->HasUserData(id_);
  }
  // Returns 1 if the map contains an element with key `n`, 0 otherwise.
  size_t count(Node* n) const {
    CheckValidId(n);
    if (!HasPackage()) {
      return 0;
    }
    return n->HasUserData(id_) ? 1 : 0;
  }
  // Returns a reference to the value mapped to key `n`. If no such element
  // exists, this function CHECK-fails.
  ValueT& at(Node* n) {
    EnsureValidId(n);
    CHECK(contains(n)) << "Nothing was ever set for " << n;
    return reinterpret_cast<DataHolder*>(n->GetUserData(id_))->value.second;
  }
  // Returns a reference to the value mapped to key `n`, inserting a
  // default-constructed value if `n` is not already present.
  ValueT& operator[](Node* n) {
    EnsureValidId(n);
    if (contains(n)) {
      return at(n);
    }
    auto holder = std::make_unique<DataHolder>(n);
    DataHolder* holder_ptr = holder.get();
    n->SetUserData(id_, EraseType(std::move(holder)));
    values_->push_front(holder_ptr);
    holder_ptr->iter = values_->begin();
    holder_ptr->configured_list = values_.get();
    return holder_ptr->value.second;
  }
  // Returns a const reference to the value mapped to key `n`. If no such
  // element exists, this function CHECK-fails.
  const ValueT& operator[](Node* n) const { return at(n); }
  // Returns a const reference to the value mapped to key `n`. If no such
  // element exists, this function CHECK-fails.
  const ValueT& at(Node* n) const {
    CheckValidId(n);
    CHECK(contains(n)) << "Nothing was ever set for " << n;
    return reinterpret_cast<DataHolder*>(n->GetUserData(id_))->value.second;
  }

  // Erases the element with key `n` if it exists.
  void erase(Node* n) {
    CheckValidId(n);
    if (contains(n)) {
      std::optional<TypeErasedUniquePtr> data = n->TakeUserData(id_);
      DCHECK(data);
      // The DataHolder could remove itself from the list when its destructor
      // runs. It seems better to just be explicit
      DataHolder* holder = reinterpret_cast<DataHolder*>(data->get());
      DCHECK(holder->configured_list == values_.get());
      holder->configured_list = nullptr;
      values_->erase(holder->iter);
    }
  }

  // Erases the element pointed to by `it`. Returns an iterator to the
  // element following the erased element.
  const_iterator erase(const_iterator it) {
    CheckValidId(it->first);
    auto res = it;
    ++res;
    erase(it->first);
    return res;
  }

  // Erases the element pointed to by `it`. Returns an iterator to the
  // element following the erased element.
  iterator erase(iterator it) {
    CheckValidId(it->first);
    auto res = it;
    ++res;
    erase(it->first);
    return res;
  }

  // Removes all elements from the map.
  ABSL_ATTRIBUTE_REINITIALIZES void clear() {
    CheckValidId();
    if (pkg_ == nullptr) {
      return;
    }
    for (DataHolder* v : *values_) {
      std::optional<TypeErasedUniquePtr> data =
          v->value.first->TakeUserData(id_);
      // We can't remove the current iterator position.
      reinterpret_cast<DataHolder*>(data->get())->configured_list = nullptr;
    }
    values_->clear();
    // Release the id.
    pkg_->ReleaseNodeUserDataId(id_);
  }
  // Swaps the contents of this map with `other`.
  void swap(NodeMap& other) {
    std::swap(pkg_, other.pkg_);
    std::swap(id_, other.id_);
    values_.swap(other.values_);
  }

  // Returns an iterator to the first element in the map.
  iterator begin() {
    CheckValidId();
    return Iterator(values_->begin());
  }
  // Returns an iterator to the element following the last element in the map.
  iterator end() {
    CheckValidId();
    return Iterator(values_->end());
  }
  // Returns a const iterator to the first element in the map.
  const_iterator cbegin() const {
    CheckValidId();
    return ConstIterator(values_->cbegin());
  }
  // Returns a const iterator to the element following the last element in the
  // map.
  const_iterator cend() const {
    CheckValidId();
    return ConstIterator(values_->cend());
  }
  // Returns a const iterator to the first element in the map.
  const_iterator begin() const {
    CheckValidId();
    return cbegin();
  }
  // Returns a const iterator to the element following the last element in the
  // map.
  const_iterator end() const {
    CheckValidId();
    return cend();
  }

  // Finds an element with key `n`.
  // Returns an iterator to the element if found, or `end()` otherwise.
  iterator find(Node* n) {
    CheckValidId(n);
    if (!contains(n)) {
      return end();
    }
    return Iterator(reinterpret_cast<DataHolder*>(n->GetUserData(id_))->iter);
  }
  // Finds an element with key `n`.
  // Returns a const iterator to the element if found, or `end()` otherwise.
  const_iterator find(Node* n) const {
    CheckValidId(n);
    if (!contains(n)) {
      return end();
    }
    return ConstIterator(
        reinterpret_cast<DataHolder*>(n->GetUserData(id_))->iter);
  }

  // Inserts a key-value pair into the map if the key does not already exist.
  // Returns a pair consisting of an iterator to the inserted element (or to
  // the element that prevented the insertion) and a bool denoting whether
  // the insertion took place.
  std::pair<iterator, bool> insert(Node* n, ValueT value) {
    EnsureValidId(n);
    if (contains(n)) {
      return std::make_pair(find(n), false);
    }
    auto holder = std::make_unique<DataHolder>(n, std::move(value));
    DataHolder* holder_ptr = holder.get();
    n->SetUserData(id_, EraseType(std::move(holder)));
    values_->push_front(holder_ptr);
    holder_ptr->iter = values_->begin();
    holder_ptr->configured_list = values_.get();
    return std::make_pair(begin(), true);
  }

  // Inserts a key-value pair into the map or assigns to the existing value if
  // the key already exists.
  // Returns a pair consisting of an iterator to the inserted element (or to
  // the element that prevented the insertion) and a bool denoting whether
  // the insertion took place.
  std::pair<iterator, bool> insert_or_assign(Node* n, ValueT value) {
    EnsureValidId(n);
    if (contains(n)) {
      Iterator f = find(n);
      f->second = std::move(value);
      return std::make_pair(f, false);
    }
    auto holder = std::make_unique<DataHolder>(n, std::move(value));
    DataHolder* holder_ptr = holder.get();
    n->SetUserData(id_, EraseType(std::move(holder)));
    values_->push_front(holder_ptr);
    holder_ptr->iter = values_->begin();
    holder_ptr->configured_list = values_.get();
    return std::make_pair(begin(), true);
  }

  // Inserts an element constructed in-place if the key does not already exist.
  // Note: Unlike `try_emplace`, `emplace` may construct `ValueT` from `args`
  // even if insertion does not occur.
  // Returns a pair consisting of an iterator to the inserted element (or to
  // the element that prevented the insertion) and a bool denoting whether
  // the insertion took place.
  template <typename... Args>
  std::pair<iterator, bool> emplace(Node* n, Args&&... args) {
    EnsureValidId(n);
    // If key already exists, construct elements but don't insert.
    // This is to match std::map::emplace behavior where element construction
    // might happen before check for duplication and value is discarded.
    auto holder = std::make_unique<DataHolder>(n, std::forward<Args>(args)...);
    if (contains(n)) {
      return std::make_pair(find(n), false);
    }
    DataHolder* holder_ptr = holder.get();
    n->SetUserData(id_, EraseType(std::move(holder)));
    values_->push_front(holder_ptr);
    holder_ptr->iter = values_->begin();
    holder_ptr->configured_list = values_.get();
    return std::make_pair(begin(), true);
  }

  // Inserts an element constructed in-place if the key does not already exist.
  // If the key already exists, no element is constructed.
  // Returns a pair consisting of an iterator to the inserted element (or to
  // the element that prevented the insertion) and a bool denoting whether
  // the insertion took place.
  template <typename... Args>
  std::pair<iterator, bool> try_emplace(Node* n, Args&&... args) {
    EnsureValidId(n);
    if (contains(n)) {
      return std::make_pair(find(n), false);
    }
    auto holder = std::make_unique<DataHolder>(n, std::forward<Args>(args)...);
    DataHolder* holder_ptr = holder.get();
    n->SetUserData(id_, EraseType(std::move(holder)));
    values_->push_front(holder_ptr);
    holder_ptr->iter = values_->begin();
    holder_ptr->configured_list = values_.get();
    return std::make_pair(begin(), true);
  }

  friend bool operator==(const Iterator& a, const ConstIterator& b) {
    return a.iter_ == b.iter_;
  }
  friend bool operator==(const ConstIterator& a, const Iterator& b) {
    return a.iter_ == b.iter_;
  }
  friend bool operator!=(const Iterator& a, const ConstIterator& b) {
    return a.iter_ != b.iter_;
  }
  friend bool operator!=(const ConstIterator& a, const Iterator& b) {
    return a.iter_ != b.iter_;
  }

 private:
  void CheckValid() const {
#ifdef DEBUG
    CHECK(HasPackage());
    CheckValidId();
#endif
  }
  void CheckValidId() const {
#ifdef DEBUG
    if (pkg_ != nullptr) {
      CHECK(pkg_->IsLiveUserDataId(id_)) << id_;
    }
#endif
  }

  // Check that this map has a valid id and correct package.
  void CheckValidId(Node* n) const {
#ifdef DEBUG
    CheckValidId();
    if (HasPackage()) {
      CHECK_EQ(n->package(), pkg_)
          << "Incorrect package for " << n << " got " << n->package()->name()
          << " expected " << pkg_->name();
    }
#endif
  }
  // Force this map to have a user-data id if it doesn't already
  void EnsureValidId(Node* n) {
    if (!HasPackage()) {
      pkg_ = n->package();
      DCHECK(pkg_ != nullptr)
          << "Cannot add a node " << n << " without a package.";
      id_ = pkg_->AllocateNodeUserDataId();
    }
    CheckValidId(n);
  }

  Package* pkg_;
  int64_t id_;
  std::unique_ptr<std::list<DataHolder*>> values_;
};

}  // namespace xls

#endif  // XLS_IR_NODE_MAP_H_
