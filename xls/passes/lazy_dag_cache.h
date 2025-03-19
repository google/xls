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

#ifndef XLS_PASSES_LAZY_DAG_CACHE_H_
#define XLS_PASSES_LAZY_DAG_CACHE_H_

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

// An implementation of an invalidating/"re-validating" cache for any analysis
// on a DAG that can be written in terms of only local information about a node
// (represented by a `Key`) and its inputs.
//
// This class implements a cache with a simple state machine; each key has
// either no recorded value (kUnknown), a value that may be out-of-date
// (kUnverified), a value that is valid if all inputs prove to be up-to-date
// (kInputsUnverified), a value that is known to be current & correct
// (kKnown), or information with a forced value (kForced).
//
// When a `key` is queried, we query for the values for all of its inputs, and
// then re-compute the value for `key` if absent or unverified. If `key` was in
// state kUnverified & this does change its associated value, we mark any direct
// users that were in state kInputsUnverified as kUnverified, since their inputs
// have changed.
//
// Any node that is in the kForced state will remain in the same state
// regardless of other changes to the graph. Changing the value a node is Forced
// to will invalidate all its users and force a recalculation of downstream
// values.
//
// NOTE: If `key` is in state kInputsUnverified after we queried all of its
//       inputs, then their values did not change, so we have verified that
//       `key`'s value is up-to-date without having to recompute it! This is the
//       main advantage of this cache over a more typical invalidating cache.
template <typename Key, typename Value>
class LazyDagCache {
 public:
  // A pure-virtual interface for providing information about a DAG.
  class DagProvider {
   public:
    virtual ~DagProvider() = default;

    virtual std::string GetName(const Key& key) const = 0;
    virtual absl::Span<const Key> GetInputs(const Key& key) const = 0;
    virtual absl::Span<const Key> GetUsers(const Key& key) const = 0;

    virtual absl::StatusOr<Value> ComputeValue(
        const Key& key, absl::Span<const Value* const> input_values) const = 0;
  };

  LazyDagCache<Key, Value>(DagProvider* provider) : provider_(provider) {}

  LazyDagCache<Key, Value>(DagProvider* provider,
                           const LazyDagCache<Key, Value>& other)
      : provider_(provider) {
    cache_ = other.cache_;
  }

  LazyDagCache<Key, Value>(DagProvider* provider,
                           LazyDagCache<Key, Value>&& other)
      : provider_(provider) {
    cache_ = std::move(other.cache_);
  }

  LazyDagCache<Key, Value>(const LazyDagCache<Key, Value>& other) = delete;
  LazyDagCache<Key, Value>& operator=(const LazyDagCache<Key, Value>& other) =
      delete;
  LazyDagCache<Key, Value>(LazyDagCache<Key, Value>&& other) = delete;
  LazyDagCache<Key, Value>& operator=(LazyDagCache<Key, Value>&& other) =
      delete;

  // Erase all knowledge of the values of all keys.
  void Clear() { cache_.clear(); }
  // Erase all knowledge of the value of all keys except for 'Forced' values.
  void ClearNonForced() {
    absl::erase_if(cache_, [](const auto& v) {
      return v.second.state != CacheState::kForced;
    });
  }

  // Entirely remove knowledge of this key. This includes erasing any Forced
  // data.
  void Forget(const Key& key) {
    cache_.erase(key);
    for (const Key& user : provider_->GetUsers(key)) {
      MarkInputsUnverified(user);
    }
  }

  // Set the key as having the immutable, authoritative 'value'.
  //
  // *This is a dangerous operation and should be used with care.* It tells the
  // cache to never call the ComputeValue callback and to instead consider
  // 'value' to be associated with 'key' now and forever. This knowledge may
  // only be removed by calling 'Forget' or 'Clear'.
  void SetForced(const Key& key, std::unique_ptr<Value> value) {
    cache_.insert_or_assign(key, CacheEntry{.state = CacheState::kForced,
                                            .value = std::move(value)});
    MarkUsersUnverified(key);
  }

  // Set the key as having the immutable, authoritative 'value'.
  //
  // *This is a dangerous operation and should be used with care.* It tells the
  // cache to never call the ComputeValue callback and to instead consider
  // 'value' to be associated with 'key' now and forever. This knowledge may
  // only be removed by calling 'Forget' or 'Clear'.
  void SetForced(const Key& key, Value value) {
    SetForced(key, std::make_unique<Value>(std::move(value)));
  }

  void AddUnverified(const Key& key, Value value) {
    cache_.insert_or_assign(
        key, CacheEntry{.state = CacheState::kUnverified,
                        .value = std::make_unique<Value>(std::move(value))});
  }
  void AddUnverified(const Key& key, std::unique_ptr<Value> value) {
    cache_.insert_or_assign(key, CacheEntry{.state = CacheState::kUnverified,
                                            .value = std::move(value)});
  }

  // Request recomputation of any users of this key.
  //
  // Mark as full unverified to force recomputation.
  void MarkUsersUnverified(const Key& key) {
    for (const Key& user : provider_->GetUsers(key)) {
      MarkUnverified(user);
    }
  }

  void MarkUnverified(const Key& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return;
    }
    auto& [state, _] = it->second;
    if (state == CacheState::kUnverified) {
      return;
    }
    if (state == CacheState::kForced) {
      VLOG(1) << "Mark unverified called on forced entry "
              << provider_->GetName(key);
      return;
    }
    state = CacheState::kUnverified;
    for (const Key& user : provider_->GetUsers(key)) {
      if (GetCacheState(user) == CacheState::kKnown) {
        MarkInputsUnverified(user);
      }
    }
  }

  void MarkInputsUnverified(const Key& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return;
    }
    auto& [state, _] = it->second;
    if (state != CacheState::kKnown) {
      return;
    }
    state = CacheState::kInputsUnverified;
    absl::Span<const Key> users = provider_->GetUsers(key);
    std::vector<Key> worklist(users.begin(), users.end());
    while (!worklist.empty()) {
      Key descendant = std::move(worklist.back());
      worklist.pop_back();
      auto descendant_it = cache_.find(descendant);
      if (descendant_it == cache_.end()) {
        continue;
      }
      auto& [descendant_state, _] = descendant_it->second;
      if (descendant_state != CacheState::kKnown) {
        continue;
      }
      descendant_state = CacheState::kInputsUnverified;
      worklist.insert(worklist.end(), descendant->users().begin(),
                      descendant->users().end());
    }
  }

  Value* QueryValue(const Key& key) {
    // If `key` is already known, return a pointer to the cached value.
    if (auto it = cache_.find(key);
        it != cache_.end() && (it->second.state == CacheState::kKnown ||
                               it->second.state == CacheState::kForced)) {
      return it->second.value.get();
    }

    // Retrieve the values for all of this node's inputs.
    std::vector<const Value*> input_values;
    for (const Key& input : provider_->GetInputs(key)) {
      input_values.push_back(QueryValue(input));
    }

    // Find the CacheEntry for this node; if not present, this will insert a
    // default-constructed CacheEntry, with state kUnknown and empty value,
    // which we will populate below.
    auto& [state, cached_value] = cache_[key];

    // If this node was previously in state kInputsUnverified and any input
    // changed, QueryValue on that input would have automatically downgraded
    // this node to kUnverified. Therefore, if we're still in state
    // kInputsUnverified, the stored value is still valid; we can skip
    // recomputation.
    if (state == CacheState::kInputsUnverified) {
      state = CacheState::kKnown;
      return cached_value.get();
    }

    absl::StatusOr<Value> new_value =
        provider_->ComputeValue(key, input_values);
    CHECK_OK(new_value);
    if (state == CacheState::kUnverified && *new_value == *cached_value) {
      // The value didn't change; the stored value is still valid.
      state = CacheState::kKnown;
      return cached_value.get();
    }

    CHECK_NE(state, CacheState::kForced);
    state = CacheState::kKnown;
    cached_value = std::make_unique<Value>(*std::move(new_value));

    // Our stored value changed; make sure we downgrade any users that were
    // previously kInputsUnverified to kUnverified.
    for (const Key& user : provider_->GetUsers(key)) {
      if (GetCacheState(user) == CacheState::kInputsUnverified) {
        MarkUnverified(user);
      }
    }
    return cached_value.get();
  }

  // Eagerly computes the values for all nodes in the DAG that do not have known
  // values. This is expensive and should only be used for testing and
  // measurement. `topo_sorted_keys` must be a topological sort of the keys for
  // all nodes in the DAG.
  absl::Status EagerlyPopulate(absl::Span<const Key> topo_sorted_keys) {
    for (const Key& key : topo_sorted_keys) {
      if (GetNonForcedCacheState(key) == CacheState::kKnown) {
        continue;
      }
      std::vector<const Value*> input_values;
      absl::Span<const Key> inputs = provider_->GetInputs(key);
      input_values.reserve(inputs.size());
      for (const Key& input : inputs) {
        const CacheEntryView input_entry = GetCacheEntry(input);
        XLS_RET_CHECK_EQ(input_entry.state, CacheState::kKnown);
        input_values.push_back(input_entry.value);
      }
      XLS_ASSIGN_OR_RETURN(Value value,
                           provider_->ComputeValue(key, input_values));
      cache_.insert_or_assign(
          key, CacheEntry{.state = CacheState::kKnown,
                          .value = std::make_unique<Value>(std::move(value))});
    }
    return absl::OkStatus();
  }

  // Verifies that the cache's current state is correct where expected &
  // consistent regardless. This is an expensive operation, intended for use in
  // tests. `topo_sorted_keys` must be a topological sort of the keys for all
  // nodes in the DAG.
  //
  // Note that Forced values are always considered consistent.
  absl::Status CheckConsistency(absl::Span<const Key> topo_sorted_keys) const;

  enum class CacheState : uint8_t {
    kUnknown,  // NOTE: This state corresponds to a key that's not in the cache,
               //       so is only temporarily stored while we're computing the
               //       value.
    kUnverified,
    kInputsUnverified,
    kKnown,
    // A value has been provided via external information and should be
    // considered authoritatively known.
    //
    // This node can never be put to unverified state.
    //
    // It is possible that the value this key is forced to is one that cannot be
    // arrived at through the normal update sequence.
    kForced,
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CacheState& state) {
    switch (state) {
      case CacheState::kUnknown:
        absl::Format(&sink, "UNKNOWN");
        return;
      case CacheState::kUnverified:
        absl::Format(&sink, "UNVERIFIED");
        return;
      case CacheState::kInputsUnverified:
        absl::Format(&sink, "INPUTS_UNVERIFIED");
        return;
      case CacheState::kKnown:
        absl::Format(&sink, "KNOWN");
        return;
      case CacheState::kForced:
        absl::Format(&sink, "FORCED");
        return;
    }
    LOG(FATAL) << "Unknown CacheState: " << static_cast<int>(state);
  }
  friend std::ostream& operator<<(std::ostream& os, const CacheState& state) {
    return os << absl::StrCat(state);
  }

  CacheState GetCacheState(const Key& key) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return CacheState::kUnknown;
    }
    return it->second.state;
  }
  // Get the cache state with forced values being considered kKnown
  CacheState GetNonForcedCacheState(const Key& key) const {
    CacheState s = GetCacheState(key);
    if (s == CacheState::kForced) {
      return CacheState::kKnown;
    }
    return s;
  }
  const Value* GetCachedValue(const Key& key) const {
    auto it = cache_.find(key);
    if (it == cache_.end() || it->second.value == nullptr) {
      return nullptr;
    }
    return it->second.value.get();
  }

 private:
  DagProvider* const provider_;

  struct CacheEntry {
    CacheState state = CacheState::kUnknown;
    std::unique_ptr<Value> value = nullptr;
  };
  absl::flat_hash_map<Key, CacheEntry> cache_;

  struct CacheEntryView {
    CacheState state = CacheState::kUnknown;
    const Value* value = nullptr;
  };
  CacheEntryView GetCacheEntry(const Key& key) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return {.state = CacheState::kUnknown, .value = nullptr};
    }
    return {.state = it->second.state, .value = it->second.value.get()};
  }
};

template <typename Key, typename Value>
absl::Status LazyDagCache<Key, Value>::CheckConsistency(
    absl::Span<const Key> topo_sorted_keys) const {
  absl::flat_hash_set<Key> correct_values;
  for (const Key& key : topo_sorted_keys) {
    const auto& [state, value] = GetCacheEntry(key);

    if (state == CacheState::kUnknown) {
      XLS_RET_CHECK_EQ(value, nullptr) << "Key " << provider_->GetName(key)
                                       << " is UNKNOWN but has a stored value.";
      continue;
    }
    XLS_RET_CHECK_NE(value, nullptr)
        << "Key " << provider_->GetName(key)
        << " is not UNKNOWN but has no stored value.";

    if (state == CacheState::kKnown) {
      for (const Key& input : provider_->GetInputs(key)) {
        XLS_RET_CHECK_EQ(GetNonForcedCacheState(input), CacheState::kKnown)
            << "Non-KNOWN input for KNOWN key " << provider_->GetName(key)
            << ": " << provider_->GetName(input) << " (input is "
            << GetCacheState(input) << ")";
      }
    }
    if (state == CacheState::kInputsUnverified) {
      for (const Key& input : provider_->GetInputs(key)) {
        XLS_RET_CHECK_NE(GetCacheState(input), CacheState::kUnknown)
            << "UNKNOWN input for INPUTS_UNVERIFIED key "
            << provider_->GetName(key) << ": " << provider_->GetName(input);
      }
    }

    // NB state FORCED & UNVERIFIED has no requirements on its inputs.

    if (absl::c_any_of(provider_->GetInputs(key), [&](const Key& input) {
          return !correct_values.contains(input);
        })) {
      // We can only check the consistency/correctness of `node`'s stored value
      // if all of its inputs have correct stored values.
      continue;
    }

    if (state == CacheState::kForced) {
      // Forced values are definitionally correct.
      correct_values.insert(key);
      continue;
    }

    std::vector<const Value*> input_values;
    absl::Span<const Key> inputs = provider_->GetInputs(key);
    input_values.reserve(inputs.size());
    for (const Key& input : inputs) {
      input_values.push_back(GetCachedValue(input));
    }

    XLS_ASSIGN_OR_RETURN(Value recomputed_value,
                         provider_->ComputeValue(key, input_values));
    if (*value == recomputed_value) {
      correct_values.insert(key);
    }

    // If the key is KNOWN or INPUTS_UNVERIFIED (with all inputs' stored
    // values verified correct), we can check that the stored value is
    // consistent with the value we would compute from its inputs.
    if (state == CacheState::kKnown || state == CacheState::kInputsUnverified) {
      XLS_RET_CHECK(*value == recomputed_value)
          << state << " key " << provider_->GetName(key)
          << " has a stored value that is not consistent with its inputs' "
             "stored values";
    }
  }
  return absl::OkStatus();
}

}  // namespace xls

#endif  // XLS_PASSES_LAZY_DAG_CACHE_H_
