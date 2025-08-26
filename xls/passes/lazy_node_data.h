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

#ifndef XLS_PASSES_LAZY_NODE_DATA_H_
#define XLS_PASSES_LAZY_NODE_DATA_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A base class for node-data cache that can be lazily populated; implements an
// invalidating/"re-validating" cache for any analysis that depends only on the
// transitive inputs to a node.
//
// Derived classes must implement:
// - `ComputeInfo`: called when a node's information needs to be recalculated,
//    and
// - `MergeWithGiven`: called to merge computed information with given
//    information,
//
// This abstracts away the details of the information being stored and the ways
// it's used to answer queries, allowing this class to focus entirely on the
// caching logic.
//
// This class implements a cache with a simple state machine; each node has
// either no recorded information (kUnknown), information that may be
// out-of-date (kUnverified), information that is valid if all inputs prove to
// be up-to-date (kInputsUnverified), information that is known to be current
// & correct (kKnown), or information with a forced value (kForced).
//
// On populating with a FunctionBase, this class listens for change events. When
// a node changes, we mark its information as potentially out-of-date
// (kUnverified), and update all transitive users that are in state kKnown to
// kInputsUnverified. (In some exceptional circumstances, we can guess a node's
// value and mark it kUnverified if this would allow us to keep other nodes
// kInputsUnverified.)
//
// When a node `n` is queried, we query for the information for all of its
// operands, and then re-compute the information for `n` if absent or
// unverified. If `n` was in state kUnverified & this does change its associated
// information, we mark any direct users that were in state kInputsUnverified as
// kUnverified, since their inputs have changed.
//
// Any node that is in the kForced state will remain in the same state
// regardless of other changes to the graph. Changing the value a node is Forced
// to will invalidate all its users.
//
// A node may not have both 'given' data and 'forced' data at the same time.
//
// NOTE: If `n` is in state kInputsUnverified after we queried all of its
//       operands, then their values did not change, so we have verified that
//       `n`'s data is up-to-date without having to recompute it! This is
//       the main advantage of this cache over a more typical invalidating
//       cache.
template <typename CacheValueT>
class LazyNodeData : public ChangeListener,
                     public LazyDagCache<Node*, CacheValueT>::DagProvider {
 private:
  using CacheState = LazyDagCache<Node*, CacheValueT>::CacheState;

 public:
  LazyNodeData<CacheValueT>() : cache_(this) {}
  ~LazyNodeData() override {
    if (f_ != nullptr) {
      f_->UnregisterChangeListener(this);
    }
  }

  LazyNodeData(const LazyNodeData<CacheValueT>& other)
      : f_(other.f_), cache_(this, other.cache_), givens_(other.givens_) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyNodeData& operator=(const LazyNodeData<CacheValueT>& other) {
    if (f_ != other.f_) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
      }
      f_ = other.f_;
    }
    cache_ = LazyDagCache<Node*, CacheValueT>(this, other.cache_);
    givens_ = other.givens_;
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
    return *this;
  }

  LazyNodeData(LazyNodeData<CacheValueT>&& other)
      : f_(other.f_),
        cache_(this, std::move(other.cache_)),
        givens_(std::move(other.givens_)) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyNodeData& operator=(LazyNodeData<CacheValueT>&& other) {
    if (f_ != other.f_) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
      }
      f_ = other.f_;
      if (other.f_ != nullptr) {
        other.f_->UnregisterChangeListener(&other);
        f_->RegisterChangeListener(this);
      }
    }
    cache_ = LazyDagCache<Node*, CacheValueT>(this, std::move(other.cache_));
    givens_ = std::move(other.givens_);
    return *this;
  }

  // Bind the node data to the given function.
  virtual absl::StatusOr<ReachedFixpoint> AttachWithGivens(
      FunctionBase* f, absl::flat_hash_map<Node*, CacheValueT> givens) {
    ReachedFixpoint rf = ReachedFixpoint::Unchanged;
    if (f_ != f) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
        cache_.Clear();
        givens_.clear();
        rf = ReachedFixpoint::Changed;
      }

      if (f != nullptr) {
        f_ = f;
        f_->RegisterChangeListener(this);
        rf = ReachedFixpoint::Changed;
      }
    }

    if (givens_ != givens) {
      absl::flat_hash_map<Node*, CacheValueT> old_givens =
          std::exchange(givens_, std::move(givens));
      for (Node* node : f->nodes()) {
        // If the given information for a node could have changed, we need to
        // mark it (and its descendants) for possible recomputation.
        auto old_given_it = old_givens.find(node);
        auto new_given_it = givens_.find(node);
        const bool had_old_given = old_given_it != old_givens.end();
        const bool has_new_given = new_given_it != givens_.end();
        if (!had_old_given && !has_new_given) {
          continue;
        }
        cache_.MarkUnverified(node);
        rf = ReachedFixpoint::Changed;
      }
    }

    return rf;
  }

  // Set the node to a single immutable forced value.
  //
  // This is different from Givens since it is not combined with the calculated
  // values from earlier in the tree but instead considered a-priori known.
  //
  // Note that any forced value may not have a given associated with it as the
  // given value will be ignored.
  //
  // Care should be taken when using this since existing information is utterly
  // ignored. In general AddGiven is a better choice.
  absl::StatusOr<ReachedFixpoint> SetForced(Node* node,
                                            CacheValueT forced_value) {
    XLS_RET_CHECK(givens_.find(node) == givens_.end())
        << node << " already has given information.";
    if (cache_.GetCacheState(node) == CacheState::kForced &&
        *cache_.GetCachedValue(node) == forced_value) {
      return ReachedFixpoint::Unchanged;
    }
    cache_.SetForced(node, std::move(forced_value));
    return ReachedFixpoint::Changed;
  }

  absl::StatusOr<ReachedFixpoint> RemoveForced(Node* node) {
    XLS_RET_CHECK_EQ(cache_.GetCacheState(node), CacheState::kForced)
        << node << " has no forced value associated with it.";
    if (cache_.GetCacheState(node) != CacheState::kForced) {
      return ReachedFixpoint::Unchanged;
    }
    cache_.Forget(node);
    return ReachedFixpoint::Changed;
  }

  // Bind the node data to the given function.
  absl::StatusOr<ReachedFixpoint> Attach(FunctionBase* f) {
    return AttachWithGivens(f, {});
  }

  // Set the value 'given' as being assumed for the given node. This data is
  // combined with the already calculated data. If one needs to set the state
  // directly to the given value use SetForced instead. A node may not have both
  // 'Forced' and 'Given' data associated with it at the same time.
  absl::StatusOr<ReachedFixpoint> AddGiven(Node* node,
                                           CacheValueT given_value) {
    XLS_RET_CHECK_NE(cache_.GetCacheState(node), CacheState::kForced)
        << node << " has a forced value associated with it.";
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      givens_.emplace(node, std::move(given_value));
      cache_.MarkUnverified(node);
      return ReachedFixpoint::Changed;
    }
    CacheValueT new_given = given_value;
    XLS_RETURN_IF_ERROR(MergeWithGiven(new_given, it->second));
    return ReplaceGiven(node, std::move(new_given));
  }
  ReachedFixpoint RemoveGiven(Node* node) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      return ReachedFixpoint::Unchanged;
    }
    givens_.erase(it);
    cache_.MarkUnverified(node);
    return ReachedFixpoint::Changed;
  }

  absl::StatusOr<ReachedFixpoint> ReplaceGiven(Node* node, CacheValueT given) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      XLS_RET_CHECK_NE(cache_.GetCacheState(node), CacheState::kForced)
          << node << " has a forced value associated with it.";
      givens_.emplace(node, std::move(given));
      cache_.MarkUnverified(node);
      return ReachedFixpoint::Changed;
    }
    CacheValueT prev_given = std::exchange(it->second, std::move(given));
    cache_.MarkUnverified(node);
    return ReachedFixpoint::Changed;
  }

  const CacheValueT* GetInfo(Node* node) const {
    return cache_.QueryValue(node);
  }

  // No action necessary on NodeAdded.

  void NodeDeleted(Node* node) override { cache_.Forget(node); }

  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override {
    CacheState prev_state = cache_.GetCacheState(node);
    if (prev_state != CacheState::kKnown &&
        prev_state != CacheState::kInputsUnverified) {
      return;
    }
    bool operand_info_changed = false;
    if (const CacheValueT* old_operand_info =
            cache_.GetCachedValue(old_operand);
        old_operand_info != nullptr) {
      for (int64_t operand_no : operand_nos) {
        Node* new_operand = node->operand(operand_no);
        if (new_operand->GetType() != old_operand->GetType()) {
          operand_info_changed = true;
          continue;
        }
        const CacheValueT* new_operand_info =
            cache_.GetCachedValue(new_operand);
        if (new_operand_info == nullptr) {
          // We don't know anything about the new operand - but we can guess
          // that it may be equivalent to the old operand.
          cache_.AddUnverified(
              new_operand, std::make_unique<CacheValueT>(*old_operand_info));
        } else if (*new_operand_info != *old_operand_info) {
          operand_info_changed = true;
        }
      }
    }
    if (operand_info_changed) {
      cache_.MarkUnverified(node);
    } else {
      cache_.MarkInputsUnverified(node);
    }
  }

  void OperandRemoved(Node* node, Node* old_operand) override {
    cache_.MarkUnverified(node);
  }

  void OperandAdded(Node* node) override { cache_.MarkUnverified(node); }

  void ForceRecompute(Node* node) { cache_.MarkUnverified(node); }

  // Eagerly computes the values for all nodes in the function that do not have
  // known values. This is expensive and should only be used for testing and
  // measurement.
  absl::Status EagerlyPopulate(FunctionBase* f) {
    XLS_RETURN_IF_ERROR(Attach(f).status());
    return cache_.EagerlyPopulate(TopoSort(f));
  }

  // Verifies that the query engine's current state is consistent; e.g., for
  // lazy query engines, checks that the current state of the cache is correct
  // where expected & consistent regardless. This is an expensive operation,
  // intended for use in tests.
  //
  // Note that Forced values are always considered consistent.
  absl::Status CheckCacheConsistency() const {
    XLS_RET_CHECK(f_ != nullptr) << "Unattached info";
    return cache_.CheckConsistency(TopoSort(f_));
  }

  // Implementation for LazyDagCache::DagProvider.
  std::string GetName(Node* const& node) const override {
    return node->GetName();
  }
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->operands();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->users();
  }
  absl::StatusOr<CacheValueT> ComputeValue(
      Node* const& node,
      absl::Span<const CacheValueT* const> operand_infos) const override {
    CacheValueT new_info = ComputeInfo(node, operand_infos);
    if (auto it = givens_.find(node); it != givens_.end()) {
      const CacheValueT& given_value = it->second;
      XLS_RETURN_IF_ERROR(MergeWithGiven(new_info, given_value));
    }
    return new_info;
  }

  // The function that this cache is bound on.
  FunctionBase* bound_function() const { return f_; }

 protected:
  virtual CacheValueT ComputeInfo(
      Node* node, absl::Span<const CacheValueT* const> operand_infos) const = 0;

  virtual absl::Status MergeWithGiven(CacheValueT& info,
                                      const CacheValueT& given) const = 0;

 private:
  FunctionBase* f_ = nullptr;

  mutable LazyDagCache<Node*, CacheValueT> cache_;
  absl::flat_hash_map<Node*, CacheValueT> givens_;
};

}  // namespace xls

#endif  // XLS_PASSES_LAZY_NODE_DATA_H_
