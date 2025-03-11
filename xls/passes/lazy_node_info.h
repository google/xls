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

#ifndef XLS_PASSES_LAZY_NODE_INFO_H_
#define XLS_PASSES_LAZY_NODE_INFO_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A base class for node-info cache that can be lazily populated; implements an
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
// be up-to-date (kInputsUnverified), or information that is known to be current
// & correct (kKnown).
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
// NOTE: If `n` is in state kInputsUnverified after we queried all of its
//       operands, then their values did not change, so we have verified that
//       `n`'s information is up-to-date without having to recompute it! This is
//       the main advantage of this cache over a more typical invalidating
//       cache.
template <typename Info>
class LazyNodeInfo
    : public ChangeListener,
      public LazyDagCache<Node*, LeafTypeTree<Info>>::DagProvider {
 private:
  using CacheState = LazyDagCache<Node*, LeafTypeTree<Info>>::CacheState;

 public:
  LazyNodeInfo<Info>() : cache_(this) {}
  ~LazyNodeInfo() override {
    if (f_ != nullptr) {
      f_->UnregisterChangeListener(this);
    }
  }

  LazyNodeInfo(const LazyNodeInfo<Info>& other)
      : f_(other.f_), cache_(this, other.cache_), givens_(other.givens_) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyNodeInfo& operator=(const LazyNodeInfo<Info>& other) {
    if (f_ != other.f_) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
      }
      f_ = other.f_;
    }
    cache_ = LazyDagCache<Node*, LeafTypeTree<Info>>(this, other.cache_);
    givens_ = other.givens_;
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
    return *this;
  }

  LazyNodeInfo(LazyNodeInfo<Info>&& other)
      : f_(other.f_),
        cache_(this, std::move(other.cache_)),
        givens_(std::move(other.givens_)) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyNodeInfo& operator=(LazyNodeInfo<Info>&& other) {
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
    cache_ =
        LazyDagCache<Node*, LeafTypeTree<Info>>(this, std::move(other.cache_));
    givens_ = std::move(other.givens_);
    return *this;
  }

  // Bind the node info to the given function.
  virtual absl::StatusOr<ReachedFixpoint> AttachWithGivens(
      FunctionBase* f, absl::flat_hash_map<Node*, LeafTypeTree<Info>> givens) {
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
      absl::flat_hash_map<Node*, LeafTypeTree<Info>> old_givens =
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

  // Bind the node info to the given function.
  absl::StatusOr<ReachedFixpoint> Attach(FunctionBase* f) {
    return AttachWithGivens(f, {});
  }

  absl::StatusOr<ReachedFixpoint> AddGiven(Node* node,
                                           LeafTypeTree<Info> given_ltt) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      givens_.emplace(node, std::move(given_ltt));
      cache_.MarkUnverified(node);
      return ReachedFixpoint::Changed;
    }
    XLS_RETURN_IF_ERROR((leaf_type_tree::UpdateFrom<Info, Info>(
        given_ltt.AsMutableView(), it->second.AsView(),
        [this](Type*, Info& info, const Info& given,
               absl::Span<const int64_t>) {
          return MergeWithGiven(info, given);
        })));
    return ReplaceGiven(node, std::move(given_ltt));
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
  ReachedFixpoint ReplaceGiven(Node* node, LeafTypeTree<Info> given) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      givens_.emplace(node, std::move(given));
      cache_.MarkUnverified(node);
      return ReachedFixpoint::Changed;
    }
    LeafTypeTree<Info> prev_given = std::exchange(it->second, std::move(given));
    cache_.MarkUnverified(node);
    return ReachedFixpoint::Changed;
  }

  std::optional<SharedLeafTypeTree<Info>> GetInfo(Node* node) const {
    LeafTypeTree<Info>* info = cache_.QueryValue(node);
    if (info == nullptr) {
      return std::nullopt;
    }
    return info->AsView().AsShared();
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
    if (const LeafTypeTree<Info>* old_operand_info =
            cache_.GetCachedValue(old_operand);
        old_operand_info != nullptr) {
      for (int64_t operand_no : operand_nos) {
        Node* new_operand = node->operand(operand_no);
        if (new_operand->GetType() != old_operand->GetType()) {
          operand_info_changed = true;
          continue;
        }
        const LeafTypeTree<Info>* new_operand_info =
            cache_.GetCachedValue(new_operand);
        if (new_operand_info == nullptr) {
          // We don't know anything about the new operand - but we can guess
          // that it may be equivalent to the old operand.
          cache_.AddUnverified(new_operand, leaf_type_tree::CloneToHeap(
                                                old_operand_info->AsView()));
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
  absl::StatusOr<LeafTypeTree<Info>> ComputeValue(
      Node* const& node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos)
      const override {
    LeafTypeTree<Info> new_info = ComputeInfo(node, operand_infos);
    if (auto it = givens_.find(node); it != givens_.end()) {
      const LeafTypeTree<Info>& given_ltt = it->second;
      XLS_RETURN_IF_ERROR((leaf_type_tree::UpdateFrom<Info, Info>(
          new_info.AsMutableView(), given_ltt.AsView(),
          [this](Type*, Info& info, const Info& given,
                 absl::Span<const int64_t>) {
            return MergeWithGiven(info, given);
          })));
    }
    return new_info;
  }

  // The function that this cache is bound on.
  FunctionBase* bound_function() const { return f_; }

 protected:
  virtual LeafTypeTree<Info> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos) const = 0;

  virtual absl::Status MergeWithGiven(Info& info, const Info& given) const = 0;

 private:
  FunctionBase* f_ = nullptr;

  mutable LazyDagCache<Node*, LeafTypeTree<Info>> cache_;
  absl::flat_hash_map<Node*, LeafTypeTree<Info>> givens_;
};

}  // namespace xls

#endif  // XLS_PASSES_LAZY_NODE_INFO_H_
