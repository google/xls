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

#ifndef XLS_PASSES_LAZY_QUERY_ENGINE_H_
#define XLS_PASSES_LAZY_QUERY_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

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
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/passes/query_engine.h"

namespace xls {

// A base class for query engines that can be lazily populated; implements an
// invalidating/"re-validating" cache for any analysis that depends only on the
// transitive inputs to a node.
//
// Derived classes must implement:
// - `ComputeInfo`: called when a node's information needs to be recalculated,
//    and
// - `MergeWithGiven`: called to merge computed information with given
//    information,
// and then must implement all pure-virtual methods of `QueryEngine`, but may
// use `QueryInfo` to retrieve the information for a node.
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
class LazyQueryEngine : public QueryEngine, public ChangeListener {
 public:
  LazyQueryEngine<Info>() = default;
  ~LazyQueryEngine() override {
    if (f_ != nullptr) {
      f_->UnregisterChangeListener(this);
    }
  }

  LazyQueryEngine(const LazyQueryEngine<Info>& other)
      : f_(other.f_), cache_(other.cache_), givens_(other.givens_) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyQueryEngine& operator=(const LazyQueryEngine<Info>& other) {
    if (f_ != other.f_) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
      }
      f_ = other.f_;
    }
    cache_ = other.cache_;
    givens_ = other.givens_;
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
    return *this;
  }

  LazyQueryEngine(LazyQueryEngine<Info>&& other)
      : f_(other.f_),
        cache_(std::move(other.cache_)),
        givens_(std::move(other.givens_)) {
    if (f_ != nullptr) {
      f_->RegisterChangeListener(this);
    }
  }
  LazyQueryEngine& operator=(LazyQueryEngine<Info>&& other) {
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
    cache_ = std::move(other.cache_);
    givens_ = std::move(other.givens_);
    return *this;
  }

  absl::StatusOr<ReachedFixpoint> PopulateWithGivens(
      FunctionBase* f, absl::flat_hash_map<Node*, LeafTypeTree<Info>> givens) {
    ReachedFixpoint rf = ReachedFixpoint::Unchanged;
    if (f_ != f) {
      if (f_ != nullptr) {
        f_->UnregisterChangeListener(this);
        cache_.clear();
        givens_.clear();
        rf = ReachedFixpoint::Changed;
      }

      if (f != nullptr) {
        f_ = f;
        cache_.reserve(f->node_count());
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
        MarkUnverified(node);
        rf = ReachedFixpoint::Changed;
      }
    }

    return rf;
  }
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return PopulateWithGivens(f, {});
  }

  absl::StatusOr<ReachedFixpoint> AddGiven(Node* node,
                                           LeafTypeTree<Info> given_ltt) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      givens_.emplace(node, std::move(given_ltt));
      MarkUnverified(node);
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
    MarkUnverified(node);
    return ReachedFixpoint::Changed;
  }
  ReachedFixpoint ReplaceGiven(Node* node, LeafTypeTree<Info> given) {
    auto it = givens_.find(node);
    if (it == givens_.end()) {
      givens_.emplace(node, std::move(given));
      MarkUnverified(node);
      return ReachedFixpoint::Changed;
    }
    LeafTypeTree<Info> prev_given = std::exchange(it->second, std::move(given));
    MarkUnverified(node);
    return ReachedFixpoint::Changed;
  }

  bool IsTracked(Node* node) const override {
    return node->function_base() == f_;
  }

  std::optional<SharedLeafTypeTree<Info>> GetInfo(Node* node) const {
    LeafTypeTree<Info>* info = QueryInfo(node);
    if (info == nullptr) {
      return std::nullopt;
    }
    return info->AsView().AsShared();
  }

  // No action necessary on NodeAdded.

  void NodeDeleted(Node* node) override { cache_.erase(node); }

  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override {
    CacheState prev_state = GetCacheState(node);
    if (prev_state != CacheState::kKnown &&
        prev_state != CacheState::kInputsUnverified) {
      return;
    }
    bool operand_info_changed = false;
    if (const LeafTypeTree<Info>* old_operand_info = GetCachedInfo(old_operand);
        old_operand_info != nullptr) {
      for (int64_t operand_no : operand_nos) {
        Node* new_operand = node->operand(operand_no);
        if (new_operand->GetType() != old_operand->GetType()) {
          operand_info_changed = true;
          continue;
        }
        const LeafTypeTree<Info>* new_operand_info = GetCachedInfo(new_operand);
        if (new_operand_info == nullptr) {
          // We don't know anything about the new operand - but we can guess
          // that it may be equivalent to the old operand.
          cache_[new_operand] = CacheEntry{
              .state = CacheState::kUnverified,
              .info = leaf_type_tree::CloneToHeap(old_operand_info->AsView())};
        } else if (*new_operand_info != *old_operand_info) {
          operand_info_changed = true;
        }
      }
    }
    if (operand_info_changed) {
      MarkUnverified(node);
    } else {
      MarkInputsUnverified(node);
    }
  }

  void OperandRemoved(Node* node, Node* old_operand) override {
    MarkUnverified(node);
  }

  void OperandAdded(Node* node) override { MarkUnverified(node); }

  void ForceRecompute(Node* node) { MarkUnverified(node); }

  // Eagerly computes the values for all nodes in the function that do not have
  // known values. This is expensive and should only be used for testing and
  // measurement.
  absl::Status EagerlyPopulate(FunctionBase* f) {
    XLS_RETURN_IF_ERROR(Populate(f).status());
    for (Node* node : TopoSort(f_)) {
      if (GetCacheState(node) == CacheState::kKnown) {
        continue;
      }
      std::vector<const LeafTypeTree<Info>*> operand_infos;
      operand_infos.reserve(node->operands().size());
      for (Node* operand : node->operands()) {
        const CacheEntryView operand_entry = GetCacheEntry(operand);
        XLS_RET_CHECK_EQ(operand_entry.state, CacheState::kKnown);
        operand_infos.push_back(operand_entry.info);
      }
      cache_.insert_or_assign(
          node, CacheEntry{.state = CacheState::kKnown,
                           .info = std::make_unique<LeafTypeTree<Info>>(
                               ComputeInfo(node, operand_infos))});
    }
    return absl::OkStatus();
  }

  // Verifies that the query engine's current state is consistent; e.g., for
  // lazy query engines, checks that the current state of the cache is correct
  // where expected & consistent regardless. This is an expensive operation,
  // intended for use in tests.
  absl::Status CheckConsistency() const override;

 protected:
  virtual LeafTypeTree<Info> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos) const = 0;

  virtual absl::Status MergeWithGiven(Info& info, const Info& given) const = 0;

 private:
  FunctionBase* f_ = nullptr;

  enum class CacheState : uint8_t {
    kUnknown,  // NOTE: This state corresponds to a node that's not in the
               //       cache, so is only temporarily stored while we're
               //       computing the value.
    kUnverified,
    kInputsUnverified,
    kKnown,
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
    }
    LOG(FATAL) << "Unknown CacheState: " << static_cast<int>(state);
  }
  friend std::ostream& operator<<(std::ostream& os, const CacheState& state) {
    return os << absl::StrCat(state);
  }

  struct CacheEntry {
    CacheState state = CacheState::kUnknown;
    std::unique_ptr<LeafTypeTree<Info>> info = nullptr;
  };
  mutable absl::flat_hash_map<Node*, CacheEntry> cache_;
  absl::flat_hash_map<Node*, LeafTypeTree<Info>> givens_;

  void MarkUnverified(Node* node) const {
    auto it = cache_.find(node);
    if (it == cache_.end()) {
      return;
    }
    auto& [state, _] = it->second;
    if (state == CacheState::kUnverified) {
      return;
    }
    state = CacheState::kUnverified;
    for (Node* user : node->users()) {
      if (GetCacheState(user) == CacheState::kKnown) {
        MarkInputsUnverified(user);
      }
    }
  }

  void MarkInputsUnverified(Node* node) const {
    auto it = cache_.find(node);
    if (it == cache_.end()) {
      return;
    }
    auto& [state, _] = it->second;
    if (state != CacheState::kKnown) {
      return;
    }
    state = CacheState::kInputsUnverified;
    std::vector<Node*> worklist(node->users().begin(), node->users().end());
    while (!worklist.empty()) {
      Node* descendant = worklist.back();
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

  CacheState GetCacheState(Node* node) const {
    auto it = cache_.find(node);
    if (it == cache_.end()) {
      return CacheState::kUnknown;
    }
    return it->second.state;
  }
  const LeafTypeTree<Info>* GetCachedInfo(Node* node) const {
    auto it = cache_.find(node);
    if (it == cache_.end() || it->second.info == nullptr) {
      return nullptr;
    }
    return it->second.info.get();
  }

  struct CacheEntryView {
    CacheState state = CacheState::kUnknown;
    const LeafTypeTree<Info>* info = nullptr;
  };
  CacheEntryView GetCacheEntry(Node* node) const {
    auto it = cache_.find(node);
    if (it == cache_.end()) {
      return {.state = CacheState::kUnknown, .info = nullptr};
    }
    return {.state = it->second.state, .info = it->second.info.get()};
  }

  LeafTypeTree<Info> ComputeInfoWithGivens(
      Node* node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos) const {
    LeafTypeTree<Info> new_info = ComputeInfo(node, operand_infos);
    if (auto it = givens_.find(node); it != givens_.end()) {
      const LeafTypeTree<Info>& given_ltt = it->second;
      CHECK_OK((leaf_type_tree::UpdateFrom<Info, Info>(
          new_info.AsMutableView(), given_ltt.AsView(),
          [this](Type*, Info& info, const Info& given,
                 absl::Span<const int64_t>) {
            return MergeWithGiven(info, given);
          })));
    }
    return new_info;
  }

  LeafTypeTree<Info>* QueryInfo(Node* node) const {
    CHECK_EQ(node->function_base(), f_);
    // If `node` is already known, return a pointer to the cached information.
    if (auto it = cache_.find(node);
        it != cache_.end() && it->second.state == CacheState::kKnown) {
      return it->second.info.get();
    }

    // Retrieve the information for all of this node's operands.
    std::vector<const LeafTypeTree<Info>*> operand_infos;
    for (Node* operand : node->operands()) {
      operand_infos.push_back(QueryInfo(operand));
    }

    // Find the CacheEntry for this node; if not present, this will insert a
    // default-constructed CacheEntry, with state kUnknown and empty info, which
    // we will populate below.
    auto& [state, cached_info] = cache_[node];

    // If this node was previously in state kInputsUnverified and any operand
    // changed, QueryInfo on that operand would have automatically downgraded
    // this node to kUnverified. Therefore, if we're still in state
    // kInputsUnverified, the stored information is still valid; we can skip
    // recomputation.
    if (state == CacheState::kInputsUnverified) {
      state = CacheState::kKnown;
      return cached_info.get();
    }

    LeafTypeTree<Info> new_info = ComputeInfoWithGivens(node, operand_infos);
    if (state == CacheState::kUnverified && new_info == *cached_info) {
      // The information didn't change; the stored information is still valid.
      state = CacheState::kKnown;
      return cached_info.get();
    }

    state = CacheState::kKnown;
    cached_info = std::make_unique<LeafTypeTree<Info>>(std::move(new_info));

    // Our stored information changed; make sure we downgrade any users that
    // were previously kInputsUnverified to kUnverified.
    for (Node* user : node->users()) {
      if (GetCacheState(user) == CacheState::kInputsUnverified) {
        MarkUnverified(user);
      }
    }
    return cached_info.get();
  }
};

template <typename Info>
absl::Status LazyQueryEngine<Info>::CheckConsistency() const {
  absl::flat_hash_set<Node*> correct_values;
  for (Node* node : TopoSort(f_)) {
    const auto& [state, info] = GetCacheEntry(node);

    if (state == CacheState::kUnknown) {
      XLS_RET_CHECK_EQ(info, nullptr)
          << "Node " << node->GetName() << " is UNKNOWN but has stored info.";
      continue;
    }
    XLS_RET_CHECK_NE(info, nullptr)
        << "Node " << node->GetName()
        << " is not UNKNOWN but has no stored info.";

    if (state == CacheState::kKnown) {
      for (Node* operand : node->operands()) {
        XLS_RET_CHECK_EQ(GetCacheState(operand), CacheState::kKnown)
            << "Non-KNOWN operand for KNOWN node " << node->GetName() << ": "
            << operand->GetName();
      }
    }
    if (state == CacheState::kInputsUnverified) {
      for (Node* operand : node->operands()) {
        XLS_RET_CHECK_NE(GetCacheState(operand), CacheState::kUnknown)
            << "UNKNOWN operand for INPUTS_UNVERIFIED node " << node->GetName()
            << ": " << operand->GetName();
      }
    }

    if (absl::c_any_of(node->operands(), [&](Node* operand) {
          return !correct_values.contains(operand);
        })) {
      // We can only check the consistency/correctness of `node`'s stored value
      // if all of its operands have correct stored values.
      continue;
    }

    std::vector<const LeafTypeTree<Info>*> operand_infos;
    operand_infos.reserve(node->operand_count());
    for (Node* operand : node->operands()) {
      operand_infos.push_back(GetCachedInfo(operand));
    }

    LeafTypeTree<Info> recomputed_info =
        ComputeInfoWithGivens(node, operand_infos);
    if (*info == recomputed_info) {
      correct_values.insert(node);
    }

    // If the node is KNOWN or INPUTS_UNVERIFIED (with all operands' stored
    // information verified correct), we can check that the stored information
    // is consistent with the information we would compute from its operands.
    if (state == CacheState::kKnown || state == CacheState::kInputsUnverified) {
      XLS_RET_CHECK(*info == recomputed_info)
          << state << " node " << node->GetName()
          << " has a stored value that is not consistent with its operands' "
             "stored values";
    }
  }
  return absl::OkStatus();
}

}  // namespace xls

#endif  // XLS_PASSES_LAZY_QUERY_ENGINE_H_
