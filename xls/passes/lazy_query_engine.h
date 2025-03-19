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

#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/lazy_node_info.h"
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
class LazyQueryEngine : public QueryEngine {
 private:
  class QueryEngineNodeInfo : public LazyNodeInfo<Info> {
   public:
    explicit QueryEngineNodeInfo(LazyQueryEngine<Info>* owner)
        : owner_(owner) {}
    QueryEngineNodeInfo(const QueryEngineNodeInfo&) = default;
    QueryEngineNodeInfo(QueryEngineNodeInfo&&) = default;
    QueryEngineNodeInfo& operator=(const QueryEngineNodeInfo&) = default;
    QueryEngineNodeInfo& operator=(QueryEngineNodeInfo&&) = default;

    QueryEngineNodeInfo WithOwner(LazyQueryEngine<Info>* owner) const& {
      QueryEngineNodeInfo info(*this);
      info.owner_ = owner;
      return info;
    }

    QueryEngineNodeInfo TakeOwner(LazyQueryEngine<Info>* owner) && {
      QueryEngineNodeInfo info(std::move(*this));
      info.owner_ = owner;
      return info;
    }

   protected:
    LeafTypeTree<Info> ComputeInfo(
        Node* node,
        absl::Span<const LeafTypeTree<Info>* const> operand_infos) const final;

    absl::Status MergeWithGiven(Info& info, const Info& given) const final;

   private:
    LazyQueryEngine<Info>* owner_;
  };

 public:
  LazyQueryEngine<Info>() : info_(this) {}
  ~LazyQueryEngine() override {}

  LazyQueryEngine(const LazyQueryEngine<Info>& other)
      : info_(other.info_.WithOwner(this)) {}
  LazyQueryEngine& operator=(const LazyQueryEngine<Info>& other) {
    info_ = other.info_.WithOwner(this);
    return *this;
  }
  LazyQueryEngine(LazyQueryEngine<Info>&& other)
      : info_(std::move(other).info_.TakeOwner(this)) {}
  LazyQueryEngine& operator=(LazyQueryEngine<Info>&& other) {
    info_ = std::move(other).info_.TakeOwner(this);
    return *this;
  }

  absl::StatusOr<ReachedFixpoint> PopulateWithGivens(
      FunctionBase* f, absl::flat_hash_map<Node*, LeafTypeTree<Info>> givens) {
    return info_.AttachWithGivens(f, std::move(givens));
  }
  absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) override {
    return info_.Attach(f);
  }

  bool IsTracked(Node* node) const override {
    return node->function_base() == info_.bound_function();
  }

  // Check that the query engine is consistent. Note that Forced values are
  // always considered consistent.
  absl::Status CheckConsistency() const override {
    return info_.CheckCacheConsistency();
  }

  // Access to the underlying data store for this query engine. Use this to
  // directly add givens if required.
  LazyNodeInfo<Info>& info() { return info_; }

  // Access to the underlying data store for this query engine. Use this to
  // directly add givens if required.
  const LazyNodeInfo<Info>& info() const { return info_; }

  std::optional<SharedLeafTypeTree<Info>> GetInfo(Node* node) const {
    return info_.GetInfo(node);
  }

  // Eagerly computes the values for all nodes in the function that do not have
  // known values. This is expensive and should only be used for testing and
  // measurement.
  absl::Status EagerlyPopulate(FunctionBase* f) {
    return info_.EagerlyPopulate(f);
  }

  void ForceRecompute(Node* node) { info_.ForceRecompute(node); }

  // Helpers for adding/removing givens.
  absl::StatusOr<ReachedFixpoint> AddGiven(Node* node,
                                           LeafTypeTree<Info> given_ltt) {
    return info_.AddGiven(node, std::move(given_ltt));
  }
  absl::StatusOr<ReachedFixpoint> ReplaceGiven(Node* node,
                                               LeafTypeTree<Info> given_ltt) {
    return info_.ReplaceGiven(node, std::move(given_ltt));
  }
  ReachedFixpoint RemoveGiven(Node* node) { return info_.RemoveGiven(node); }

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
                                            LeafTypeTree<Info> forced_ltt) {
    return info_.SetForced(node, std::move(forced_ltt));
  }
  // Removed forced information.
  absl::StatusOr<ReachedFixpoint> RemoveForced(Node* node) {
    return info_.RemoveForced(node);
  }

 protected:
  virtual LeafTypeTree<Info> ComputeInfo(
      Node* node,
      absl::Span<const LeafTypeTree<Info>* const> operand_infos) const = 0;

  virtual absl::Status MergeWithGiven(Info& info, const Info& given) const = 0;

 private:
  QueryEngineNodeInfo info_;
};

template <typename Info>
absl::Status LazyQueryEngine<Info>::QueryEngineNodeInfo::MergeWithGiven(
    Info& info, const Info& given) const {
  return owner_->MergeWithGiven(info, given);
}

template <typename Info>
LeafTypeTree<Info> LazyQueryEngine<Info>::QueryEngineNodeInfo::ComputeInfo(
    Node* node,
    absl::Span<const LeafTypeTree<Info>* const> operand_infos) const {
  return owner_->ComputeInfo(node, operand_infos);
}

}  // namespace xls

#endif  // XLS_PASSES_LAZY_QUERY_ENGINE_H_
