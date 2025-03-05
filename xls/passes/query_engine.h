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

#ifndef XLS_PASSES_QUERY_ENGINE_H_
#define XLS_PASSES_QUERY_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/passes/predicate_state.h"

namespace xls {

// Abstraction representing a particular bit of a particular XLS Node.
class TreeBitLocation {
 public:
  TreeBitLocation() : node_(nullptr), bit_index_(0) {}

  TreeBitLocation(Node* node, int64_t bit_index,
                  absl::Span<const int64_t> tree_index = {})
      : node_(node),
        bit_index_(bit_index),
        tree_index_(tree_index.begin(), tree_index.end()) {}

  Node* node() const { return node_; }

  int64_t bit_index() const { return bit_index_; }

  absl::Span<const int64_t> tree_index() const { return tree_index_; }

  friend bool operator==(const TreeBitLocation& x, const TreeBitLocation& y) {
    return (x.node_ == y.node_) && (x.tree_index_ == y.tree_index_) &&
           (x.bit_index_ == y.bit_index_);
  }

  template <typename H>
  friend H AbslHashValue(H h, const TreeBitLocation& tbl) {
    return H::combine(std::move(h), tbl.node_, tbl.tree_index_, tbl.bit_index_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TreeBitLocation& tbl) {
    if (tbl.tree_index_.empty()) {
      absl::Format(&sink, "%s[%d]", tbl.node_->GetName(), tbl.bit_index_);
    } else {
      absl::Format(&sink, "%s@{%s}[%d]", tbl.node_->GetName(),
                   absl::StrJoin(tbl.tree_index_, ", "), tbl.bit_index_);
    }
  }

 private:
  Node* node_;
  int64_t bit_index_;
  std::vector<int64_t> tree_index_;
};

using TernaryTree = LeafTypeTree<TernaryVector>;
using TernaryTreeView = LeafTypeTreeView<TernaryVector>;
using SharedTernaryTree = SharedLeafTypeTree<TernaryVector>;
using MutableTernaryTreeView = MutableLeafTypeTreeView<TernaryVector>;

using IntervalSetTree = LeafTypeTree<IntervalSet>;
using IntervalSetTreeView = LeafTypeTreeView<IntervalSet>;
using MutableIntervalSetTreeView = MutableLeafTypeTreeView<IntervalSet>;

struct ValueKnowledge {
  std::optional<TernaryTree> ternary;
  std::optional<IntervalSetTree> intervals;
};

enum class ReachedFixpoint { Unchanged, Changed, Unknown };

// An abstract base class providing an interface for answering queries about the
// values of and relationship between bits in an XLS function. Information
// provided include statically known bit values and implications between bits in
// the graph.
//
// Generally query methods returning a boolean value return true if the
// condition is known to be true, and false if the condition cannot be
// determined.  This means a false return value does *not* mean that the
// condition is necessarily false. For example, KnownEqual(a, b) returning false
// does not mean that 'a' is necessarily not equal 'b'. Rather, the false return
// value indicates that analysis could not determine whether 'a' and 'b' are
// necessarily equal.
class QueryEngine {
 public:
  virtual ~QueryEngine() = default;

  // LINT.IfChanged - All virtual functions must be forwarded by
  // ForwardingQueryEngine.

  // Populate or prepare the query engine for calls to other functions.
  virtual absl::StatusOr<ReachedFixpoint> Populate(FunctionBase* f) = 0;

  // Returns whether any information is available for this node.
  virtual bool IsTracked(Node* node) const = 0;

  // Returns a `SharedLeafTypeTree<TernaryVector>` indicating which bits have
  // known values for the given node and what that bit's known value is. May
  // return `std::nullopt` if no bits are known.
  virtual std::optional<SharedLeafTypeTree<TernaryVector>> GetTernary(
      Node* node) const = 0;

  // Return a query engine which is specialized with the given predicates. The
  // reference has an lifetime of the source engine.  For now no query-engine
  // supports a state set with more than a single element. This is CHECK'd
  // internally in some query engines to avoid surprising non-deterministic
  // behavior. In the future we might relax this restriction.
  virtual std::unique_ptr<QueryEngine> SpecializeGivenPredicate(
      const absl::flat_hash_set<PredicateState>& state) const;

  // Return a query engine which is specialized with the given information. The
  // reference has an lifetime of the source engine.
  virtual std::unique_ptr<QueryEngine> SpecializeGiven(
      const absl::flat_hash_map<Node*, ValueKnowledge>& givens) const;

  // Returns a `LeafTypeTree<IntervalSet>` indicating which interval sets the
  // various parts of the value for a given node can exist in.
  virtual LeafTypeTree<IntervalSet> GetIntervals(Node* node) const;

  // Returns true if at most one of the given bits can be true.
  virtual bool AtMostOneTrue(absl::Span<TreeBitLocation const> bits) const = 0;

  // Returns true if at least one of the given bits is true.
  virtual bool AtLeastOneTrue(absl::Span<TreeBitLocation const> bits) const = 0;

  // Returns true if 'a' implies 'b'.
  virtual bool Implies(const TreeBitLocation& a,
                       const TreeBitLocation& b) const = 0;

  // If a particular value of 'node' (true or false for all bits)
  // is implied when the bits in 'predicate_bit_values' have the given values,
  // the implied value of 'node' is returned.
  virtual std::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const = 0;

  // If we learn any bits of 'node' (true or false for all bits) when the bits
  // in 'predicate_bit_values' have the given values, the corresponding
  // TernaryVector is returned.
  virtual std::optional<TernaryVector> ImpliedNodeTernary(
      absl::Span<const std::pair<TreeBitLocation, bool>> predicate_bit_values,
      Node* node) const = 0;

  // Returns true if 'a' equals 'b'
  virtual bool KnownEquals(const TreeBitLocation& a,
                           const TreeBitLocation& b) const = 0;

  // Returns true if 'a' is the inverse of 'b'
  virtual bool KnownNotEquals(const TreeBitLocation& a,
                              const TreeBitLocation& b) const = 0;

  // Returns true if at most/at least/exactly one of the bits in 'node' is true.
  // 'node' must be bits-typed.
  virtual bool AtMostOneBitTrue(Node* node) const;
  virtual bool AtLeastOneBitTrue(Node* node) const;
  virtual bool ExactlyOneBitTrue(Node* node) const;

  // Returns whether the value of the output bit of the given node at the given
  // index is known (definitely zero or one).
  virtual bool IsKnown(const TreeBitLocation& bit) const;

  // Returns the value of the output bit of the given node at the given index,
  // if known; otherwise returns std::nullopt.
  virtual std::optional<bool> KnownValue(const TreeBitLocation& bit) const;

  // Returns the output of the given node at the given index, if completely
  // known; otherwise returns std::nullopt.
  virtual std::optional<Value> KnownValue(Node* node) const;

  // Returns whether every bit in the output of the given node is definitely
  // zero (or one). Always returns false if the node includes a token.
  virtual bool IsAllZeros(Node* node) const;
  virtual bool IsAllOnes(Node* node) const;

  // Returns true if every output bit of the given node is known (definitely
  // zero or one).
  virtual bool IsFullyKnown(Node* n) const;

  // Returns the maximum unsigned value that the node can be.
  virtual Bits MaxUnsignedValue(Node* node) const;

  // Returns the minimum unsigned value that the node can be.
  virtual Bits MinUnsignedValue(Node* node) const;

  // Returns the number of leading bits which are known to be zero or nullopt if
  // the node is not bits type or the result is not known.
  virtual std::optional<int64_t> KnownLeadingZeros(Node* node) const;
  // Returns the number of leading bits which are known to be one or nullopt if
  // the node is not bits type or the result is not known.
  virtual std::optional<int64_t> KnownLeadingOnes(Node* node) const;
  // Returns the number of leading bits which are known to be the same as the
  // leading bit or nullopt if the node is not bits type or the result is not
  // known. NB For any bits type (which the query engine has any data about at
  // least) will have a value of at least 1 since the sign bit always
  // definitionally equal to the sign bit.
  virtual std::optional<int64_t> KnownLeadingSignBits(Node* node) const;

  // LINT.ThenChange(//xls/passes/forwarding_query_engine.h)

  // Returns the location of the unknown bit within the node if exactly one of
  // the given bits is unknown.
  std::optional<TreeBitLocation> ExactlyOneBitUnknown(Node* node) const;

  // Returns true if at most/least one of the values in 'preds' is true. Each
  // value in 'preds' must be a single-bit bits-typed value.
  bool AtMostOneNodeTrue(absl::Span<Node* const> preds) const;
  bool AtLeastOneNodeTrue(absl::Span<Node* const> preds) const;

  // Returns the output of the given node at the given index, if completely
  // known; otherwise returns std::nullopt. Precondition: 'node' must be
  // bits-typed.
  std::optional<Bits> KnownValueAsBits(Node* node) const;

  // Returns if the most-significant bit is known of 'node'.
  bool IsMsbKnown(Node* node) const;

  // Returns the value of the most-significant bit of 'node'. Precondition: the
  // most-significant bit must be known (IsMsbKnown returns true),
  bool GetKnownMsb(Node* node) const;

  // Returns whether the value of the output bit of the given node at the given
  // index is definitely one (or zero).
  bool IsOne(const TreeBitLocation& bit) const;
  bool IsZero(const TreeBitLocation& bit) const;

  // Returns whether *all* the bits are known for 'node' (which must be
  // bits-type).
  bool AllBitsKnown(Node* node) const {
    CHECK(node->GetType()->IsBits());
    return IsFullyKnown(node);
  }

  // Returns true if the values of the two nodes are known to be equal when
  // interpreted as unsigned numbers. The nodes can be of different widths.
  bool NodesKnownUnsignedEquals(Node* a, Node* b) const;

  // Returns true if the values of the two nodes are known *NOT* to be equal
  // when interpreted as unsigned numbers. The nodes can be of different
  // widths.
  bool NodesKnownUnsignedNotEquals(Node* a, Node* b) const;

  // Returns the known bits information of the given node as a string of ternary
  // symbols (0, 1, or X) with a '0b' prefix. For example: 0b1XX0.
  std::string ToString(Node* node) const;

  virtual absl::Status CheckConsistency() const { return absl::OkStatus(); }
};

}  // namespace xls

#endif  // XLS_PASSES_QUERY_ENGINE_H_
