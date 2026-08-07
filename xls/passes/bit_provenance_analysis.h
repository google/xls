// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_BIT_PROVENANCE_ANALYSIS_H_
#define XLS_PASSES_BIT_PROVENANCE_ANALYSIS_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/query_engine.h"

namespace xls {

// Object which holds information about where each bit comes from.
class TreeBitSources {
 public:
  // Where a particular segment of bits come from.
  class BitRange {
   public:
    // A set of bits from 'bit-index-low' for bit_width at the given tree-index
    // in the given node.
    BitRange(Node* source_node, int64_t source_bit_index_low,
             int64_t dest_bit_index_low, int64_t bit_width,
             absl::Span<const int64_t> source_tree_index = {})
        : source_node_(source_node),
          source_bit_index_low_(source_bit_index_low),
          dest_bit_index_low_(dest_bit_index_low),
          bit_width_(bit_width),
          source_tree_index_(source_tree_index.begin(),
                             source_tree_index.end()) {}
    BitRange(const BitRange&) = default;
    BitRange(BitRange&&) = default;
    BitRange& operator=(const BitRange&) = default;
    BitRange& operator=(BitRange&&) = default;

    // What node these bits come from.
    //
    // NOTE: `source_node()` is not necessarily bits-typed if the bit originated
    // from an aggregate type (tuple or array); see `source_tree_index()`.
    Node* source_node() const { return source_node_; }

    // How many bits wide is this segment.
    int64_t bit_width() const { return bit_width_; }

    // first bit in the source node which is in this bit-segment
    int64_t source_bit_index_low() const { return source_bit_index_low_; }
    int64_t source_bit_index_high() const {
      return source_bit_index_low_ + (bit_width_ - 1);
    }

    // First bit in the final node which has its source as this bit segment
    int64_t dest_bit_index_low() const { return dest_bit_index_low_; }
    int64_t dest_bit_index_high() const {
      return dest_bit_index_low_ + (bit_width_ - 1);
    }

    // Where in the tree of source are these bits from
    absl::Span<const int64_t> source_tree_index() const {
      return source_tree_index_;
    }

    friend bool operator==(const BitRange& x, const BitRange& y) {
      return x.source_node() == y.source_node() &&
             x.source_bit_index_low() == y.source_bit_index_low() &&
             x.dest_bit_index_low() == y.dest_bit_index_low() &&
             x.bit_width() == y.bit_width() &&
             x.source_tree_index() == y.source_tree_index();
    }

    template <typename H>
    friend H AbslHashValue(H h, const BitRange& tbs) {
      return H::combine(std::move(h), tbs.source_node(),
                        tbs.source_bit_index_low(), tbs.bit_width(),
                        tbs.source_tree_index());
    }

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const BitRange& tbs) {
      absl::Format(&sink,
                   "{source_node: %s, source_bit_index_low: %d, "
                   "dest_bit_index_low: %d, bit_width: %d, "
                   "source_tree_index: [%s]}",
                   tbs.source_node()->ToString(), tbs.source_bit_index_low(),
                   tbs.dest_bit_index_low(), tbs.bit_width(),
                   absl::StrJoin(tbs.source_tree_index(), ", "));
    }

   private:
    Node* source_node_;
    int64_t source_bit_index_low_;
    int64_t dest_bit_index_low_;
    int64_t bit_width_;
    std::vector<int64_t> source_tree_index_;
  };

  explicit TreeBitSources(std::vector<BitRange>&& range)
      : bit_ranges_(Minimize(std::move(range))) {}
  TreeBitSources(const TreeBitSources&) = default;
  TreeBitSources(TreeBitSources&&) = default;
  TreeBitSources& operator=(const TreeBitSources&) = default;
  TreeBitSources& operator=(TreeBitSources&&) = default;

  // Get the sources of each bit segment. Segments are sorted from low-bit to
  // high-bit of the result node.
  absl::Span<const BitRange> ranges() const& { return bit_ranges_; }
  TreeBitLocation GetBitSource(int64_t bit_index) const;

  friend bool operator==(const TreeBitSources& x, const TreeBitSources& y) {
    return x.ranges() == y.ranges();
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TreeBitSources& tbs) {
    absl::Format(&sink, "ranges: [%s]", absl::StrJoin(tbs.ranges(), ", "));
  }

 private:
  static std::vector<BitRange> Minimize(std::vector<BitRange>&& orig);

  std::vector<BitRange> bit_ranges_;

  friend class BitProvenanceAnalysis;
};

namespace internal {

class BitProvenanceVisitor;

}  // namespace internal

// A class which provides information about which (if any) node a particular
// single bit of a value comes from. Similar information is also possible to
// obtain from a BDD analysis but this is a significantly simplified analysis
// only concerning itself with tracking bits that vary together precisely.
class BitProvenanceAnalysis {
 public:
  // Create BitProvenanceAnalysis and eagerly evaluate every node in the given
  // function. The analysis does not own or listen to the function and becomes
  // invalid if the function is modified.
  static absl::StatusOr<BitProvenanceAnalysis> CreatePrepopulated(
      FunctionBase* func);
  static absl::StatusOr<BitProvenanceAnalysis> CreatePrepopulated(
      FunctionBase* func, OptimizationContext& context);

  // constructors and destructors need to be declared here and implemented in
  // the .cc file to avoid the compiler inserting constructors and destructors
  // in the header file and then failing to resolve the forward declared type of
  // the BitProvenanceVisitor member field.
  explicit BitProvenanceAnalysis();
  ~BitProvenanceAnalysis();
  BitProvenanceAnalysis(const BitProvenanceAnalysis& other) = delete;
  BitProvenanceAnalysis& operator=(const BitProvenanceAnalysis& other) = delete;
  BitProvenanceAnalysis(BitProvenanceAnalysis&& other);
  BitProvenanceAnalysis& operator=(BitProvenanceAnalysis&& other);

  absl::Status Populate(FunctionBase* func);
  absl::Status Populate(FunctionBase* func, OptimizationContext& context);

  // Get the tree-bit-location which provides the original source of the given
  // bit.
  //
  // NOTE: `TreeBitLocation::node()` is not necessarily bits-typed. If the bit
  // originated from an aggregate type (e.g. tuple or array parameter), `node()`
  // will be that aggregate node with `tree_index()` specifying the path to the
  // leaf element. To obtain a bits-typed Node* representing the bit, use
  // `MaterializeTreeBit`.
  absl::StatusOr<TreeBitLocation> GetSource(const TreeBitLocation& bit) const;

  bool IsTracked(Node* n) const;

  // Get all the sources for a given node.
  absl::StatusOr<LeafTypeTreeView<TreeBitSources>> GetBitSources(Node* n) const;

  // Removes ranges that repeat the first or last bit of a source; useful when
  // intending to operate on the underlying source that was bit-extended.
  static LeafTypeTree<TreeBitSources> TrimRepeatedSourceBits(
      const LeafTypeTreeView<TreeBitSources>& tree);

 private:
  std::unique_ptr<internal::BitProvenanceVisitor> visitor_;
};

// Materializes a 1-bit Bits-typed Node* representing the exact bit at
// `location`.
//
// NOTE: `location.node()` is not necessarily bits-typed (e.g., if the bit
// originated from an aggregate parameter). This function handles compound
// types by traversing `location.tree_index()` with `TupleIndex` / `ArrayIndex`
// nodes down to the leaf element, and creating a `BitSlice` if the leaf element
// is wider than 1 bit. Reuses existing users when available.
absl::StatusOr<Node*> MaterializeTreeBit(const TreeBitLocation& location);

// Materializes a `width`-bit Bits-typed Node* starting at `location`.
// Note: `location.node()` is not necessarily bits-typed.
absl::StatusOr<Node*> MaterializeTreeBitRange(const TreeBitLocation& location,
                                              int64_t width);

// Materializes a Bits-typed Node* representing the given BitRange source.
absl::StatusOr<Node*> MaterializeTreeBitRange(
    const TreeBitSources::BitRange& range);

}  // namespace xls

#endif  // XLS_PASSES_BIT_PROVENANCE_ANALYSIS_H_
