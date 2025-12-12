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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
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

// A class which provides information about which (if any) node a particular
// single bit of a value comes from. Similar information is also possible to
// obtain from a BDD analysis but this is a significantly simplified analysis
// only concerning itself with tracking bits that vary together precisely.
class BitProvenanceAnalysis {
 public:
  // Create a provenance analysis.
  static absl::StatusOr<BitProvenanceAnalysis> Create(FunctionBase* function);

  // Get the tree-bit-location which provides the original source of the given
  // bit.
  TreeBitLocation GetSource(const TreeBitLocation& bit) const;

  bool IsTracked(Node* n) const { return sources_.contains(n); }

  // Get all the sources for a given node.
  LeafTypeTreeView<TreeBitSources> GetBitSources(Node* n) const {
    CHECK(IsTracked(n)) << n;
    return sources_.at(n)->AsView();
  }

 private:
  explicit BitProvenanceAnalysis(
      absl::flat_hash_map<
          Node*, std::unique_ptr<SharedLeafTypeTree<TreeBitSources>>>&& sources)
      : sources_(std::move(sources)) {}
  // Map from a node to the nodes which are the source of each of its bits.
  absl::flat_hash_map<Node*,
                      std::unique_ptr<SharedLeafTypeTree<TreeBitSources>>>
      sources_;
};

}  // namespace xls

#endif  // XLS_PASSES_BIT_PROVENANCE_ANALYSIS_H_
