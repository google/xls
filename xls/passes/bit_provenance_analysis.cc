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

#include "xls/passes/bit_provenance_analysis.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/query_engine.h"

namespace xls {
namespace {
using TreeBitRange = TreeBitSources::BitRange;

class BitProvenanceVisitor final : public DataflowVisitor<TreeBitSources> {
 public:
  // TODO(allight): We could do bit-ops (and/or/nor/xor/nand/etc) and even some
  // math-ops plus some other things like DynamicBitSlice and BitSliceUpdate
  // with a query-engine to get at known values (b/c X + 0 has all bits from X,
  // etc). Not clear how useful this would be and it could easily end up getting
  // very complicated however.

  // TODO(allight): With a query engine passed down to DataflowVisitor we could
  // do a better job picking which select branches etc are possible.

  absl::Status DefaultHandler(Node* node) override {
    // Generic node is the sole source of all of its bits.
    XLS_ASSIGN_OR_RETURN(
        auto tree,
        LeafTypeTree<TreeBitSources>::CreateFromFunction(
            node->GetType(),
            [&](Type* t, absl::Span<const int64_t> idx)
                -> absl::StatusOr<TreeBitSources> {
              return TreeBitSources(
                  /*range=*/std::vector<TreeBitRange>{
                      TreeBitRange(
                          /*source_node=*/node, /*source_bit_index_low=*/0,
                          /*dest_bit_index_low=*/0,
                          /*bit_width=*/t->GetFlatBitCount(),
                          /*source_tree_index=*/idx),
                  });
            }));
    return SetValue(node, std::move(tree));
  }

  absl::Status HandleConcat(Concat* concat) override {
    std::vector<TreeBitRange> elements;
    int64_t bit_off = 0;
    // Loop from LSB to MSB
    for (auto it = concat->operands().crbegin();
         it != concat->operands().crend(); ++it) {
      Node* operand = *it;
      XLS_RET_CHECK(operand->GetType()->IsBits());
      const TreeBitSources& elem = GetValue(operand).Get({});
      absl::c_transform(elem.ranges(), std::back_inserter(elements),
                        [&](const TreeBitRange& tbr) -> TreeBitRange {
                          return TreeBitRange(
                              tbr.source_node(), tbr.source_bit_index_low(),
                              tbr.dest_bit_index_low() + bit_off,
                              tbr.bit_width(), tbr.source_tree_index());
                        });
      bit_off += operand->BitCountOrDie();
    }
    return SetValue(
        concat, LeafTypeTree<TreeBitSources>::CreateSingleElementTree(
                    concat->GetType(), TreeBitSources(std::move(elements))));
  }

  absl::Status HandleBitSlice(BitSlice* bs) override {
    const TreeBitSources& arg = GetValue(bs->operand(0)).Get({});
    std::vector<TreeBitRange> elements;
    int64_t remaining_bits = bs->width();
    elements.reserve(arg.ranges().size());
    for (const TreeBitRange& tbr : arg.ranges()) {
      if (remaining_bits == 0) {
        break;
      }
      if (tbr.dest_bit_index_high() < bs->start() ||
          tbr.dest_bit_index_low() >= bs->start() + bs->width()) {
        // Does not overlap
        continue;
      }
      int64_t off = tbr.dest_bit_index_low() < bs->start()
                        ? bs->start() - tbr.dest_bit_index_low()
                        : 0;
      CHECK_LT(off, tbr.bit_width());
      int64_t final_width = std::min(remaining_bits, tbr.bit_width() - off);
      elements.push_back(TreeBitRange(
          tbr.source_node(),
          /*source_bit_index_low=*/tbr.source_bit_index_low() + off,
          /*dest_bit_index_low=*/bs->width() - remaining_bits,
          /*bit_width=*/final_width,
          /*source_tree_index=*/tbr.source_tree_index()));
      remaining_bits -= final_width;
    }
    XLS_RET_CHECK(bs->width() == 0 || !elements.empty()) << "No slice " << bs;
    return SetValue(bs,
                    LeafTypeTree<TreeBitSources>::CreateSingleElementTree(
                        bs->GetType(), TreeBitSources(std::move(elements))));
  }

  // Extend ops take the low bits from the source and the high bits are
  // self-generated.
  absl::Status HandleExtend(ExtendOp* ext) {
    Node* source = ext->operand(0);
    const TreeBitSources& arg = GetValue(source).Get({});
    if (source->BitCountOrDie() == ext->new_bit_count()) {
      return SetValue(
          ext,
          LeafTypeTree<TreeBitSources>::CreateSingleElementTree(
              ext->GetType(), TreeBitSources(std::vector<TreeBitRange>(
                                  arg.ranges().begin(), arg.ranges().end()))));
    }
    // All original sources stay the same
    std::vector<TreeBitRange> elements(arg.ranges().begin(),
                                       arg.ranges().end());
    // Add a new one at the end of the new bits
    elements.push_back(TreeBitRange(
        /*source_node=*/ext, /*source_bit_index_low=*/source->BitCountOrDie(),
        /*dest_bit_index_low=*/source->BitCountOrDie(),
        /*bit_width=*/ext->new_bit_count() - source->BitCountOrDie()));
    return SetValue(ext,
                    LeafTypeTree<TreeBitSources>::CreateSingleElementTree(
                        ext->GetType(), TreeBitSources(std::move(elements))));
  }

  absl::Status HandleSignExtend(ExtendOp* ext) override {
    // TODO(allight): Argurably the source for the new high-bits is the old MSB
    // in sign-extend. It's not clear this would ever really be useful however.
    return HandleExtend(ext);
  }
  absl::Status HandleZeroExtend(ExtendOp* ext) override {
    return HandleExtend(ext);
  }

 protected:
  template <typename Func>
    requires(std::is_invocable_r_v<absl::Status, Func, const TreeBitRange&,
                                   const TreeBitRange&, int64_t, int64_t>)
  absl::Status ForEachSegment(absl::Span<const TreeBitRange> left,
                              absl::Span<const TreeBitRange> right,
                              Func f) const {
    auto it_l = left.begin();
    auto it_r = right.begin();
    XLS_RET_CHECK(it_l != left.end());
    XLS_RET_CHECK(it_r != right.end());
    int64_t low = 0;
    do {
      int64_t high =
          std::min(it_l->dest_bit_index_high(), it_r->dest_bit_index_high());
      XLS_RETURN_IF_ERROR(f(*it_l, *it_r, low, high));
      if (it_l->dest_bit_index_high() == high) {
        ++it_l;
      }
      if (it_r->dest_bit_index_high() == high) {
        ++it_r;
      }
      low = high + 1;
    } while (it_l != left.end() && it_r != right.end());
    // Both must be the same overall length so they should both hit the end at
    // the same time.
    XLS_RET_CHECK(it_l == left.end() && it_r == right.end())
        << "Interseting segments are not the same length.";
    return absl::OkStatus();
  }

  // Intersect the in-progress left span with the expanded version of right.
  absl::StatusOr<std::vector<TreeBitRange>> Intersect(
      absl::Span<const TreeBitRange> left, absl::Span<const TreeBitRange> right,
      Node* result_node, absl::Span<const int64_t> tree_idx = {}) const {
    if (left.empty()) {
      XLS_RET_CHECK(right.empty()) << "Type difference!";
      return std::vector<TreeBitRange>{};
    }
    std::vector<TreeBitRange> res;
    res.reserve(std::max(left.size(), right.size()));
    auto merge_or_add = [&](TreeBitRange tbr) {
      if (res.empty() || res.back().source_node() != tbr.source_node() ||
          res.back().source_bit_index_high() + 1 !=
              tbr.source_bit_index_low() ||
          res.back().source_tree_index() != tbr.source_tree_index()) {
        res.push_back(std::move(tbr));
      } else {
        TreeBitRange new_back(
            /*source_node=*/tbr.source_node(),
            /*source_bit_index_low=*/res.back().source_bit_index_low(),
            /*dest_bit_index_low=*/res.back().dest_bit_index_low(),
            /*bit_width=*/res.back().bit_width() + tbr.bit_width(),
            /*source_tree_index=*/res.back().source_tree_index());
        res.back() = std::move(new_back);
      }
    };
    XLS_RETURN_IF_ERROR(ForEachSegment(
        left, right,
        [&](const TreeBitRange& l, const TreeBitRange& r, int64_t low_bit,
            int64_t high_bit) -> absl::Status {
          int64_t l_off = low_bit - l.dest_bit_index_low();
          int64_t r_off = low_bit - r.dest_bit_index_low();
          if (l.source_node() == r.source_node() &&
              l.source_tree_index() == r.source_tree_index() &&
              l.source_bit_index_low() + l_off ==
                  r.source_bit_index_low() + r_off) {
            // intersection

            merge_or_add(TreeBitRange(
                /*source_node=*/l.source_node(),
                /*source_bit_index_low=*/l.source_bit_index_low() + l_off,
                /*dest_bit_index_low=*/low_bit,
                /*bit_width=*/1 + (high_bit - low_bit),
                /*source_tree_index=*/l.source_tree_index()));
          } else {
            // no intersection. These bits have to source from the current node.
            merge_or_add(TreeBitRange(/*source_node=*/result_node,
                                      /*source_bit_index_low=*/low_bit,
                                      /*dest_bit_index_low*/ low_bit,
                                      /*bit_width=*/1 + (high_bit - low_bit),
                                      /*source_tree_index=*/tree_idx));
          }
          return absl::OkStatus();
        }));
    return res;
  }
  absl::StatusOr<TreeBitSources> JoinElements(
      Type* element_type, absl::Span<const TreeBitSources* const> data_sources,
      absl::Span<const LeafTypeTreeView<TreeBitSources>> control_sources,
      Node* node, absl::Span<const int64_t> index) override {
    // TODO Find overlaps
    std::vector<TreeBitRange> range(data_sources.front()->ranges().begin(),
                                    data_sources.front()->ranges().end());
    for (const TreeBitSources* const s : data_sources.subspan(1)) {
      XLS_ASSIGN_OR_RETURN(range, Intersect(range, s->ranges(), node, index));
    }
    return TreeBitSources(std::move(range));
  }
};

}  // namespace

/* static */ absl::StatusOr<BitProvenanceAnalysis>
BitProvenanceAnalysis::Create(FunctionBase* function) {
  BitProvenanceVisitor prov;
  XLS_RETURN_IF_ERROR(function->Accept(&prov));
  return BitProvenanceAnalysis(std::move(prov).ToStoredValues());
}

TreeBitLocation BitProvenanceAnalysis::GetSource(
    const TreeBitLocation& bit) const {
  CHECK(IsTracked(bit.node())) << bit;
  return GetBitSources(bit.node())
      .Get(bit.tree_index())
      .GetBitSource(bit.bit_index());
}

TreeBitLocation TreeBitSources::GetBitSource(int64_t bit) const {
  auto it = absl::c_upper_bound(bit_ranges_, bit,
                                [](int64_t bit, const TreeBitRange& v) {
                                  return bit < v.dest_bit_index_low();
                                });
  CHECK(it != bit_ranges_.begin());
  auto segment = it - 1;
  CHECK_GE(segment->dest_bit_index_high(), bit);
  CHECK_LE(segment->dest_bit_index_low(), bit);

  int64_t index_off = bit - segment->dest_bit_index_low();

  return TreeBitLocation(
      /*node=*/segment->source_node(),
      /*bit_index=*/segment->source_bit_index_low() + index_off,
      /*tree_index=*/segment->source_tree_index());
}

/* static */ std::vector<TreeBitRange> TreeBitSources::Minimize(
    std::vector<TreeBitRange>&& orig) {
  std::vector<TreeBitRange> res;
  res.reserve(orig.size());
  for (auto it = orig.begin(); it != orig.end(); ++it) {
    if (it->bit_width() == 0) {
      continue;
    }
    if (res.empty() || res.back().source_node() != it->source_node() ||
        res.back().source_bit_index_high() + 1 != it->source_bit_index_low() ||
        res.back().source_tree_index() != it->source_tree_index()) {
      res.push_back(std::move(*it));
    } else {
      TreeBitRange new_back(
          /*source_node=*/it->source_node(),
          /*source_bit_index_low=*/res.back().source_bit_index_low(),
          /*dest_bit_index_low=*/res.back().dest_bit_index_low(),
          /*bit_width=*/res.back().bit_width() + it->bit_width(),
          /*source_tree_index=*/res.back().source_tree_index());
      res.back() = std::move(new_back);
    }
  }
  return res;
}

}  // namespace xls
