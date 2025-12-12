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

#include "xls/passes/select_simplification_pass.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/data_structures/algorithm.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/partial_info_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Slice out changed bits and store them into a tuple.
struct RemoveUnchangedBits {
  const TreeBitSources& source;
  // What source_node denotes bits that need to be kept.
  Node* changed_bit_src;
  absl::StatusOr<Node*> operator()(Node* src) const {
    XLS_RET_CHECK(changed_bit_src->GetType()->IsBits()) << changed_bit_src;
    XLS_RET_CHECK(src->GetType()->IsBits());
    std::vector<Node*> pieces;
    FunctionBase* fb = src->function_base();
    pieces.reserve(source.ranges().size());
    for (const auto& range : source.ranges()) {
      if (range.source_node() == changed_bit_src) {
        XLS_RET_CHECK_EQ(range.dest_bit_index_low(),
                         range.source_bit_index_low())
            << "Invalid range.";
        XLS_ASSIGN_OR_RETURN(
            Node * sliced,
            fb->MakeNodeWithName<BitSlice>(
                src->loc(), src, range.dest_bit_index_low(), range.bit_width(),
                src->HasAssignedName()
                    ? absl::StrFormat("%s_bits_%d_width_%d", src->GetName(),
                                      range.dest_bit_index_low(),
                                      range.bit_width())
                    : ""));
        pieces.push_back(sliced);
      }
    }
    return fb->MakeNodeWithName<Tuple>(
        src->loc(), pieces,
        src->HasAssignedName() ? absl::StrFormat("%s_squeezed", src->GetName())
                               : "");
  }
};

// Reconstitute the tuple into the real values.
struct RestoreUnchangedBits {
  const TreeBitSources& source;
  // What source_node denotes bits that have been kept.
  Node* changed_bit_src;
  absl::StatusOr<Node*> operator()(Node* src) const {
    XLS_RET_CHECK(changed_bit_src->GetType()->IsBits()) << changed_bit_src;
    FunctionBase* fb = src->function_base();
    std::vector<Node*> concat_args;
    concat_args.reserve(source.ranges().size());
    int64_t piece_count = 0;
    for (const auto& range : source.ranges()) {
      if (range.source_node() == changed_bit_src) {
        Node* sliced;
        XLS_ASSIGN_OR_RETURN(
            sliced, fb->MakeNodeWithName<TupleIndex>(
                        src->loc(), src, piece_count,
                        src->HasAssignedName()
                            ? absl::StrFormat(
                                  "%s_portion_%d_width_%d", src->GetName(),
                                  range.dest_bit_index_low(), range.bit_width())
                            : ""));
        concat_args.push_back(sliced);
        piece_count++;
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * unsliced,
            GetNodeAtIndex(range.source_node(), range.source_tree_index()));
        if (unsliced->BitCountOrDie() == range.bit_width()) {
          concat_args.push_back(unsliced);
        } else {
          XLS_ASSIGN_OR_RETURN(
              Node * sliced,
              fb->MakeNodeWithName<BitSlice>(
                  unsliced->loc(), unsliced, range.source_bit_index_low(),
                  range.bit_width(),
                  unsliced->HasAssignedName()
                      ? absl::StrFormat(
                            "%s_portion_%d_width_%d", unsliced->GetName(),
                            range.source_bit_index_low(), range.bit_width())
                      : ""));
          concat_args.push_back(sliced);
        }
      }
    }
    absl::c_reverse(concat_args);
    return fb->MakeNodeWithName<Concat>(
        src->loc(), concat_args,
        src->HasAssignedName()
            ? absl::StrFormat("%s_unsqueezed", src->GetName())
            : "");
  }
};
struct SqueezeConstantBits {
  const Bits& const_msb;
  const Bits& const_lsb;

  absl::StatusOr<Node*> operator()(Node* src) const {
    return src->function_base()->MakeNodeWithName<BitSlice>(
        src->loc(), src, const_lsb.bit_count(),
        src->BitCountOrDie() - (const_msb.bit_count() + const_lsb.bit_count()),
        src->HasAssignedName() ? absl::StrFormat("%s_squeezed", src->GetName())
                               : "");
  }
};
struct UnsqueezeConstantBits {
  const Bits& const_msb;
  const Bits& const_lsb;
  absl::StatusOr<Node*> operator()(Node* src) const {
    auto fmt_name = [&](std::string_view postfix) -> std::string {
      return src->HasAssignedName() ? absl::StrCat(src->GetName(), postfix)
                                    : "";
    };
    FunctionBase* fb = src->function_base();
    XLS_ASSIGN_OR_RETURN(Node * msbs, fb->MakeNodeWithName<Literal>(
                                          src->loc(), Value(const_msb),
                                          fmt_name("_const_msb_bits")));
    XLS_ASSIGN_OR_RETURN(Node * lsbs, fb->MakeNodeWithName<Literal>(
                                          src->loc(), Value(const_lsb),
                                          fmt_name("_const_lsb_bits")));
    return fb->MakeNodeWithName<Concat>(
        src->loc(), absl::Span<Node* const>{msbs, src, lsbs},
        fmt_name("_unsqueeze"));
  }
};

// Given a SelectT node (either OneHotSelect or Select), squeezes the const_msb
// and const_lsb values out of the output, and slices all the operands to
// correspond to the non-const run of bits in the center.
template <typename SelectT, typename SqueezeF, typename UnsqueezeF,
          typename MakeSelectF>
  requires(std::is_invocable_r_v<absl::StatusOr<Node*>, SqueezeF, Node*> &&
           std::is_invocable_r_v<absl::StatusOr<Node*>, UnsqueezeF, Node*> &&
           std::is_invocable_r_v<absl::StatusOr<Node*>, MakeSelectF, SelectT*,
                                 absl::Span<Node* const>>)
absl::Status SqueezeSelect(SelectT* select, SqueezeF squeeze,
                           UnsqueezeF unsqueeze, MakeSelectF make_select) {
  Node* sel_node = select;
  std::vector<Node*> new_cases;
  new_cases.reserve(select->operands().size() - 1);
  for (Node* n : select->operands().subspan(SelectT::kSelectorOperand + 1)) {
    XLS_ASSIGN_OR_RETURN(Node * squeezed, squeeze(n));
    new_cases.push_back(squeezed);
  }
  XLS_ASSIGN_OR_RETURN(Node * squeezed_sel, make_select(select, new_cases));
  if (!squeezed_sel->HasAssignedName() && sel_node->HasAssignedName()) {
    squeezed_sel->SetName(absl::StrFormat("%s_squeezed", sel_node->GetName()));
  }
  VLOG(2) << absl::StreamFormat("Squeezed select %s into %d bits",
                                select->ToString(),
                                squeezed_sel->GetType()->GetFlatBitCount());
  std::optional<std::string> orig_name =
      sel_node->HasAssignedName() ? std::make_optional(sel_node->GetName())
                                  : std::nullopt;
  XLS_ASSIGN_OR_RETURN(Node * unsqueezed_sel, unsqueeze(squeezed_sel));
  XLS_RETURN_IF_ERROR(sel_node->ReplaceUsesWith(unsqueezed_sel));
  if (orig_name) {
    // Take over name of original select.
    sel_node->ClearName();
    unsqueezed_sel->SetNameDirectly(*orig_name);
  }
  return absl::OkStatus();
}

template <typename SelectT>
absl::StatusOr<SelectT*> MakeSelect(SelectT* original,
                                    absl::Span<Node* const> new_cases) {
  if constexpr (std::is_same_v<Select, SelectT>) {
    std::optional<Node*> new_default;

    if (original->default_value().has_value()) {
      new_default = new_cases.back();
      new_cases = new_cases.subspan(0, new_cases.size() - 1);
    }
    return original->function_base()->template MakeNode<Select>(
        original->loc(), original->selector(), new_cases, new_default);
  }
  if constexpr (std::is_same_v<OneHotSelect, SelectT>) {
    return original->function_base()->template MakeNode<OneHotSelect>(
        original->loc(), original->selector(), new_cases);
  }
  if constexpr (std::is_same_v<PrioritySelect, SelectT>) {
    Node* new_default = new_cases.back();
    new_cases = new_cases.subspan(0, new_cases.size() - 1);
    return original->function_base()->template MakeNode<PrioritySelect>(
        original->loc(), original->selector(), new_cases, new_default);
  }
  return absl::UnimplementedError("Not a select");
}

template <typename SelectT>
  requires(std::is_same_v<SelectT, Select> ||
           std::is_same_v<SelectT, OneHotSelect> ||
           std::is_same_v<SelectT, PrioritySelect>)
absl::StatusOr<bool> TrySqueezeSelect(SelectT* sel,
                                      const QueryEngine& query_engine,
                                      const BitProvenanceAnalysis& provenance,
                                      bool with_range_analysis) {
  Node* node = sel;
  // If we have range analysis check for signed reduction. We want to see if the
  // values are all around signed-zero
  int64_t min_signed_size;
  if (with_range_analysis) {
    min_signed_size = interval_ops::MinimumSignedBitCount(
        query_engine.GetIntervals(node).Get({}));
  } else {
    // If we don't have range analysis this will at best be the same as
    // ternary-based constant leading-trailing bits analysis. Therefore, don't
    // bother to do the expensive interval calculation.
    min_signed_size = node->BitCountOrDie();
  }
  // Figure out common known constant MSB & LSB bits
  auto is_squeezable_mux = [&](Bits* msb, Bits* lsb) {
    std::optional<SharedLeafTypeTree<TernaryVector>> ternary =
        query_engine.GetTernary(node);
    if (!ternary.has_value()) {
      return false;
    }
    TernaryVector ternary_vec = ternary->Get({});
    Bits known_bits = ternary_ops::ToKnownBits(ternary_vec);
    int64_t leading_known = bits_ops::CountLeadingOnes(known_bits);
    int64_t trailing_known = bits_ops::CountTrailingOnes(known_bits);
    if (leading_known == 0 && trailing_known == 0) {
      return false;
    }
    int64_t bit_count = node->BitCountOrDie();
    *msb = ternary_ops::ToKnownBitsValues(ternary_vec)
               .Slice(/*start=*/bit_count - leading_known,
                      /*width=*/leading_known);
    if (leading_known == trailing_known && leading_known == bit_count) {
      // This is just a constant value, just say we only have high constant
      // bits, the replacement will be the same.
      return true;
    }
    *lsb = ternary_ops::ToKnownBitsValues(ternary_vec)
               .Slice(/*start=*/0, /*width=*/trailing_known);
    return true;
  };
  Bits const_msb, const_lsb;
  bool const_squeezable = is_squeezable_mux(&const_msb, &const_lsb);
  int64_t const_mux_width =
      node->BitCountOrDie() - (const_msb.bit_count() + const_lsb.bit_count());
  if (const_mux_width == 0) {
    // Just a constant. Other narrowing will handle this.
    return false;
  }
  const TreeBitSources& prov_bits = provenance.GetBitSources(node).Get({});
  int64_t prov_width = absl::c_accumulate(
      prov_bits.ranges(), int64_t{0},
      [&](int64_t v, const TreeBitSources::BitRange& r) -> int64_t {
        if (r.source_node() == node) {
          return v + r.bit_width();
        }
        return v;
      });
  if ((!const_squeezable || const_mux_width == node->BitCountOrDie()) &&
      prov_width == node->BitCountOrDie() &&
      min_signed_size == node->BitCountOrDie()) {
    // Can't narrow
    return false;
  }
  // Technically we could do all 3 (since these might narrow different bits) but
  // (1) that makes whichever goes second more complicated and (2) this pass is
  // run in fixed-point (for non-range analysis) anyway so all it will do is
  // save a pass-group run. Hopefully one range based run will suffice since it
  // is too slow to run in a fixed-point.
  //
  // When multiple choices have the same net effect, const-mux is prioritized
  // mostly to avoid having to rewrite all the tests for it. Next provenance is
  // used finally we use sign-reduction.
  enum class SqueezeType : int8_t {
    kConstants = 1,
    kBitProvenance = 2,
    kSignExtend = 3
  };
  struct SqueezeOption {
    int64_t bit_count;
    SqueezeType type;
  };
  auto options = std::to_array<SqueezeOption>(
      {{.bit_count = const_mux_width, .type = SqueezeType::kConstants},
       {.bit_count = prov_width, .type = SqueezeType::kBitProvenance},
       {.bit_count = min_signed_size, .type = SqueezeType::kSignExtend}});
  SqueezeType option =
      absl::c_min_element(options, [](const SqueezeOption& l,
                                      const SqueezeOption& r) {
        return l.bit_count < r.bit_count ||
               (l.bit_count == r.bit_count &&
                static_cast<int8_t>(l.type) < static_cast<int8_t>(r.type));
      })->type;

  VLOG(3) << "Options of squeeze for " << sel
          << " are: mux: " << const_mux_width << ", provenance: " << prov_width
          << " sign_ext: " << min_signed_size;
  switch (option) {
    case SqueezeType::kConstants:
      VLOG(2) << "Squeezing select using constants : " << sel << " to "
              << const_mux_width << " bits";
      XLS_RETURN_IF_ERROR(SqueezeSelect(
          sel,
          SqueezeConstantBits{.const_msb = const_msb, .const_lsb = const_lsb},
          UnsqueezeConstantBits{.const_msb = const_msb, .const_lsb = const_lsb},
          MakeSelect<SelectT>));
      break;
    case SqueezeType::kBitProvenance:
      VLOG(2) << "Squeezing select using bit-prov: " << sel << " to "
              << prov_width << " bits";
      XLS_RETURN_IF_ERROR(SqueezeSelect(
          sel, RemoveUnchangedBits{.source = prov_bits, .changed_bit_src = sel},
          RestoreUnchangedBits{.source = prov_bits, .changed_bit_src = sel},
          MakeSelect<SelectT>));
      break;
    case SqueezeType::kSignExtend:
      VLOG(2) << "Squeezing select using sign-ext: " << sel << " to "
              << min_signed_size << " bits";
      XLS_RETURN_IF_ERROR(SqueezeSelect(
          sel,
          [&](Node* src) -> absl::StatusOr<Node*> {
            return src->function_base()->MakeNodeWithName<BitSlice>(
                src->loc(), src, /*start=*/0, /*width=*/min_signed_size,
                src->HasAssignedName()
                    ? absl::StrFormat("%s_squeezed", src->GetName())
                    : "");
          },
          [&](Node* src) -> absl::StatusOr<Node*> {
            return src->function_base()->MakeNodeWithName<ExtendOp>(
                src->loc(), src, node->BitCountOrDie(), Op::kSignExt,
                src->HasAssignedName()
                    ? absl::StrFormat("%s_unsqueezed", src->GetName())
                    : "");
          },
          MakeSelect<SelectT>));
      break;
  }
  return true;
}

// The source of a bit. Can be either a literal 0/1 or a bit at a particular
// index of a Node.
using BitSource = std::variant<bool, TreeBitLocation>;

// Traces the bit at the given node and bit index through bit slices and concats
// and returns its source.
BitSource GetBitSource(Node* node, int64_t bit_index,
                       const QueryEngine& query_engine,
                       const BitProvenanceAnalysis& provenance) {
  if (node->GetType()->IsBits() &&
      query_engine.IsKnown(TreeBitLocation(node, bit_index))) {
    return query_engine.IsOne(TreeBitLocation(node, bit_index));
  }
  if (provenance.IsTracked(node)) {
    return provenance.GetSource(TreeBitLocation(node, bit_index));
  }
  // Not able to find anything, maybe this is because other changes invalidated
  // the provenance information but in that case we should have optimized
  // already anyway.
  return TreeBitLocation(node, bit_index);
}

std::string ToString(const BitSource& bit_source) {
  return absl::visit(Visitor{[](bool value) { return absl::StrCat(value); },
                             [](const TreeBitLocation& p) {
                               return absl::StrFormat("%s[%d]",
                                                      p.node()->GetName(),
                                                      p.bit_index());
                             }},
                     bit_source);
}

using MatchedPairs = std::vector<std::pair<int64_t, int64_t>>;

// Returns the pairs of indices into 'nodes' for which the indexed Nodes have
// the same of bits sources at the given bit index. The returned indices are
// indices into the given 'nodes' span. For example, given the following:
//
//  GetBitSource(a, 42) = BitSource{true}
//  GetBitSource(b, 42) = BitSource{foo, 7}
//  GetBitSource(c, 42) = BitSource{foo, 7}
//  GetBitSource(d, 42) = BitSource{true}
//  GetBitSource(e, 42) = BitSource{false}
//
// PairsOfBitsWithSameSource({a, b, c, d, e}, 42) would return [(0, 3), (1, 2)]
MatchedPairs PairsOfBitsWithSameSource(
    absl::Span<Node* const> nodes, int64_t bit_index,
    const QueryEngine& query_engine, const BitProvenanceAnalysis& provenance) {
  std::vector<BitSource> bit_sources;
  for (Node* node : nodes) {
    bit_sources.push_back(
        GetBitSource(node, bit_index, query_engine, provenance));
  }
  MatchedPairs matching_pairs;
  for (int64_t i = 0; i < bit_sources.size(); ++i) {
    for (int64_t j = i + 1; j < bit_sources.size(); ++j) {
      if (bit_sources[i] == bit_sources[j]) {
        matching_pairs.push_back({i, j});
      }
    }
  }
  return matching_pairs;
}

std::string ToString(const MatchedPairs& pairs) {
  std::string ret;
  for (const auto& p : pairs) {
    absl::StrAppend(&ret, "(", p.first, ", ", p.second, ") ");
  }
  return ret;
}

// Checks if two comparison nodes are opposite to each other for any way they
// are arranged.
bool CanSimplifyThreeWayCompare(Node* first_compare, Node* second_compare,
                                const QueryEngine& query_engine,
                                bool inversed_compares = false) {
  // The second compare must only be used once otherwise using an EqualOp would
  // just add a component.
  if (second_compare->users().size() != 1) {
    return false;
  }
  // The compares must have two operands.
  if (first_compare->operand_count() != 2 ||
      second_compare->operand_count() != 2) {
    return false;
  }
  Node* first_compare_lhs = first_compare->operands()[0];
  Node* first_compare_rhs = first_compare->operands()[1];
  Node* second_compare_lhs = second_compare->operands()[0];
  Node* second_compare_rhs = second_compare->operands()[1];
  // The compares must contain only bits operands.
  if (!first_compare_lhs->GetType()->IsBits() ||
      !first_compare_rhs->GetType()->IsBits() ||
      !second_compare_lhs->GetType()->IsBits() ||
      !second_compare_rhs->GetType()->IsBits()) {
    return false;
  }
  std::vector<Op> gt_ops(2);
  std::vector<Op> lt_ops(2);
  // Use equality comparisons if the compares should be inverted.
  if (inversed_compares) {
    gt_ops = {Op::kUGe, Op::kSGe};
    lt_ops = {Op::kULe, Op::kSLe};
  } else {
    gt_ops = {Op::kUGt, Op::kSGt};
    lt_ops = {Op::kULt, Op::kSLt};
  }
  auto is_gt_op = [&](Node* node) { return node->OpIn(gt_ops); };
  auto is_lt_op = [&](Node* node) { return node->OpIn(lt_ops); };
  // Check if the nodes operands are equal to each other in order.
  bool operands_equal = query_engine.NodesKnownUnsignedEquals(
                            first_compare_lhs, second_compare_lhs) &&
                        query_engine.NodesKnownUnsignedEquals(
                            first_compare_rhs, second_compare_rhs);
  // Check if the nodes operands are equal to each other in flipped order.
  bool flipped_operands_equal = query_engine.NodesKnownUnsignedEquals(
                                    first_compare_lhs, second_compare_rhs) &&
                                query_engine.NodesKnownUnsignedEquals(
                                    first_compare_rhs, second_compare_lhs);
  if (is_gt_op(first_compare) && is_lt_op(second_compare)) {
    return operands_equal;
  }
  if (is_lt_op(first_compare) && is_gt_op(second_compare)) {
    return operands_equal;
  }
  if (is_gt_op(first_compare) && is_gt_op(second_compare)) {
    return flipped_operands_equal;
  }
  if (is_lt_op(first_compare) && is_lt_op(second_compare)) {
    return flipped_operands_equal;
  }
  return false;
}

// Simplify the select if it is a three way compare.
absl::StatusOr<bool> TrySimplifyThreeWayCompareSelect(
    Select* sel, const QueryEngine& query_engine) {
  if (IsBinarySelect(sel->cases()[0])) {
    // sel(ugt, [sel(ult, [eq_result, ult_result]), ugt_result]) becomes
    // sel(ugt, [sel(eq, [ult_result, eq_result]), ugt_result])
    Select* sub_sel = sel->cases()[0]->As<Select>();
    Node* first_compare = sel->selector();
    Node* second_compare = sub_sel->selector();
    // Make sure the sub select is only used once to avoid duplicates.
    if (sub_sel->users().size() != 1 ||
        !CanSimplifyThreeWayCompare(first_compare, second_compare,
                                    query_engine)) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq, sel->function_base()->MakeNode<CompareOp>(
                           second_compare->loc(), second_compare->operands()[0],
                           second_compare->operands()[1], Op::kEq));
    XLS_ASSIGN_OR_RETURN(
        Node * new_sub_sel,
        sel->function_base()->MakeNode<Select>(
            sub_sel->loc(), new_eq,
            std::vector<Node*>{sub_sel->cases()[1], sub_sel->cases()[0]},
            std::nullopt));
    // Replace all uses of the select with the new select because they are
    // equivalent.
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<Select>(
                               sel->selector(),
                               std::vector<Node*>{new_sub_sel, sel->cases()[1]},
                               std::nullopt)
                            .status());
    return true;
  } else if (IsBinarySelect(sel->cases()[1])) {
    // sel(ule, [ugt_result, sel(uge, [ult_result, eq_result])) becomes
    // sel(ule, [ugt_result, sel(eq, [ult_result, eq_result])])
    Select* sub_sel = sel->cases()[1]->As<Select>();
    Node* first_compare = sel->selector();
    Node* second_compare = sub_sel->selector();
    if (sub_sel->users().size() != 1 ||
        !CanSimplifyThreeWayCompare(first_compare, second_compare, query_engine,
                                    /*inversed_compares=*/true)) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq, sel->function_base()->MakeNode<CompareOp>(
                           second_compare->loc(), second_compare->operands()[0],
                           second_compare->operands()[1], Op::kEq));
    XLS_ASSIGN_OR_RETURN(
        Node * new_sub_sel,
        sel->function_base()->MakeNode<Select>(
            sub_sel->loc(), new_eq,
            std::vector<Node*>{sub_sel->cases()[0], sub_sel->cases()[1]},
            std::nullopt));
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<Select>(
                               sel->selector(),
                               std::vector<Node*>{sel->cases()[0], new_sub_sel},
                               std::nullopt)
                            .status());
    return true;
  }
  return false;
}

// Simplify the priority select if it is a three way compare.
absl::StatusOr<bool> TrySimplifyThreeWayComparePrioritySelect(
    PrioritySelect* sel, const QueryEngine& query_engine) {
  Node* selector = sel->selector();
  if (selector->op() == Op::kConcat && selector->operand_count() == 2) {
    // priority_sel(concat(ugt, ult), [ult_result, ugt_result], eq_result)
    // becomes
    // priority_sel(concat(ugt, eq), [eq_result, ugt_result], ult_result)
    Node* first_compare = selector->operands()[0];
    Node* second_compare = selector->operands()[1];
    if (selector->users().size() != 1 ||
        !CanSimplifyThreeWayCompare(first_compare, second_compare,
                                    query_engine)) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq, sel->function_base()->MakeNode<CompareOp>(
                           second_compare->loc(), second_compare->operands()[0],
                           second_compare->operands()[1], Op::kEq));
    XLS_ASSIGN_OR_RETURN(
        Node * new_concat,
        sel->function_base()->MakeNode<Concat>(
            selector->loc(), std::vector<Node*>{first_compare, new_eq}));
    XLS_RETURN_IF_ERROR(
        sel->ReplaceUsesWithNew<PrioritySelect>(
               new_concat,
               std::vector<Node*>{sel->default_value(), sel->cases()[1]},
               sel->cases()[0])
            .status());
    return true;
  } else if (IsBinaryPrioritySelect(sel) &&
             IsBinaryPrioritySelect(sel->default_value())) {
    // priority_sel(ugt, [ugt_result],
    //              priority_sel(ult, [ult_result], eq_result)) becomes
    // priority_sel(ugt, [ugt_result],
    //              priority_sel(eq, [eq_result], ult_result))
    PrioritySelect* sub_sel = sel->default_value()->As<PrioritySelect>();
    Node* first_compare = selector;
    Node* second_compare = sub_sel->selector();
    if (sub_sel->users().size() != 1 ||
        !CanSimplifyThreeWayCompare(first_compare, second_compare,
                                    query_engine)) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq, sel->function_base()->MakeNode<CompareOp>(
                           second_compare->loc(), second_compare->operands()[0],
                           second_compare->operands()[1], Op::kEq));
    XLS_ASSIGN_OR_RETURN(
        Node * new_sub_sel,
        sel->function_base()->MakeNode<PrioritySelect>(
            sub_sel->loc(), new_eq,
            std::vector<Node*>{sub_sel->default_value()}, sub_sel->cases()[0]));
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<PrioritySelect>(
                               sel->selector(),
                               std::vector<Node*>{sel->cases()[0]}, new_sub_sel)
                            .status());
    return true;
  } else if (IsBinaryPrioritySelect(sel) &&
             IsBinaryPrioritySelect(sel->cases()[0])) {
    // priority_sel(ule,
    //              [priority_sel(uge, [eq_result], ult_result)], ugt_result)
    // becomes
    // priority_sel(ule,
    //              [priority_sel(eq, [eq_result], ult_result)], ugt_result)
    PrioritySelect* sub_sel = sel->cases()[0]->As<PrioritySelect>();
    Node* first_compare = selector;
    Node* second_compare = sub_sel->selector();
    if (!CanSimplifyThreeWayCompare(first_compare, second_compare, query_engine,
                                    /*inversed_compares=*/true)) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * new_eq, sel->function_base()->MakeNode<CompareOp>(
                           second_compare->loc(), second_compare->operands()[0],
                           second_compare->operands()[1], Op::kEq));
    XLS_ASSIGN_OR_RETURN(
        Node * new_sub_sel,
        sel->function_base()->MakeNode<PrioritySelect>(
            sub_sel->loc(), new_eq, std::vector<Node*>{sub_sel->cases()[0]},
            sub_sel->default_value()));
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<PrioritySelect>(
                               sel->selector(), std::vector<Node*>{new_sub_sel},
                               sel->default_value())
                            .status());
    return true;
  }
  return false;
}

// Returns a bit-based select instruction which selects a slice of the given
// bit-based select's cases. The cases are sliced with the given start and width
// and then selected with a new bit-based select which is returned.
absl::StatusOr<Node*> SliceBitBasedSelect(Node* bbs, int64_t start,
                                          int64_t width) {
  Node* selector;
  absl::Span<Node* const> cases;
  std::optional<Node*> default_value = std::nullopt;
  if (bbs->Is<OneHotSelect>()) {
    selector = bbs->As<OneHotSelect>()->selector();
    cases = bbs->As<OneHotSelect>()->cases();
  } else {
    XLS_RET_CHECK(bbs->Is<PrioritySelect>());
    selector = bbs->As<PrioritySelect>()->selector();
    cases = bbs->As<PrioritySelect>()->cases();
    default_value = bbs->As<PrioritySelect>()->default_value();
  }

  std::vector<Node*> case_slices;
  std::optional<Node*> default_value_slice = std::nullopt;
  for (Node* cas : cases) {
    XLS_ASSIGN_OR_RETURN(Node * case_slice,
                         bbs->function_base()->MakeNode<BitSlice>(
                             bbs->loc(), cas, /*start=*/start,
                             /*width=*/width));
    case_slices.push_back(case_slice);
  }
  if (default_value.has_value()) {
    XLS_ASSIGN_OR_RETURN(default_value_slice,
                         bbs->function_base()->MakeNode<BitSlice>(
                             bbs->loc(), *default_value, /*start=*/start,
                             /*width=*/width));
  }

  std::string new_bbs_name;
  if (bbs->HasAssignedName()) {
    new_bbs_name =
        absl::StrFormat("%s__%d_to_%d", bbs->GetName(), start, start + width);
  }
  if (bbs->Is<OneHotSelect>()) {
    XLS_RET_CHECK(!default_value_slice.has_value());
    return bbs->function_base()->MakeNodeWithName<OneHotSelect>(
        bbs->loc(), selector, case_slices, new_bbs_name);
  }
  XLS_RET_CHECK(bbs->Is<PrioritySelect>());
  XLS_RET_CHECK(default_value_slice.has_value());
  return bbs->function_base()->MakeNodeWithName<PrioritySelect>(
      bbs->loc(), selector, case_slices, *default_value_slice, new_bbs_name);
}

// Returns the length of the run of bit indices starting at 'start' for which
// there exists at least one pair of elements in 'cases' which have the same bit
// source at the respective bit indices in the entire run. For example, given
// the following
//
//   a = Literal(value=0b110011)
//   b = Literal(value=0b100010)
//   c = Literal(value=0b101010)
//
// RunOfNonDistinctCaseBits({a, b, c}, 1) returns 3 because bits 1, 2, and 3 of
// 'a', and 'b' are the same (have the same BitSource).
int64_t RunOfNonDistinctCaseBits(absl::Span<Node* const> cases, int64_t start,
                                 const QueryEngine& query_engine,
                                 const BitProvenanceAnalysis& provenance) {
  VLOG(5) << "Finding runs of non-distinct bits starting at " << start;
  // Do a reduction via intersection of the set of matching pairs within
  // 'cases'. When the intersection is empty, the run is over.
  MatchedPairs matches;
  int64_t i = start;
  while (i < cases.front()->BitCountOrDie()) {
    if (i == start) {
      matches = PairsOfBitsWithSameSource(cases, i, query_engine, provenance);
    } else {
      MatchedPairs new_matches;
      absl::c_set_intersection(
          PairsOfBitsWithSameSource(cases, i, query_engine, provenance),
          matches, std::back_inserter(new_matches));
      matches = std::move(new_matches);
    }

    VLOG(5) << "  " << i << ": " << ToString(matches);
    if (matches.empty()) {
      break;
    }
    ++i;
  }
  VLOG(5) << " run of " << i - start;
  return i - start;
}

// Returns the length of the run of bit indices starting at 'start' for which
// the indexed bits of the given cases are distinct at each
// bit index. For example:
int64_t RunOfDistinctCaseBits(absl::Span<Node* const> cases, int64_t start,
                              const QueryEngine& query_engine,
                              const BitProvenanceAnalysis& provenance) {
  VLOG(5) << "Finding runs of distinct case bit starting at " << start;
  int64_t i = start;
  while (
      i < cases.front()->BitCountOrDie() &&
      PairsOfBitsWithSameSource(cases, i, query_engine, provenance).empty()) {
    ++i;
  }
  VLOG(5) << " run of " << i - start << " bits";
  return i - start;
}

// Try to split OneHotSelect/PrioritySelect instructions into separate
// instructions which have common cases. For example, if some of the cases of a
// OneHotSelect have the same first three bits, then this transformation will
// slice off these three bits (and the remainder) into separate OneHotSelect
// operation and replace the original OneHotSelect with a concat of the sharded
// OneHotSelects.
//
// Returns the newly created bit-based select instructions if the transformation
// succeeded.
absl::StatusOr<std::vector<Node*>> MaybeSplitBitBasedSelect(
    Node* bbs, const QueryEngine& query_engine,
    const BitProvenanceAnalysis& provenance) {
  XLS_RET_CHECK(bbs->Is<OneHotSelect>() || bbs->Is<PrioritySelect>());
  // For *very* wide one-hot-selects this optimization can be very slow and make
  // a mess of the graph so limit it to 64 bits.
  if (!bbs->GetType()->IsBits() || bbs->GetType()->GetFlatBitCount() > 64) {
    return std::vector<Node*>();
  }
  std::vector<Node*> bbs_cases;
  if (bbs->Is<OneHotSelect>()) {
    absl::c_copy(bbs->As<OneHotSelect>()->cases(),
                 std::back_inserter(bbs_cases));
  } else {
    XLS_RET_CHECK(bbs->Is<PrioritySelect>());
    absl::c_copy(bbs->As<PrioritySelect>()->cases(),
                 std::back_inserter(bbs_cases));
    bbs_cases.push_back(bbs->As<PrioritySelect>()->default_value());
  }

  VLOG(4) << "Trying to split: " << bbs->ToString();
  if (VLOG_IS_ON(4)) {
    for (int64_t i = 0; i < bbs_cases.size(); ++i) {
      Node* cas = bbs_cases[i];
      VLOG(4) << "  case (" << i << "): " << cas->ToString();
      for (int64_t j = 0; j < cas->BitCountOrDie(); ++j) {
        VLOG(4) << "    bit " << j << ": "
                << ToString(GetBitSource(cas, j, query_engine, provenance));
      }
    }
  }

  int64_t start = 0;
  std::vector<Node*> bbs_slices;
  std::vector<Node*> new_bbses;
  while (start < bbs->BitCountOrDie()) {
    int64_t run =
        RunOfDistinctCaseBits(bbs_cases, start, query_engine, provenance);
    if (run == 0) {
      run =
          RunOfNonDistinctCaseBits(bbs_cases, start, query_engine, provenance);
    }
    XLS_RET_CHECK_GT(run, 0);
    if (run == bbs->BitCountOrDie()) {
      // If all the cases are distinct (or have a matching pair) then just
      // return as there is nothing to slice.
      return std::vector<Node*>();
    }
    XLS_ASSIGN_OR_RETURN(Node * bbs_slice, SliceBitBasedSelect(bbs,
                                                               /*start=*/start,
                                                               /*width=*/run));
    new_bbses.push_back(bbs_slice);
    bbs_slices.push_back(bbs_slice);
    start += run;
  }
  std::reverse(bbs_slices.begin(), bbs_slices.end());
  VLOG(2) << absl::StrFormat("Splitting bit-based-select: %s", bbs->ToString());
  XLS_RETURN_IF_ERROR(bbs->ReplaceUsesWithNew<Concat>(bbs_slices).status());
  return new_bbses;
}

// Any type of select with only one non-literal-zero arm can be replaced with
// an AND.
//
//  sel(p, cases=[x, 0]) => and(sign_ext(p == 0), x)
//  sel(p, cases=[0, x]) => and(sign_ext(p == 1), x)
//  one_hot_select(p, cases=[x, 0]) => and(sign_ext(p[0]), x)
//  one_hot_select(p, cases=[0, x]) => and(sign_ext(p[1]), x)
//  priority_select(p, cases=[x, 0],
//                     default=0)   => and(sign_ext(p[0]), x)
//  priority_select(p, cases=[0, x],
//                     default=0)   => and(sign_ext(p == 2), x)
//
//  sel(p, cases=[x], default_value=0)  => and(sign_ext(p == 0), x)
//  one_hot_select(p, cases=[x])        => and(sign_ext(p[0]), x)
//  priority_select(p, cases=[x],
//                     default=0)       => and(sign_ext(p), x)
//
//  sel(p, cases=[0], default_value=x) => and(sign_ext(p != 0), x)
//  priority_select(p, cases=[0],
//                     default=x)      => and(sign_ext(p != 0), x)
//
// If the result is not bits-typed, we can still reduce it to a two-arm select
// against a literal zero. (If a non-bits-typed select only has two arms,
// there's no benefit, so we won't simplify the node.)
//
absl::StatusOr<bool> MaybeConvertSelectToMask(Node* node,
                                              const QueryEngine& query_engine) {
  if (!node->OpIn({Op::kSel, Op::kOneHotSel, Op::kPrioritySel})) {
    return false;
  }
  if (!node->GetType()->IsBits() && node->operands().size() <= 3) {
    // We already have a select with at most two arms; we can't simplify this
    // any further for non-bits-typed operands.
    return false;
  }

  std::optional<Node*> only_nonzero_value = std::nullopt;
  Node* nonzero_condition = nullptr;
  switch (node->op()) {
    default:
      return false;
    case Op::kSel: {
      Select* sel = node->As<Select>();
      std::optional<int64_t> nonzero_arm = std::nullopt;
      if (sel->default_value().has_value() &&
          !query_engine.IsAllZeros(*sel->default_value())) {
        nonzero_arm = -1;
        only_nonzero_value = sel->default_value();
      }
      for (int64_t arm = 0; arm < sel->cases().size(); ++arm) {
        Node* case_value = sel->get_case(arm);
        if (query_engine.IsAllZeros(case_value)) {
          continue;
        }
        if (only_nonzero_value.has_value()) {
          // More than one non-zero value;
          return false;
        }

        nonzero_arm = arm;
        only_nonzero_value = case_value;
      }
      if (nonzero_arm.has_value()) {
        VLOG(2) << absl::StrFormat("Select with one non-zero case: %s",
                                   node->ToString());
        if (*nonzero_arm == -1) {
          XLS_ASSIGN_OR_RETURN(
              Node * num_cases,
              node->function_base()->MakeNode<Literal>(
                  node->loc(), Value(UBits(sel->cases().size(),
                                           sel->selector()->BitCountOrDie()))));
          XLS_ASSIGN_OR_RETURN(
              nonzero_condition,
              node->function_base()->MakeNode<CompareOp>(
                  sel->loc(), sel->selector(), num_cases, Op::kUGe));
        } else if (sel->selector()->BitCountOrDie() == 1) {
          if (*nonzero_arm == 0) {
            XLS_ASSIGN_OR_RETURN(nonzero_condition,
                                 node->function_base()->MakeNode<UnOp>(
                                     sel->loc(), sel->selector(), Op::kNot));
          } else {
            XLS_RET_CHECK_EQ(*nonzero_arm, 1);
            nonzero_condition = sel->selector();
          }
        } else {
          XLS_ASSIGN_OR_RETURN(
              Node * arm_number,
              node->function_base()->MakeNode<Literal>(
                  node->loc(), Value(UBits(*nonzero_arm,
                                           sel->selector()->BitCountOrDie()))));
          XLS_ASSIGN_OR_RETURN(
              nonzero_condition,
              node->function_base()->MakeNode<CompareOp>(
                  sel->loc(), sel->selector(), arm_number, Op::kEq));
        }
      }
      break;
    }
    case Op::kOneHotSel: {
      OneHotSelect* sel = node->As<OneHotSelect>();
      std::optional<int64_t> nonzero_arm = std::nullopt;
      for (int64_t arm = 0; arm < sel->cases().size(); ++arm) {
        Node* case_value = sel->get_case(arm);
        if (query_engine.IsAllZeros(case_value)) {
          continue;
        }
        if (only_nonzero_value.has_value()) {
          // More than one non-zero value;
          return false;
        }

        nonzero_arm = arm;
        only_nonzero_value = case_value;
      }
      if (nonzero_arm.has_value()) {
        VLOG(2) << absl::StrFormat("One-hot select with one non-zero case: %s",
                                   node->ToString());
        if (sel->selector()->BitCountOrDie() == 1) {
          XLS_RET_CHECK_EQ(*nonzero_arm, 0);
          nonzero_condition = sel->selector();
        } else {
          XLS_ASSIGN_OR_RETURN(
              nonzero_condition,
              node->function_base()->MakeNode<BitSlice>(
                  sel->loc(), sel->selector(), /*start=*/*nonzero_arm,
                  /*width=*/1));
        }
      }
      break;
    }
    case Op::kPrioritySel: {
      PrioritySelect* sel = node->As<PrioritySelect>();
      std::optional<int64_t> nonzero_arm = std::nullopt;
      if (!query_engine.IsAllZeros(sel->default_value())) {
        nonzero_arm = -1;
        only_nonzero_value = sel->default_value();
      }
      for (int64_t arm = 0; arm < sel->cases().size(); ++arm) {
        Node* case_value = sel->get_case(arm);
        if (query_engine.IsAllZeros(case_value)) {
          continue;
        }
        if (only_nonzero_value.has_value()) {
          // More than one non-zero value;
          return false;
        }

        nonzero_arm = arm;
        only_nonzero_value = case_value;
      }
      if (nonzero_arm.has_value()) {
        VLOG(2) << absl::StrFormat("Priority select with one non-zero case: %s",
                                   node->ToString());
        Node* truncated_selector;
        if (*nonzero_arm == -1 ||
            *nonzero_arm == sel->selector()->BitCountOrDie() - 1) {
          truncated_selector = sel->selector();
        } else {
          XLS_ASSIGN_OR_RETURN(truncated_selector,
                               node->function_base()->MakeNode<BitSlice>(
                                   sel->loc(), sel->selector(), /*start=*/0,
                                   /*width=*/*nonzero_arm + 1));
        }
        if (*nonzero_arm == 0) {
          nonzero_condition = truncated_selector;
        } else if (*nonzero_arm > 0) {
          XLS_ASSIGN_OR_RETURN(
              Node * matching_value,
              node->function_base()->MakeNode<Literal>(
                  sel->loc(),
                  Value(Bits::PowerOfTwo(*nonzero_arm, *nonzero_arm + 1))));
          XLS_ASSIGN_OR_RETURN(
              nonzero_condition,
              node->function_base()->MakeNode<CompareOp>(
                  sel->loc(), truncated_selector, matching_value, Op::kEq));
        } else {
          XLS_RET_CHECK_EQ(*nonzero_arm, -1);
          XLS_ASSIGN_OR_RETURN(
              Node * selector_zero,
              node->function_base()->MakeNode<Literal>(
                  sel->loc(),
                  Value(UBits(0, sel->selector()->BitCountOrDie()))));
          XLS_ASSIGN_OR_RETURN(
              nonzero_condition,
              node->function_base()->MakeNode<CompareOp>(
                  sel->loc(), sel->selector(), selector_zero, Op::kEq));
        }
      }
      break;
    }
  }

  if (!only_nonzero_value.has_value()) {
    // The select can't return any non-zero value.
    VLOG(2) << absl::StrFormat("select with no non-zero cases: %s",
                               node->ToString());
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<Literal>(ZeroOfType(node->GetType()))
            .status());
    return true;
  }

  XLS_RET_CHECK_NE(nonzero_condition, nullptr);
  if (node->GetType()->IsBits()) {
    Node* mask;
    if (node->BitCountOrDie() == 1) {
      mask = nonzero_condition;
    } else {
      XLS_ASSIGN_OR_RETURN(
          mask, node->function_base()->MakeNode<ExtendOp>(
                    node->loc(), nonzero_condition,
                    /*new_bit_count=*/node->BitCountOrDie(), Op::kSignExt));
    }
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{*only_nonzero_value, mask}, Op::kAnd)
            .status());
    return true;
  }
  XLS_ASSIGN_OR_RETURN(Node * literal_zero,
                       node->function_base()->MakeNode<Literal>(
                           node->loc(), ZeroOfType(node->GetType())));
  XLS_RETURN_IF_ERROR(
      node->ReplaceUsesWithNew<Select>(nonzero_condition,
                                       std::vector<Node*>({literal_zero}),
                                       /*default_value=*/*only_nonzero_value)
          .status());
  return true;
}

// If a select's selector is just a unary function of some other value, we can
// encode the function for free by reordering the cases. For example:
//
//   sel(x + 1, cases=[a, b, c, d]) => sel(x, cases=[b, c, d, a])
//   sel(x * 2, cases=[a, b, c, d]) => sel(x, cases=[b, d, b, d])
//   sel(-x, cases=[a, b, c, d]) => sel(x, cases=[a, d, c, b])
absl::StatusOr<bool> MaybeReorderSelect(Node* node,
                                        const QueryEngine& query_engine) {
  if (!node->Is<Select>()) {
    return false;
  }
  Select* sel = node->As<Select>();
  Node* selector = sel->selector();

  // TODO(epastor): It would be nice to handle default values, but doing this
  // without adding more cases requires proving that all cases that end up
  // defaulted are equivalent... or else deciding how many cases we can add
  // without the cost being too high.
  if (sel->default_value().has_value()) {
    return false;
  }

  // Run through all of the selector's operands, recording the Values of any
  // that are known constant; this simplification applies if all but one are
  // known.
  absl::flat_hash_set<Node*> unknown_operands;
  std::vector<std::variant<Node*, Value>> selector_operands;
  selector_operands.reserve(selector->operands().size());
  for (Node* operand : selector->operands()) {
    if (std::optional<Value> known_value = query_engine.KnownValue(operand);
        known_value.has_value()) {
      selector_operands.push_back(*known_value);
    } else {
      selector_operands.push_back(operand);
      unknown_operands.insert(operand);
    }
  }
  if (unknown_operands.size() != 1) {
    return false;
  }
  Node* base_selector = *unknown_operands.begin();
  if (!base_selector->GetType()->IsBits()) {
    return false;
  }
  if (base_selector->BitCountOrDie() > selector->BitCountOrDie()) {
    // This would produce a wider selector than the original; avoid this.
    return false;
  }

  // Loop through all values of `base_selector`, recording which case ends up
  // selected. We know that this will run through at most 2^N values, where N is
  // the bit count of `base_selector` - and since `base_selector` is no wider
  // than the original `selector`, this is bounded by the number of cases in the
  // original select.
  std::vector<Node*> new_cases;
  new_cases.reserve(sel->cases().size());
  absl::FixedArray<Value> operand_values(selector_operands.size());
  Bits base_selector_bits(base_selector->BitCountOrDie());
  do {
    Value base_selector_value(base_selector_bits);

    for (int64_t i = 0; i < selector_operands.size(); ++i) {
      if (std::holds_alternative<Node*>(selector_operands[i])) {
        CHECK_EQ(std::get<Node*>(selector_operands[i]), base_selector);
        operand_values[i] = base_selector_value;
      } else {
        CHECK(std::holds_alternative<Value>(selector_operands[i]));
        operand_values[i] = std::get<Value>(selector_operands[i]);
      }
    }

    XLS_ASSIGN_OR_RETURN(Value selector_value,
                         InterpretNode(selector, operand_values));
    int64_t selected_case =
        static_cast<int64_t>(selector_value.bits().ToUint64().value());
    new_cases.push_back(sel->get_case(selected_case));

    base_selector_bits = bits_ops::Increment(base_selector_bits);
  } while (!base_selector_bits.IsZero());

  XLS_RETURN_IF_ERROR(
      sel->ReplaceUsesWithNew<Select>(base_selector, new_cases,
                                      /*default_value=*/std::nullopt)
          .status());
  return true;
}

absl::StatusOr<bool> SimplifyNode(Node* node, const QueryEngine& query_engine,
                                  const BitProvenanceAnalysis& provenance,
                                  int64_t opt_level, bool range_analysis) {
  // Select with a constant selector can be replaced with the respective
  // case.
  if (node->Is<Select>() &&
      query_engine.IsFullyKnown(node->As<Select>()->selector())) {
    Select* sel = node->As<Select>();
    const Bits selector = *query_engine.KnownValueAsBits(sel->selector());
    VLOG(2) << absl::StrFormat("Simplifying select with constant selector: %s",
                               node->ToString());
    if (bits_ops::UGreaterThan(
            selector, UBits(sel->cases().size() - 1, selector.bit_count()))) {
      XLS_RET_CHECK(sel->default_value().has_value());
      XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(*sel->default_value()));
    } else {
      XLS_ASSIGN_OR_RETURN(uint64_t i, selector.ToUint64());
      XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(sel->get_case(i)));
    }
    return true;
  }

  // Priority select where we know the selector ends with a one followed by
  // zeros can be replaced with the selected case.
  if (node->Is<PrioritySelect>()) {
    PrioritySelect* sel = node->As<PrioritySelect>();
    XLS_RET_CHECK(sel->selector()->GetType()->IsBits());
    std::optional<SharedLeafTypeTree<TernaryVector>> selector_ltt =
        query_engine.GetTernary(sel->selector());
    if (selector_ltt.has_value()) {
      const TernaryVector selector = selector_ltt->Get({});
      auto first_nonzero_case = absl::c_find_if(selector, [](TernaryValue v) {
        return v != TernaryValue::kKnownZero;
      });
      if (first_nonzero_case == selector.end()) {
        // All zeros; priority select with a zero selector returns the default
        // value.
        XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(sel->default_value()));
        return true;
      }
      if (*first_nonzero_case == TernaryValue::kKnownOne) {
        // Ends with a one followed by zeros; returns the corresponding case.
        int64_t case_num = std::distance(selector.begin(), first_nonzero_case);
        XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(sel->get_case(case_num)));
        return true;
      }
      // Has an unknown bit before the first known one, so the result is
      // unknown.
    }
  }

  // Three Way Compare with Select:
  // Checking if a > b followed by a < b with an else is a common coding
  // practice:
  //
  // if (a > b) {
  // } else if (a < b) {
  // } else {}  // a == b
  //
  // This can be simplified to use a comparison op for the default case instead
  // of an equality op because the comparison op is more expensive:
  //
  // if (a > b) {
  // } else if (a == b) {
  // } else {}  // a < b
  if (IsBinarySelect(node)) {
    Select* sel = node->As<Select>();
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TrySimplifyThreeWayCompareSelect(sel, query_engine));
    if (changed) {
      return true;
    }
  }

  // Three Way Compare with PrioritySelect:
  // Same as the above optimizations but uses PrioritySelect:
  if (node->Is<PrioritySelect>()) {
    PrioritySelect* sel = node->As<PrioritySelect>();
    XLS_ASSIGN_OR_RETURN(bool changed, TrySimplifyThreeWayComparePrioritySelect(
                                           sel, query_engine));
    if (changed) {
      return true;
    }
  }

  // One-hot-select with a constant selector can be replaced with OR of the
  // activated cases.
  if (node->Is<OneHotSelect>() &&
      query_engine.IsFullyKnown(node->As<OneHotSelect>()->selector()) &&
      node->GetType()->IsBits()) {
    OneHotSelect* sel = node->As<OneHotSelect>();
    const Bits selector = *query_engine.KnownValueAsBits(sel->selector());
    Node* replacement = nullptr;
    for (int64_t i = 0; i < selector.bit_count(); ++i) {
      if (selector.Get(i)) {
        if (replacement == nullptr) {
          replacement = sel->get_case(i);
        } else {
          XLS_ASSIGN_OR_RETURN(
              replacement,
              node->function_base()->MakeNode<NaryOp>(
                  node->loc(),
                  std::vector<Node*>{replacement, sel->get_case(i)}, Op::kOr));
        }
      }
    }
    if (replacement == nullptr) {
      XLS_ASSIGN_OR_RETURN(
          replacement,
          node->function_base()->MakeNode<Literal>(
              node->loc(), Value(UBits(0, node->BitCountOrDie()))));
    }
    VLOG(2) << absl::StrFormat(
        "Simplifying one-hot-select with constant selector: %s",
        node->ToString());
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(replacement));
    return true;
  }

  // Select with identical cases can be replaced with the value.
  if (node->Is<Select>()) {
    Select* sel = node->As<Select>();
    if (sel->AllCases(
            [&](Node* other_case) { return other_case == sel->any_case(); })) {
      VLOG(2) << absl::StrFormat("Simplifying select with identical cases: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(sel->any_case()));
      return true;
    }
  }

  // OneHotSelect with identical cases can be replaced with a select between one
  // of the identical case and the default value where the selector is: original
  // selector == 0
  if (node->Is<OneHotSelect>() && node->GetType()->IsBits() &&
      node->BitCountOrDie() > 1) {
    Node* selector = node->As<OneHotSelect>()->selector();
    absl::Span<Node* const> cases = node->As<OneHotSelect>()->cases();
    if (absl::c_all_of(cases, [&](Node* c) { return c == cases[0]; })) {
      FunctionBase* f = node->function_base();
      Node* is_nonzero;
      if (selector->GetType()->IsBits() && selector->BitCountOrDie() == 1) {
        is_nonzero = selector;
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * selector_zero,
            f->MakeNode<Literal>(node->loc(), ZeroOfType(selector->GetType())));
        XLS_ASSIGN_OR_RETURN(is_nonzero,
                             f->MakeNode<CompareOp>(node->loc(), selector,
                                                    selector_zero, Op::kNe));
      }
      XLS_ASSIGN_OR_RETURN(
          Node * default_value,
          f->MakeNode<Literal>(node->loc(), ZeroOfType(node->GetType())));
      VLOG(2) << absl::StrFormat(
          "Simplifying one-hot-select with identical cases: %s",
          node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Select>(
                                  is_nonzero,
                                  std::vector<Node*>{default_value, cases[0]},
                                  /*default_value=*/std::nullopt)
                              .status());
      return true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool reordered_select,
                       MaybeReorderSelect(node, query_engine));
  if (reordered_select) {
    return true;
  }

  // Replace a select among tuples to a tuple of selects. Handles all of select,
  // one-hot-select, and priority-select.
  if (node->GetType()->IsTuple() &&
      node->OpIn({Op::kSel, Op::kOneHotSel, Op::kPrioritySel})) {
    // Construct a vector containing the element at 'tuple_index' for each
    // case of the select.
    auto elements_at_tuple_index =
        [&](absl::Span<Node* const> nodes,
            int64_t tuple_index) -> absl::StatusOr<std::vector<Node*>> {
      std::vector<Node*> elements;
      for (Node* n : nodes) {
        XLS_ASSIGN_OR_RETURN(Node * element,
                             node->function_base()->MakeNode<TupleIndex>(
                                 node->loc(), n, tuple_index));
        elements.push_back(element);
      }
      return elements;
    };

    if (node->Is<OneHotSelect>()) {
      OneHotSelect* sel = node->As<OneHotSelect>();
      std::vector<Node*> selected_elements;
      for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(std::vector<Node*> case_elements,
                             elements_at_tuple_index(sel->cases(), i));
        XLS_ASSIGN_OR_RETURN(Node * selected_element,
                             node->function_base()->MakeNode<OneHotSelect>(
                                 node->loc(), sel->selector(), case_elements));
        selected_elements.push_back(selected_element);
      }
      VLOG(2) << absl::StrFormat("Decomposing tuple-typed one-hot-select: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Tuple>(selected_elements).status());
      return true;
    }

    if (node->Is<Select>()) {
      Select* sel = node->As<Select>();
      std::vector<Node*> selected_elements;
      for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(std::vector<Node*> case_elements,
                             elements_at_tuple_index(sel->cases(), i));
        std::optional<Node*> default_element = std::nullopt;
        if (sel->default_value().has_value()) {
          XLS_ASSIGN_OR_RETURN(default_element,
                               node->function_base()->MakeNode<TupleIndex>(
                                   node->loc(), *sel->default_value(), i));
        }
        XLS_ASSIGN_OR_RETURN(
            Node * selected_element,
            node->function_base()->MakeNode<Select>(
                node->loc(), sel->selector(), case_elements, default_element));
        selected_elements.push_back(selected_element);
      }
      VLOG(2) << absl::StrFormat("Decomposing tuple-typed select: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Tuple>(selected_elements).status());
      return true;
    }

    if (node->Is<PrioritySelect>()) {
      PrioritySelect* sel = node->As<PrioritySelect>();
      std::vector<Node*> selected_elements;
      for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
        XLS_ASSIGN_OR_RETURN(std::vector<Node*> case_elements,
                             elements_at_tuple_index(sel->cases(), i));
        XLS_ASSIGN_OR_RETURN(Node * default_element,
                             node->function_base()->MakeNode<TupleIndex>(
                                 node->loc(), sel->default_value(), i));
        XLS_ASSIGN_OR_RETURN(
            Node * selected_element,
            node->function_base()->MakeNode<PrioritySelect>(
                node->loc(), sel->selector(), case_elements, default_element));
        selected_elements.push_back(selected_element);
      }
      VLOG(2) << absl::StrFormat("Decomposing tuple-typed priority select: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Tuple>(selected_elements).status());
      return true;
    }
  }

  // Common out equivalent cases in a one hot select.
  if (NarrowingEnabled(opt_level) && node->Is<OneHotSelect>()) {
    FunctionBase* f = node->function_base();
    OneHotSelect* sel = node->As<OneHotSelect>();
    if (!sel->cases().empty() &&
        absl::flat_hash_set<Node*>(sel->cases().begin(), sel->cases().end())
                .size() != sel->cases().size()) {
      // For any case that's equal to another case, we or together the one hot
      // selectors and common out the value to squeeze the width of the one hot
      // select.
      std::vector<Node*> new_selectors;
      std::vector<Node*> new_cases;
      for (int64_t i = 0; i < sel->cases().size(); ++i) {
        Node* old_case = sel->get_case(i);
        XLS_ASSIGN_OR_RETURN(Node * old_selector,
                             f->MakeNode<BitSlice>(node->loc(), sel->selector(),
                                                   /*start=*/i, 1));
        auto it = std::find_if(
            new_cases.begin(), new_cases.end(),
            [old_case](Node* new_case) { return old_case == new_case; });
        if (it == new_cases.end()) {
          new_selectors.push_back(old_selector);
          new_cases.push_back(old_case);
        } else {
          // Or together the selectors, no need to append the old case.
          int64_t index = std::distance(new_cases.begin(), it);
          XLS_ASSIGN_OR_RETURN(
              new_selectors[index],
              f->MakeNode<NaryOp>(
                  node->loc(),
                  std::vector<Node*>{new_selectors[index], old_selector},
                  Op::kOr));
        }
      }
      std::reverse(new_selectors.begin(), new_selectors.end());
      XLS_ASSIGN_OR_RETURN(Node * new_selector,
                           f->MakeNode<Concat>(node->loc(), new_selectors));
      VLOG(2) << absl::StrFormat("One-hot select with equivalent cases: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
              .status());
      return true;
    }
  }

  // Common out equivalent cases in a priority select.
  if (SplitsEnabled(opt_level) && node->Is<PrioritySelect>() &&
      !node->As<PrioritySelect>()->cases().empty()) {
    FunctionBase* f = node->function_base();
    PrioritySelect* sel = node->As<PrioritySelect>();

    // We can merge adjacent cases with the same outputs by OR-ing together
    // the relevant bits of the selector.
    struct SelectorRange {
      int64_t start;
      int64_t width = 1;
    };
    std::vector<SelectorRange> new_selector_ranges;
    std::vector<Node*> new_cases;
    new_selector_ranges.push_back({.start = 0});
    new_cases.push_back(sel->get_case(0));
    for (int64_t i = 1; i < sel->cases().size(); ++i) {
      Node* old_case = sel->get_case(i);
      if (old_case == new_cases.back()) {
        new_selector_ranges.back().width++;
      } else {
        new_selector_ranges.push_back({.start = i});
        new_cases.push_back(old_case);
      }
    }
    if (new_cases.size() < sel->cases().size()) {
      std::vector<Node*> new_selector_slices;
      std::optional<SelectorRange> current_original_slice = std::nullopt;
      auto commit_original_slice = [&]() -> absl::Status {
        if (!current_original_slice.has_value()) {
          return absl::OkStatus();
        }
        XLS_ASSIGN_OR_RETURN(
            Node * selector_slice,
            f->MakeNode<BitSlice>(node->loc(), sel->selector(),
                                  current_original_slice->start,
                                  current_original_slice->width));
        new_selector_slices.push_back(selector_slice);
        current_original_slice.reset();
        return absl::OkStatus();
      };
      for (const SelectorRange& range : new_selector_ranges) {
        if (range.width == 1 && current_original_slice.has_value()) {
          current_original_slice->width++;
          continue;
        }

        XLS_RETURN_IF_ERROR(commit_original_slice());
        if (range.width == 1) {
          current_original_slice = SelectorRange{.start = range.start};
        } else {
          XLS_ASSIGN_OR_RETURN(
              Node * selector_slice,
              f->MakeNode<BitSlice>(node->loc(), sel->selector(), range.start,
                                    range.width));
          XLS_ASSIGN_OR_RETURN(Node * selector_bit,
                               f->MakeNode<BitwiseReductionOp>(
                                   node->loc(), selector_slice, Op::kOrReduce));
          new_selector_slices.push_back(selector_bit);
        }
      }
      XLS_RETURN_IF_ERROR(commit_original_slice());
      absl::c_reverse(new_selector_slices);
      XLS_ASSIGN_OR_RETURN(
          Node * new_selector,
          f->MakeNode<Concat>(node->loc(), new_selector_slices));
      VLOG(2) << absl::StrFormat("Priority select with equivalent cases: %s",
                                 node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<PrioritySelect>(
                                  new_selector, new_cases, sel->default_value())
                              .status());
      return true;
    }
  }

  // Absorb inverted 1-bit selectors into their selects.
  if (node->OpIn({Op::kSel, Op::kPrioritySel}) &&
      node->operand(0)->op() == Op::kNot &&
      node->operand(0)->BitCountOrDie() == 1) {
    VLOG(2) << absl::StrFormat("Select with an inverted one-bit selector: %s",
                               node->ToString());
    XLS_RETURN_IF_ERROR(
        node->ReplaceOperandNumber(0, node->operand(0)->operand(0)));
    node->SwapOperands(1, 2);
    return true;
  }

  // We explode single-bit muxes into their constituent gates to expose more
  // optimization opportunities. Since this creates more ops in the general
  // case, we look for certain sub-cases:
  //
  // * At least one of the selected values is a constant.
  // * One of the selected values is also the selector.
  //
  // If the one-bit MUX is a one-hot select, one of the selected values is
  // always a constant, since the default value is always zero.
  auto is_one_bit_mux = [&] {
    if (!node->GetType()->IsBits() || node->BitCountOrDie() != 1) {
      return false;
    }
    if (node->Is<Select>()) {
      return node->As<Select>()->selector()->BitCountOrDie() == 1;
    }
    if (node->Is<PrioritySelect>()) {
      return node->As<PrioritySelect>()->selector()->BitCountOrDie() == 1;
    }
    if (node->Is<OneHotSelect>()) {
      return node->As<OneHotSelect>()->selector()->BitCountOrDie() == 1;
    }
    return false;
  };
  if (NarrowingEnabled(opt_level) && is_one_bit_mux() &&
      (node->Is<OneHotSelect>() ||
       query_engine.IsFullyKnown(node->operand(1)) ||
       query_engine.IsFullyKnown(node->operand(2)) ||
       node->operand(0) == node->operand(1) ||
       node->operand(0) == node->operand(2))) {
    FunctionBase* f = node->function_base();
    Node* s;
    Node* on_true;
    std::optional<Node*> on_false;
    if (node->Is<Select>()) {
      Select* select = node->As<Select>();
      s = select->selector();
      on_false = select->get_case(0);
      on_true = select->default_value().has_value()
                    ? select->default_value().value()
                    : select->get_case(1);
    } else if (node->Is<PrioritySelect>()) {
      s = node->As<PrioritySelect>()->selector();
      on_true = node->As<PrioritySelect>()->get_case(0);
      on_false = node->As<PrioritySelect>()->default_value();
    } else {
      XLS_RET_CHECK(node->Is<OneHotSelect>());
      s = node->As<OneHotSelect>()->selector();
      on_true = node->As<OneHotSelect>()->get_case(0);
      on_false = std::nullopt;
    }
    VLOG(3) << absl::StrFormat("Decomposing single-bit select: %s",
                               node->ToString());
    if (!on_false.has_value()) {
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<NaryOp>(
                                  std::vector<Node*>{s, on_true}, Op::kAnd)
                              .status());
      return true;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * lhs, f->MakeNode<NaryOp>(
                        node->loc(), std::vector<Node*>{s, on_true}, Op::kAnd));
    XLS_ASSIGN_OR_RETURN(Node * s_not,
                         f->MakeNode<UnOp>(node->loc(), s, Op::kNot));
    XLS_ASSIGN_OR_RETURN(
        Node * rhs,
        f->MakeNode<NaryOp>(node->loc(), std::vector<Node*>{s_not, *on_false},
                            Op::kAnd));
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(std::vector<Node*>{lhs, rhs}, Op::kOr)
            .status());
    return true;
  }

  // Since masking with an 'and' can't be reasoned through as easily (e.g., by
  // conditional specialization), we want to avoid doing this until fairly
  // late in the pipeline.
  if (SplitsEnabled(opt_level)) {
    XLS_ASSIGN_OR_RETURN(bool converted_to_mask,
                         MaybeConvertSelectToMask(node, query_engine));
    if (converted_to_mask) {
      return true;
    }
  }

  // Priority selects can be narrowed to the smallest range of bits known to
  // have at least one bit set, with the last value turned into the new default
  // value.
  if (NarrowingEnabled(opt_level) && node->Is<PrioritySelect>()) {
    PrioritySelect* sel = node->As<PrioritySelect>();
    std::vector<TreeBitLocation> trailing_bits;
    int64_t last_bit = 0;
    for (; last_bit < sel->selector()->BitCountOrDie(); ++last_bit) {
      trailing_bits.push_back(TreeBitLocation(sel->selector(), last_bit));
      if (query_engine.AtLeastOneTrue(trailing_bits)) {
        break;
      }
    }
    if (last_bit == 0) {
      XLS_RETURN_IF_ERROR(sel->ReplaceUsesWith(sel->get_case(0)));
      return true;
    }
    if (last_bit < sel->selector()->BitCountOrDie()) {
      // We can drop at least one bit from the selector, and turn the last case
      // into the default value.
      XLS_ASSIGN_OR_RETURN(
          Node * narrowed_selector,
          node->function_base()->MakeNode<BitSlice>(
              node->loc(), sel->selector(), /*start=*/0, /*width=*/last_bit));
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<PrioritySelect>(
                                  narrowed_selector,
                                  sel->cases().subspan(0, last_bit),
                                  /*default_value=*/sel->get_case(last_bit))
                              .status());
      return true;
    }
  }

  // Cases matching the default value or positions where the selector is zero
  // can be removed from OneHotSelects and priority selects.
  if (NarrowingEnabled(opt_level) &&
      (node->Is<OneHotSelect>() || node->Is<PrioritySelect>())) {
    Node* selector = node->Is<OneHotSelect>()
                         ? node->As<OneHotSelect>()->selector()
                         : node->As<PrioritySelect>()->selector();
    absl::Span<Node* const> cases = node->Is<OneHotSelect>()
                                        ? node->As<OneHotSelect>()->cases()
                                        : node->As<PrioritySelect>()->cases();
    if (query_engine.IsTracked(selector)) {
      std::optional<SharedLeafTypeTree<TernaryVector>> selector_ltt =
          query_engine.GetTernary(selector);
      TernaryVector selector_bits =
          selector_ltt.has_value() ? selector_ltt->Get({})
                                   : TernaryVector(selector->BitCountOrDie(),
                                                   TernaryValue::kUnknown);
      // For one-hot-selects if either the selector bit or the case value is
      // zero, the case can be removed. For priority selects, the case can be
      // removed only if the selector bit is zero, or if *all later* cases are
      // removable.
      bool all_later_cases_removable = false;
      auto is_removable_case = [&](int64_t c) {
        if (all_later_cases_removable) {
          return true;
        }
        if (node->Is<PrioritySelect>() &&
            selector_bits[c] == TernaryValue::kKnownOne) {
          all_later_cases_removable = true;
          return false;
        }
        if (selector_bits[c] == TernaryValue::kKnownZero) {
          return true;
        }
        return node->Is<OneHotSelect>() && query_engine.IsAllZeros(cases[c]);
      };
      bool has_removable_case = false;
      std::vector<int64_t> nonremovable_indices;
      for (int64_t i = 0; i < cases.size(); ++i) {
        if (is_removable_case(i)) {
          has_removable_case = true;
        } else {
          nonremovable_indices.push_back(i);
        }
      }
      if (node->Is<PrioritySelect>()) {
        // Go back and check the trailing cases; we can remove trailing cases
        // that match the default.
        while (!nonremovable_indices.empty() &&
               cases[nonremovable_indices.back()] ==
                   node->As<PrioritySelect>()->default_value()) {
          nonremovable_indices.pop_back();
          has_removable_case = true;
        }
      }
      if (!SplitsEnabled(opt_level) && !nonremovable_indices.empty() &&
          has_removable_case) {
        // No splitting, so we can only remove the leading and trailing cases.
        int64_t first_nonremovable_index = nonremovable_indices.front();
        int64_t last_nonremovable_index = nonremovable_indices.back();
        nonremovable_indices.clear();
        for (int64_t i = first_nonremovable_index; i <= last_nonremovable_index;
             ++i) {
          nonremovable_indices.push_back(i);
        }
        if (nonremovable_indices.size() == cases.size()) {
          // No cases are removable.
          has_removable_case = false;
        }
      }
      if (has_removable_case) {
        // Assemble the slices of the selector which correspond to non-zero
        // cases.
        if (nonremovable_indices.empty()) {
          // If all cases were zeros, just replace the op with the default
          // value.
          if (node->Is<PrioritySelect>()) {
            XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(
                node->As<PrioritySelect>()->default_value()));
          } else {
            XLS_RETURN_IF_ERROR(
                node->ReplaceUsesWithNew<Literal>(ZeroOfType(node->GetType()))
                    .status());
          }
          return true;
        }
        XLS_ASSIGN_OR_RETURN(Node * new_selector,
                             GatherBits(selector, nonremovable_indices));
        std::vector<Node*> new_cases =
            GatherFromSequence(cases, nonremovable_indices);
        VLOG(2) << absl::StrFormat(
            "Literal zero cases removed from %s-select: %s",
            node->Is<OneHotSelect>() ? "one-hot" : "priority",
            node->ToString());
        if (node->Is<OneHotSelect>()) {
          XLS_RETURN_IF_ERROR(
              node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
                  .status());
        } else {
          XLS_RETURN_IF_ERROR(
              node->ReplaceUsesWithNew<PrioritySelect>(
                      new_selector, new_cases,
                      node->As<PrioritySelect>()->default_value())
                  .status());
        }
        return true;
      }
    }
  }

  // PrioritySelect with two distinct cases (including the default value) can be
  // reduced to a one-bit PrioritySelect by checking for just the non-default
  // cases.
  //
  // By the time we get here, this should only apply to PrioritySelects of the
  // following forms:
  //
  //   priority_sel(s, cases=[x, y], default=x)
  //   priority_sel(s, cases=[y, x, y], default=x)
  //   priority_sel(s, cases=[x, y, x, y], default=x)
  //   ...
  //
  // where 'x' and 'y' are distinct values, due to collapsing of consecutive
  // equivalent cases.
  auto distinct_case_count = [&](PrioritySelect* sel) {
    absl::Span<Node* const> cases = sel->cases();
    absl::flat_hash_set<Node*> distinct_cases(cases.begin(), cases.end());
    distinct_cases.insert(sel->default_value());
    return distinct_cases.size();
  };
  if (SplitsEnabled(opt_level) && node->Is<PrioritySelect>() &&
      node->As<PrioritySelect>()->cases().size() > 1 &&
      distinct_case_count(node->As<PrioritySelect>()) == 2) {
    VLOG(2) << absl::StrFormat(
        "Simplifying priority-select with two distinct cases: %s",
        node->ToString());
    PrioritySelect* sel = node->As<PrioritySelect>();
    Node* nondefault_case = nullptr;
    std::vector<Node*> nondefault_selectors;
    for (int64_t i = 0; i < sel->cases().size(); ++i) {
      if (sel->get_case(i) == sel->default_value()) {
        continue;
      }
      nondefault_case = sel->get_case(i);
      Node* case_selector;
      if (i == 0) {
        XLS_ASSIGN_OR_RETURN(case_selector,
                             sel->function_base()->MakeNode<BitSlice>(
                                 node->loc(), sel->selector(),
                                 /*start=*/0, /*width=*/1));
      } else {
        Node* selector_slice;
        if (i + 1 == sel->selector()->BitCountOrDie()) {
          selector_slice = sel->selector();
        } else {
          XLS_ASSIGN_OR_RETURN(selector_slice,
                               sel->function_base()->MakeNode<BitSlice>(
                                   node->loc(), sel->selector(),
                                   /*start=*/0, /*width=*/i + 1));
        }
        XLS_ASSIGN_OR_RETURN(
            Node * selector_case,
            sel->function_base()->MakeNode<Literal>(
                node->loc(), Value(Bits::PowerOfTwo(i, i + 1))));
        XLS_ASSIGN_OR_RETURN(
            case_selector,
            sel->function_base()->MakeNode<CompareOp>(
                node->loc(), selector_slice, selector_case, Op::kEq));
      }
      nondefault_selectors.push_back(case_selector);
    }
    CHECK_NE(nondefault_case, nullptr);
    CHECK(!nondefault_selectors.empty());
    Node* new_selector;
    if (nondefault_selectors.size() == 1) {
      new_selector = nondefault_selectors.front();
    } else {
      XLS_ASSIGN_OR_RETURN(new_selector,
                           sel->function_base()->MakeNode<NaryOp>(
                               node->loc(), nondefault_selectors, Op::kOr));
    }
    XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<PrioritySelect>(
                               /*selector=*/new_selector,
                               /*cases=*/absl::MakeConstSpan({nondefault_case}),
                               /*default_value=*/sel->default_value())
                            .status());
    return true;
  }

  // "Squeeze" the width of the mux when bits are known to reduce the cost of
  // the operation.
  //
  // Sel(...) => Concat(Known, Sel(...), Known)
  if (SplitsEnabled(opt_level)) {
    if (node->GetType()->IsBits()) {
      bool squeezed = false;
      if (node->Is<Select>()) {
        XLS_ASSIGN_OR_RETURN(squeezed,
                             TrySqueezeSelect(node->As<Select>(), query_engine,
                                              provenance, range_analysis));
      } else if (node->Is<OneHotSelect>()) {
        XLS_ASSIGN_OR_RETURN(
            squeezed, TrySqueezeSelect(node->As<OneHotSelect>(), query_engine,
                                       provenance, range_analysis));
      } else if (node->Is<PrioritySelect>()) {
        XLS_ASSIGN_OR_RETURN(
            squeezed, TrySqueezeSelect(node->As<PrioritySelect>(), query_engine,
                                       provenance, range_analysis));
      }
      if (squeezed) {
        return true;
      }
    }
  }

  // Collapse consecutive two-ways selects which have share a common case. For
  // example:
  //
  //   s1 = select(p1, [y, x])
  //   s0 = select(p0, [s_1, x])
  //
  // In this case, 'x' is a common case between the two selects and the above
  // can be replaced with:
  //
  //   p' = or(p0, p1)
  //   s0 = select(p', [x, y])
  //
  // There are four different cases to consider depending upon whether the
  // common case is on the LHS or RHS of the selects.
  auto is_2way_select = [](Node* n) {
    return n->Is<Select>() &&
           n->As<Select>()->selector()->BitCountOrDie() == 1 &&
           n->As<Select>()->cases().size() == 2;
  };
  if (is_2way_select(node)) {
    //  The variable names correspond to the names of the nodes in the
    //  diagrams below.
    Select* sel0 = node->As<Select>();
    Node* p0 = sel0->selector();
    // The values below are by each matching cases below.
    Node* x = nullptr;
    Node* y = nullptr;
    // The predicate to select the common case 'x' in the newly constructed
    // select.
    Node* p_x = nullptr;
    if (is_2way_select(sel0->get_case(0))) {
      Select* sel1 = sel0->get_case(0)->As<Select>();
      Node* p1 = sel1->selector();
      if (sel0->get_case(1) == sel1->get_case(0)) {
        //       x   y
        //        \ /
        //  p1 -> sel1   x
        //           \   /
        //      p0 -> sel0
        //
        // p_x = p0 | !p1
        x = sel0->get_case(1);
        y = sel1->get_case(1);
        XLS_ASSIGN_OR_RETURN(
            Node * not_p1,
            sel0->function_base()->MakeNode<UnOp>(sel0->loc(), p1, Op::kNot));
        XLS_ASSIGN_OR_RETURN(
            p_x, sel0->function_base()->MakeNode<NaryOp>(
                     sel0->loc(), std::vector<Node*>{p0, not_p1}, Op::kOr));
      } else if (sel0->get_case(1) == sel1->get_case(1)) {
        //         y   x
        //          \ /
        //   p1 -> sel1   x
        //            \   /
        //       p0 -> sel0
        //
        // p_x = p0 | p1
        x = sel0->get_case(1);
        y = sel1->get_case(0);
        XLS_ASSIGN_OR_RETURN(
            p_x, sel0->function_base()->MakeNode<NaryOp>(
                     sel0->loc(), std::vector<Node*>{p0, p1}, Op::kOr));
      }
    } else if (is_2way_select(sel0->get_case(1))) {
      Select* sel1 = sel0->get_case(1)->As<Select>();
      Node* p1 = sel1->selector();
      if (sel0->get_case(0) == sel1->get_case(0)) {
        //  x    x   y
        //   \    \ /
        //    \  sel1 <- p1
        //     \  /
        //      sel0 <- p0
        //
        // p_x = nand(p0, p1)
        x = sel0->get_case(0);
        y = sel1->get_case(1);
        XLS_ASSIGN_OR_RETURN(
            p_x, sel0->function_base()->MakeNode<NaryOp>(
                     sel0->loc(), std::vector<Node*>{p0, p1}, Op::kNand));
      } else if (sel0->get_case(0) == sel1->get_case(1)) {
        //  x    y   x
        //   \    \ /
        //    \  sel1 <- p1
        //     \  /
        //      sel0 <- p0
        //
        // p_x = !p0 | p1
        x = sel0->get_case(0);
        y = sel1->get_case(0);
        XLS_ASSIGN_OR_RETURN(
            Node * not_p0,
            sel0->function_base()->MakeNode<UnOp>(sel0->loc(), p0, Op::kNot));
        XLS_ASSIGN_OR_RETURN(
            p_x, sel0->function_base()->MakeNode<NaryOp>(
                     sel0->loc(), std::vector<Node*>{not_p0, p1}, Op::kOr));
      }
    }
    if (x != nullptr) {
      VLOG(2) << absl::StrFormat(
          "Consecutive binary select with common cases: %s", node->ToString());
      XLS_ASSIGN_OR_RETURN(p_x, sel0->ReplaceUsesWithNew<Select>(
                                    p_x, std::vector<Node*>{y, x},
                                    /*default_value=*/std::nullopt));
      return true;
    }
  }

  // Consecutive selects which share a selector can be collapsed into a single
  // select. If sel0 selects sel1 on when p is false:
  //
  //  a   b
  //   \ /
  //   sel1 ----+-- p       a   c
  //    |       |       =>   \ /
  //    |  c    |            sel -- p
  //    | /     |             |
  //   sel0 ----+
  //    |
  //
  // If sel0 selects sel1 on when p is true:
  //
  //    a   b
  //     \ /
  //     sel1 -+-- p       c   b
  //      |    |       =>   \ /
  //   c  |    |            sel -- p
  //    \ |    |             |
  //     sel0 -+
  //      |
  //
  // TODO(meheff): Generalize this to multi-way selects and possibly
  // one-hot-selects.
  if (is_2way_select(node)) {
    Select* sel0 = node->As<Select>();
    if (is_2way_select(sel0->get_case(0))) {
      Select* sel1 = sel0->get_case(0)->As<Select>();
      if (sel0->selector() == sel1->selector()) {
        XLS_RETURN_IF_ERROR(sel0->ReplaceOperandNumber(1, sel1->get_case(0)));
        return true;
      }
    }
    if (is_2way_select(sel0->get_case(1))) {
      Select* sel1 = sel0->get_case(1)->As<Select>();
      if (sel0->selector() == sel1->selector()) {
        XLS_RETURN_IF_ERROR(sel0->ReplaceOperandNumber(2, sel1->get_case(1)));
        return true;
      }
    }
  }

  // Decompose single-bit, two-way OneHotSelects into ANDs and ORs.
  if (SplitsEnabled(opt_level) && node->Is<OneHotSelect>() &&
      node->GetType()->IsBits() && node->BitCountOrDie() == 1 &&
      node->As<OneHotSelect>()->cases().size() == 2) {
    OneHotSelect* ohs = node->As<OneHotSelect>();
    XLS_ASSIGN_OR_RETURN(Node * sel0,
                         node->function_base()->MakeNode<BitSlice>(
                             node->loc(), ohs->selector(), /*start=*/0,
                             /*width=*/1));
    XLS_ASSIGN_OR_RETURN(Node * sel1,
                         node->function_base()->MakeNode<BitSlice>(
                             node->loc(), ohs->selector(), /*start=*/1,
                             /*width=*/1));
    XLS_ASSIGN_OR_RETURN(
        Node * and0,
        node->function_base()->MakeNode<NaryOp>(
            node->loc(), std::vector<Node*>{sel0, ohs->get_case(0)}, Op::kAnd));
    XLS_ASSIGN_OR_RETURN(
        Node * and1,
        node->function_base()->MakeNode<NaryOp>(
            node->loc(), std::vector<Node*>{sel1, ohs->get_case(1)}, Op::kAnd));
    VLOG(2) << absl::StrFormat("Decompose single-bit one-hot-select: %s",
                               node->ToString());
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<NaryOp>(
                                std::vector<Node*>{and0, and1}, Op::kOr)
                            .status());
    return true;
  }

  // Replace one-hot-select using a one-hotted selector with a priority select,
  // if the selector's one-hot has no uses other than as a selector for
  // other one-hot-selects.
  auto is_only_a_one_hot_selector = [](Node* selector) {
    if (selector->function_base()->HasImplicitUse(selector)) {
      return false;
    }
    for (Node* user : selector->users()) {
      if (!user->Is<OneHotSelect>()) {
        return false;
      }
      absl::Span<Node* const> cases = user->As<OneHotSelect>()->cases();
      if (absl::c_find(cases, selector) != cases.end()) {
        // This uses `selector` as something other than a one-hot selector.
        return false;
      }
    }
    return true;
  };
  if (node->Is<OneHotSelect>() &&
      node->As<OneHotSelect>()->selector()->Is<OneHot>() &&
      is_only_a_one_hot_selector(node->As<OneHotSelect>()->selector())) {
    OneHotSelect* ohs = node->As<OneHotSelect>();
    Node* selector = ohs->selector()->As<OneHot>()->operand(0);
    LsbOrMsb priority = ohs->selector()->As<OneHot>()->priority();

    // OneHot extends the value with an extra bit that's set iff the original
    // value is exactly zero; PrioritySelect's custom default value lets us
    // recreate that effect without the extra bit.
    absl::Span<Node* const> cases = ohs->cases();
    Node* default_value = cases.back();
    cases.remove_suffix(1);

    Node* new_selector = nullptr;
    std::vector<Node*> new_cases(cases.begin(), cases.end());
    switch (priority) {
      case LsbOrMsb::kLsb: {
        new_selector = selector;
        break;
      }
      case LsbOrMsb::kMsb: {
        XLS_ASSIGN_OR_RETURN(new_selector,
                             node->function_base()->MakeNode<UnOp>(
                                 node->loc(), selector, Op::kReverse));
        std::reverse(new_cases.begin(), new_cases.end());
        break;
      }
    }
    XLS_RET_CHECK_NE(new_selector, nullptr)
        << absl::StreamFormat("invalid OneHot priority: %v", priority);

    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<PrioritySelect>(
                                new_selector, new_cases, default_value)
                            .status());
    return true;
  }

  // Replace a single-bit input kOneHot with the concat of the input and its
  // inverse.
  if (NarrowingEnabled(opt_level) && node->Is<OneHot>() &&
      node->BitCountOrDie() == 2) {
    XLS_ASSIGN_OR_RETURN(Node * inv_operand,
                         node->function_base()->MakeNode<UnOp>(
                             node->loc(), node->operand(0), Op::kNot));
    VLOG(2) << absl::StrFormat("Replace single-bit input one-hot to concat: %s",
                               node->ToString());
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<Concat>(
                std::vector<Node*>{inv_operand, node->operand(0)})
            .status());
    return true;
  }

  // Remove kOneHot operations with an input that is mutually exclusive.
  if (node->Is<OneHot>()) {
    if (query_engine.AtMostOneBitTrue(node->operand(0))) {
      XLS_ASSIGN_OR_RETURN(
          Node * zero,
          node->function_base()->MakeNode<Literal>(
              node->loc(),
              Value(UBits(0,
                          /*bit_count=*/node->operand(0)->BitCountOrDie()))));
      XLS_ASSIGN_OR_RETURN(Node * operand_eq_zero,
                           node->function_base()->MakeNode<CompareOp>(
                               node->loc(), node->operand(0), zero, Op::kEq));
      VLOG(2) << absl::StrFormat(
          "Replace one-hot with mutually exclusive input: %s",
          node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Concat>(
                  std::vector{operand_eq_zero, node->operand(0)})
              .status());
      return true;
    }

    if (std::optional<TreeBitLocation> unknown_bit =
            query_engine.ExactlyOneBitUnknown(node->operand(0));
        unknown_bit.has_value()) {
      Node* input = node->operand(0);
      // When only one bit is unknown there are only two possible values, so
      // we can strength reduce this to a select between the two possible
      // values based on the unknown bit, which should unblock more subsequent
      // optimizations.
      // 1. Determine the unknown bit (for use as a selector).
      XLS_ASSIGN_OR_RETURN(
          Node * selector,
          node->function_base()->MakeNode<BitSlice>(
              node->loc(), input, /*start=*/unknown_bit->bit_index(),
              /*width=*/1));

      // 2. Create the literals we select among based on whether the bit is
      //    populated or not.
      const int64_t input_bit_count =
          input->GetType()->AsBitsOrDie()->bit_count();

      // Build up inputs for the case where the unknown value is true and
      // false, respectively.
      InlineBitmap input_on_true(input_bit_count);
      InlineBitmap input_on_false(input_bit_count);
      int64_t seen_unknown = 0;
      for (int64_t bitno = 0; bitno < input_bit_count; ++bitno) {
        TreeBitLocation tree_location(input, bitno);
        std::optional<bool> known_value =
            query_engine.KnownValue(tree_location);
        if (known_value.has_value()) {
          input_on_false.Set(bitno, known_value.value());
          input_on_true.Set(bitno, known_value.value());
        } else {
          seen_unknown++;
          input_on_false.Set(bitno, false);
          input_on_true.Set(bitno, true);
        }
      }
      CHECK_EQ(seen_unknown, 1)
          << "Query engine noted exactly one bit was unknown; saw unexpected "
             "number of unknown bits";

      // Wrapper lambda that invokes the right priority for the one hot op
      // based on the node metadata.
      auto do_one_hot = [&](const Bits& input) {
        OneHot* one_hot = node->As<OneHot>();
        if (one_hot->priority() == LsbOrMsb::kLsb) {
          return bits_ops::OneHotLsbToMsb(input);
        }
        return bits_ops::OneHotMsbToLsb(input);
      };

      Bits output_on_false = do_one_hot(Bits::FromBitmap(input_on_false));
      Bits output_on_true = do_one_hot(Bits::FromBitmap(input_on_true));
      VLOG(2) << absl::StrFormat(
          "input_on_false: %s input_on_true: %s output_on_false: %s "
          "output_on_true: %s",
          Bits::FromBitmap(input_on_false).ToDebugString(),
          Bits::FromBitmap(input_on_true).ToDebugString(),
          output_on_false.ToDebugString(), output_on_true.ToDebugString());
      XLS_ASSIGN_OR_RETURN(Node * on_false,
                           node->function_base()->MakeNode<Literal>(
                               node->loc(), Value(std::move(output_on_false))));
      XLS_ASSIGN_OR_RETURN(Node * on_true,
                           node->function_base()->MakeNode<Literal>(
                               node->loc(), Value(std::move(output_on_true))));

      // 3. Create the select.
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<Select>(
                                  selector,
                                  std::vector<Node*>{on_false, on_true},
                                  /*default_value=*/std::nullopt)
                              .status());
      return true;
    }
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> SelectSimplificationPassBase::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  QueryEngine* value_engine;
  if (range_analysis_) {
    value_engine = context.SharedQueryEngine<PartialInfoQueryEngine>(func);
  } else {
    value_engine = context.SharedQueryEngine<LazyTernaryQueryEngine>(func);
  }
  VLOG(2) << "Range analysis is " << std::boolalpha << range_analysis_;

  auto query_engine =
      UnionQueryEngine::Of(StatelessQueryEngine(), value_engine);
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  XLS_ASSIGN_OR_RETURN(BitProvenanceAnalysis provenance,
                       BitProvenanceAnalysis::Create(func));

  bool changed = false;
  for (Node* node : context.TopoSort(func)) {
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyNode(node, query_engine, provenance,
                                      options.opt_level, range_analysis_));
    changed = changed || node_changed;
  }

  // Use a worklist to split OneHotSelects & PrioritySelects based on common
  // bits in the cases because this transformation creates many more
  // OneHotSelects & PrioritySelects exposing further opportunities for
  // optimizations.
  if (options.splits_enabled()) {
    // We need to recalculate provenance and qe if changes happened.
    std::optional<BitProvenanceAnalysis> post_simplify_provenance;
    if (changed) {
      XLS_ASSIGN_OR_RETURN(post_simplify_provenance,
                           BitProvenanceAnalysis::Create(func));
      XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());
    } else {
      post_simplify_provenance = std::move(provenance);
    }
    std::deque<Node*> worklist;
    for (Node* node : func->nodes()) {
      if (node->Is<OneHotSelect>() || node->Is<PrioritySelect>()) {
        worklist.push_back(node);
      }
    }
    while (!worklist.empty()) {
      Node* bit_based_select = worklist.front();
      worklist.pop_front();
      // Note that query_engine may be stale at this point but that is
      // ok; we'll fall back on the stateless query engine.
      XLS_ASSIGN_OR_RETURN(
          std::vector<Node*> new_bit_based_selects,
          MaybeSplitBitBasedSelect(bit_based_select, query_engine,
                                   *post_simplify_provenance));
      if (!new_bit_based_selects.empty()) {
        changed = true;
        worklist.insert(worklist.end(), new_bit_based_selects.begin(),
                        new_bit_based_selects.end());
      }
    }
  }
  return changed;
}

REGISTER_OPT_PASS(SelectSimplificationPass);
REGISTER_OPT_PASS(SelectRangeSimplificationPass);

}  // namespace xls
