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
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/data_structures/algorithm.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

// Given a SelectT node (either OneHotSelect or Select), squeezes the const_msb
// and const_lsb values out of the output, and slices all the operands to
// correspond to the non-const run of bits in the center.
template <typename SelectT>
absl::StatusOr<bool> SqueezeSelect(
    const Bits& const_msb, const Bits& const_lsb,
    const std::function<absl::StatusOr<SelectT*>(SelectT*, std::vector<Node*>)>&
        make_select,
    SelectT* select) {
  FunctionBase* f = select->function_base();
  int64_t bit_count = select->BitCountOrDie();
  auto slice = [&](Node* n) -> absl::StatusOr<Node*> {
    int64_t new_width =
        bit_count - const_msb.bit_count() - const_lsb.bit_count();
    return f->MakeNode<BitSlice>(select->loc(), n,
                                 /*start=*/const_lsb.bit_count(),
                                 /*width=*/new_width);
  };
  std::vector<Node*> new_cases;
  absl::Span<Node* const> cases = select->operands().subspan(1);
  for (Node* old_case : cases) {
    XLS_ASSIGN_OR_RETURN(Node * new_case, slice(old_case));
    new_cases.push_back(new_case);
  }
  XLS_ASSIGN_OR_RETURN(Node * msb_literal,
                       f->MakeNode<Literal>(select->loc(), Value(const_msb)));
  XLS_ASSIGN_OR_RETURN(Node * lsb_literal,
                       f->MakeNode<Literal>(select->loc(), Value(const_lsb)));
  XLS_ASSIGN_OR_RETURN(Node * new_select, make_select(select, new_cases));
  Node* select_node = select;
  XLS_VLOG(2) << absl::StrFormat("Squeezing select: %s", select->ToString());
  XLS_RETURN_IF_ERROR(select_node
                          ->ReplaceUsesWithNew<Concat>(std::vector<Node*>{
                              msb_literal, new_select, lsb_literal})
                          .status());
  return true;
}

// The source of a bit. Can be either a literal 0/1 or a bit at a particular
// index of a Node.
using BitSource = std::variant<bool, std::pair<Node*, int64_t>>;

// Traces the bit at the given node and bit index through bit slices and concats
// and returns its source.
// TODO(meheff): Combine this into TernaryQueryEngine.
BitSource GetBitSource(Node* node, int64_t bit_index,
                       const QueryEngine& query_engine) {
  if (node->Is<BitSlice>()) {
    return GetBitSource(node->operand(0),
                        bit_index + node->As<BitSlice>()->start(),
                        query_engine);
  } else if (node->Is<Concat>()) {
    int64_t offset = 0;
    for (int64_t i = node->operand_count() - 1; i >= 0; --i) {
      Node* operand = node->operand(i);
      if (bit_index - offset < operand->BitCountOrDie()) {
        return GetBitSource(operand, bit_index - offset, query_engine);
      }
      offset += operand->BitCountOrDie();
    }
    LOG(FATAL) << "Bit index " << bit_index << " too large for "
               << node->ToString();
  } else if (node->Is<Literal>()) {
    return node->As<Literal>()->value().bits().Get(bit_index);
  } else if (node->GetType()->IsBits() &&
             query_engine.IsKnown(TreeBitLocation(node, bit_index))) {
    return query_engine.IsOne(TreeBitLocation(node, bit_index));
  }
  return std::make_pair(node, bit_index);
}

std::string ToString(const BitSource& bit_source) {
  return absl::visit(Visitor{[](bool value) { return absl::StrCat(value); },
                             [](const std::pair<Node*, int64_t>& p) {
                               return absl::StrCat(p.first->GetName(), "[",
                                                   p.second, "]");
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
MatchedPairs PairsOfBitsWithSameSource(absl::Span<Node* const> nodes,
                                       int64_t bit_index,
                                       const QueryEngine& query_engine) {
  std::vector<BitSource> bit_sources;
  for (Node* node : nodes) {
    bit_sources.push_back(GetBitSource(node, bit_index, query_engine));
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

// Returns a OneHotSelect instruction which selects a slice of the given
// OneHotSelect's cases. The cases are sliced with the given start and width and
// then selected with a new OnehotSelect which is returned.
absl::StatusOr<OneHotSelect*> SliceOneHotSelect(OneHotSelect* ohs,
                                                int64_t start, int64_t width) {
  std::vector<Node*> case_slices;
  for (Node* cas : ohs->cases()) {
    XLS_ASSIGN_OR_RETURN(Node * case_slice,
                         ohs->function_base()->MakeNode<BitSlice>(
                             ohs->loc(), cas, /*start=*/start,
                             /*width=*/width));
    case_slices.push_back(case_slice);
  }
  return ohs->function_base()->MakeNode<OneHotSelect>(
      ohs->loc(), ohs->selector(), case_slices);
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
                                 const QueryEngine& query_engine) {
  XLS_VLOG(5) << "Finding runs of non-distinct bits starting at " << start;
  // Do a reduction via intersection of the set of matching pairs within
  // 'cases'. When the intersection is empty, the run is over.
  MatchedPairs matches;
  int64_t i = start;
  while (i < cases.front()->BitCountOrDie()) {
    if (i == start) {
      matches = PairsOfBitsWithSameSource(cases, i, query_engine);
    } else {
      MatchedPairs new_matches;
      absl::c_set_intersection(
          PairsOfBitsWithSameSource(cases, i, query_engine), matches,
          std::back_inserter(new_matches));
      matches = std::move(new_matches);
    }

    XLS_VLOG(5) << "  " << i << ": " << ToString(matches);
    if (matches.empty()) {
      break;
    }
    ++i;
  }
  XLS_VLOG(5) << " run of " << i - start;
  return i - start;
}

// Returns the length of the run of bit indices starting at 'start' for which
// the indexed bits of the given cases are distinct at each
// bit index. For example:
int64_t RunOfDistinctCaseBits(absl::Span<Node* const> cases, int64_t start,
                              const QueryEngine& query_engine) {
  XLS_VLOG(5) << "Finding runs of distinct case bit starting at " << start;
  int64_t i = start;
  while (i < cases.front()->BitCountOrDie() &&
         PairsOfBitsWithSameSource(cases, i, query_engine).empty()) {
    ++i;
  }
  XLS_VLOG(5) << " run of " << i - start << " bits";
  return i - start;
}

// Try to split OneHotSelect instructions into separate OneHotSelect
// instructions which have common cases. For example, if some of the cases of a
// OneHotSelect have the same first three bits, then this transformation will
// slice off these three bits (and the remainder) into separate OneHotSelect
// operation and replace the original OneHotSelect with a concat of thes sharded
// OneHotSelects.
//
// Returns the newly created OneHotSelect instructions if the transformation
// succeeded.
absl::StatusOr<std::vector<OneHotSelect*>> MaybeSplitOneHotSelect(
    OneHotSelect* ohs, const QueryEngine& query_engine) {
  // For *very* wide one-hot-selects this optimization can be very slow and make
  // a mess of the graph so limit it to 64 bits.
  if (!ohs->GetType()->IsBits() || ohs->GetType()->GetFlatBitCount() > 64) {
    return std::vector<OneHotSelect*>();
  }

  XLS_VLOG(4) << "Trying to split: " << ohs->ToString();
  if (VLOG_IS_ON(4)) {
    for (int64_t i = 0; i < ohs->cases().size(); ++i) {
      Node* cas = ohs->get_case(i);
      XLS_VLOG(4) << "  case (" << i << "): " << cas->ToString();
      for (int64_t j = 0; j < cas->BitCountOrDie(); ++j) {
        XLS_VLOG(4) << "    bit " << j << ": "
                    << ToString(GetBitSource(cas, j, query_engine));
      }
    }
  }

  int64_t start = 0;
  std::vector<Node*> ohs_slices;
  std::vector<OneHotSelect*> new_ohses;
  while (start < ohs->BitCountOrDie()) {
    int64_t run = RunOfDistinctCaseBits(ohs->cases(), start, query_engine);
    if (run == 0) {
      run = RunOfNonDistinctCaseBits(ohs->cases(), start, query_engine);
    }
    XLS_RET_CHECK_GT(run, 0);
    if (run == ohs->BitCountOrDie()) {
      // If all the cases are distinct (or have a matching pair) then just
      // return as there is nothing to slice.
      return std::vector<OneHotSelect*>();
    }
    XLS_ASSIGN_OR_RETURN(OneHotSelect * ohs_slice,
                         SliceOneHotSelect(ohs,
                                           /*start=*/start,
                                           /*width=*/run));
    new_ohses.push_back(ohs_slice);
    ohs_slices.push_back(ohs_slice);
    start += run;
  }
  std::reverse(ohs_slices.begin(), ohs_slices.end());
  XLS_VLOG(2) << absl::StrFormat("Splitting one-hot-select: %s",
                                 ohs->ToString());
  XLS_RETURN_IF_ERROR(ohs->ReplaceUsesWithNew<Concat>(ohs_slices).status());
  return new_ohses;
}

absl::StatusOr<bool> SimplifyNode(Node* node, const QueryEngine& query_engine,
                                  int64_t opt_level) {
  // Select with a constant selector can be replaced with the respective
  // case.
  if (node->Is<Select>() && node->As<Select>()->selector()->Is<Literal>()) {
    Select* sel = node->As<Select>();
    const Bits& selector = sel->selector()->As<Literal>()->value().bits();
    XLS_VLOG(2) << absl::StrFormat(
        "Simplifying select with constant selector: %s", node->ToString());
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

  // One-hot-select with a constant selector can be replaced with OR of the
  // activated cases.
  if (node->Is<OneHotSelect>() &&
      node->As<OneHotSelect>()->selector()->Is<Literal>() &&
      node->GetType()->IsBits()) {
    OneHotSelect* sel = node->As<OneHotSelect>();
    const Bits& selector = sel->selector()->As<Literal>()->value().bits();
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
    XLS_VLOG(2) << absl::StrFormat(
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
      XLS_VLOG(2) << absl::StrFormat(
          "Simplifying select with identical cases: %s", node->ToString());
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(sel->any_case()));
      return true;
    }
  }

  // OneHotSelect with identical cases can be replaced with a select between one
  // of the identical case and the value zero where the selector is: original
  // selector == 0
  if (node->Is<OneHotSelect>() && node->GetType()->IsBits()) {
    OneHotSelect* sel = node->As<OneHotSelect>();
    if (std::all_of(sel->cases().begin(), sel->cases().end(),
                    [&](Node* c) { return c == sel->get_case(0); })) {
      FunctionBase* f = node->function_base();
      XLS_ASSIGN_OR_RETURN(
          Node * selector_zero,
          f->MakeNode<Literal>(
              node->loc(), Value(UBits(0, sel->selector()->BitCountOrDie()))));
      XLS_ASSIGN_OR_RETURN(Node * is_zero,
                           f->MakeNode<CompareOp>(node->loc(), sel->selector(),
                                                  selector_zero, Op::kEq));
      XLS_ASSIGN_OR_RETURN(
          Node * selected_zero,
          f->MakeNode<Literal>(node->loc(),
                               Value(UBits(0, sel->BitCountOrDie()))));
      XLS_VLOG(2) << absl::StrFormat(
          "Simplifying one-hot-select with identical cases: %s",
          node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Select>(
                  is_zero, std::vector<Node*>{sel->get_case(0), selected_zero},
                  /*default_value=*/std::nullopt)
              .status());
      return true;
    }
  }

  // Replace a select among tuples to a tuple of selects. Handles both
  // kOneHotSelect and kSelect.
  if ((node->Is<Select>() || node->Is<OneHotSelect>()) &&
      node->GetType()->IsTuple()) {
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
      XLS_VLOG(2) << absl::StrFormat(
          "Decomposing tuple-typed one-hot-select: %s", node->ToString());
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
      XLS_VLOG(2) << absl::StrFormat("Decomposing tuple-typed select: %s",
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
      XLS_VLOG(2) << absl::StrFormat("Select with equivalent cases: %s",
                                     node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
              .status());
      return true;
    }
  }

  // We explode single-bit muxes into their constituent gates to expose more
  // optimization opportunities. Since this creates more ops in the general
  // case, we look for certain sub-cases:
  //
  // * At least one of the selected values is a literal.
  // * One of the selected values is also the selector.
  //
  // TODO(meheff): Handle one-hot select here as well.
  auto is_one_bit_mux = [&] {
    return node->Is<Select>() && node->GetType()->IsBits() &&
           node->BitCountOrDie() == 1 && node->operand(0)->BitCountOrDie() == 1;
  };
  if (NarrowingEnabled(opt_level) && is_one_bit_mux() &&
      (node->operand(1)->Is<Literal>() || node->operand(2)->Is<Literal>() ||
       (node->operand(0) == node->operand(1) ||
        node->operand(0) == node->operand(2)))) {
    FunctionBase* f = node->function_base();
    Select* select = node->As<Select>();
    XLS_RET_CHECK(!select->default_value().has_value()) << select->ToString();
    Node* s = select->operand(0);
    Node* on_false = select->get_case(0);
    Node* on_true = select->get_case(1);
    XLS_ASSIGN_OR_RETURN(
        Node * lhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s, on_true},
                            Op::kAnd));
    XLS_ASSIGN_OR_RETURN(Node * s_not,
                         f->MakeNode<UnOp>(select->loc(), s, Op::kNot));
    XLS_ASSIGN_OR_RETURN(
        Node * rhs,
        f->MakeNode<NaryOp>(select->loc(), std::vector<Node*>{s_not, on_false},
                            Op::kAnd));
    XLS_VLOG(2) << absl::StrFormat("Decomposing single-bit select: %s",
                                   node->ToString());
    XLS_RETURN_IF_ERROR(
        select
            ->ReplaceUsesWithNew<NaryOp>(std::vector<Node*>{lhs, rhs}, Op::kOr)
            .status());
    return true;
  }

  // Merge consecutive one-hot-select instructions if the predecessor operation
  // has only a single use.
  if (NarrowingEnabled(opt_level) && node->Is<OneHotSelect>()) {
    OneHotSelect* select = node->As<OneHotSelect>();
    absl::Span<Node* const> cases = select->cases();
    auto is_single_user_ohs = [](Node* n) {
      return n->Is<OneHotSelect>() && HasSingleUse(n);
    };
    if (std::any_of(cases.begin(), cases.end(), is_single_user_ohs)) {
      // Cases for the replacement one-hot-select.
      std::vector<Node*> new_cases;
      // Pieces of the selector for the replacement one-hot-select. These are
      // concatted together.
      std::vector<Node*> new_selector_parts;
      // When iterating through the cases to perform this optimization, cases
      // which are to remain unmodified (ie, not a single-use one-hot-select)
      // are passed over. This lambda gathers the passed over cases and updates
      // new_cases and new_selector_parts.
      int64_t unhandled_selector_bits = 0;
      auto add_unhandled_selector_bits = [&](int64_t index) -> absl::Status {
        if (unhandled_selector_bits != 0) {
          XLS_ASSIGN_OR_RETURN(Node * selector_part,
                               node->function_base()->MakeNode<BitSlice>(
                                   select->loc(), select->selector(),
                                   /*start=*/index - unhandled_selector_bits,
                                   /*width=*/
                                   unhandled_selector_bits));
          new_selector_parts.push_back(selector_part);
          for (int64_t i = index - unhandled_selector_bits; i < index; ++i) {
            new_cases.push_back(select->get_case(i));
          }
        }
        unhandled_selector_bits = 0;
        return absl::OkStatus();
      };
      // Iterate through the cases merging single-use one-hot-select cases.
      for (int64_t i = 0; i < cases.size(); ++i) {
        if (is_single_user_ohs(cases[i])) {
          OneHotSelect* ohs_operand = cases[i]->As<OneHotSelect>();
          XLS_RETURN_IF_ERROR(add_unhandled_selector_bits(i));
          // The selector bits for the predecessor one-hot-select need to be
          // ANDed with the original selector bit in the successor
          // one-hot-select. Example:
          //
          //   X = one_hot_select(selector={A, B, C},
          //                      cases=[x, y z])
          //   Y = one_hot_select(selector={..., S, ...},
          //                      cases=[..., X, ...])
          // Becomes:
          //
          //   Y = one_hot_select(
          //     selector={..., S & A, S & B, S & C, ...},
          //     cases=[..., A, B, C, ...])
          //
          XLS_ASSIGN_OR_RETURN(Node * selector_bit,
                               node->function_base()->MakeNode<BitSlice>(
                                   select->loc(), select->selector(),
                                   /*start=*/i, /*width=*/1));
          XLS_ASSIGN_OR_RETURN(
              Node * selector_bit_mask,
              node->function_base()->MakeNode<ExtendOp>(
                  select->loc(), selector_bit,
                  /*new_bit_count=*/ohs_operand->cases().size(), Op::kSignExt));
          XLS_ASSIGN_OR_RETURN(Node * masked_selector,
                               node->function_base()->MakeNode<NaryOp>(
                                   select->loc(),
                                   std::vector<Node*>{selector_bit_mask,
                                                      ohs_operand->selector()},
                                   Op::kAnd));
          new_selector_parts.push_back(masked_selector);
          for (Node* subcase : ohs_operand->cases()) {
            new_cases.push_back(subcase);
          }
        } else {
          unhandled_selector_bits++;
        }
      }
      XLS_RETURN_IF_ERROR(add_unhandled_selector_bits(cases.size()));
      // Reverse selector parts because concat operand zero is the msb.
      std::reverse(new_selector_parts.begin(), new_selector_parts.end());
      XLS_ASSIGN_OR_RETURN(Node * new_selector,
                           node->function_base()->MakeNode<Concat>(
                               select->loc(), new_selector_parts));
      XLS_VLOG(2) << absl::StrFormat("Merging consecutive one-hot-selects: %s",
                                     node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
              .status());
      return true;
    }
  }

  // Literal zero cases can be removed from OneHotSelects.
  if (SplitsEnabled(opt_level) && node->Is<OneHotSelect>() &&
      std::any_of(node->As<OneHotSelect>()->cases().begin(),
                  node->As<OneHotSelect>()->cases().end(),
                  [](Node* n) { return IsLiteralZero(n); })) {
    // Assemble the slices of the selector which correspond to non-zero cases.
    OneHotSelect* select = node->As<OneHotSelect>();
    std::vector<int64_t> nonzero_indices =
        IndicesWhereNot<Node*>(select->cases(), IsLiteralZero);
    if (nonzero_indices.empty()) {
      // If all cases were literal zeros, just replace with literal zero (chosen
      // arbitrarily as the first case).
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(select->cases().front()));
      return true;
    }
    XLS_ASSIGN_OR_RETURN(Node * new_selector,
                         GatherBits(select->selector(), nonzero_indices));
    std::vector<Node*> new_cases =
        GatherFromSequence(select->cases(), nonzero_indices);
    XLS_VLOG(2) << absl::StrFormat(
        "Literal zero cases removed from one-hot-select: %s", node->ToString());
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<OneHotSelect>(new_selector, new_cases)
            .status());
    return true;
  }

  // A two-way select with one arm which is literal zero can be replaced with an
  // AND.
  //
  //  sel(p cases=[x, 0]) => and(sign_ext(p), x)
  //
  // Since 'and' can't be reasoned through by conditional specialization and
  // other passes as easily we want to avoid doing this until fairly late in the
  // pipeline.
  auto is_select_with_zero = [&](Node* n) {
    if (!n->Is<Select>()) {
      return false;
    }
    Select* sel = n->As<Select>();
    return sel->GetType()->IsBits() && sel->selector()->BitCountOrDie() == 1 &&
           sel->cases().size() == 2 &&
           (IsLiteralZero(sel->get_case(0)) || IsLiteralZero(sel->get_case(1)));
  };
  if (SplitsEnabled(opt_level) && is_select_with_zero(node)) {
    Select* sel = node->As<Select>();
    int64_t nonzero_case_no = IsLiteralZero(sel->get_case(0)) ? 1 : 0;
    Node* selector = sel->selector();
    if (nonzero_case_no == 0) {
      XLS_ASSIGN_OR_RETURN(selector, sel->function_base()->MakeNode<UnOp>(
                                         sel->loc(), selector, Op::kNot));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * sign_ext_selector,
        node->function_base()->MakeNode<ExtendOp>(
            node->loc(), selector,
            /*new_bit_count=*/sel->BitCountOrDie(), Op::kSignExt));
    XLS_VLOG(2) << absl::StrFormat("Binary select with zero case: %s",
                                   node->ToString());
    XLS_RETURN_IF_ERROR(
        node->ReplaceUsesWithNew<NaryOp>(
                std::vector<Node*>{sel->get_case(nonzero_case_no),
                                   sign_ext_selector},
                Op::kAnd)
            .status());
    return true;
  }

  // "Squeeze" the width of the mux when bits are known to reduce the cost of
  // the operation.
  //
  // Sel(...) => Concat(Known, Sel(...), Known)
  if (SplitsEnabled(opt_level)) {
    auto is_squeezable_mux = [&](Bits* msb, Bits* lsb) {
      if (!node->Is<Select>() || !node->GetType()->IsBits() ||
          !query_engine.IsTracked(node)) {
        return false;
      }
      int64_t leading_known = bits_ops::CountLeadingOnes(
          ternary_ops::ToKnownBits(query_engine.GetTernary(node).Get({})));
      int64_t trailing_known = bits_ops::CountTrailingOnes(
          ternary_ops::ToKnownBits(query_engine.GetTernary(node).Get({})));
      if (leading_known == 0 && trailing_known == 0) {
        return false;
      }
      int64_t bit_count = node->BitCountOrDie();
      *msb =
          ternary_ops::ToKnownBitsValues(query_engine.GetTernary(node).Get({}))
              .Slice(/*start=*/bit_count - leading_known,
                     /*width=*/leading_known);
      if (leading_known == trailing_known && leading_known == bit_count) {
        // This is just a constant value, just say we only have high constant
        // bits, the replacement will be the same.
        return true;
      }
      *lsb =
          ternary_ops::ToKnownBitsValues(query_engine.GetTernary(node).Get({}))
              .Slice(/*start=*/0, /*width=*/trailing_known);
      return true;
    };
    Bits const_msb, const_lsb;
    if (is_squeezable_mux(&const_msb, &const_lsb)) {
      std::function<absl::StatusOr<Select*>(Select*, std::vector<Node*>)>
          make_select =
              [](Select* original,
                 std::vector<Node*> new_cases) -> absl::StatusOr<Select*> {
        std::optional<Node*> new_default;
        if (original->default_value().has_value()) {
          new_default = new_cases.back();
          new_cases.pop_back();
        }
        return original->function_base()->MakeNode<Select>(
            original->loc(), original->selector(), new_cases, new_default);
      };
      return SqueezeSelect(const_msb, const_lsb, make_select,
                           node->As<Select>());
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
    //  The variable names correspond to the names of the nodes in the diagrams
    //  below.
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
      XLS_VLOG(2) << absl::StrFormat(
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
    XLS_VLOG(2) << absl::StrFormat("Decompose single-bit one-hot-select: %s",
                                   node->ToString());
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<NaryOp>(
                                std::vector<Node*>{and0, and1}, Op::kOr)
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
    XLS_VLOG(2) << absl::StrFormat(
        "Replace single-bit input one-hot to concat: %s", node->ToString());
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
              Value(
                  UBits(0, /*bit_count=*/node->operand(0)->BitCountOrDie()))));
      XLS_ASSIGN_OR_RETURN(Node * operand_eq_zero,
                           node->function_base()->MakeNode<CompareOp>(
                               node->loc(), node->operand(0), zero, Op::kEq));
      XLS_VLOG(2) << absl::StrFormat(
          "Replace one-hot with mutually exclusive input: %s",
          node->ToString());
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<Concat>(
                  std::vector{operand_eq_zero, node->operand(0)})
              .status());
      return true;
    }
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> SelectSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results) const {
  TernaryQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());
  bool changed = false;
  for (Node* node : TopoSort(func)) {
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyNode(node, query_engine, opt_level_));
    changed = changed || node_changed;
  }

  // Use a worklist to split OneHotSelects based on common bits in the cases
  // because this transformation creates many more OneHotSelects exposing
  // further opportunities for optimizations.
  if (SplitsEnabled(opt_level_)) {
    std::deque<OneHotSelect*> worklist;
    for (Node* node : func->nodes()) {
      if (node->Is<OneHotSelect>()) {
        worklist.push_back(node->As<OneHotSelect>());
      }
    }
    while (!worklist.empty()) {
      OneHotSelect* ohs = worklist.front();
      worklist.pop_front();
      // Note that query_engine may be stale at this point but that is
      // ok. TernaryQueryEngine::IsTracked will return false for new nodes which
      // have not been analyzed.
      XLS_ASSIGN_OR_RETURN(std::vector<OneHotSelect*> new_ohses,
                           MaybeSplitOneHotSelect(ohs, query_engine));
      if (!new_ohses.empty()) {
        changed = true;
        worklist.insert(worklist.end(), new_ohses.begin(), new_ohses.end());
      }
    }
  }
  return changed;
}

REGISTER_OPT_PASS(SelectSimplificationPass, pass_config::kOptLevel);

}  // namespace xls
