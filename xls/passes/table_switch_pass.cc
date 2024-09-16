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
#include "xls/passes/table_switch_pass.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

// If the given node is an Op::kEq or Op::kNe node which compares against a
// bits-typed constant which fits in a uint64_t, then return the two operands of
// the comparison as a Uint64Comparison. Returns std::nullopt otherwise.
struct Uint64Comparison {
  Node* index;
  uint64_t key;
  Op comparison_op;
};
std::optional<Uint64Comparison> MatchCompareEqAgainstUint64(
    Node* node, QueryEngine& query_engine) {
  if (node->op() != Op::kEq && node->op() != Op::kNe) {
    return std::nullopt;
  }
  auto to_uint64_constant =
      [&query_engine](Node* n) -> std::optional<uint64_t> {
    if (!n->GetType()->IsBits()) {
      return std::nullopt;
    }
    const std::optional<Bits> constant = query_engine.KnownValueAsBits(n);
    if (!constant.has_value() || !constant->FitsInUint64()) {
      return std::nullopt;
    }
    return *constant->ToUint64();
  };
  if (std::optional<uint64_t> constant = to_uint64_constant(node->operand(0));
      constant.has_value()) {
    // Constant is the lhs.
    return Uint64Comparison{.index = node->operand(1),
                            .key = *constant,
                            .comparison_op = node->op()};
  }
  if (std::optional<uint64_t> constant = to_uint64_constant(node->operand(1));
      constant.has_value()) {
    // Constant is the rhs.
    return Uint64Comparison{.index = node->operand(0),
                            .key = *constant,
                            .comparison_op = node->op()};
  }
  return std::nullopt;
}

// Return type of the MatchLinks function. See MatchLinks comment for details.
struct Link {
  Node* node;
  Node* index;
  uint64_t key;
  const Value& value;
  Node* next;
  std::optional<int64_t> next_case = std::nullopt;
};

// Matches the given node against the following Select instruction patterns:
//
//  Sel({index} == {key}, cases=[{next}, {value}])
//  Sel({key} == {index}, cases=[{next}, {value}])
//  Sel({index} != {key}, cases=[{value}, {next}])
//  Sel({key} != {index}, cases=[{value}, {next}])
//
//  and the equivalents with a default value, as well as:
//
//  PrioritySel({index} == {key}, cases=[{value}], default={next})
//  PrioritySel({key} == {index}, cases=[{value}], default={next})
//  PrioritySel({index} != {key}, cases=[{next}], default={value})
//  PrioritySel({key} != {index}, cases=[{next}], default={value})
//
// Where:
//   {key}   : A literal whose value fits in a uint64_t.
//   {index} : Equal to the argument 'index' if 'index' is non-null.
//   {next}  : Arbitrary node.
//   {value} : A literal node.
//
// We also try to recognize PrioritySelects that merge multiple links in the
// chain, where the selector is a concat of single bit selectors. In this case,
// we return multiple Links, and all intermediate Links include a {next_case}
// entry to indicate which case of the PrioritySelect picks up from there.
//
// If a match is found, the respective Link fields are filled in (as named
// above). Otherwise an empty vector is returned.
std::vector<Link> MatchLinks(QueryEngine& query_engine, Node* node,
                             Node* index = nullptr, int64_t first_case = 0) {
  if (!node->OpIn({Op::kSel, Op::kPrioritySel})) {
    return {};
  }

  if (node->Is<Select>() ||
      node->As<PrioritySelect>()->selector()->BitCountOrDie() == 1) {
    Node* selector;
    Node* false_case;
    Node* true_case;
    if (node->Is<Select>()) {
      Select* sel = node->As<Select>();
      if (sel->cases().size() + (sel->default_value().has_value() ? 1 : 0) !=
          2) {
        return {};
      }
      selector = sel->selector();
      false_case = sel->get_case(0);
      if (sel->default_value().has_value()) {
        true_case = *sel->default_value();
      } else {
        true_case = sel->get_case(1);
      }
    } else {
      PrioritySelect* sel = node->As<PrioritySelect>();
      if (sel->cases().size() != 1) {
        return {};
      }
      selector = sel->selector();
      true_case = sel->get_case(0);
      false_case = sel->default_value();
    }

    // The selector must be a comparison to a literal which fits in a uint64_t.
    std::optional<Uint64Comparison> match =
        MatchCompareEqAgainstUint64(selector, query_engine);
    if (!match.has_value()) {
      return {};
    }

    Node* next;
    Node* value_node;
    if (match->comparison_op == Op::kEq) {
      next = false_case;
      value_node = true_case;
    } else {
      CHECK_EQ(match->comparison_op, Op::kNe);
      next = true_case;
      value_node = false_case;
    }

    // The select instruction must have a literal value for the index-match
    // case.
    if (!value_node->Is<Literal>()) {
      return {};
    }
    const Value& value = value_node->As<Literal>()->value();

    // The index, if given, must match the non-literal operand of the eq.
    if (index != nullptr && index != match->index) {
      return {};
    }

    return {Link{.node = node,
                 .index = match->index,
                 .key = match->key,
                 .value = value,
                 .next = next}};
  }

  PrioritySelect* sel = node->As<PrioritySelect>();
  CHECK_LT(first_case, sel->cases().size());

  // We currently only support priority selects with a single bit selector, or
  // where the selector is a concat of single bit selectors.
  if (!sel->selector()->Is<Concat>()) {
    return {};
  }

  absl::Span<Node* const> selector_operands = sel->selector()->operands();
  std::vector<Node*> selectors(selector_operands.begin(),
                               selector_operands.end());
  // Reverse the selectors to match the order of the cases.
  absl::c_reverse(selectors);

  int64_t current_selector = -1;
  int64_t selector_bit = first_case;
  for (int64_t i = 0; i < selectors.size(); ++i) {
    int64_t bit_count = selectors[i]->BitCountOrDie();
    if (selector_bit < bit_count) {
      current_selector = i;
      break;
    }
    selector_bit -= bit_count;
  }
  CHECK_GE(current_selector, 0);

  std::vector<Link> links;
  std::optional<int64_t> next_case;
  for (int64_t i = first_case; i < sel->cases().size(); ++i) {
    Node* selector = selectors[current_selector];
    if (selector->BitCountOrDie() > 1) {
      // We only recognize single-bit selectors as potential links in a chain.
      break;
    }
    // The selector must be a comparison to a literal which fits in a uint64_t.
    std::optional<Uint64Comparison> match =
        MatchCompareEqAgainstUint64(selector, query_engine);
    if (!match.has_value()) {
      break;
    }

    Node* true_case = sel->get_case(i);

    Node* false_case;
    if (i < sel->cases().size() - 1) {
      false_case = sel;
      next_case = i + 1;
    } else {
      false_case = sel->default_value();
      next_case = std::nullopt;
    }

    Node* next;
    Node* value_node;
    if (match->comparison_op == Op::kEq) {
      value_node = true_case;
      next = false_case;
    } else if (match->comparison_op == Op::kNe && !next_case.has_value()) {
      value_node = false_case;
      next = true_case;
    } else {
      break;
    }

    // The select instruction must have a literal value for the index-match
    // case.
    if (!value_node->Is<Literal>()) {
      break;
    }
    const Value& value = value_node->As<Literal>()->value();

    // The index, if given, must match the non-literal operand of the eq.
    if (index == nullptr) {
      index = match->index;
    } else if (index != match->index) {
      break;
    }

    links.push_back({.node = node,
                     .index = match->index,
                     .key = match->key,
                     .value = value,
                     .next = next,
                     .next_case = next_case});
    current_selector++;
  }
  return links;
}

// Returns an array Value of the table lookup effectively performed by the given
// chain of selects. For example, given the following chain:
//
//                        else_value
//                             |    Value_2
//                             |   /
// link[2]:  (index == 2) -> Select
//                             |    Value_1
//                             |   /
// link[1]:  (index == 1) -> Select
//                             |    Value_0
//                             |   /
// link[0]:  (index == 0) -> Select
//                             |
//
// The returned array might be: {Value_0, Value_1, Value_2, else_value}
//
// Returns std::nullopt if the chain cannot be represented as an index into a
// literal array.
absl::StatusOr<std::optional<Value>> LinksToTable(
    absl::Span<const Link> links) {
  if (links.empty()) {
    VLOG(3) << "Empty chain.";
    return std::nullopt;
  }

  // Compute the size of the space indexed by the index of the select
  // chain. Used to test if the selects cover the entire index space. If the
  // index space is huge (>=2^64) we set this value to std::nullopt and
  // consider the index space size to be infinitely large for the purposes of
  // this transformation.
  int64_t index_width = links.front().index->GetType()->GetFlatBitCount();
  std::optional<uint64_t> index_space_size = std::nullopt;
  if (index_width < 63) {
    index_space_size = uint64_t{1} << index_width;
  }

  // Gather all selectable Values in a map indexed by the uint64_t index
  // associated with the Value.
  absl::flat_hash_map<uint64_t, Value> map;
  uint64_t min_key = std::numeric_limits<uint64_t>::max();
  uint64_t max_key = 0;
  for (const Link& link : links) {
    if (map.contains(link.key)) {
      // We're iterating from the bottom of the chain up, so if a key appears
      // more than once then the value associated with the later instance is
      // dead and can be ignored.
      continue;
    }
    map[link.key] = link.value;
    min_key = std::min(min_key, link.key);
    max_key = std::max(max_key, link.key);
  }

  // Converts the dense map of Values into an array of Values.
  auto map_to_array_value = [](const absl::flat_hash_map<uint64_t, Value>& m)
      -> absl::StatusOr<Value> {
    std::vector<Value> values(m.size());
    for (auto& [key, value] : m) {
      XLS_RET_CHECK_LT(key, values.size());
      values[key] = value;
    }
    return Value::Array(values);
  };

  if (index_space_size.has_value() && map.size() == index_space_size.value()) {
    // The on-true cases of the selects cover the entire index space. The final
    // on-false case (else_value in the diagram above) is dead and need not be
    // considered.
    XLS_RET_CHECK_EQ(min_key, 0);
    XLS_RET_CHECK_EQ(max_key, index_space_size.value() - 1);
    XLS_ASSIGN_OR_RETURN(Value array, map_to_array_value(map));
    return array;
  }

  // The entire index space is not covered so the final on-false case
  // (else_value in the diagram above) is not dead and must be a literal in
  // order for this to be converted into a table lookup.
  if (!links.back().next->Is<Literal>()) {
    VLOG(3) << "Final fall-through case is not a literal: "
            << links.back().next->ToString();
    return std::nullopt;
  }
  const Value& else_value = links.back().next->As<Literal>()->value();

  VLOG(3) << "Index width: " << index_width;
  if (index_space_size.has_value()) {
    VLOG(3) << "Index space size: " << index_space_size.value();
  }
  VLOG(3) << "map.size(): " << map.size();

  // As a special case, even when dealing with an "infinite" index space, we can
  // rely on the saturating semantics of ArrayIndex to convert the select
  // sequence to a table lookup. Specifically, if the index is OOB, ArrayIndex
  // returns the element of the array at the max index - so as long as we fill
  // the last entry with the else_value, we can proceed as if the index space
  // was [0, max_key + 1] (with size = max_key + 2).
  index_space_size =
      std::min(index_space_size.value_or(std::numeric_limits<uint64_t>::max()),
               max_key + 2);
  if (map.size() < *index_space_size && map.size() * 2 > *index_space_size) {
    // There are holes in the index space, but most of the index space is
    // covered. Necessarily, if the index assumes one of the missing values the
    // expression returns else_value so fill in the holes with else_value and
    // return.
    for (uint64_t i = 0; i < index_space_size; ++i) {
      if (!map.contains(i)) {
        map[i] = else_value;
      }
    }
    XLS_RET_CHECK_EQ(map.size(), index_space_size.value());
    XLS_ASSIGN_OR_RETURN(Value array, map_to_array_value(map));
    return array;
  }

  // Possible improvements that could be handled here:
  //  - Handling large start indices (e.g., shifted index spaces).
  //  - Support non-1 index strides:
  //    - Extract a constant factor, e.g., 0, 4, 8, ... -> 0, 1, 2, ...
  //  - Handle "partial" chains - those that only cover part of the match space

  VLOG(3) << "Cannot convert link chain to table lookup; min_key: " << min_key
          << ", max_key: " << max_key << ", map.size(): " << map.size();
  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> TableSwitchPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  StatelessQueryEngine query_engine;

  bool changed = false;
  absl::flat_hash_set<Node*> transformed;
  for (Node* node : ReverseTopoSort(f)) {
    VLOG(3) << "Considering node: " << node->ToString();
    if (transformed.contains(node)) {
      VLOG(3) << absl::StreamFormat("Already transformed %s", node->GetName());
      continue;
    }
    // Check if this node is the start of a chain of selects. This also
    // identifies the common index.
    int64_t first_case = 0;
    std::vector<Link> links;
    if (node->Is<PrioritySelect>()) {
      for (; first_case < node->As<PrioritySelect>()->cases().size();
           ++first_case) {
        links = MatchLinks(query_engine, node,
                           /*index=*/nullptr, first_case);
        if (!links.empty()) {
          break;
        }
      }
    } else {
      links = MatchLinks(query_engine, node);
    }
    if (links.empty()) {
      VLOG(3) << absl::StreamFormat("%s is not the start of a chain.",
                                    node->GetName());
      continue;
    }

    // Walk up the graph adding as many links as possible. When done, each
    // element in 'links' represents one select instruction in the chain. For
    // example:
    //
    //                             next_n
    //                               |    Value_n
    //                               |   /
    // link[n]:  (index == C_n) -> Select
    //                               |
    //                              ...
    //                               |    Value_1
    //                               |   /
    // link[1]:  (index == C_1) -> Select
    //                               |    Value_0
    //                               |   /
    // link[0]:  (index == C_0) -> Select
    //                               |
    //
    // In each link, the 'next' value points up to the next element in the
    // chain.
    std::vector<Link> new_links =
        MatchLinks(query_engine, links.back().next, links.back().index,
                   links.back().next_case.value_or(0));
    while (!new_links.empty()) {
      absl::c_move(new_links, std::back_inserter(links));
      new_links =
          MatchLinks(query_engine, links.back().next, links.back().index,
                     links.back().next_case.value_or(0));
    }

    VLOG(3) << absl::StreamFormat("Chain of length %d found", links.size());
    if (VLOG_IS_ON(4)) {
      for (auto it = links.rbegin(); it != links.rend(); ++it) {
        VLOG(4) << absl::StreamFormat("  %s = (%s == %d) ? %s : %s",
                                      it->node->GetName(), it->index->GetName(),
                                      it->key, it->value.ToString(),
                                      it->next->GetName());
      }
    }

    if (links.size() <= 2) {
      VLOG(3) << "Chain is too short.";
      continue;
    }

    // Try to convert the chain into a table representing the lookup being
    // performed.
    XLS_ASSIGN_OR_RETURN(std::optional<Value> table, LinksToTable(links));
    if (!table.has_value()) {
      continue;
    }

    VLOG(3) << absl::StreamFormat(
        "Replacing chain starting at %s with index of array: %s",
        node->GetName(), table.value().ToString());

    XLS_ASSIGN_OR_RETURN(Literal * array_literal,
                         f->MakeNode<Literal>(node->loc(), table.value()));
    if (first_case > 0) {
      XLS_RET_CHECK(node->Is<PrioritySelect>());
      PrioritySelect* sel = node->As<PrioritySelect>();
      XLS_ASSIGN_OR_RETURN(Node * array_index,
                           sel->function_base()->MakeNode<ArrayIndex>(
                               sel->loc(), array_literal,
                               std::vector<Node*>({links.back().index})));
      XLS_ASSIGN_OR_RETURN(
          Node * truncated_selector,
          sel->function_base()->MakeNode<BitSlice>(
              node->loc(), sel->selector(), /*start=*/0, /*width=*/first_case));
      XLS_RETURN_IF_ERROR(sel->ReplaceUsesWithNew<PrioritySelect>(
                                 truncated_selector,
                                 sel->cases().subspan(0, /*len=*/first_case),
                                 /*default_value=*/array_index)
                              .status());
    } else {
      XLS_RETURN_IF_ERROR(
          node->ReplaceUsesWithNew<ArrayIndex>(
                  array_literal, std::vector<Node*>({links.back().index}))
              .status());
    }

    // Mark the replaced nodes as being transformed to avoid quadratic
    // behavior. These nodes will be skipped in future iterations.
    for (const Link& link : links) {
      transformed.insert(link.node);
    }
    changed = true;
  }

  return changed;
}

REGISTER_OPT_PASS(TableSwitchPass);

}  // namespace xls
