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
#include <limits>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

// If the given node is an Op::kEq or Op::kNe node which compares against a
// bits-typed literal which fits in a uint64_t, then return the two operands of
// the comparison as a Uint64Comparison. Returns std::nullopt otherwise.
struct Uint64Comparison {
  Node* index;
  uint64_t key;
  Op comparison_op;
};
static std::optional<Uint64Comparison> MatchCompareEqAgainstUint64(Node* node) {
  if (node->op() != Op::kEq && node->op() != Op::kNe) {
    return std::nullopt;
  }
  auto is_uint64_literal = [](Node* n) {
    return n->Is<Literal>() && n->As<Literal>()->value().IsBits() &&
           n->As<Literal>()->value().bits().FitsInUint64();
  };
  if (is_uint64_literal(node->operand(0))) {
    // Literal is the lhs.
    return Uint64Comparison{
        node->operand(1),
        node->operand(0)->As<Literal>()->value().bits().ToUint64().value(),
        node->op()};
  } else if (is_uint64_literal(node->operand(1))) {
    // Literal is the rhs.
    return Uint64Comparison{
        node->operand(0),
        node->operand(1)->As<Literal>()->value().bits().ToUint64().value(),
        node->op()};
  }
  return std::nullopt;
}

// Return type of the MatchLink function. See MatchLink comment for details.
struct Link {
  Select* node;
  Node* index;
  uint64_t key;
  const Value& value;
  Node* next;
};

// Matches the given node against the following Select instruction patterns:
//
//  Sel({index} == {key}, cases=[{next}, {value}])
//  Sel({key} == {index}, cases=[{next}, {value}])
//  Sel({index} != {key}, cases=[{value}, {next}])
//  Sel({key} != {index}, cases=[{value}, {next}])
//
// Where:
//   {key}   : A literal whose value fits in a uint64_t.
//   {index} : Equal to the argument 'index' if 'index' is non-null.
//   {next}  : Arbitrary node.
//   {value} : A literal node.
//
// If a match is found, the respective Link fields are filled in (as named
// above). Otherwise std::nullopt is returned.
static std::optional<Link> MatchLink(Node* node, Node* index = nullptr) {
  if (!IsBinarySelect(node)) {
    return std::nullopt;
  }
  Select* select = node->As<Select>();

  // The selector must be a comparison to a literal which fits in a uint64_t.
  std::optional<Uint64Comparison> match =
      MatchCompareEqAgainstUint64(select->selector());
  if (!match.has_value()) {
    return std::nullopt;
  }

  Node* next;
  Node* value_node;
  if (match->comparison_op == Op::kEq) {
    next = select->get_case(0);
    value_node = select->get_case(1);
  } else {
    CHECK_EQ(match->comparison_op, Op::kNe);
    next = select->get_case(1);
    value_node = select->get_case(0);
  }

  // The select instruction must have a literal value for the selector is true
  // case.
  if (!value_node->Is<Literal>()) {
    return std::nullopt;
  }
  const Value& value = value_node->As<Literal>()->value();

  // The index, if given, must match the non-literal operand of the eq.
  if (index != nullptr && index != match->index) {
    return std::nullopt;
  }

  return Link{select, match->index, match->key, value, next};
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
static absl::StatusOr<std::optional<Value>> LinksToTable(
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

  if (index_space_size.has_value() && map.size() < index_space_size.value() &&
      map.size() * 2 > index_space_size.value()) {
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

  // As a special case we can rely on the saturating semantics of ArrayIndex to
  // convert the select sequence to a table lookup. Specifically, if the index
  // is OOB, ArrayIndex returns the element of the array at the max index. We
  // put the else_value at the maximum index. For this to work, the map must be
  // dense from zero on up.
  if (map.size() == max_key + 1) {
    // The condition above should imply that the min key is zero.
    XLS_RET_CHECK_EQ(min_key, 0);
    map[max_key + 1] = else_value;
    XLS_ASSIGN_OR_RETURN(Value array, map_to_array_value(map));
    return array;
  }

  // Possible improvements that could be handled here:
  //  - Handling non-zero start indexes.
  //  - Support non-1 index strides:
  //    - Extract a constant factor, e.g., 0, 4, 8, ... -> 0, 1, 2, ...
  //  - Handle "partial" chains - those that only cover part of the match space

  VLOG(3) << "Cannot convert link chain to table lookup";
  return std::nullopt;
}

absl::StatusOr<bool> TableSwitchPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
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
    std::optional<Link> start = MatchLink(node);
    if (!start.has_value()) {
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
    Node* next = start->next;
    Node* index = start->index;
    std::vector<Link> links = {start.value()};
    while (std::optional<Link> link = MatchLink(next, index)) {
      next = link->next;
      links.push_back(link.value());
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
    XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<ArrayIndex>(
                                array_literal, std::vector<Node*>({index}))
                            .status());

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
