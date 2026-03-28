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

#include "xls/ir/node_util.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {
namespace {

// Produces a vector of nodes from the given span removing duplicates. The
// first instance of each node is kept and subsequent duplicates are dropped,
// e.g. RemoveRedundantNodes({a, b, a, c, a, d}) -> {a, b, c, d}.
std::vector<Node*> RemoveRedundantNodes(
    absl::Span<Node* const> values,
    std::optional<std::function<bool(const Value&)>> drop_literal =
        std::nullopt) {
  absl::flat_hash_set<Node*> unique_values_set(values.begin(), values.end());
  std::vector<Node*> unique_values_vector;
  unique_values_vector.reserve(unique_values_set.size());
  for (Node* value : values) {
    if (drop_literal.has_value() && value->Is<Literal>() &&
        (*drop_literal)(value->As<Literal>()->value())) {
      continue;
    }
    if (auto extracted = unique_values_set.extract(value)) {
      unique_values_vector.push_back(extracted.value());
    }
  }
  return unique_values_vector;
}

}  // namespace

std::optional<ShiftedBitView> IsOneShiftedBit(Node* node) {
  // Match: shll(zext(b), literal(k))
  if (node->op() == Op::kShll) {
    Node* shift_base = node->operand(0);
    Node* shift_amount = node->operand(1);
    if (shift_base->op() == Op::kZeroExt &&
        IsSingleBitType(shift_base->operand(0)) &&
        shift_amount->Is<Literal>()) {
      absl::StatusOr<uint64_t> k_u64 =
          shift_amount->As<Literal>()->value().bits().ToUint64();
      if (!k_u64.ok()) {
        return std::nullopt;
      }
      return ShiftedBitView{.b = shift_base->operand(0),
                            .k = static_cast<int64_t>(*k_u64)};
    }
  }

  // Match: concat(0..., b, 0...)
  if (node->Is<Concat>()) {
    std::optional<int64_t> b_operand_index;
    for (int64_t i = 0; i < node->operand_count(); ++i) {
      Node* operand = node->operand(i);
      if (!IsSingleBitType(operand)) {
        continue;
      }
      if (b_operand_index.has_value()) {
        // More than one 1-bit operand.
        return std::nullopt;
      }
      b_operand_index = i;
    }
    if (!b_operand_index.has_value()) {
      return std::nullopt;
    }

    for (int64_t i = 0; i < node->operand_count(); ++i) {
      if (i == *b_operand_index) {
        continue;
      }
      if (!IsLiteralZero(node->operand(i))) {
        return std::nullopt;
      }
    }

    int64_t k = 0;
    for (int64_t i = *b_operand_index + 1; i < node->operand_count(); ++i) {
      k += node->operand(i)->BitCountOrDie();
    }
    return ShiftedBitView{.b = node->operand(*b_operand_index), .k = k};
  }

  return std::nullopt;
}

bool IsLiteralWithRunOfSetBits(Node* node, int64_t* leading_zero_count,
                               int64_t* set_bit_count,
                               int64_t* trailing_zero_count) {
  if (!node->Is<Literal>()) {
    return false;
  }
  if (!node->GetType()->IsBits()) {
    return false;
  }
  Literal* literal = node->As<Literal>();
  const Bits& bits = literal->value().bits();
  return bits.HasSingleRunOfSetBits(leading_zero_count, set_bit_count,
                                    trailing_zero_count);
}

absl::StatusOr<Node*> GatherUnknownBits(Node* node, TernarySpan ts) {
  return GatherBits(node, bits_ops::Not(ternary_ops::ToKnownBits(ts)));
}

absl::StatusOr<Node*> GatherBits(Node* node,
                                 absl::Span<int64_t const> indices) {
  XLS_RET_CHECK(node->GetType()->IsBits());

  absl::flat_hash_set<int64_t> unique_indices;
  unique_indices.insert(indices.begin(), indices.end());
  XLS_RET_CHECK_EQ(unique_indices.size(), indices.size())
      << "Gather indices not unique.";

  InlineBitmap mask(node->BitCountOrDie());
  for (int64_t index : indices) {
    mask.Set(index);
  }
  return GatherBits(node, Bits::FromBitmap(std::move(mask)));
}

// TODO(allight): We should clean this up and deduplicate this and
// RemoveKnownBits.
absl::StatusOr<Node*> GatherBits(
    Node* node, LeafTypeTreeView<std::vector<int64_t>> positions) {
  XLS_RET_CHECK(node->GetType()->IsBits());

  absl::StatusOr<LeafTypeTree<Bits>> mask =
      leaf_type_tree::MapIndex<Bits, std::vector<int64_t>>(
          positions,
          [](Type* leaf_type, absl::Span<const int64_t> indices,
             absl::Span<const int64_t>) -> absl::StatusOr<Bits> {
            absl::flat_hash_set<int64_t> unique_indices;
            unique_indices.insert(indices.begin(), indices.end());
            XLS_RET_CHECK_EQ(unique_indices.size(), indices.size())
                << "Gather indices not unique.";

            InlineBitmap mask(leaf_type->GetFlatBitCount());
            for (int64_t index : indices) {
              mask.Set(index);
            }
            return Bits::FromBitmap(std::move(mask));
          });
  if (!mask.ok()) {
    return std::move(mask).status();
  }
  return GatherBits(node, mask->AsView());
}

absl::StatusOr<Node*> GatherBits(Node* node, const Bits& mask) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  XLS_RET_CHECK_EQ(node->BitCountOrDie(), mask.bit_count());

  FunctionBase* f = node->function_base();
  if (mask.IsZero()) {
    // Return a literal with a Bits value of zero width.
    return f->MakeNode<Literal>(node->loc(), Value(Bits()));
  }
  if (mask.IsAllOnes()) {
    return node;
  }

  std::vector<Node*> slices;
  int64_t pos = 0;
  while (pos < mask.bit_count()) {
    if (!mask.Get(pos)) {
      pos++;
      continue;
    }

    int64_t width = 1;
    while (pos + width < mask.bit_count() && mask.Get(pos + width)) {
      width++;
    }
    XLS_ASSIGN_OR_RETURN(Node * slice,
                         f->MakeNode<BitSlice>(node->loc(), node,
                                               /*start=*/pos, width));
    slices.push_back(slice);
    pos += width;
  }
  if (slices.size() == 1) {
    return slices[0];
  }
  absl::c_reverse(slices);
  return f->MakeNode<Concat>(node->loc(), slices);
}

absl::StatusOr<Node*> FillPattern(TernarySpan pattern, Node* node) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  const int64_t unknown_bits = absl::c_count_if(
      pattern, [](TernaryValue v) { return v == TernaryValue::kUnknown; });
  XLS_RET_CHECK_EQ(node->BitCountOrDie(), unknown_bits);
  if (unknown_bits == pattern.size()) {
    return node;
  }
  FunctionBase* f = node->function_base();
  Bits known_values = ternary_ops::ToKnownBitsValues(pattern);

  std::vector<Node*> slices;
  int64_t result_pos = 0;
  int64_t node_pos = 0;
  while (result_pos < pattern.size()) {
    Node* slice;
    int64_t width = 1;
    while (result_pos + width < pattern.size() &&
           (pattern[result_pos + width] == TernaryValue::kUnknown) ==
               (pattern[result_pos] == TernaryValue::kUnknown)) {
      width++;
    }
    if (pattern[result_pos] == TernaryValue::kUnknown) {
      XLS_ASSIGN_OR_RETURN(slice,
                           f->MakeNode<BitSlice>(node->loc(), node,
                                                 /*start=*/node_pos, width));
      node_pos += width;
    } else {
      XLS_ASSIGN_OR_RETURN(
          slice, f->MakeNode<Literal>(node->loc(), Value(known_values.Slice(
                                                       result_pos, width))));
    }
    slices.push_back(slice);
    result_pos += width;
  }
  XLS_RET_CHECK_EQ(node_pos, node->BitCountOrDie());
  if (slices.size() == 1) {
    return slices[0];
  }
  absl::c_reverse(slices);
  return f->MakeNode<Concat>(node->loc(), slices);
}

namespace {
absl::Status ToTreeOfNodesInPlace(Node* node,
                                  MutableLeafTypeTreeView<Node*> segment) {
  if (segment.size() == 0) {
    return absl::OkStatus();
  }
  XLS_RET_CHECK_EQ(segment.type(), node->GetType());
  if (node->GetType()->IsBits() || node->GetType()->IsToken() ||
      (node->GetType()->IsTuple() &&
       node->GetType()->AsTupleOrDie()->element_types().empty()) ||
      (node->GetType()->IsArray() &&
       node->GetType()->AsArrayOrDie()->empty())) {
    segment.elements().front() = node;
    return absl::OkStatus();
  }
  if (node->GetType()->IsTuple()) {
    for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * tup_idx,
          node->function_base()->MakeNodeWithName<TupleIndex>(
              node->loc(), node, i,
              node->HasAssignedName()
                  ? absl::StrFormat("%s_idx_%d", node->GetNameView(), i)
                  : ""));
      XLS_RETURN_IF_ERROR(
          ToTreeOfNodesInPlace(tup_idx, segment.AsMutableView({i})));
    }
    return absl::OkStatus();
  }
  XLS_RET_CHECK(node->GetType()->IsArray()) << "unexpected type for " << node;
  for (int64_t i = 0; i < node->GetType()->AsArrayOrDie()->size(); ++i) {
    XLS_ASSIGN_OR_RETURN(Node * idx, node->function_base()->MakeNode<Literal>(
                                         node->loc(), Value(UBits(i, 64))));
    XLS_ASSIGN_OR_RETURN(
        Node * arr_idx,
        node->function_base()->MakeNodeWithName<ArrayIndex>(
            node->loc(), node, absl::Span<Node* const>{idx},
            /*assumed_in_bounds=*/true,
            node->HasAssignedName()
                ? absl::StrFormat("%s_idx_%d", node->GetNameView(), i)
                : ""));
    XLS_RETURN_IF_ERROR(
        ToTreeOfNodesInPlace(arr_idx, segment.AsMutableView({i})));
  }
  return absl::OkStatus();
}
}  // namespace
absl::StatusOr<LeafTypeTree<Node*>> ToTreeOfNodes(Node* node) {
  LeafTypeTree<Node*> result(node->GetType(), nullptr);
  XLS_RETURN_IF_ERROR(ToTreeOfNodesInPlace(node, result.AsMutableView()))
      << "Failed to convert " << node << " into a ltt";
  return result;
}

absl::StatusOr<Node*> FromTreeOfNodes(FunctionBase* f,
                                      LeafTypeTreeView<Node*> tree,
                                      std::string_view name, SourceInfo loc) {
  if (tree.type()->IsArray()) {
    ArrayType* type = tree.type()->AsArrayOrDie();
    std::vector<Node*> elements;
    elements.reserve(type->size());
    for (int64_t i = 0; i < type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Node * element,
                           FromTreeOfNodes(f, tree.AsView({i})));
      elements.push_back(element);
    }
    return f->MakeNodeWithName<Array>(loc, elements, type->element_type(),
                                      name);
  }
  if (tree.type()->IsTuple()) {
    TupleType* type = tree.type()->AsTupleOrDie();
    std::vector<Node*> elements;
    elements.reserve(type->size());
    for (int64_t i = 0; i < type->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Node * element,
                           FromTreeOfNodes(f, tree.AsView({i})));
      elements.push_back(element);
    }
    return f->MakeNodeWithName<Tuple>(loc, elements, name);
  }

  // Flat type; this is a leaf node.
  CHECK(tree.type()->IsBits() || tree.type()->IsToken());
  CHECK_EQ(tree.size(), 1);

  Node* leaf = tree.elements().front();
  XLS_RET_CHECK_EQ(leaf->function_base(), f);
  return leaf;
}

// TODO(allight): We should clean this up and deduplicate this and
// RemoveKnownBits.
absl::StatusOr<Node*> GatherBits(Node* node, LeafTypeTreeView<Bits> mask) {
  std::vector<Node*> gathered_bits;
  XLS_RET_CHECK_EQ(mask.type(), node->GetType()) << "Type mismatch";
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> tree_nodes, ToTreeOfNodes(node));
  for (const auto& [n, mask] :
       iter::zip(tree_nodes.elements(), mask.elements())) {
    if (mask.IsZero()) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Node * gathered, GatherBits(n, mask));
    gathered_bits.push_back(gathered);
  }

  XLS_RET_CHECK(!gathered_bits.empty());
  if (gathered_bits.size() == 1) {
    return gathered_bits[0];
  }
  absl::c_reverse(gathered_bits);
  return node->function_base()->MakeNode<Concat>(node->loc(), gathered_bits);
}

absl::StatusOr<Node*> AndReduceTrailing(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info) {
  SourceInfo loc = source_info.value_or(node->loc());
  FunctionBase* f = node->function_base();
  // Reducing zero bits should return one (identity of AND).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(loc, Value(UBits(1, 1)));
  }
  const int64_t width = node->BitCountOrDie();
  CHECK_LE(bit_count, width);
  XLS_ASSIGN_OR_RETURN(
      Node * bit_slice,
      f->MakeNode<BitSlice>(loc, node, /*start=*/0, /*width=*/bit_count));
  if (bit_count == 1) {
    return bit_slice;
  }
  return f->MakeNode<BitwiseReductionOp>(loc, bit_slice, Op::kAndReduce);
}

absl::StatusOr<Node*> NorReduceTrailing(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info) {
  SourceInfo loc = source_info.value_or(node->loc());
  FunctionBase* f = node->function_base();
  // Reducing zero bits returns one (documented convention).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(loc, Value(UBits(1, 1)));
  }
  const int64_t width = node->BitCountOrDie();
  CHECK_LE(bit_count, width);
  XLS_ASSIGN_OR_RETURN(Node * bit_slice,
                       f->MakeNode<BitSlice>(loc, node,
                                             /*start=*/0,
                                             /*width=*/bit_count));
  Node* reduced = bit_slice;
  if (bit_count > 1) {
    XLS_ASSIGN_OR_RETURN(reduced, f->MakeNode<BitwiseReductionOp>(
                                      loc, bit_slice, Op::kOrReduce));
  }
  return f->MakeNode<UnOp>(loc, reduced, Op::kNot);
}

absl::StatusOr<Node*> OrReduceLeading(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info) {
  SourceInfo loc = source_info.value_or(node->loc());
  FunctionBase* f = node->function_base();
  // Reducing zero bits should return zero (identity of OR).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(loc, Value(UBits(0, 1)));
  }
  const int64_t width = node->BitCountOrDie();
  CHECK_LE(bit_count, width);
  XLS_ASSIGN_OR_RETURN(Node * bit_slice,
                       f->MakeNode<BitSlice>(loc, node,
                                             /*start=*/width - bit_count,
                                             /*width=*/bit_count));
  if (bit_count == 1) {
    return bit_slice;
  }
  return f->MakeNode<BitwiseReductionOp>(loc, bit_slice, Op::kOrReduce);
}

absl::StatusOr<Node*> NorReduceLeading(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info) {
  SourceInfo loc = source_info.value_or(node->loc());
  FunctionBase* f = node->function_base();
  // Reducing zero bits returns one (documented convention).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(loc, Value(UBits(1, 1)));
  }
  const int64_t width = node->BitCountOrDie();
  CHECK_LE(bit_count, width);
  XLS_ASSIGN_OR_RETURN(Node * bit_slice,
                       f->MakeNode<BitSlice>(loc, node,
                                             /*start=*/width - bit_count,
                                             /*width=*/bit_count));
  Node* reduced = bit_slice;
  if (bit_count > 1) {
    XLS_ASSIGN_OR_RETURN(reduced, f->MakeNode<BitwiseReductionOp>(
                                      loc, bit_slice, Op::kOrReduce));
  }
  return f->MakeNode<UnOp>(loc, reduced, Op::kNot);
}

absl::StatusOr<Node*> ConcatIfNeeded(FunctionBase* f,
                                     absl::Span<Node* const> operands,
                                     std::string_view name,
                                     const SourceInfo& source_info) {
  XLS_RET_CHECK(!operands.empty());

  if (operands.size() == 1) {
    return operands[0];
  }
  return f->MakeNodeWithName<Concat>(source_info, operands, name);
}

absl::StatusOr<Node*> NaryAndIfNeeded(FunctionBase* f,
                                      absl::Span<Node* const> operands,
                                      std::string_view name,
                                      const SourceInfo& source_info,
                                      bool drop_literal_one_operands) {
  if (operands.empty()) {
    return f->MakeNodeWithName<Literal>(source_info, Value(UBits(1, 1)), name);
  }

  std::vector<Node*> unique_operands = RemoveRedundantNodes(
      operands,
      /*drop_literal=*/drop_literal_one_operands
          ? std::make_optional([](const Value& v) { return v.IsAllOnes(); })
          : std::nullopt);
  if (unique_operands.empty()) {
    return f->MakeNodeWithName<Literal>(source_info, Value(UBits(1, 1)), name);
  }

  if (unique_operands.size() == 1) {
    return unique_operands[0];
  }
  return f->MakeNodeWithName<NaryOp>(source_info, unique_operands, Op::kAnd,
                                     name);
}

absl::StatusOr<Node*> NaryOrIfNeeded(FunctionBase* f,
                                     absl::Span<Node* const> operands,
                                     std::string_view name,
                                     const SourceInfo& source_info,
                                     bool drop_literal_zero_operands) {
  if (operands.empty()) {
    return f->MakeNodeWithName<Literal>(source_info, Value(UBits(0, 1)), name);
  }

  std::vector<Node*> unique_operands = RemoveRedundantNodes(
      operands,
      /*drop_literal=*/drop_literal_zero_operands
          ? std::make_optional([](const Value& v) { return v.IsAllZeros(); })
          : std::nullopt);
  if (unique_operands.empty()) {
    return f->MakeNodeWithName<Literal>(source_info, Value(UBits(0, 1)), name);
  }

  if (unique_operands.size() == 1) {
    return unique_operands[0];
  }
  return f->MakeNodeWithName<NaryOp>(source_info, unique_operands, Op::kOr,
                                     name);
}

absl::StatusOr<Node*> NaryNorIfNeeded(FunctionBase* f,
                                      absl::Span<Node* const> operands,
                                      std::string_view name,
                                      const SourceInfo& source_info,
                                      bool drop_literal_zero_operands) {
  XLS_RET_CHECK(!operands.empty());

  std::vector<Node*> unique_operands = RemoveRedundantNodes(
      operands,
      /*drop_literal=*/drop_literal_zero_operands
          ? std::make_optional([](const Value& v) { return v.IsAllZeros(); })
          : std::nullopt);
  if (unique_operands.empty()) {
    return f->MakeNodeWithName<Literal>(source_info, Value(UBits(1, 1)), name);
  }

  if (unique_operands.size() == 1) {
    return f->MakeNodeWithName<UnOp>(source_info, unique_operands[0], Op::kNot,
                                     name);
  }
  return f->MakeNodeWithName<NaryOp>(source_info, unique_operands, Op::kNor,
                                     name);
}

absl::StatusOr<Node*> JoinWithAnd(FunctionBase* f,
                                  absl::Span<Node* const> operands,
                                  bool combine_literals, std::string_view name,
                                  std::optional<SourceInfo> loc) {
  XLS_RET_CHECK(!operands.empty());
  for (Node* operand : operands) {
    XLS_RET_CHECK_EQ(operand->function_base(), f);
  }

  std::vector<Node*> new_operands;
  new_operands.reserve(operands.size());
  for (Node* operand : operands) {
    if (operand->op() == Op::kAnd) {
      absl::c_copy(operand->operands(), std::back_inserter(new_operands));
    } else {
      new_operands.push_back(operand);
    }
  }

  std::optional<Bits> literal_value;
  if (combine_literals) {
    for (Node* operand : new_operands) {
      if (operand->Is<Literal>()) {
        Value operand_value = operand->As<Literal>()->value();
        CHECK(operand_value.IsBits());
        if (!literal_value.has_value()) {
          literal_value = operand_value.bits();
        } else {
          literal_value = bits_ops::And(*literal_value, operand_value.bits());
        }
      }
    }
    if (literal_value.has_value() && literal_value->IsZero()) {
      return f->MakeNodeWithName<Literal>(loc.value_or(SourceInfo()),
                                          Value(*literal_value), name);
    }
    if (literal_value.has_value() && literal_value->IsAllOnes()) {
      literal_value = std::nullopt;
    }
  }

  std::vector<Node*> unique_operands = RemoveRedundantNodes(
      new_operands,
      /*drop_literal=*/combine_literals
          ? std::make_optional([](const Value&) { return true; })
          : std::nullopt);
  if (literal_value.has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * literal,
                         f->MakeNode<Literal>(loc.value_or(SourceInfo()),
                                              Value(*literal_value)));
    unique_operands.push_back(literal);
  }

  if (unique_operands.empty()) {
    return f->MakeNodeWithName<Literal>(
        loc.value_or(SourceInfo()), AllOnesOfType(operands.front()->GetType()),
        name);
  }

  if (unique_operands.size() == 1) {
    return unique_operands.front();
  }

  if (!loc.has_value()) {
    loc = unique_operands.front()->loc();
    for (Node* operand : unique_operands) {
      loc = loc->Extend(operand->loc());
    }
  }

  return NaryAndIfNeeded(f, unique_operands, name, *loc);
}

absl::StatusOr<Node*> ReplaceWithAnd(Node* old_node,
                                     absl::Span<Node* const> new_nodes,
                                     bool combine_literals,
                                     std::string_view name,
                                     std::optional<SourceInfo> loc) {
  FunctionBase* f = old_node->function_base();
  std::vector<Node*> operands;
  operands.reserve(new_nodes.size() + 1);
  operands.push_back(old_node);
  absl::c_copy(new_nodes, std::back_inserter(operands));

  XLS_ASSIGN_OR_RETURN(Node * replacement,
                       JoinWithAnd(f, operands, combine_literals, name, loc));
  if (f->IsStaged(old_node) && !f->IsStaged(replacement)) {
    XLS_ASSIGN_OR_RETURN(int64_t stage_index,
                         old_node->function_base()->GetStageIndex(old_node));
    XLS_RETURN_IF_ERROR(old_node->function_base()
                            ->AddNodeToStage(stage_index, replacement)
                            .status());
  }
  // Take over the name of the old node, if possible; not all replacement nodes
  // have configurable names (e.g. ports).
  if (!old_node->OpIn({Op::kInputPort, Op::kOutputPort, Op::kParam}) &&
      !replacement->OpIn({Op::kInputPort, Op::kOutputPort, Op::kParam}) &&
      old_node->HasAssignedName() &&
      (name.empty() || name == old_node->GetNameView())) {
    std::string old_name = old_node->GetName();
    old_node->ClearName();
    replacement->SetNameDirectly(old_name);
  }
  XLS_RETURN_IF_ERROR(old_node->ReplaceUsesWith(replacement));
  return replacement;
}

absl::StatusOr<Node*> JoinWithOr(FunctionBase* f,
                                 absl::Span<Node* const> operands,
                                 bool combine_literals, std::string_view name,
                                 std::optional<SourceInfo> loc) {
  XLS_RET_CHECK(!operands.empty());
  for (Node* operand : operands) {
    XLS_RET_CHECK_EQ(operand->function_base(), f);
  }

  std::vector<Node*> new_operands;
  new_operands.reserve(operands.size());
  for (Node* operand : operands) {
    if (operand->op() == Op::kOr) {
      absl::c_copy(operand->operands(), std::back_inserter(new_operands));
    } else {
      new_operands.push_back(operand);
    }
  }

  std::optional<Bits> literal_value;
  if (combine_literals) {
    for (Node* operand : new_operands) {
      if (operand->Is<Literal>()) {
        Value operand_value = operand->As<Literal>()->value();
        CHECK(operand_value.IsBits());
        if (!literal_value.has_value()) {
          literal_value = operand_value.bits();
        } else {
          literal_value = bits_ops::Or(*literal_value, operand_value.bits());
        }
      }
    }
    if (literal_value.has_value() && literal_value->IsAllOnes()) {
      return f->MakeNodeWithName<Literal>(loc.value_or(SourceInfo()),
                                          Value(*literal_value), name);
    }
    if (literal_value.has_value() && literal_value->IsZero()) {
      literal_value = std::nullopt;
    }
  }

  std::vector<Node*> unique_operands = RemoveRedundantNodes(
      new_operands,
      /*drop_literal=*/combine_literals
          ? std::make_optional([](const Value&) { return true; })
          : std::nullopt);
  if (literal_value.has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * literal,
                         f->MakeNode<Literal>(loc.value_or(SourceInfo()),
                                              Value(*literal_value)));
    unique_operands.push_back(literal);
  }

  if (unique_operands.empty()) {
    return f->MakeNodeWithName<Literal>(
        loc.value_or(SourceInfo()),
        Value(ZeroOfType(operands.front()->GetType())), name);
  }

  if (unique_operands.size() == 1) {
    return unique_operands.front();
  }

  if (!loc.has_value()) {
    loc = operands.front()->loc();
    for (Node* operand : operands) {
      loc = loc->Extend(operand->loc());
    }
  }

  return NaryOrIfNeeded(f, unique_operands, name, *loc);
}

absl::StatusOr<Node*> ReplaceWithOr(Node* old_node,
                                    absl::Span<Node* const> new_nodes,
                                    bool combine_literals,
                                    std::string_view name,
                                    std::optional<SourceInfo> loc) {
  FunctionBase* f = old_node->function_base();
  std::vector<Node*> operands;
  operands.reserve(new_nodes.size() + 1);
  operands.push_back(old_node);
  absl::c_copy(new_nodes, std::back_inserter(operands));

  XLS_ASSIGN_OR_RETURN(Node * replacement,
                       JoinWithOr(f, operands, combine_literals, name, loc));
  if (f->IsStaged(old_node) && !f->IsStaged(replacement)) {
    XLS_ASSIGN_OR_RETURN(int64_t stage_index,
                         old_node->function_base()->GetStageIndex(old_node));
    XLS_RETURN_IF_ERROR(old_node->function_base()
                            ->AddNodeToStage(stage_index, replacement)
                            .status());
  }
  // Take over the name of the old node, if possible; not all replacement nodes
  // have configurable names (e.g. ports).
  if (!old_node->OpIn({Op::kInputPort, Op::kOutputPort, Op::kParam}) &&
      !replacement->OpIn({Op::kInputPort, Op::kOutputPort, Op::kParam}) &&
      old_node->HasAssignedName() &&
      (name.empty() || name == old_node->GetNameView())) {
    std::string old_name = old_node->GetName();
    old_node->ClearName();
    replacement->SetNameDirectly(old_name);
  }
  XLS_RETURN_IF_ERROR(old_node->ReplaceUsesWith(replacement));
  return replacement;
}

bool IsUnsignedCompare(Node* node) {
  switch (node->op()) {
    case Op::kEq:
    case Op::kNe:
    case Op::kULe:
    case Op::kULt:
    case Op::kUGe:
    case Op::kUGt:
      return true;
    default:
      return false;
  }
}

bool IsSignedCompare(Node* node) {
  switch (node->op()) {
    case Op::kEq:
    case Op::kNe:
    case Op::kSLe:
    case Op::kSLt:
    case Op::kSGe:
    case Op::kSGt:
      return true;
    default:
      return false;
  }
}

absl::StatusOr<Op> OpToNonReductionOp(Op reduce_op) {
  switch (reduce_op) {
    case Op::kAndReduce:
      return Op::kAnd;
      break;
    case Op::kOrReduce:
      return Op::kOr;
      break;
    case Op::kXorReduce:
      return Op::kXor;
      break;
    default:
      return absl::InternalError("Unexpected bitwise reduction op");
  }
}

absl::StatusOr<Channel*> GetChannelUsedByNode(Node* node) {
  if (!node->Is<ChannelNode>()) {
    return absl::NotFoundError(
        absl::StrFormat("No channel associated with node %s", node->GetName()));
  }
  return node->package()->GetChannel(node->As<ChannelNode>()->channel_name());
}

absl::StatusOr<std::optional<Node*>> GetPredicateUsedByNode(Node* node) {
  switch (node->op()) {
    case Op::kSend: {
      return node->As<Send>()->predicate();
    }
    case Op::kReceive: {
      return node->As<Receive>()->predicate();
    }
    case Op::kNext: {
      return node->As<Next>()->predicate();
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %s to be a send, receive, or next value.",
                          node->GetName()));
  }
}

absl::StatusOr<Node*> GetNodeAtIndex(Node* base,
                                     absl::Span<const int64_t> index) {
  if (index.empty()) {
    return base;
  }
  if (base->GetType()->IsTuple()) {
    if (index.front() >= base->GetType()->AsTupleOrDie()->size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("index %d out of range for type %s", index.front(),
                          base->GetType()->ToString()));
    }
    XLS_ASSIGN_OR_RETURN(
        Node * nxt,
        base->function_base()->MakeNodeWithName<TupleIndex>(
            base->loc(), base, index.front(),
            base->HasAssignedName()
                ? absl::StrFormat("%s_tup%d", base->GetName(), index.front())
                : ""));
    return GetNodeAtIndex(nxt, index.subspan(1));
  }
  if (base->GetType()->IsArray()) {
    FunctionBase* fb = base->function_base();
    int64_t dims = 0;
    Type* t = base->GetType();
    std::vector<Node*> idxs;
    auto it = index.cbegin();
    while (t->IsArray() && it != index.cend()) {
      if (*it >= t->AsArrayOrDie()->size()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("index %d out of range for type %s", index.front(),
                            base->GetType()->ToString()));
      }
      XLS_ASSIGN_OR_RETURN(
          Node * idx,
          fb->MakeNode<Literal>(SourceInfo(), Value(UBits(*it, 64))));
      idxs.push_back(idx);

      dims++;
      t = t->AsArrayOrDie()->element_type();
      ++it;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * nxt,
        fb->MakeNodeWithName<ArrayIndex>(
            base->loc(), base, idxs, /*assumed_in_bounds=*/true,
            base->HasAssignedName()
                ? absl::StrFormat("%s_arr%s", base->GetName(),
                                  absl::StrJoin(index.subspan(0, dims), "_"))
                : ""));
    return GetNodeAtIndex(nxt, index.subspan(dims));
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("%s has an invalid index path", base->ToString()));
}

namespace {
absl::flat_hash_set<Node*> GetTransitiveUsers(Node* node) {
  absl::flat_hash_set<Node*> users;
  std::deque<Node*> deque{node};
  while (!deque.empty()) {
    Node* front = deque.front();
    users.insert(front);
    for (Node* user : front->users()) {
      if (!users.contains(user)) {
        deque.push_back(user);
      }
    }
    deque.pop_front();
  }
  return users;
}
}  // namespace

absl::StatusOr<Node*> ReplaceTupleElementsWith(
    Node* node, const absl::flat_hash_map<int64_t, Node*>& replacements) {
  if (!node->GetType()->IsTuple()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Node %v must be type tuple, got %v.", *node, *node->GetType()));
  }
  TupleType* tuple_type = node->GetType()->AsTupleOrDie();
  int64_t num_elements = tuple_type->size();
  std::vector<Node*> elements;
  elements.reserve(num_elements);

  // Check that replacements have valid indices.
  const auto [min_index_replacement, max_index_replacement] =
      std::minmax_element(
          replacements.begin(), replacements.end(),
          [](std::pair<int64_t, Node*> lhs, std::pair<int64_t, Node*> rhs) {
            return lhs.first < rhs.first;
          });
  if (min_index_replacement->first < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Replacement index %d (with node %v) is negative.",
        min_index_replacement->first, *min_index_replacement->second));
  }
  if (max_index_replacement->first >= num_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Replacement index %d (with node %v) is too large: "
                        "tuple only has %d elements.",
                        max_index_replacement->first,
                        *max_index_replacement->second, num_elements));
  }

  // Check that replacements have valid values.
  absl::flat_hash_set<Node*> users = GetTransitiveUsers(node);
  for (auto& [index, replacement] : replacements) {
    if (users.contains(replacement)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Replacement index %d (%v) depends on node %v.",
                          index, *replacement, *node));
    }
  }

  // First, make an empty tuple. We're going to replace everything here, we
  // just need something to exist before calling ReplaceUsesWith() on node.
  // After we call ReplaceUsesWith(), we'll replace each operand of the tuple
  // with a TupleIndex from node.
  for (int64_t index = 0; index < num_elements; ++index) {
    XLS_ASSIGN_OR_RETURN(
        Literal * element_lit,
        node->function_base()->MakeNode<Literal>(
            SourceInfo(), ZeroOfType(tuple_type->element_type(index))));
    elements.push_back(element_lit);
  }
  XLS_ASSIGN_OR_RETURN(
      Tuple * new_tuple,
      node->function_base()->MakeNode<Tuple>(SourceInfo(), elements));
  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(new_tuple));

  for (int64_t index = 0; index < num_elements; ++index) {
    Node* replacement_element;
    auto itr = replacements.find(index);
    if (itr != replacements.end()) {
      replacement_element = itr->second;
    } else {
      XLS_ASSIGN_OR_RETURN(replacement_element,
                           node->function_base()->MakeNode<TupleIndex>(
                               SourceInfo(), node, index));
    }
    XLS_RETURN_IF_ERROR(
        new_tuple->ReplaceOperandNumber(index, replacement_element));
  }

  // Now remove the old literals.
  for (int64_t index = 0; index < num_elements; ++index) {
    XLS_RETURN_IF_ERROR(node->function_base()->RemoveNode(elements[index]));
  }

  return new_tuple;
}

absl::StatusOr<Node*> ReplaceTupleElementsWith(Node* node, int64_t index,
                                               Node* replacement_element) {
  return ReplaceTupleElementsWith(node, {{index, replacement_element}});
}

bool IsBinarySelectTwoCases(Node* node) {
  if (!node->Is<Select>()) {
    return false;
  }
  Select* sel = node->As<Select>();
  return sel->cases().size() == 2 && !sel->default_value().has_value();
}

bool IsBinaryPrioritySelect(Node* node) {
  if (!node->Is<PrioritySelect>()) {
    return false;
  }
  PrioritySelect* sel = node->As<PrioritySelect>();
  return sel->cases().size() == 1;
}

absl::StatusOr<std::optional<BinarySelectView>> MatchBinarySelectLike(
    Node* node) {
  if (IsBinarySelectTwoCases(node)) {
    Select* sel = node->As<Select>();
    XLS_RET_CHECK_EQ(sel->selector()->BitCountOrDie(), 1);
    return BinarySelectView{.selector = sel->selector(),
                            .on_false = sel->get_case(0),
                            .on_true = sel->get_case(1)};
  }
  // A Select with one case plus a default can also represent a binary mux when
  // the selector is 1-bit:
  //
  //   sel(p, cases=[x], default=y)  =>  p==0 ? x : y
  //
  // In this case we treat `x` as the on-false arm and `y` as the on-true arm.
  if (node->Is<Select>()) {
    Select* sel = node->As<Select>();
    if (sel->cases().size() == 1 && sel->default_value().has_value()) {
      if (sel->selector()->BitCountOrDie() != 1) {
        return std::nullopt;
      }
      return BinarySelectView{.selector = sel->selector(),
                              .on_false = sel->get_case(0),
                              .on_true = *sel->default_value()};
    }
  }
  if (IsBinaryPrioritySelect(node)) {
    PrioritySelect* sel = node->As<PrioritySelect>();
    XLS_RET_CHECK_EQ(sel->selector()->BitCountOrDie(), 1);
    return BinarySelectView{.selector = sel->selector(),
                            .on_false = sel->default_value(),
                            .on_true = sel->get_case(0)};
  }
  return std::nullopt;
}

absl::StatusOr<absl::flat_hash_map<Channel*, std::vector<Node*>>> ChannelUsers(
    Package* package) {
  absl::flat_hash_map<Channel*, std::vector<Node*>> channel_users;
  for (std::unique_ptr<Proc>& proc : package->procs()) {
    for (Node* node : proc->nodes()) {
      if (!node->Is<ChannelNode>()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_users[channel].push_back(node);
    }
  }
  return channel_users;
}

namespace {
// TODO(allight): replace with absl::c_contains once absl is updated.
template <typename Lst, typename Element>
bool c_contains(const Lst& lst, const Element& e) {
  return absl::c_find(lst, e) != lst.end();
}
}  // namespace

absl::StatusOr<Node*> CompareLiteral(Node* lhs, int64_t rhs, Op cmp,
                                     const std::optional<std::string>& name) {
  XLS_RET_CHECK(c_contains(CompareOp::kOps, cmp)) << "Bad op " << cmp;
  XLS_RET_CHECK(lhs->GetType()->IsBits());
  bool is_signed = c_contains(
      absl::Span<Op const>{Op::kSLe, Op::kSLt, Op::kSGe, Op::kSGt}, cmp);
  int64_t rhs_bit_cnt = std::max(lhs->BitCountOrDie(),
                                 is_signed ? Bits::MinBitCountSigned(rhs)
                                           : Bits::MinBitCountUnsigned(rhs));
  Bits val = is_signed ? SBits(rhs, rhs_bit_cnt) : UBits(rhs, rhs_bit_cnt);
  XLS_ASSIGN_OR_RETURN(
      Node * lit,
      lhs->function_base()->MakeNodeWithName<Literal>(
          lhs->loc(), Value(val),
          name && !name->empty() ? absl::StrCat(*name, "_literal") : ""));
  if (lhs->BitCountOrDie() < val.bit_count()) {
    XLS_ASSIGN_OR_RETURN(lhs, lhs->function_base()->MakeNodeWithName<ExtendOp>(
                                  lhs->loc(), lhs, lit->BitCountOrDie(),
                                  is_signed ? Op::kSignExt : Op::kZeroExt,
                                  lhs->HasAssignedName()
                                      ? absl::StrCat(lhs->GetNameView(), "_ext")
                                      : ""));
  }
  return lhs->function_base()->MakeNodeWithName<CompareOp>(
      lhs->loc(), lhs, lit, cmp, name.value_or(""));
}

absl::StatusOr<Node*> CompareNumeric(Node* lhs, Node* rhs, Op cmp,
                                     const std::optional<std::string>& name) {
  XLS_RET_CHECK(c_contains(CompareOp::kOps, cmp)) << "Bad op " << cmp;
  bool is_signed = c_contains(
      absl::Span<Op const>{Op::kSLe, Op::kSLt, Op::kSGe, Op::kSGt}, cmp);
  Op ext_op = is_signed ? Op::kSignExt : Op::kZeroExt;
  XLS_RET_CHECK(lhs->GetType()->IsBits()) << lhs;
  XLS_RET_CHECK(rhs->GetType()->IsBits()) << rhs;
  if (lhs->BitCountOrDie() < rhs->BitCountOrDie()) {
    XLS_ASSIGN_OR_RETURN(lhs, lhs->function_base()->MakeNodeWithName<ExtendOp>(
                                  lhs->loc(), lhs, rhs->BitCountOrDie(), ext_op,
                                  lhs->HasAssignedName()
                                      ? absl::StrCat(lhs->GetNameView(), "_ext")
                                      : ""));
  } else if (lhs->BitCountOrDie() > rhs->BitCountOrDie()) {
    XLS_ASSIGN_OR_RETURN(rhs, rhs->function_base()->MakeNodeWithName<ExtendOp>(
                                  rhs->loc(), rhs, lhs->BitCountOrDie(), ext_op,
                                  rhs->HasAssignedName()
                                      ? absl::StrCat(rhs->GetNameView(), "_ext")
                                      : ""));
  }
  return lhs->function_base()->MakeNodeWithName<CompareOp>(
      lhs->loc(), lhs, rhs, cmp, name.value_or(""));
}

absl::StatusOr<Node*> UnsignedBoundByLiterals(Node* v, int64_t low_bound,
                                              int64_t high_bound) {
  XLS_RET_CHECK_LE(low_bound, high_bound);
  XLS_RET_CHECK(v->GetType()->IsBits()) << v;
  XLS_RET_CHECK_LE(Bits::MinBitCountUnsigned(low_bound), v->BitCountOrDie())
      << "Low bound (" << low_bound << ") not representable by " << v;
  if (low_bound == high_bound) {
    return v->function_base()->MakeNodeWithName<Literal>(
        v->loc(), Value(UBits(low_bound, v->BitCountOrDie())),
        v->HasAssignedName() ? absl::StrFormat("%s_bounded", v->GetName())
                             : "");
  }
  auto extend_name =
      [&](std::string_view suffix) -> std::optional<std::string> {
    if (v->HasAssignedName()) {
      return absl::StrCat(v->GetName(), suffix);
    }
    return std::nullopt;
  };
  std::vector<Node*> select_compares;
  std::vector<Node*> result_literals;
  if (low_bound != 0) {
    XLS_ASSIGN_OR_RETURN(Node * lt_low,
                         CompareLiteral(v, low_bound, Op::kULt,
                                        extend_name("_low_bound_check")));
    XLS_ASSIGN_OR_RETURN(
        Node * low, v->function_base()->MakeNodeWithName<Literal>(
                        v->loc(), Value(UBits(low_bound, v->BitCountOrDie())),
                        extend_name("_low_bound").value_or("")));
    select_compares.push_back(lt_low);
    result_literals.push_back(low);
  }
  if (v->BitCountOrDie() >= Bits::MinBitCountUnsigned(high_bound) &&
      !UBits(high_bound, v->BitCountOrDie()).IsAllOnes()) {
    XLS_ASSIGN_OR_RETURN(Node * gt_high,
                         CompareLiteral(v, high_bound, Op::kUGt,
                                        extend_name("_high_bound_check")));
    XLS_ASSIGN_OR_RETURN(
        Node * high, v->function_base()->MakeNodeWithName<Literal>(
                         v->loc(), Value(UBits(high_bound, v->BitCountOrDie())),
                         extend_name("_high_bound").value_or("")));
    select_compares.push_back(gt_high);
    result_literals.push_back(high);
  }
  if (select_compares.empty()) {
    // Nothing to actually bound.
    return v;
  }
  // Need MSB to be first
  Node* selector;
  if (select_compares.size() == 1) {
    selector = select_compares.front();
  } else {
    absl::c_reverse(select_compares);
    XLS_ASSIGN_OR_RETURN(selector, v->function_base()->MakeNodeWithName<Concat>(
                                       v->loc(), select_compares,
                                       extend_name("_checks").value_or("")));
  }
  return v->function_base()->MakeNodeWithName<PrioritySelect>(
      v->loc(), selector, result_literals, v,
      extend_name("_bounded").value_or(""));
}

bool AreAllLiteral(absl::Span<Node* const> nodes) {
  // Check if all indices are literals
  return absl::c_all_of(nodes, [](Node* i) -> bool { return IsLiteral(i); });
}

namespace {

class NodeSearch : public DfsVisitorWithDefault {
 public:
  explicit NodeSearch(Node* target) : target_(target) {}

  absl::Status DefaultHandler(Node* node) override {
    if (node == target_) {
      // We've found our target, and can cancel the search. (This causes
      // Node::Accept to return early, since we've already accomplished our
      // goal.)
      return absl::CancelledError();
    }
    return absl::OkStatus();
  }

 private:
  Node* target_;
};

}  // namespace

bool IsAncestorOf(Node* a, Node* b) {
  CHECK_NE(a, nullptr);
  CHECK_NE(b, nullptr);

  if (a->function_base() != b->function_base()) {
    return false;
  }
  if (a == b) {
    return false;
  }

  NodeSearch visitor(a);
  absl::Status visitor_status = b->Accept(&visitor);
  CHECK(visitor_status.ok() || absl::IsCancelled(visitor_status));
  return visitor.IsVisited(a);
}

absl::StatusOr<Node*> RemoveNodeFromBooleanExpression(Node* to_remove,
                                                      Node* expression,
                                                      bool favored_outcome) {
  XLS_RET_CHECK(expression->GetType()->IsBits());
  XLS_RET_CHECK_EQ(expression->GetType()->AsBitsOrDie()->bit_count(), 1)
      << expression->ToString();

  if (expression == to_remove) {
    return expression->function_base()->MakeNode<Literal>(
        expression->loc(), Value(UBits((favored_outcome ? 1 : 0), 1)));
  }

  if (expression->op() == Op::kNot) {
    XLS_ASSIGN_OR_RETURN(
        Node * new_operand,
        RemoveNodeFromBooleanExpression(to_remove, expression->operand(0),
                                        /*favored_outcome=*/!favored_outcome));
    if (new_operand == expression->operand(0)) {
      // No change was necessary; apparently `to_remove` was not present.
      return expression;
    } else {
      return expression->function_base()->MakeNode<UnOp>(expression->loc(),
                                                         new_operand, Op::kNot);
    }
  }

  if (expression->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor})) {
    const bool favored_operand =
        favored_outcome ^ expression->OpIn({Op::kNand, Op::kNor});

    std::vector<Node*> new_operands;
    new_operands.reserve(expression->operands().size());
    bool changed = false;
    for (Node* operand : expression->operands()) {
      XLS_ASSIGN_OR_RETURN(
          Node * new_operand,
          RemoveNodeFromBooleanExpression(to_remove, operand, favored_operand));
      new_operands.push_back(new_operand);
      if (new_operand != operand) {
        changed = true;
      }
    }
    if (changed) {
      return expression->function_base()->MakeNode<NaryOp>(
          expression->loc(), new_operands, expression->op());
    } else {
      // No change was necessary; apparently `to_remove` was not present.
      return expression;
    }
  }

  if (IsAncestorOf(to_remove, expression)) {
    // We're unable to remove `to_remove` from `expression` directly, but it is
    // an ancestor; just replace the entire expression with a literal.
    return expression->function_base()->MakeNode<Literal>(
        expression->loc(), Value(UBits((favored_outcome ? 1 : 0), 1)));
  }
  return expression;
}

absl::StatusOr<Type*> RemoveKnownBitsType(
    TypeManager& arena, Type* ty, LeafTypeTreeView<TernaryVector> mask) {
  XLS_RET_CHECK_EQ(ty, mask.type());
  if (ty->GetFlatBitCount() == 0 ||
      absl::c_none_of(mask.elements(), ternary_ops::AnyKnown)) {
    return ty;
  }
  if (ty->IsBits()) {
    return arena.GetBitsType(
        absl::c_count_if(mask.Get({}), ternary_ops::IsUnknown));
  }
  if (ty->IsTuple()) {
    std::vector<Type*> res;
    res.reserve(ty->AsTupleOrDie()->size());
    for (int64_t i = 0; i < ty->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          std::back_inserter(res),
          RemoveKnownBitsType(arena, ty->AsTupleOrDie()->element_type(i),
                              mask.AsView({i})));
    }
    return arena.GetTupleType(res);
  }
  // token handled by bitcount == 0
  XLS_RET_CHECK(ty->IsArray()) << "unexpected type " << ty;
  // We encode the array as a tuple to allow for individual elements to be
  // narrowed.
  std::vector<Type*> res;
  res.reserve(ty->AsArrayOrDie()->size());
  for (int64_t i = 0; i < ty->AsArrayOrDie()->size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        std::back_inserter(res),
        RemoveKnownBitsType(arena, ty->AsArrayOrDie()->element_type(),
                            mask.AsView({i})));
  }
  return arena.GetTupleType(res);
}
namespace {
absl::StatusOr<LeafTypeTree<Node*>> RemoveKnownBitsFromTree(
    LeafTypeTreeView<Node*> node, LeafTypeTreeView<TernaryVector> mask) {
  XLS_RET_CHECK_GT(node.size(), 0) << "Zero-size tree?";
  FunctionBase* fb = node.elements().front()->function_base();
  XLS_ASSIGN_OR_RETURN(
      Type * new_ty,
      RemoveKnownBitsType(fb->package()->type_manager(), node.type(), mask));
  XLS_ASSIGN_OR_RETURN(
      LeafTypeTree<Node*> res_old_type,
      (leaf_type_tree::ZipIndex<Node*, Node*, TernaryVector>(
          node.AsView(), mask.AsView(),
          [&](Type*, Node* n, TernarySpan t,
              absl::Span<int64_t const>) -> absl::StatusOr<Node*> {
            if (ternary_ops::AllUnknown(t)) {
              // No splitting.
              return n;
            }
            // segment the bits.
            XLS_ASSIGN_OR_RETURN(auto res, GatherUnknownBits(n, t),
                                 _ << "Invalid narrowing: " << n);
            return res;
          })));
  return LeafTypeTree<Node*>(new_ty, std::move(res_old_type).elements());
}

absl::StatusOr<LeafTypeTree<Node*>> RestoreKnownBitsFromTree(
    FunctionBase* fb, LeafTypeTreeView<Node*> node,
    LeafTypeTreeView<TernaryVector> mask) {
  XLS_RET_CHECK_GT(mask.size(), 0) << "Zero-size tree?";
  return leaf_type_tree::MapIndex<Node*, TernaryVector>(
      mask.AsView(),
      [&](Type* ty, TernarySpan ts,
          absl::Span<int64_t const> index) -> absl::StatusOr<Node*> {
        if (ternary_ops::AllUnknown(ts)) {
          // No need to split.
          return node.Get(index);
        }
        if (ternary_ops::IsFullyKnown(ts) && ty->IsBits()) {
          XLS_ASSIGN_OR_RETURN(
              Node * lit,
              fb->MakeNode<Literal>(SourceInfo(),
                                    Value(ternary_ops::ToKnownBitsValues(ts))));
          return lit;
        }
        auto view = node.AsView(index);
        XLS_RET_CHECK(view.type()->IsBits())
            << view.ToString([](Node* n) { return n->ToString(); });
        XLS_RET_CHECK_EQ(view.type()->GetFlatBitCount() +
                             absl::c_count_if(ts, ternary_ops::IsKnown),
                         ty->GetFlatBitCount())
            << "bits of " << view.Get({}) << " don't match " << ts;
        return FillPattern(ts, view.Get({}));
      });
}
}  // namespace
absl::StatusOr<Node*> RemoveKnownBits(Node* node,
                                      LeafTypeTreeView<TernaryVector> mask,
                                      bool* any_changed) {
  if (absl::c_none_of(mask.elements(), ternary_ops::AnyKnown)) {
    if (any_changed != nullptr) {
      *any_changed = false;
    }
    return node;
  }
  if (any_changed != nullptr) {
    *any_changed = true;
  }
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> split, ToTreeOfNodes(node));
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> narrowed,
                       RemoveKnownBitsFromTree(split.AsView(), mask));
  return FromTreeOfNodes(node->function_base(), narrowed.AsView(),
                         node->HasAssignedName()
                             ? absl::StrCat(node->GetNameView(), "_narrowed")
                             : "",
                         node->loc());
}
absl::StatusOr<Node*> RestoreKnownBits(Node* split,
                                       LeafTypeTreeView<TernaryVector> mask,
                                       bool* any_changed) {
  if (absl::c_none_of(mask.elements(), ternary_ops::AnyKnown)) {
    if (any_changed != nullptr) {
      *any_changed = false;
    }
    return split;
  }
  if (any_changed != nullptr) {
    *any_changed = true;
  }
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> split_tree, ToTreeOfNodes(split));
  XLS_ASSIGN_OR_RETURN(
      auto expected_type,
      RemoveKnownBitsType(split->package()->type_manager(), mask.type(), mask));
  XLS_RET_CHECK_EQ(split_tree.type(), expected_type)
      << "Incorrect type restoration.";
  XLS_ASSIGN_OR_RETURN(LeafTypeTree<Node*> restored,
                       RestoreKnownBitsFromTree(split->function_base(),
                                                split_tree.AsView(), mask));
  return FromTreeOfNodes(split->function_base(), restored.AsView(),
                         split->HasAssignedName()
                             ? absl::StrCat(split->GetNameView(), "_restored")
                             : "",
                         split->loc());
}

absl::StatusOr<Node*> SliceTuple(Node* tuple, int64_t start,
                                 std::optional<int64_t> count) {
  XLS_RET_CHECK(tuple->GetType()->IsTuple()) << "Not a tuple: " << tuple;
  TupleType* type = tuple->GetType()->AsTupleOrDie();
  int64_t limit = type->size();
  if (count.has_value()) {
    limit = std::min(limit, start + *count);
  }
  std::string result_name =
      tuple->HasAssignedName()
          ? absl::StrFormat("%s_%d_to_%d", tuple->GetNameView(), start, limit)
          : "";
  if (start >= type->size() || count == 0) {
    return tuple->function_base()->MakeNodeWithName<Literal>(
        tuple->loc(), Value::Tuple({}), result_name);
  }
  std::vector<Node*> elements;
  elements.reserve(count.value_or(type->size() - start));
  for (int64_t i = start; i < limit; ++i) {
    XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                         GetNodeAtIndex(tuple, {i}));
  }
  return tuple->function_base()->MakeNodeWithName<Tuple>(tuple->loc(), elements,
                                                         result_name);
}

absl::StatusOr<Node*> SetTupleIndex(Node* tuple, Node* value,
                                    absl::Span<int64_t const> indices) {
  XLS_RET_CHECK(!indices.empty());
  XLS_RET_CHECK(tuple->GetType()->IsTuple()) << "Not a tuple: " << tuple;
  TupleType* type = tuple->GetType()->AsTupleOrDie();
  XLS_RET_CHECK_GT(type->size(), indices.front())
      << "No element addressed in " << tuple;
  std::vector<Node*> elements;
  elements.reserve(type->size());
  for (int64_t i = 0; i < type->size(); ++i) {
    if (i != indices.front()) {
      XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                           GetNodeAtIndex(tuple, {i}));
    } else if (indices.size() == 1) {
      elements.push_back(value);
    } else {
      XLS_ASSIGN_OR_RETURN(Node * front,
                           GetNodeAtIndex(tuple, {indices.front()}));
      XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                           SetTupleIndex(front, value, indices.subspan(1)),
                           _ << "Failed to set tuple index " << indices.front()
                             << " for " << tuple);
    }
  }
  return tuple->function_base()->MakeNodeWithName<Tuple>(tuple->loc(), elements,
                                                         tuple->GetNameView());
}

absl::StatusOr<Node*> InsertIntoTuple(Node* tuple, Node* value,
                                      absl::Span<int64_t const> indices) {
  XLS_RET_CHECK(!indices.empty());
  XLS_RET_CHECK(tuple->GetType()->IsTuple()) << "Not a tuple: " << tuple;
  TupleType* type = tuple->GetType()->AsTupleOrDie();
  XLS_RET_CHECK_GE(type->size(), indices.front())
      << "No element addressed in " << tuple;
  bool inserted = false;
  std::vector<Node*> elements;
  elements.reserve(type->size());
  for (int64_t i = 0; i < type->size(); ++i) {
    if (inserted || i != indices.front()) {
      XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                           GetNodeAtIndex(tuple, {i}));
    } else {
      // NB This can only be run at most once since we set 'inserted' which
      // forces us into the simple append state above.
      inserted = true;
      if (indices.size() == 1) {
        // This is the actual location we are inserting things.
        elements.push_back(value);
        inserted = true;
        // Insert the i'th element we are pushing back.
        i--;
      } else {
        // We are inserting into the tuple at this index.
        XLS_ASSIGN_OR_RETURN(Node * front,
                             GetNodeAtIndex(tuple, {indices.front()}));
        XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                             SetTupleIndex(front, value, indices.subspan(1)),
                             _ << "Failed to set tuple index "
                               << indices.front() << " for " << tuple);
      }
    }
  }
  return tuple->function_base()->MakeNodeWithName<Tuple>(tuple->loc(), elements,
                                                         tuple->GetNameView());
}

absl::StatusOr<Node*> RemoveFromTuple(Node* tuple,
                                      absl::Span<int64_t const> indices) {
  XLS_RET_CHECK(!indices.empty());
  XLS_RET_CHECK(tuple->GetType()->IsTuple()) << "Not a tuple: " << tuple;
  TupleType* type = tuple->GetType()->AsTupleOrDie();
  XLS_RET_CHECK_GT(type->size(), indices.front())
      << "No element addressed in " << tuple;
  std::vector<Node*> elements;
  elements.reserve(type->size());
  for (int64_t i = 0; i < type->size(); ++i) {
    if (i != indices.front()) {
      XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                           GetNodeAtIndex(tuple, {i}));
    } else if (indices.size() != 1) {
      XLS_ASSIGN_OR_RETURN(Node * front,
                           GetNodeAtIndex(tuple, {indices.front()}));
      XLS_ASSIGN_OR_RETURN(std::back_inserter(elements),
                           RemoveFromTuple(front, indices.subspan(1)),
                           _ << "Failed to remove element from tuple index "
                             << indices.front() << " for " << tuple);
    }
  }
  return tuple->function_base()->MakeNodeWithName<Tuple>(tuple->loc(), elements,
                                                         tuple->GetNameView());
}
// Create and return a new node which is true if case 'i' is selected.
absl::StatusOr<Node*> GenericSelect::MakePredicateForCase(int64_t i) const {
  FunctionBase* fb = AsNode()->function_base();
  auto append_name = [&](std::string_view sv) -> std::string {
    if (AsNode()->HasAssignedName()) {
      return absl::StrCat(AsNode()->GetNameView(), "_", sv);
    }
    return "";
  };
  std::string case_name = append_name(absl::StrFormat("%d_case_value", i));
  std::string cmp_name = append_name(absl::StrFormat("%d_case_cmp", i));
  return std::visit(
      Visitor{
          [&](OneHotSelect* select) -> absl::StatusOr<Node*> {
            return fb->MakeNodeWithName<BitSlice>(AsNode()->loc(), selector(),
                                                  /*start=*/i, /*count=*/1,
                                                  cmp_name);
          },
          [&](Select* select) -> absl::StatusOr<Node*> {
            if (Bits::MinBitCountUnsigned(i) > selector()->BitCountOrDie()) {
              return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(0, 1)));
            }
            XLS_ASSIGN_OR_RETURN(
                Node * i_const,
                fb->MakeNodeWithName<Literal>(
                    AsNode()->loc(),
                    Value(UBits(i, selector()->BitCountOrDie())), case_name));
            return fb->MakeNodeWithName<CompareOp>(AsNode()->loc(), selector(),
                                                   i_const, Op::kEq, cmp_name);
          },
          [&](PrioritySelect* select) -> absl::StatusOr<Node*> {
            XLS_RET_CHECK_LT(i, selector()->BitCountOrDie());
            XLS_ASSIGN_OR_RETURN(
                Node * slice,
                fb->MakeNodeWithName<BitSlice>(
                    AsNode()->loc(), selector(),
                    /*start=*/0, /*count=*/i + 1,
                    append_name(absl::StrFormat("predicate_piece_%d", i))));
            InlineBitmap tgt(i + 1);
            tgt.Set(i, true);
            XLS_ASSIGN_OR_RETURN(
                Node * target_value,
                fb->MakeNodeWithName<Literal>(
                    AsNode()->loc(), Value(Bits::FromBitmap(std::move(tgt))),
                    case_name));
            return fb->MakeNodeWithName<CompareOp>(
                AsNode()->loc(), slice, target_value, Op::kEq, cmp_name);
          }},
      sel_);
}

// Create and return a new node which is true if the default case is selected.
absl::StatusOr<Node*> GenericSelect::MakePredicateForDefault() const {
  FunctionBase* fb = AsNode()->function_base();
  auto append_name = [&](std::string_view sv) -> std::string {
    if (AsNode()->HasAssignedName()) {
      return absl::StrCat(AsNode()->GetNameView(), "_", sv);
    }
    return "";
  };
  std::string case_name = append_name("default_case_value");
  std::string cmp_name = append_name("default_case_cmp");
  return std::visit(
      Visitor{
          [&](OneHotSelect* select) -> absl::StatusOr<Node*> {
            return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(0, 1)));
          },
          [&](Select* select) -> absl::StatusOr<Node*> {
            if (!select->default_value()) {
              return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(0, 1)));
            }
            if (select->cases().empty()) {
              return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(1, 1)));
            }
            XLS_ASSIGN_OR_RETURN(Node * i_const,
                                 fb->MakeNodeWithName<Literal>(
                                     AsNode()->loc(),
                                     Value(UBits(select->cases().size() - 1,
                                                 selector()->BitCountOrDie())),
                                     case_name));
            return fb->MakeNodeWithName<CompareOp>(AsNode()->loc(), selector(),
                                                   i_const, Op::kUGt, cmp_name);
          },
          [&](PrioritySelect* select) -> absl::StatusOr<Node*> {
            if (!select->default_value()) {
              return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(0, 1)));
            }
            if (select->cases().empty()) {
              return fb->MakeNode<Literal>(AsNode()->loc(), Value(UBits(1, 1)));
            }
            XLS_ASSIGN_OR_RETURN(
                Node * const_val,
                fb->MakeNode<Literal>(selector()->loc(),
                                      ZeroOfType(selector()->GetType())));
            return fb->MakeNodeWithName<CompareOp>(
                AsNode()->loc(), selector(), const_val, Op::kEq, cmp_name);
          }},
      sel_);
}
/* static */ absl::StatusOr<GenericSelect> GenericSelect::From(Node* n) {
  switch (n->op()) {
    case Op::kOneHotSel:
      return GenericSelect(n->As<OneHotSelect>());
    case Op::kSel:
      return GenericSelect(n->As<Select>());
    case Op::kPrioritySel:
      return GenericSelect(n->As<PrioritySelect>());
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("%s is not a select like operation.", n->ToString()));
  }
}

absl::StatusOr<Node*> GenericSelect::MakeSelectLikeWithNewArms(
    absl::Span<Node* const> new_cases, std::optional<Node*> new_default_value,
    const SourceInfo& loc) const {
  XLS_RET_CHECK(valid());
  XLS_RET_CHECK_EQ(new_cases.size(), cases().size());
  FunctionBase* fb = AsNode()->function_base();
  return std::visit(
      Visitor{[&](Select* /*unused*/) -> absl::StatusOr<Node*> {
                return fb->MakeNode<Select>(loc, selector(), new_cases,
                                            new_default_value);
              },
              [&](PrioritySelect* /*unused*/) -> absl::StatusOr<Node*> {
                XLS_RET_CHECK(new_default_value.has_value());
                return fb->MakeNode<PrioritySelect>(loc, selector(), new_cases,
                                                    *new_default_value);
              },
              [&](OneHotSelect* /*unused*/) -> absl::StatusOr<Node*> {
                XLS_RET_CHECK(!new_default_value.has_value());
                return fb->MakeNode<OneHotSelect>(loc, selector(), new_cases);
              }},
      sel_);
}

}  // namespace xls
