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
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/value_helpers.h"

namespace xls {

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

absl::StatusOr<Node*> GatherBits(Node* node,
                                 absl::Span<int64_t const> indices) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  FunctionBase* f = node->function_base();
  if (indices.empty()) {
    // Return a literal with a Bits value of zero width.
    return f->MakeNode<Literal>(node->loc(), Value(Bits()));
  }
  XLS_RET_CHECK(absl::c_is_sorted(indices)) << "Gather indices not sorted.";
  for (int64_t i = 1; i < indices.size(); ++i) {
    XLS_RET_CHECK_NE(indices[i - 1], indices[i])
        << "Gather indices not unique.";
  }
  if (indices.size() == node->BitCountOrDie()) {
    // Gathering all the bits. Just return the value.
    return node;
  }
  std::vector<Node*> segments;
  std::vector<Node*> slices;
  auto add_bit_slice = [&](int64_t start, int64_t end) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(
        Node * slice, f->MakeNode<BitSlice>(node->loc(), node, /*start=*/start,
                                            /*width=*/end - start));
    slices.push_back(slice);
    return absl::OkStatus();
  };
  int64_t slice_start = indices.front();
  int64_t slice_end = slice_start + 1;
  for (int64_t i = 1; i < indices.size(); ++i) {
    int64_t index = indices[i];
    if (index == slice_end) {
      slice_end++;
    } else {
      XLS_RETURN_IF_ERROR(add_bit_slice(slice_start, slice_end));
      slice_start = index;
      slice_end = index + 1;
    }
  }
  XLS_RETURN_IF_ERROR(add_bit_slice(slice_start, slice_end));
  if (slices.size() == 1) {
    return slices[0];
  }
  std::reverse(slices.begin(), slices.end());
  return f->MakeNode<Concat>(node->loc(), slices);
}

absl::StatusOr<Node*> AndReduceTrailing(Node* node, int64_t bit_count) {
  FunctionBase* f = node->function_base();
  // Reducing zero bits should return one (identity of AND).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(node->loc(), Value(UBits(1, 1)));
  }
  std::vector<Node*> bits;
  for (int64_t i = 0; i < bit_count; ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * bit,
        f->MakeNode<BitSlice>(node->loc(), node, /*start=*/i, /*width=*/1));
    bits.push_back(bit);
  }
  return f->MakeNode<NaryOp>(node->loc(), bits, Op::kAnd);
}

absl::StatusOr<Node*> OrReduceLeading(Node* node, int64_t bit_count) {
  FunctionBase* f = node->function_base();
  // Reducing zero bits should return zero (identity of OR).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(node->loc(), Value(UBits(0, 1)));
  }
  const int64_t width = node->BitCountOrDie();
  XLS_CHECK_LE(bit_count, width);
  std::vector<Node*> bits;
  for (int64_t i = 0; i < bit_count; ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * bit,
        f->MakeNode<BitSlice>(node->loc(), node,
                              /*start=*/width - bit_count + i, /*width=*/1));
    bits.push_back(bit);
  }
  return f->MakeNode<NaryOp>(node->loc(), bits, Op::kOr);
}

bool IsUnsignedCompare(Node* node) {
  switch (node->op()) {
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

bool IsChannelNode(Node* node) {
  return node->Is<Send>() || node->Is<Receive>();
}

absl::StatusOr<Channel*> GetChannelUsedByNode(Node* node) {
  int64_t channel_id;
  if (node->Is<Send>()) {
    channel_id = node->As<Send>()->channel_id();
  } else if (node->Is<Receive>()) {
    channel_id = node->As<Receive>()->channel_id();
  } else {
    return absl::NotFoundError(
        absl::StrFormat("No channel associated with node %s", node->GetName()));
  }
  return node->package()->GetChannel(channel_id);
}

absl::Status ReplaceChannelUsedByNode(Node* node, int64_t new_channel_id) {
  switch (node->op()) {
    case Op::kSend:
      node->As<Send>()->ReplaceChannel(new_channel_id);
      return absl::OkStatus();
    case Op::kReceive:
      node->As<Receive>()->ReplaceChannel(new_channel_id);
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Operation %s does not have a channel, must be send or receive.",
          node->GetName()));
  }
}

absl::StatusOr<std::optional<Node*>> GetPredicateUsedByNode(Node* node) {
  switch (node->op()) {
    case Op::kSend: {
      return node->As<Send>()->predicate();
    }
    case Op::kReceive: {
      return node->As<Receive>()->predicate();
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expected %s to be a send or receive.", node->GetName()));
  }
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

  // First, make an empty tuple. We're going to replace everything here, we just
  // need something to exist before calling ReplaceUsesWith() on node. After we
  // call ReplaceUsesWith(), we'll replace each operand of the tuple with a
  // TupleIndex from node.
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


bool IsBinarySelect(Node* node) {
  if (!node->Is<Select>()) {
    return false;
  }
  Select* sel = node->As<Select>();
  return sel->cases().size() == 2 && !sel->default_value().has_value();
}


}  // namespace xls
