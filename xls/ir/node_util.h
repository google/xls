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

// Helper predicate functions for interrogating a Node*.

#ifndef XLS_IR_NODE_UTIL_H_
#define XLS_IR_NODE_UTIL_H_

#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/channel.h"
#include "xls/ir/nodes.h"

namespace xls {

inline bool IsLiteralZero(Node* node) {
  return node->Is<Literal>() && node->As<Literal>()->value().IsBits() &&
         node->As<Literal>()->value().bits().IsZero();
}

// Returns true if the given node is a literal with the value one when
// interpreted as an unsigned number
inline bool IsLiteralUnsignedOne(Node* node) {
  return node->Is<Literal>() && node->As<Literal>()->value().IsBits() &&
         node->As<Literal>()->value().bits().IsOne();
}

// Returns true if the given node is a literal with the value one when
// interpreted as a signed number. This is identical to IsLiteralUnsignedOne
// except for the case of one-bit numbers where 1'b1 is equal to -1.
inline bool IsLiteralSignedOne(Node* node) {
  return node->Is<Literal>() && node->As<Literal>()->value().IsBits() &&
         node->As<Literal>()->value().bits().IsOne() &&
         node->As<Literal>()->BitCountOrDie() > 1;
}

inline bool IsLiteralAllOnes(Node* node) {
  return node->Is<Literal>() && node->As<Literal>()->value().IsBits() &&
         node->As<Literal>()->value().bits().IsAllOnes();
}

// Checks whether the node is a literal that holds a bits value where
// HasSingleRunOfSetBits (see bits_ops.h) is true.
bool IsLiteralWithRunOfSetBits(Node* node, int64_t* leading_zero_count,
                               int64_t* set_bit_count,
                               int64_t* trailing_zero_count);

// Returns whether node is a literal "mask" value, which is of the form:
// 0b[0]*[1]+
//
// Note that "leading_zeros" is easily calculated as width - "trailing_ones", it
// is populated purely for caller convenience.
inline bool IsLiteralMask(Node* node, int64_t* leading_zeros,
                          int64_t* trailing_ones) {
  int64_t trailing_zeros = -1;
  if (IsLiteralWithRunOfSetBits(node, leading_zeros, trailing_ones,
                                &trailing_zeros) &&
      trailing_zeros == 0) {
    return true;
  }
  return false;
}

// For use as a predicate function.
inline bool IsLiteral(Node* node) { return node->Is<Literal>(); }
inline bool IsBitSlice(Node* node) { return node->Is<BitSlice>(); }

inline bool AnyOperandWhere(Node* node,
                            const std::function<bool(Node*)>& predicate) {
  return std::any_of(node->operands().begin(), node->operands().end(),
                     predicate);
}

inline bool AnyTwoOperandsWhere(Node* node,
                            const std::function<bool(Node*)>& predicate) {
  int64_t count = 0;
  for (Node* operand : node->operands()) {
    count += predicate(operand);
    if (count >= 2) {
      return true;
    }
  }
  return false;
}

inline bool SoleUserSatisfies(Node* node,
                              const std::function<bool(Node*)>& predicate) {
  if (node->users().size() != 1) {
    return false;
  }
  return predicate(*node->users().begin());
}

inline bool IsNotOf(const Node* node, const Node* inverted) {
  return node->op() == Op::kNot && node->operand(0) == inverted;
}

// For use in e.g. absl::StrJoin.
inline void NodeFormatter(std::string* out, Node* node) {
  absl::StrAppend(out, node->GetName());
}

// Returns an IR expression whose value is equal to the bits of 'operand' at the
// given indices concated together. 'indices' must be unique and sorted in an
// ascending order.
absl::StatusOr<Node*> GatherBits(Node* node, absl::Span<int64_t const> indices);

// And-reduces the trailing (least significant) "bit_count" bits of node.
//
// TODO(b/150557922): Create a dedicated opcode for and-reductions of multi-bit
// values.
absl::StatusOr<Node*> AndReduceTrailing(Node* node, int64_t bit_count);

// Or-reduces the leading (most significant) "bit_count" bits of node.
//
// TODO(b/150557922): Create a dedicated opcode for or-reductions of multi-bit
// values.
absl::StatusOr<Node*> OrReduceLeading(Node* node, int64_t bit_count);

// Returns whether the given node is a signed/unsigned comparison operation (for
// example, ULe or SGt).
bool IsUnsignedCompare(Node* node);
bool IsSignedCompare(Node* node);

// For <AndReduce, OrReduce, XorReduce>, returns <And, Or, Xor>.
absl::StatusOr<Op> OpToNonReductionOp(Op reduce_op);

// Returns true if the given node is a send/sendif/receive/recieveif node.
bool IsChannelNode(Node* node);

// Returns the channel used by the given node. If node is not a
// send/sendif/receive/receiveif node then an error is returned.
absl::StatusOr<Channel*> GetChannelUsedByNode(Node* node);

}  // namespace xls

#endif  // XLS_IR_NODE_UTIL_H_
