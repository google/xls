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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"

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

// Returns true if `node` is of single-bit Bits type.
inline bool IsSingleBitType(const Node* node) {
  return node->GetType()->IsBits() && node->BitCountOrDie() == 1;
}

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

inline bool HasSingleUse(Node* node) {
  if (node->function_base()->HasImplicitUse(node)) {
    return node->users().empty();
  }
  return node->users().size() == 1;
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

// Returns an IR expression whose value is equal to the bits of 'operand' at the
// given indices concated together. All 'indices' must be unique.
absl::StatusOr<Node*> GatherBits(Node* node, absl::Span<int64_t const> indices);

// Returns an IR expression whose value is equal to the bits of 'node' at the
// indices where 'mask' is true, discarding all bits where 'mask' is false.
// 'node' must be bits-typed, and 'mask' must have the same bit count as 'node'.
absl::StatusOr<Node*> GatherBits(Node* node, const Bits& mask);

// Returns an IR expression whose bits is equal to the values in 'pattern' where
// known, and otherwise fills in with the bits from 'node' (both going in
// LSB-first order). 'node' must be bits-typed, and must have bit count equal to
// the number of unknown bits in 'pattern'.
absl::StatusOr<Node*> FillPattern(TernarySpan pattern, Node* node);

// And-reduces the trailing (least significant) "bit_count" bits of node.
// If `source_info` is provided, will set all generated nodes to have the given
// `source_info`; otherwise, will use `node->loc()`.
//
// If bit_count == 0, returns a literal 1.
absl::StatusOr<Node*> AndReduceTrailing(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info = std::nullopt);

// Nor-reduces the trailing (least significant) "bit_count" bits of node.
// If `source_info` is provided, will set all generated nodes to have the given
// `source_info`.
//
// If bit_count == 0, returns a literal 1.
absl::StatusOr<Node*> NorReduceTrailing(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info = std::nullopt);

// Or-reduces the leading (most significant) "bit_count" bits of node.
// If `source_info` is provided, will set all generated nodes to have the given
// `source_info`.
//
// If bit_count == 0, returns a literal 0.
absl::StatusOr<Node*> OrReduceLeading(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info = std::nullopt);

// Nor-reduces the leading (most significant) "bit_count" bits of node.
// If `source_info` is provided, will set all generated nodes to have the given
// `source_info`.
//
// If bit_count == 0, returns a literal 1.
absl::StatusOr<Node*> NorReduceLeading(
    Node* node, int64_t bit_count,
    const std::optional<SourceInfo>& source_info = std::nullopt);

// And-reduce the given operands if needed. If there are 2+ operands, returns an
// N-ary AND of them; if there is 1 operand, returns that operand; and if there
// are no operands, returns a literal 1.
absl::StatusOr<Node*> NaryAndIfNeeded(FunctionBase* f,
                                      absl::Span<Node* const> operands,
                                      std::string_view name = "",
                                      const SourceInfo& source_info = {});

// Or-reduce the given operands if needed. If there are 2+ operands, returns an
// N-ary OR of them; if there is 1 operand, returns that operand; and if there
// are no operands, returns a literal 0.
absl::StatusOr<Node*> NaryOrIfNeeded(FunctionBase* f,
                                     absl::Span<Node* const> operands,
                                     std::string_view name = "",
                                     const SourceInfo& source_info = {});

// Nor-reduce the given operands if needed. If there are 2+ operands, returns an
// N-ary NOR of them; if there is 1 operand, returns a negation of it; and if
// there are no operands, fails.
absl::StatusOr<Node*> NaryNorIfNeeded(FunctionBase* f,
                                      absl::Span<Node* const> operands,
                                      std::string_view name = "",
                                      const SourceInfo& source_info = {});

// Returns whether the given node is a signed/unsigned comparison operation (for
// example, ULe or SGt).
bool IsUnsignedCompare(Node* node);
bool IsSignedCompare(Node* node);

// For <AndReduce, OrReduce, XorReduce>, returns <And, Or, Xor>.
absl::StatusOr<Op> OpToNonReductionOp(Op reduce_op);

// Returns true if the given node is a send/sendif/receive/recieveif node.
bool IsChannelNode(Node* node);

// Returns the channel used by the given node. If node is not a
// send/sendif/receive/receiveif node then an error is returned. Only supported
// for old-style procs.
// TODO(https://github.com/google/xls/issues/869): Remove when all procs are
// new-style.
absl::StatusOr<Channel*> GetChannelUsedByNode(Node* node);

// Returns the ChannelReference used by the given node. If node is not a
// send/sendif/receive/receiveif node then an error is returned. Only supported
// for  new-style procs.
absl::StatusOr<ChannelReference*> GetChannelReferenceUsedByNode(Node* node);

// Returns the ChannelRef used by the given node. If node is not a
// send/sendif/receive/receiveif node then an error is returned. Works for
// old-style and new-style procs.
absl::StatusOr<ChannelRef> GetChannelRefUsedByNode(Node* node);

// Replace the channel that node is operating on. If node is not a
// send/sendif/receive/receiveif node then an error is returned.
absl::Status ReplaceChannelUsedByNode(Node* node, std::string_view new_channel);

// Returns the predicate used by the given node. If node is not a
// send/sendif/receive/receiveif node then an error is returned.
absl::StatusOr<std::optional<Node*>> GetPredicateUsedByNode(Node* node);

// Get the value of a node at the given leaf-type-tree index.
absl::StatusOr<Node*> GetNodeAtIndex(Node* base,
                                     absl::Span<const int64_t> index);

// For a tuple-typed `node`, replace the tuple elements with new values from the
// `replacements` map. This will fail if a value in `replacements` depends on
// `node`.
absl::StatusOr<Node*> ReplaceTupleElementsWith(
    Node* node, const absl::flat_hash_map<int64_t, Node*>& replacements);

// For a tuple-typed `node`, replace the tuple element at `index` with
// `replacement_element`. This will fail if replacement_element depends on
// `node`.
absl::StatusOr<Node*> ReplaceTupleElementsWith(Node* node, int64_t index,
                                               Node* replacement_element);

// Compares the ID of two nodes. Can be used for deterministic sorting of Nodes.
inline bool NodeIdLessThan(const Node* a, const Node* b) {
  return a->id() < b->id();
}

// Sorts the given vector of Nodes by ID.
template <typename NodePtrT>
void SortByNodeId(std::vector<NodePtrT>* v) {
  std::sort(v->begin(), v->end(), NodeIdLessThan);
}

// Returns a vector containing the nodes in the given set sorted by node id.
template <typename NodePtrT>
std::vector<NodePtrT> SetToSortedVector(
    const absl::flat_hash_set<NodePtrT>& s) {
  std::vector<NodePtrT> v;
  v.reserve(s.size());
  v.insert(v.begin(), s.begin(), s.end());
  SortByNodeId(&v);
  return v;
}

// Returns true if the given node is a binary select (two cases, no default).
bool IsBinarySelect(Node* node);

// Returns the op which is the inverse of the given comparison.
//
// That is (not (op L R)) == ((InvertComparisonOp op) L R).
//
// May only be called with ops where 'OpIsCompare(op)' is true.
inline absl::StatusOr<Op> InvertComparisonOp(Op op) {
  switch (op) {
    case Op::kSGe:
      return Op::kSLt;
    case Op::kSGt:
      return Op::kSLe;
    case Op::kSLe:
      return Op::kSGt;
    case Op::kSLt:
      return Op::kSGe;
    case Op::kUGe:
      return Op::kULt;
    case Op::kUGt:
      return Op::kULe;
    case Op::kULe:
      return Op::kUGt;
    case Op::kULt:
      return Op::kUGe;
    case Op::kEq:
      return Op::kNe;
    case Op::kNe:
      return Op::kEq;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("%v is not a comparison operation", op));
  }
}

// Returns the op which is the reverse of the given comparison.
//
// That is (op L R) == ((ReverseComparisonOp op) R L).
//
// May only be called with ops where 'OpIsCompare(op)' is true.
inline absl::StatusOr<Op> ReverseComparisonOp(Op op) {
  switch (op) {
    case Op::kULe:
      return Op::kUGe;
    case Op::kULt:
      return Op::kUGt;
    case Op::kUGe:
      return Op::kULe;
    case Op::kUGt:
      return Op::kULt;
    case Op::kSLe:
      return Op::kSGe;
    case Op::kSLt:
      return Op::kSGt;
    case Op::kSGe:
      return Op::kSLe;
    case Op::kSGt:
      return Op::kSLt;
    case Op::kEq:
      return Op::kEq;
    case Op::kNe:
      return Op::kNe;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("%v is not a comparison operation", op));
  }
}

inline absl::StatusOr<Op> SignedCompareToUnsigned(Op op) {
  switch (op) {
    case Op::kSGe:
      return Op::kUGe;
    case Op::kSGt:
      return Op::kUGt;
    case Op::kSLe:
      return Op::kULe;
    case Op::kSLt:
      return Op::kULt;
    default:
      return absl::InternalError(
          absl::StrFormat("Unexpected comparison op %v", op));
  }
}

absl::StatusOr<absl::flat_hash_map<Channel*, std::vector<Node*>>> ChannelUsers(
    Package* package);

}  // namespace xls

#endif  // XLS_IR_NODE_UTIL_H_
