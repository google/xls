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

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function_base.h"

namespace xls {

bool IsLiteralWithRunOfSetBits(Node* node, int64* leading_zero_count,
                               int64* set_bit_count,
                               int64* trailing_zero_count) {
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

absl::StatusOr<Node*> GatherBits(Node* node, absl::Span<int64 const> indices) {
  XLS_RET_CHECK(node->GetType()->IsBits());
  FunctionBase* f = node->function();
  if (indices.empty()) {
    // Return a literal with a Bits value of zero width.
    return f->MakeNode<Literal>(node->loc(), Value(Bits()));
  }
  XLS_RET_CHECK(absl::c_is_sorted(indices)) << "Gather indices not sorted.";
  for (int64 i = 1; i < indices.size(); ++i) {
    XLS_RET_CHECK_NE(indices[i - 1], indices[i])
        << "Gather indices not unique.";
  }
  if (indices.size() == node->BitCountOrDie()) {
    // Gathering all the bits. Just return the value.
    return node;
  }
  std::vector<Node*> segments;
  std::vector<Node*> slices;
  auto add_bit_slice = [&](int64 start, int64 end) -> absl::Status {
    XLS_ASSIGN_OR_RETURN(
        Node * slice, f->MakeNode<BitSlice>(node->loc(), node, /*start=*/start,
                                            /*width=*/end - start));
    slices.push_back(slice);
    return absl::OkStatus();
  };
  int64 slice_start = indices.front();
  int64 slice_end = slice_start + 1;
  for (int64 i = 1; i < indices.size(); ++i) {
    int64 index = indices[i];
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

absl::StatusOr<Node*> AndReduceTrailing(Node* node, int64 bit_count) {
  FunctionBase* f = node->function();
  // Reducing zero bits should return one (identity of AND).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(node->loc(), Value(UBits(1, 1)));
  }
  std::vector<Node*> bits;
  for (int64 i = 0; i < bit_count; ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * bit,
        f->MakeNode<BitSlice>(node->loc(), node, /*start=*/i, /*width=*/1));
    bits.push_back(bit);
  }
  return f->MakeNode<NaryOp>(node->loc(), bits, Op::kAnd);
}

absl::StatusOr<Node*> OrReduceLeading(Node* node, int64 bit_count) {
  FunctionBase* f = node->function();
  // Reducing zero bits should return zero (identity of OR).
  if (bit_count == 0) {
    return f->MakeNode<Literal>(node->loc(), Value(UBits(0, 1)));
  }
  const int64 width = node->BitCountOrDie();
  XLS_CHECK_LE(bit_count, width);
  std::vector<Node*> bits;
  for (int64 i = 0; i < bit_count; ++i) {
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

}  // namespace xls
