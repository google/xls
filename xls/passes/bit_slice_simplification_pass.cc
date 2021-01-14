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

#include "xls/passes/bit_slice_simplification_pass.h"

#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node_iterator.h"

namespace xls {
namespace {

// Attempts to replace the given bit slice with a simpler or more canonical
// form. Returns true if the bit slice was replaced. Any newly created
// bit-slices are added to the worklist.
absl::StatusOr<bool> SimplifyBitSlice(BitSlice* bit_slice,
                                      std::deque<BitSlice*>* worklist) {
  Node* operand = bit_slice->operand(0);
  BitsType* operand_type = operand->GetType()->AsBitsOrDie();

  // Creates a new bit slice and adds it to the worklist.
  auto make_bit_slice = [&](absl::optional<SourceLocation> loc, Node* operand,
                            int64 start,
                            int64 width) -> absl::StatusOr<BitSlice*> {
    XLS_ASSIGN_OR_RETURN(BitSlice * new_bit_slice,
                         bit_slice->function_base()->MakeNode<BitSlice>(
                             loc, operand, start, width));
    worklist->push_back(new_bit_slice);
    return new_bit_slice;
  };

  // A full width slice is a nop.
  if (bit_slice->start() == 0 &&
      bit_slice->width() == operand_type->bit_count()) {
    return bit_slice->ReplaceUsesWith(operand);
  }

  // A slice of a slice can be replaced by a single slice.
  //    BitSlice(BitSlice(val, ...), ...) -> BitSlice(val, ...)
  if (operand->Is<BitSlice>()) {
    BitSlice* operand_bit_slice = operand->As<BitSlice>();
    XLS_RETURN_IF_ERROR(
        bit_slice
            ->ReplaceUsesWithNew<BitSlice>(
                operand->operand(0),
                /*start=*/bit_slice->start() + operand_bit_slice->start(),
                /*width=*/bit_slice->width())
            .status());
    return true;
  }

  // A slice can be hoisted above a concat. The new concat will have a subset of
  // the original concats operands. The first and last operand of the concat may
  // themselves be sliced. Example:
  //
  //   BitSlice(Concat(a, b, c, d), ..) -> Concat(BitSlice(b), c, BitSlice(d))
  //
  if (operand->Is<Concat>()) {
    Concat* concat = operand->As<Concat>();
    std::vector<Node*> new_operands;
    // Inclusive bounds of the start/end of the bit slice. Values are bit
    // indices within the output of the concat.
    const int64 slice_start = bit_slice->start();
    const int64 slice_end = bit_slice->start() + bit_slice->width() - 1;

    // Start index of the current operand in the iteration.
    int64 operand_start = 0;

    // Iterate through the operands in reverse order because this is the
    // increasing order of bit indices.
    for (int64 i = concat->operand_count() - 1; i >= 0; --i) {
      Node* concat_operand = concat->operand(i);
      const int64 concat_operand_width = concat_operand->BitCountOrDie();
      // Inclusive bound on the bit offset of the operand in the concat
      // operation.
      const int64 operand_end = operand_start + concat_operand_width - 1;

      if (operand_start > slice_end) {
        // Operand (and all subsequent operands) is entirely after the end of
        // the slice.
        break;
      }

      if (slice_start > operand_end) {
        // Operand is entirely before the beginning of the slice (lower bit
        // index).
      } else if (slice_start <= operand_start && slice_end >= operand_end) {
        // Operand is entirely within the slice.
        new_operands.push_back(concat_operand);
      } else {
        // Operand is partially within the slice. Slice out the part of the
        // operand which is within the slice and add it to the list of operands
        // for the replacement concat.
        const int64 operand_slice_start = std::max(slice_start, operand_start);
        const int64 operand_slice_end = std::min(slice_end, operand_end);
        XLS_ASSIGN_OR_RETURN(
            Node * operand_slice,
            make_bit_slice(bit_slice->loc(), concat_operand,
                           /*start=*/operand_slice_start - operand_start,
                           operand_slice_end - operand_slice_start + 1));
        new_operands.push_back(operand_slice);
      }
      operand_start += concat_operand_width;
    }
    std::reverse(new_operands.begin(), new_operands.end());
    XLS_RETURN_IF_ERROR(
        bit_slice->ReplaceUsesWithNew<Concat>(new_operands).status());
    return true;
  }

  // Bit slice that is the sole consumer of an op can often lead the slice to
  // propagate into the operands to reduce the work the op has to do.
  if (operand->users().size() <= 1) {
    // For bitwise operations we can always slice the operands (because it's a
    // bit-parallel operation).
    //
    // For arithmetic operations that carry low bits into high bits, where the
    // slice starts at the LSb, we can slice the operands instead of the output.
    // (Note this is not possible for operations that carry high bits into low
    // bits like right shifts.)
    //
    // Note we can't simplify a shll without clamping the RHS; e.g.
    //
    //    x: bits[4]
    //    y: bits[4]
    //    (x << y)[2:0]
    //
    // Where "y = 8" would cause the shift amount to go from "total" to "zero"
    // if we sliced the "y" value.
    bool low_bits_of_arith_output =
        bit_slice->start() == 0 &&
        (operand->op() == Op::kAdd || operand->op() == Op::kSub ||
         operand->op() == Op::kNeg);
    if (OpIsBitWise(operand->op()) || low_bits_of_arith_output) {
      std::vector<Node*> sliced_operands;
      for (Node* o : operand->operands()) {
        XLS_ASSIGN_OR_RETURN(Node * new_operand,
                             make_bit_slice(o->loc(), o, bit_slice->start(),
                                            bit_slice->width()));
        sliced_operands.push_back(new_operand);
      }
      XLS_ASSIGN_OR_RETURN(Node * pre_sliced, operand->Clone(sliced_operands));
      XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(pre_sliced).status());
      return true;
    }
  }

  // Hoist slices above sign-extends. Let:
  //
  //   X = ...
  //   ext = sign_ext(X, new_width=nw)
  //   slice = bit_slice(ext, start=s, width=w)
  //
  // Represent ext as 'ssssssssXXXXXXXX' where the 's's are the extended sign
  // bits and the 'X's are the bits of the value being extended. There are three
  // possibilities depending upon where the slice falls relative to the sign
  // bit.
  //
  //
  //  (1) Slice entirely in sign-extend operand:
  //
  //      ssssssssssssXXXXXXXXXXXX
  //                   | slice |
  //
  //      Transformation: replace the slice the sign-extend with a slice of X
  //      (the sign-extend's operand).
  //
  //   (2) Slice spans the sign bit of the sign-extend operand.
  //
  //       ssssssssssssXXXXXXXXXXXX
  //              | slice |
  //
  //      Transformation: slice the most-significant bits from X and sign-extend
  //      the result.
  //
  //   (3) Slice is entirely within the sign extended bits.
  //
  //       ssssssssssssXXXXXXXXXXXX
  //        | slice |
  //
  //       Transformation: slice the sign bit from X and sign-extend the result.
  //
  // To avoid introducing an additional sign-extension cases (2) and (3) should
  // only be performed if the bit-slice is the only user of the sign-extend.
  if (bit_slice->operand(0)->op() == Op::kSignExt) {
    ExtendOp* ext = bit_slice->operand(0)->As<ExtendOp>();
    Node* x = ext->operand(0);
    int64 x_bit_count = x->BitCountOrDie();
    if (bit_slice->start() + bit_slice->width() <= x_bit_count) {
      // Case (1), replace with slice of sign-extend's operand.
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          make_bit_slice(bit_slice->loc(), x, bit_slice->start(),
                         bit_slice->width()));
      XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(replacement).status());
      return true;
    } else if (ext->users().size() == 1) {
      if (bit_slice->start() < x_bit_count) {
        // Case (2), slice straddles the sign bit.
        XLS_ASSIGN_OR_RETURN(
            Node * x_slice,
            make_bit_slice(bit_slice->loc(), x, /*start=*/bit_slice->start(),
                           /*width=*/x_bit_count - bit_slice->start()));
        XLS_RETURN_IF_ERROR(
            bit_slice
                ->ReplaceUsesWithNew<ExtendOp>(
                    x_slice,
                    /*new_bit_count=*/bit_slice->BitCountOrDie(), Op::kSignExt)
                .status());
      } else {
        // Case (3), slice includes only the extended bits.
        XLS_ASSIGN_OR_RETURN(
            Node * x_sign_bit,
            make_bit_slice(bit_slice->loc(), x,
                           /*start=*/x_bit_count - 1, /*width=*/1));
        XLS_RETURN_IF_ERROR(
            bit_slice
                ->ReplaceUsesWithNew<ExtendOp>(
                    x_sign_bit,
                    /*new_bit_count=*/bit_slice->BitCountOrDie(), Op::kSignExt)
                .status());
      }
      return true;
    }
  }

  // Hoist slices above left shift if the slice starts at zero and above right
  // shifts if the slice ends at the MSB. Only perform this if the slice is the
  // sole user.
  if (((bit_slice->start() == 0 && operand->op() == Op::kShll) ||
       ((bit_slice->start() + bit_slice->width() == operand->BitCountOrDie()) &&
        (operand->op() == Op::kShrl || operand->op() == Op::kShra))) &&
      operand->users().size() == 1) {
    Node* shift = operand;
    Node* to_shift = shift->operand(0);
    Node* shift_amount = shift->operand(1);
    XLS_ASSIGN_OR_RETURN(
        Node * sliced_to_shift,
        make_bit_slice(operand->loc(), to_shift, bit_slice->start(),
                       bit_slice->width()));
    XLS_RETURN_IF_ERROR(bit_slice
                            ->ReplaceUsesWithNew<BinOp>(
                                sliced_to_shift, shift_amount, shift->op())
                            .status());
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> BitSliceSimplificationPass::RunOnFunctionBase(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << "Running bit-slice simplifier on function " << f->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, f->DumpIr());

  std::deque<BitSlice*> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<BitSlice>()) {
      worklist.push_back(node->As<BitSlice>());
    }
  }
  bool changed = false;
  while (!worklist.empty()) {
    BitSlice* bit_slice = worklist.front();
    worklist.pop_front();
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyBitSlice(bit_slice, &worklist));
    changed = changed || node_changed;
  }

  // Replace dynamic bit slices with literal indices with a non-dynamic bit
  // slice.
  for (Node* node : f->nodes()) {
    if (node->Is<DynamicBitSlice>() && node->operand(1)->Is<Literal>()) {
      int64 result_width = node->BitCountOrDie();
      int64 operand_width = node->operand(0)->BitCountOrDie();
      const Bits& start_bits = node->operand(1)->As<Literal>()->value().bits();
      if (bits_ops::ULessThanOrEqual(start_bits,
                                     operand_width - result_width)) {
        XLS_ASSIGN_OR_RETURN(uint64 start, start_bits.ToUint64());
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<BitSlice>(node->operand(0),
                                               /*start=*/start,
                                               /*width=*/node->BitCountOrDie())
                .status());
        changed = true;
      }
    }
  }

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, f->DumpIr());

  return changed;
}

}  // namespace xls
