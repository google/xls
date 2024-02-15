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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// Attempts to replace the given bit slice with a simpler or more canonical
// form. Returns true if the bit slice was replaced. Any newly created
// bit-slices are added to the worklist.
absl::StatusOr<bool> SimplifyBitSlice(BitSlice* bit_slice, int64_t opt_level,
                                      std::deque<BitSlice*>* worklist) {
  Node* operand = bit_slice->operand(0);
  BitsType* operand_type = operand->GetType()->AsBitsOrDie();

  // Creates a new bit slice and adds it to the worklist.
  auto make_bit_slice = [&](const SourceInfo& loc, Node* operand, int64_t start,
                            int64_t width) -> absl::StatusOr<BitSlice*> {
    XLS_ASSIGN_OR_RETURN(BitSlice * new_bit_slice,
                         bit_slice->function_base()->MakeNode<BitSlice>(
                             loc, operand, start, width));
    worklist->push_back(new_bit_slice);
    return new_bit_slice;
  };

  // A full width slice is a nop.
  if (bit_slice->start() == 0 &&
      bit_slice->width() == operand_type->bit_count()) {
    XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(operand));
    return true;
  }

  // A slice of a slice can be replaced by a single slice.
  //    BitSlice(BitSlice(val, ...), ...) -> BitSlice(val, ...)
  if (operand->Is<BitSlice>()) {
    BitSlice* operand_bit_slice = operand->As<BitSlice>();
    XLS_ASSIGN_OR_RETURN(BitSlice * new_bitslice,
                         make_bit_slice(bit_slice->loc(), operand->operand(0),
                                        /*start=*/bit_slice->start() +
                                            operand_bit_slice->start(),
                                        /*width=*/bit_slice->width()));
    XLS_VLOG(3) << absl::StreamFormat(
        "Replacing bitslice(bitslice(x)) => bitslice(x): %s",
        bit_slice->GetName());
    XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(new_bitslice));
    return true;
  }

  // A slice can be hoisted above a concat. The new concat will have a subset of
  // the original concats operands. The first and last operand of the concat may
  // themselves be sliced. Example:
  //
  //   BitSlice(Concat(a, b, c, d), ..) -> Concat(BitSlice(b), c, BitSlice(d))
  //
  if (NarrowingEnabled(opt_level) && operand->Is<Concat>()) {
    Concat* concat = operand->As<Concat>();
    std::vector<Node*> new_operands;
    // Inclusive bounds of the start/end of the bit slice. Values are bit
    // indices within the output of the concat.
    const int64_t slice_start = bit_slice->start();
    const int64_t slice_end = bit_slice->start() + bit_slice->width() - 1;

    // Start index of the current operand in the iteration.
    int64_t operand_start = 0;

    // Iterate through the operands in reverse order because this is the
    // increasing order of bit indices.
    for (int64_t i = concat->operand_count() - 1; i >= 0; --i) {
      Node* concat_operand = concat->operand(i);
      const int64_t concat_operand_width = concat_operand->BitCountOrDie();
      // Inclusive bound on the bit offset of the operand in the concat
      // operation.
      const int64_t operand_end = operand_start + concat_operand_width - 1;

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
        const int64_t operand_slice_start =
            std::max(slice_start, operand_start);
        const int64_t operand_slice_end = std::min(slice_end, operand_end);
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
    XLS_VLOG(3) << absl::StreamFormat(
        "Replacing bitslice(concat(...)) => concat(bitslice(), ...): %s",
        bit_slice->GetName());
    XLS_RETURN_IF_ERROR(
        bit_slice->ReplaceUsesWithNew<Concat>(new_operands).status());
    return true;
  }

  // Bit slice that is the sole consumer of an op can often lead the slice to
  // propagate into the operands to reduce the work the op has to do.
  if (NarrowingEnabled(opt_level) && HasSingleUse(operand)) {
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
      XLS_VLOG(3) << absl::StreamFormat(
          "Replacing bitslice(op(...)) => op(bitslice(), bitslice(), ...): %s",
          bit_slice->GetName());
      XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(pre_sliced));
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
  if (NarrowingEnabled(opt_level) &&
      bit_slice->operand(0)->op() == Op::kSignExt) {
    ExtendOp* ext = bit_slice->operand(0)->As<ExtendOp>();
    Node* x = ext->operand(0);
    int64_t x_bit_count = x->BitCountOrDie();
    if (bit_slice->start() + bit_slice->width() <= x_bit_count) {
      // Case (1), replace with slice of sign-extend's operand.
      XLS_ASSIGN_OR_RETURN(
          Node * replacement,
          make_bit_slice(bit_slice->loc(), x, bit_slice->start(),
                         bit_slice->width()));
      XLS_VLOG(3) << absl::StreamFormat(
          "Replacing bitslice(ext(x)) => bitslice(x): %s",
          bit_slice->GetName());
      XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(replacement));
      return true;
    }
    if (HasSingleUse(ext)) {
      if (bit_slice->start() < x_bit_count) {
        // Case (2), slice straddles the sign bit.
        XLS_ASSIGN_OR_RETURN(
            Node * x_slice,
            make_bit_slice(bit_slice->loc(), x, /*start=*/bit_slice->start(),
                           /*width=*/x_bit_count - bit_slice->start()));
        XLS_VLOG(3) << absl::StreamFormat(
            "Replacing bitslice(ext(x)) => ext(bitslice(x)): %s",
            bit_slice->GetName());
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
        XLS_VLOG(3) << absl::StreamFormat(
            "Replacing bitslice(signext(x)) => ext(signbit(x)): %s",
            bit_slice->GetName());
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

  // Are all users of the operand slices entirely contained within this slice?
  // If so, we may be able to narrow the operand & eliminate this slice.
  auto all_operand_users_are_subslices = [operand, bit_slice] {
    return !operand->function_base()->HasImplicitUse(operand) &&
           absl::c_all_of(operand->users(), [bit_slice](Node* user) {
             return user->Is<BitSlice>() &&
                    user->As<BitSlice>()->start() >= bit_slice->start() &&
                    user->As<BitSlice>()->start() +
                            user->As<BitSlice>()->width() <=
                        bit_slice->start() + bit_slice->width();
           });
  };

  // Hoist slices above left shift if the slice starts at zero and above right
  // shifts if the slice ends at the MSB. Only perform this if all users are
  // slices entirely contained within this slice.
  if (((bit_slice->start() == 0 && operand->op() == Op::kShll) ||
       ((bit_slice->start() + bit_slice->width() == operand->BitCountOrDie()) &&
        (operand->op() == Op::kShrl || operand->op() == Op::kShra))) &&
      all_operand_users_are_subslices()) {
    Node* shift = operand;
    Node* to_shift = shift->operand(0);
    Node* shift_amount = shift->operand(1);
    XLS_ASSIGN_OR_RETURN(
        Node * sliced_to_shift,
        make_bit_slice(operand->loc(), to_shift, bit_slice->start(),
                       bit_slice->width()));
    XLS_VLOG(3) << absl::StreamFormat(
        "Replacing bitslice(shift(x, y)) => shift(bitslice(x), y): %s",
        bit_slice->GetName());
    XLS_ASSIGN_OR_RETURN(BinOp * new_shift,
                         bit_slice->ReplaceUsesWithNew<BinOp>(
                             sliced_to_shift, shift_amount, shift->op()));
    std::vector<Node*> users(operand->users().begin(), operand->users().end());
    for (Node* user : users) {
      if (user == bit_slice) {
        continue;
      }
      XLS_VLOG(3) << absl::StreamFormat("Replacing %s with %s in: %s",
                                        operand->GetName(),
                                        new_shift->GetName(), user->ToString());
      XLS_ASSIGN_OR_RETURN(
          BitSlice * new_user,
          make_bit_slice(user->loc(), new_shift,
                         user->As<BitSlice>()->start() - bit_slice->start(),
                         user->As<BitSlice>()->width()));
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(new_user));
    }
    return true;
  }

  // Combine slices with decode if the slice starts at zero. Only perform this
  // if all users are slices entirely contained within this slice.
  if (bit_slice->start() == 0 && operand->op() == Op::kDecode &&
      !operand->function_base()->HasImplicitUse(operand) &&
      all_operand_users_are_subslices()) {
    XLS_VLOG(3) << absl::StreamFormat(
        "Replacing bitslice(decode(x), 0, %d) => decode<u%d>(x): %s",
        bit_slice->width(), bit_slice->width(), bit_slice->GetName());
    XLS_ASSIGN_OR_RETURN(Decode * new_decode,
                         bit_slice->ReplaceUsesWithNew<Decode>(
                             operand->operand(0), bit_slice->width()));
    std::vector<Node*> users(operand->users().begin(), operand->users().end());
    for (Node* user : users) {
      if (user == bit_slice) {
        continue;
      }
      XLS_VLOG(3) << absl::StreamFormat(
          "Replacing %s with %s in: %s", operand->GetName(),
          new_decode->GetName(), user->ToString());
      XLS_ASSIGN_OR_RETURN(
          BitSlice * new_user,
          make_bit_slice(user->loc(), new_decode,
                         user->As<BitSlice>()->start() - bit_slice->start(),
                         user->As<BitSlice>()->width()));
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(new_user));
    }
    return true;
  }

  // Hoist slices above selects and one-hot-selects. Only perform this if all
  // users are slices entirely contained within this slice.
  if (NarrowingEnabled(opt_level) &&
      (operand->Is<Select>() || operand->Is<OneHotSelect>()) &&
      all_operand_users_are_subslices()) {
    Node* select = operand;
    std::vector<Node*> new_operands;
    // Operand 0 is the selector in both Select and OneHotSelect operations and
    // is unchanged.
    new_operands.push_back(select->operand(0));
    // The remaining operands are the cases and should all be sliced.
    for (int64_t i = 1; i < select->operand_count(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * sliced_operand,
          make_bit_slice(operand->loc(), select->operand(i), bit_slice->start(),
                         bit_slice->width()));

      new_operands.push_back(sliced_operand);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_select, select->Clone(new_operands));
    XLS_VLOG(3) << absl::StreamFormat(
        "Replacing bitslice(sel(s, [...])) => sel(s, [bitslice(), ...]): %s",
        bit_slice->GetName());
    XLS_RETURN_IF_ERROR(bit_slice->ReplaceUsesWith(new_select));
    std::vector<Node*> users(operand->users().begin(), operand->users().end());
    for (Node* user : users) {
      if (user == bit_slice) {
        continue;
      }
      XLS_VLOG(3) << absl::StreamFormat(
          "Replacing %s with %s in: %s", operand->GetName(),
          new_select->GetName(), user->ToString());
      XLS_ASSIGN_OR_RETURN(
          BitSlice * new_user,
          make_bit_slice(user->loc(), new_select,
                         user->As<BitSlice>()->start() - bit_slice->start(),
                         user->As<BitSlice>()->width()));
      XLS_RETURN_IF_ERROR(user->ReplaceUsesWith(new_user));
    }
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> BitSliceSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  // Replace dynamic bit slices with literal indices with a non-dynamic bit
  // slice.
  for (Node* node : f->nodes()) {
    if (node->Is<DynamicBitSlice>() && node->operand(1)->Is<Literal>()) {
      int64_t result_width = node->BitCountOrDie();
      int64_t operand_width = node->operand(0)->BitCountOrDie();
      const Bits& start_bits = node->operand(1)->As<Literal>()->value().bits();
      // TODO(meheff): Handle OOB case.
      if (bits_ops::ULessThanOrEqual(start_bits,
                                     operand_width - result_width)) {
        XLS_ASSIGN_OR_RETURN(uint64_t start, start_bits.ToUint64());
        XLS_VLOG(3) << absl::StreamFormat(
            "Replacing dynamic bitslice %s with static bitslice",
            node->GetName());
        XLS_RETURN_IF_ERROR(
            node->ReplaceUsesWithNew<BitSlice>(node->operand(0),
                                               /*start=*/start,
                                               /*width=*/node->BitCountOrDie())
                .status());
        changed = true;
      }
    }
  }

  // Replace bit_slice_update operations where the start index is constant with
  // bit slices and concats.
  for (Node* node : f->nodes()) {
    if (node->Is<BitSliceUpdate>() &&
        node->As<BitSliceUpdate>()->start()->Is<Literal>()) {
      BitSliceUpdate* update = node->As<BitSliceUpdate>();
      const Bits start = update->start()->As<Literal>()->value().bits();
      if (bits_ops::UGreaterThanOrEqual(start, update->BitCountOrDie())) {
        // Bit slice update is entirely out of bounds. This is a noop. Replace
        // the update operation with the operand which is being updated.
        XLS_RETURN_IF_ERROR(update->ReplaceUsesWith(update->to_update()));
        changed = true;
        continue;
      }
      int64_t start_int64 = start.ToUint64().value();
      int64_t orig_width = update->BitCountOrDie();
      int64_t update_width = update->update_value()->BitCountOrDie();

      // Replace bitslice update with expression of slices of the update value
      // and the original vector.
      std::vector<Node*> concat_operands;
      if (start_int64 + update_width < orig_width) {
        // Bit slice update is entirely in bounds and the most-significant
        // bit(s) of the original vector are not updated.
        //
        //            0                              N
        //  original: |==============================|
        //  update  :      |=============|
        //
        //
        //  concat(bitslice(original, start=0), update, bitslice(original, ...)
        XLS_ASSIGN_OR_RETURN(
            Node * slice,
            f->MakeNode<BitSlice>(
                node->loc(), update->to_update(),
                /*start=*/start_int64 + update_width,
                /*width=*/orig_width - (start_int64 + update_width)));
        concat_operands.push_back(slice);
        concat_operands.push_back(update->update_value());
      } else if (start_int64 + update_width == orig_width) {
        // Bit slice update extends right up to end of updated vector.
        //
        //            0                              N
        //  original: |==============================|
        //  update  :                  |=============|
        //
        //
        //  concat(bitslice(original, start=0), update)
        concat_operands.push_back(update->update_value());
      } else {
        // The update value is partially out of bounds.
        //
        //            0                              N
        //  original: |==============================|
        //  update  :                       |=============|
        //
        //
        //  concat(bitslice(original, start=0), bitslice(update))
        XLS_RET_CHECK_GT(start_int64 + update_width, orig_width);
        int64_t excess = start_int64 + update_width - orig_width;
        XLS_ASSIGN_OR_RETURN(
            Node * slice,
            f->MakeNode<BitSlice>(node->loc(), update->update_value(),
                                  /*start=*/0,
                                  /*width=*/update_width - excess));
        concat_operands.push_back(slice);
      }
      if (start_int64 > 0) {
        XLS_ASSIGN_OR_RETURN(Node * slice, f->MakeNode<BitSlice>(
                                               node->loc(), update->to_update(),
                                               /*start=*/0,
                                               /*width=*/start_int64));
        concat_operands.push_back(slice);
      }
      XLS_VLOG(3) << absl::StreamFormat(
          "Replacing bitslice update %s with constant start index with concat "
          "and bitslice operations",
          node->GetName());
      if (concat_operands.size() == 1) {
        XLS_RETURN_IF_ERROR(update->ReplaceUsesWith(concat_operands.front()));
      } else {
        XLS_RETURN_IF_ERROR(
            update->ReplaceUsesWithNew<Concat>(concat_operands).status());
      }

      changed = true;
    }
  }

  std::deque<BitSlice*> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<BitSlice>()) {
      worklist.push_back(node->As<BitSlice>());
    }
  }
  while (!worklist.empty()) {
    BitSlice* bit_slice = worklist.front();
    worklist.pop_front();
    XLS_ASSIGN_OR_RETURN(bool node_changed,
                         SimplifyBitSlice(bit_slice, opt_level_, &worklist));
    changed = changed || node_changed;
  }

  return changed;
}

REGISTER_OPT_PASS(BitSliceSimplificationPass, pass_config::kOptLevel);

}  // namespace xls
