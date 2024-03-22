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

#include "xls/codegen/node_expressions.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/flattening.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"

namespace xls {
namespace verilog {

bool OperandMustBeNamedReference(Node* node, int64_t operand_no) {
  // Returns true if the emitted expression for the specified operand is
  // necessarily indexable. Generally, if the expression emitted for a node is
  // an indexing operation and the operand is emitted as an indexable expression
  // then there is no need to make the operand a declared expression because
  // indexing/slicing can be chained... unless we might need to narrow it if it
  // uses more bits than the size of the array.
  //
  // For example, a kArrayIndex of a kArrayIndex can be emitted as a chained
  // VAST Index expression like so:
  //
  // reg [42:0] foo = bar[42][7]
  //
  // In this case, no need to make bar[42] a named temporary.
  auto operand_is_indexable = [&]() {
    switch (node->operand(operand_no)->op()) {
      case Op::kArrayIndex:
      case Op::kParam:
        // These operations are emitted as VAST Index operations which
        // can be indexed.
        return true;
      default:
        return false;
    }
  };
  switch (node->op()) {
    case Op::kBitSlice:
      CHECK_EQ(operand_no, 0);
      return !operand_is_indexable();
    case Op::kDynamicBitSlice:
      return operand_no == 0 && !operand_is_indexable();
    case Op::kArrayIndex: {
      if (operand_is_indexable()) {
        return false;
      }
      switch (operand_no) {
        case 0:
          // The array needs to be indexable.
          return true;
        case 1: {
          // The indices need to be indexable if and only if we might need to
          // truncate one to ensure we don't use too many bits for the array's
          // indexing operation.
          CHECK_EQ(operand_no, 1);
          Node* operand = node->operand(operand_no);
          Type* array_type = node->As<ArrayIndex>()->array()->GetType();
          auto index_could_be_out_of_range = [&](Node* index) {
            if (index->Is<::xls::Literal>()) {
              return bits_ops::UGreaterThanOrEqual(
                  index->As<::xls::Literal>()->value().bits(),
                  array_type->AsArrayOrDie()->size());
            }
            return index->BitCountOrDie() >=
                   Bits::MinBitCountUnsigned(
                       array_type->AsArrayOrDie()->size());
          };
          if (!operand->Is<Tuple>()) {
            return index_could_be_out_of_range(operand);
          }
          Tuple* indices = operand->As<Tuple>();
          for (int64_t i = 0; i < indices->size(); ++i) {
            if (index_could_be_out_of_range(indices->operand(i))) {
              return true;
            }
            array_type = array_type->AsArrayOrDie()->element_type();
          }
          return false;
        }
        default:
          return false;
      }
    }
    case Op::kArrayUpdate:
      return operand_no == 0 && !operand_is_indexable();
    case Op::kOneHot:
    case Op::kOneHotSel:
      return operand_no == 0 && !operand_is_indexable();
    case Op::kTupleIndex:
      // Tuples are represented as flat vectors and kTupleIndex operation is a
      // slice out of the flat vector. The exception is if the element is an
      // Array. In this case, the element must be unflattened into an unpacked
      // array which requires that it be a named reference.
      return node->GetType()->IsArray() || !operand_is_indexable();
    case Op::kShra:
      // Shra indexes the sign bit of the zero-th operand.
      return operand_no == 0;
    case Op::kSignExt:
      // For operands wider than one bit, sign extend slices out the sign bit of
      // the operand so its operand needs to be a reference.
      // TODO(meheff): It might be better to have a unified place to hold both
      // the Verilog expression and constraints for the Ops, a la
      // op_specification.py
      CHECK_EQ(operand_no, 0);
      return node->operand(operand_no)->BitCountOrDie() > 1 &&
             !operand_is_indexable();
    case Op::kEncode:
      // The expression of the encode operation indexes individual bits of the
      // operand.
      return true;
    case Op::kReverse:
      return true;
    default:
      return false;
  }
}

namespace {

// Emit select as a chain of ternary expressions. For example:
//
//   foo = sel(s, cases={a, b, c, d})
//
// becomes (parentheses added for clarity)
//
//   foo = s == 0 ? a : (s == 1 ? b : (s == 2 ? c : d))
//
absl::StatusOr<Expression*> EmitSel(Select* sel, Expression* selector,
                                    absl::Span<Expression* const> cases,
                                    int64_t caseno, VerilogFile* file) {
  if (caseno + 1 == cases.size()) {
    return cases[caseno];
  }
  XLS_ASSIGN_OR_RETURN(Expression * rhs,
                       EmitSel(sel, selector, cases, caseno + 1, file));
  // If the selector is a single bit value then instead of emitting:
  //
  //   selector == 0 ? case_i : rhs
  //
  // emit:
  //
  //   selector ? rhs : case_i
  //
  if (sel->selector()->BitCountOrDie() == 1) {
    return file->Ternary(selector, rhs, cases[caseno], sel->loc());
  }

  return file->Ternary(
      file->Equals(selector,
                   file->Literal(caseno,
                                 /*bit_count=*/sel->selector()->BitCountOrDie(),
                                 sel->loc()),
                   sel->loc()),
      cases[caseno], rhs, sel->loc());
}

absl::StatusOr<Expression*> EmitOneHot(OneHot* one_hot,
                                       IndexableExpression* input,
                                       VerilogFile* file) {
  const int64_t input_width = one_hot->operand(0)->BitCountOrDie();
  const int64_t output_width = one_hot->BitCountOrDie();
  auto do_index_input = [&](int64_t i) {
    int64_t index =
        one_hot->priority() == LsbOrMsb::kLsb ? i : input_width - i - 1;
    return file->Index(input, index, one_hot->loc());
  };
  // When LSb has priority, does: x[hi_offset:0]
  // When MSb has priority, does: x[bit_count-1:bit_count-1-hi_offset]
  auto do_slice_input = [&](int64_t hi_offset) {
    if (one_hot->priority() == LsbOrMsb::kLsb) {
      return file->Slice(input, hi_offset, 0, one_hot->loc());
    }
    return file->Slice(input, input_width - 1, input_width - 1 - hi_offset,
                       one_hot->loc());
  };
  std::vector<Expression*> one_hot_bits;
  Expression* all_zero;
  for (int64_t i = 0; i < output_width; ++i) {
    if (i == 0) {
      one_hot_bits.push_back(do_index_input(i));
      continue;
    }

    Expression* higher_priority_bits_zero;
    if (i == 1) {
      higher_priority_bits_zero =
          file->LogicalNot(do_index_input(0), one_hot->loc());
    } else {
      higher_priority_bits_zero =
          file->Equals(do_slice_input(i - 1),
                       file->Literal(UBits(0, /*bit_count=*/i), one_hot->loc()),
                       one_hot->loc());
    }

    if (i < output_width - 1) {
      one_hot_bits.push_back(file->LogicalAnd(
          do_index_input(i), higher_priority_bits_zero, one_hot->loc()));
    } else {
      // Default case when all inputs are zero.
      all_zero = higher_priority_bits_zero;
    }
  }
  if (one_hot->priority() == LsbOrMsb::kLsb) {
    one_hot_bits.push_back(all_zero);
    // Reverse the order of the bits because bit index and indexing of concat
    // elements are reversed. That is, the zero-th operand of concat becomes the
    // most-significant part (highest index) of the result.
    std::reverse(one_hot_bits.begin(), one_hot_bits.end());
  } else {
    one_hot_bits.insert(one_hot_bits.begin(), all_zero);
  }
  return file->Concat(one_hot_bits, one_hot->loc());
}

absl::StatusOr<Expression*> EmitOneHotSelect(
    OneHotSelect* sel, IndexableExpression* selector,
    absl::Span<Expression* const> inputs, VerilogFile* file) {
  int64_t sel_width = sel->GetType()->GetFlatBitCount();
  Expression* sum = nullptr;
  for (int64_t i = 0; i < inputs.size(); ++i) {
    Expression* masked_input =
        sel_width == 1
            ? file->BitwiseAnd(inputs[i], file->Index(selector, i, sel->loc()),
                               sel->loc())
            : file->BitwiseAnd(
                  inputs[i],
                  file->Concat(
                      /*replication=*/sel_width,
                      {file->Index(selector, i, sel->loc())}, sel->loc()),
                  sel->loc());
    if (sum == nullptr) {
      sum = masked_input;
    } else {
      sum = file->BitwiseOr(sum, masked_input, sel->loc());
    }
  }
  return sum;
}

// Returns an OR reduction of the given expressions. That is:
//   exprs[0] | exprs[1] | ... | exprs[n]
absl::StatusOr<Expression*> OrReduction(absl::Span<Expression* const> exprs,
                                        VerilogFile* file,
                                        const SourceInfo& loc) {
  XLS_RET_CHECK(!exprs.empty());
  Expression* reduction = exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    reduction = file->BitwiseOr(reduction, exprs[i], loc);
  }
  return reduction;
}

// Emits the given encode op and returns the Verilog expression.
absl::StatusOr<Expression*> EmitEncode(Encode* encode,
                                       IndexableExpression* operand,
                                       VerilogFile* file) {
  std::vector<Expression*> output_bits(encode->BitCountOrDie());
  // Encode produces the OR reduction of the ordinal positions of the set bits
  // of the input. For example, if bit 5 and bit 42 are set in the input, the
  // result is 5 | 42.
  for (int64_t i = 0; i < encode->BitCountOrDie(); ++i) {
    std::vector<Expression*> elements;
    for (int64_t j = 0; j < encode->operand(0)->BitCountOrDie(); ++j) {
      if (j & (1 << i)) {
        elements.push_back(file->Index(operand, j, encode->loc()));
      }
    }
    XLS_RET_CHECK(!elements.empty());
    XLS_ASSIGN_OR_RETURN(output_bits[i],
                         OrReduction(elements, file, encode->loc()));
  }
  std::reverse(output_bits.begin(), output_bits.end());
  return file->Concat(output_bits, encode->loc());
}

// Reverses the order of the bits of the operand.
absl::StatusOr<Expression*> EmitReverse(Node* reverse,
                                        IndexableExpression* operand,
                                        VerilogFile* file) {
  const int64_t width = reverse->BitCountOrDie();
  std::vector<Expression*> output_bits(width);
  for (int64_t i = 0; i < width; ++i) {
    // The concat below naturally reverse bit indices so lhs and rhs can use the
    // same index.
    output_bits[i] = file->Index(operand, i, reverse->loc());
  }
  return file->Concat(output_bits, reverse->loc());
}

// Emits a shift (shll, shrl, shra).
absl::StatusOr<Expression*> EmitShift(Node* shift, Expression* operand,
                                      Expression* shift_amount,
                                      VerilogFile* file) {
  Expression* shifted_operand;
  if (shift->op() == Op::kShra) {
    // To perform an arithmetic shift right the left operand must be cast to a
    // signed value, ie:
    //
    //   $signed(x) >>> y
    //
    // Also, wrap the expression in $unsigned to prevent the signed property
    // from leaking out into the rest of the expression.
    //
    //   $unsigned($signed(x) >>> y) op ...
    //
    // Without the unsigned the '>>>' expression would be treated as a signed
    // value potentially affecting the evaluation of 'op'.  This unsigned cast
    // is also necessary for correctness of the shift evaluation when the shift
    // appears in a ternary expression because of Verilog type rules.
    shifted_operand = file->Make<UnsignedCast>(
        shift->loc(), file->Shra(file->Make<SignedCast>(shift->loc(), operand),
                                 shift_amount, shift->loc()));
  } else if (shift->op() == Op::kShrl) {
    shifted_operand = file->Shrl(operand, shift_amount, shift->loc());
  } else {
    CHECK_EQ(shift->op(), Op::kShll);
    shifted_operand = file->Shll(operand, shift_amount, shift->loc());
  }

  // Verilog semantics are not defined for shifting by more than or equal to
  // the operand width so guard the shift with a test for overshifting (if
  // the width of the shift-amount is wide enough to overshift).
  const int64_t width = shift->BitCountOrDie();
  if (shift->operand(1)->BitCountOrDie() < Bits::MinBitCountUnsigned(width)) {
    // Shift amount is not wide enough to overshift.
    return shifted_operand;
  }

  Expression* width_expr = file->Literal(
      width, /*bit_count=*/shift->operand(1)->BitCountOrDie(), shift->loc());
  Expression* overshift_value;
  if (shift->op() == Op::kShra) {
    // Shrl: overshift value is all sign bits.
    overshift_value = file->Concat(
        /*replication=*/width,
        {file->Index(operand->AsIndexableExpressionOrDie(), width - 1,
                     shift->loc())},
        shift->loc());
  } else {
    // Shll or shrl: overshift value is zero.
    overshift_value = file->Literal(0, shift->BitCountOrDie(), shift->loc());
  }
  return file->Ternary(
      file->GreaterThanEquals(shift_amount, width_expr, shift->loc()),
      overshift_value, shifted_operand, shift->loc());
}

// Emits a decode instruction.
absl::StatusOr<Expression*> EmitDecode(Decode* decode, Expression* operand,
                                       VerilogFile* file) {
  Expression* result =
      file->Shll(file->Literal(1, decode->BitCountOrDie(), decode->loc()),
                 operand, decode->loc());
  if (Bits::MinBitCountUnsigned(decode->BitCountOrDie()) >
      decode->operand(0)->BitCountOrDie()) {
    // Output is wide enough to accommodate every possible input value. No need
    // to guard the input to avoid overshifting.
    return result;
  }
  // If operand value is greater than the width of the output, zero should be
  // emitted.
  return file->Ternary(
      file->GreaterThanEquals(
          operand,
          file->Literal(/*value=*/decode->BitCountOrDie(),
                        /*bit_count=*/decode->operand(0)->BitCountOrDie(),
                        decode->loc()),
          decode->loc()),

      file->Literal(/*value=*/0, /*bit_count=*/decode->BitCountOrDie(),
                    decode->loc()),
      file->Shll(file->Literal(1, decode->BitCountOrDie(), decode->loc()),
                 operand, decode->loc()),
      decode->loc());
}

// Emits an multiply with potentially mixed bit-widths.
absl::StatusOr<Expression*> EmitMultiply(Node* mul, Expression* lhs,
                                         Expression* rhs, VerilogFile* file) {
  // TODO(meheff): Arbitrary widths are supported via functions in module
  // builder. Unify the expression generation in this file with module builder
  // some how.
  XLS_RET_CHECK_EQ(mul->BitCountOrDie(), mul->operand(0)->BitCountOrDie());
  XLS_RET_CHECK_EQ(mul->BitCountOrDie(), mul->operand(1)->BitCountOrDie());
  XLS_RET_CHECK(mul->op() == Op::kUMul || mul->op() == Op::kSMul);
  if (mul->op() == Op::kSMul) {
    return file->Make<UnsignedCast>(
        mul->loc(),
        file->Mul(file->Make<SignedCast>(mul->loc(), lhs),
                  file->Make<SignedCast>(mul->loc(), rhs), mul->loc()));
  }
  return file->Mul(lhs, rhs, mul->loc());
}

// Decomposes `e` into its constituent array elements and adds the elements to
// `decomposition`. If `e` is not an array then `e` alone is added to the
// vector. For example, if `e` is a 3x2 array, then `decomposition` will
// contain:
//
//    {e[0][0], e[0][1], e[1][0], e[1][1], e[2][0], e[2][1]}
void DecomposeExpression(Expression* e, Type* type, VerilogFile* file,
                         const SourceInfo& loc,
                         std::vector<Expression*>* decomposition) {
  if (!type->IsArray()) {
    decomposition->push_back(e);
    return;
  }
  ArrayType* array_type = type->AsArrayOrDie();
  std::vector<Expression*> result;
  for (int64_t i = 0; i < array_type->size(); ++i) {
    DecomposeExpression(file->Index(e->AsIndexableExpressionOrDie(),
                                    file->PlainLiteral(i, loc), loc),
                        array_type->element_type(), file, loc, decomposition);
  }
}

// Emits a Eq/Ne operation of arbitrary type.
absl::StatusOr<Expression*> EmitEqOrNe(Node* node,
                                       absl::Span<Expression* const> inputs,
                                       VerilogFile* file) {
  XLS_RET_CHECK(node->op() == Op::kEq || node->op() == Op::kNe);
  std::vector<Expression*> lhs_parts;
  DecomposeExpression(inputs[0], node->operand(0)->GetType(), file, node->loc(),
                      &lhs_parts);
  std::vector<Expression*> rhs_parts;
  DecomposeExpression(inputs[1], node->operand(1)->GetType(), file, node->loc(),
                      &rhs_parts);
  std::vector<Expression*> comparisons;
  for (int64_t i = 0; i < lhs_parts.size(); ++i) {
    comparisons.push_back(
        node->op() == Op::kEq
            ? file->Equals(lhs_parts[i], rhs_parts[i], node->loc())
            : file->NotEquals(lhs_parts[i], rhs_parts[i], node->loc()));
  }
  Expression* result = comparisons[0];
  for (int64_t i = 1; i < comparisons.size(); ++i) {
    result = node->op() == Op::kEq
                 ? file->LogicalAnd(result, comparisons[i], node->loc())
                 : file->LogicalOr(result, comparisons[i], node->loc());
  }
  return result;
}

}  // namespace

absl::StatusOr<Expression*> NodeToExpression(
    Node* node, absl::Span<Expression* const> inputs, VerilogFile* file,
    const CodegenOptions& options) {
  auto unimplemented = [&]() {
    return absl::UnimplementedError(
        absl::StrFormat("Node cannot be emitted as a Verilog expression: %s",
                        node->ToString()));
  };
  auto do_nary_op =
      [&](const std::function<Expression*(Expression*, Expression*)> &f) {
        Expression* accum = inputs[0];
        for (int64_t i = 1; i < inputs.size(); ++i) {
          accum = f(accum, inputs[i]);
        }
        return accum;
      };
  switch (node->op()) {
    case Op::kAdd:
      return file->Add(inputs[0], inputs[1], node->loc());
    case Op::kAnd:
      return do_nary_op([file, node](Expression* lhs, Expression* rhs) {
        return file->BitwiseAnd(lhs, rhs, node->loc());
      });
    case Op::kAndReduce:
      return file->AndReduce(inputs[0], node->loc());
    case Op::kAssert:
      return unimplemented();
    case Op::kTrace:
      return unimplemented();
    case Op::kCover:
      return unimplemented();
    case Op::kNand:
      return file->BitwiseNot(
          do_nary_op([file, node](Expression* lhs, Expression* rhs) {
            return file->BitwiseAnd(lhs, rhs, node->loc());
          }),
          node->loc());
    case Op::kNor:
      return file->BitwiseNot(
          do_nary_op([file, node](Expression* lhs, Expression* rhs) {
            return file->BitwiseOr(lhs, rhs, node->loc());
          }),
          node->loc());
    case Op::kAfterAll:
      return absl::UnimplementedError("AfterAll not yet implemented");
    case Op::kMinDelay:
      return absl::UnimplementedError("MinDelay not yet implemented");
    case Op::kReceive:
      return absl::UnimplementedError("Receive not yet implemented");
    case Op::kSend:
      return absl::UnimplementedError("Send not yet implemented");
    case Op::kArray: {
      std::vector<Expression*> elements(inputs.begin(), inputs.end());
      return file->ArrayAssignmentPattern(elements, node->loc());
    }
    case Op::kArrayIndex:
      return ArrayIndexExpression(inputs[0]->AsIndexableExpressionOrDie(),
                                  inputs.subspan(1), node->As<ArrayIndex>(),
                                  options);
    case Op::kArrayUpdate: {
      // This is only reachable as a corner case where the "array" operand of
      // the array update is not an array type. This is only possible with empty
      // indices in which case the result of the array update is the "array"
      // operand.
      ArrayUpdate* update = node->As<ArrayUpdate>();
      XLS_RET_CHECK(update->GetType()->IsBits());
      XLS_RET_CHECK(update->indices().empty());
      // The index is empty so the result of the update operation is simply the
      // update value.
      return inputs[1];
    }
    case Op::kArrayConcat: {
      return absl::UnimplementedError("ArrayConcat not yet implemented");
    }
    case Op::kArraySlice: {
      return absl::FailedPreconditionError(
          "ArraySlice is handled in module_builder.cc");
    }
    case Op::kBitSlice: {
      BitSlice* slice = node->As<BitSlice>();
      if (slice->width() == 1) {
        return file->Index(inputs[0]->AsIndexableExpressionOrDie(),
                           slice->start(), node->loc());
      }
      return file->Slice(inputs[0]->AsIndexableExpressionOrDie(),
                         slice->start() + slice->width() - 1, slice->start(),
                         node->loc());
    }
    case Op::kBitSliceUpdate:
      return unimplemented();
    case Op::kDynamicBitSlice:
      return unimplemented();
    case Op::kConcat:
      return file->Concat(inputs, node->loc());
    case Op::kUDiv:
      return unimplemented();
    case Op::kUMod:
      return file->Mod(inputs[0], inputs[1], node->loc());
    case Op::kEq:
    case Op::kNe:
      return EmitEqOrNe(node, inputs, file);
    case Op::kUGe:
      return file->GreaterThanEquals(inputs[0], inputs[1], node->loc());
    case Op::kUGt:
      return file->GreaterThan(inputs[0], inputs[1], node->loc());
    case Op::kDecode:
      return EmitDecode(node->As<Decode>(), inputs[0], file);
    case Op::kEncode:
      return EmitEncode(node->As<Encode>(),
                        inputs[0]->AsIndexableExpressionOrDie(), file);
    case Op::kIdentity:
      return inputs[0];
    case Op::kInvoke:
      return unimplemented();
    case Op::kCountedFor:
      return unimplemented();
    case Op::kDynamicCountedFor:
      return unimplemented();
    case Op::kLiteral:
      if (!node->GetType()->IsBits()) {
        return unimplemented();
      }
      return file->Literal(node->As<xls::Literal>()->value().bits(),
                           node->loc());
    case Op::kULe:
      return file->LessThanEquals(inputs[0], inputs[1], node->loc());
    case Op::kULt:
      return file->LessThan(inputs[0], inputs[1], node->loc());
    case Op::kMap:
      return unimplemented();
    case Op::kUMul:
    case Op::kSMul:
      return EmitMultiply(node, inputs[0], inputs[1], file);
    case Op::kSMulp:
    case Op::kUMulp:
      return unimplemented();
    case Op::kNeg:
      return file->Negate(inputs[0], node->loc());
    case Op::kNot:
      return file->BitwiseNot(inputs[0], node->loc());
    case Op::kOneHot:
      return EmitOneHot(node->As<OneHot>(),
                        inputs[0]->AsIndexableExpressionOrDie(), file);
    case Op::kOneHotSel:
      return EmitOneHotSelect(node->As<OneHotSelect>(),
                              inputs[0]->AsIndexableExpressionOrDie(),
                              inputs.subspan(1), file);
    case Op::kPrioritySel:
      return unimplemented();
    case Op::kOr:
      return do_nary_op([file, node](Expression* lhs, Expression* rhs) {
        return file->BitwiseOr(lhs, rhs, node->loc());
      });
    case Op::kOrReduce:
      return file->OrReduce(inputs[0], node->loc());
    case Op::kParam:
      return unimplemented();
    case Op::kNext:
      return unimplemented();
    case Op::kRegisterRead:
      return unimplemented();
    case Op::kRegisterWrite:
      return unimplemented();
    case Op::kReverse:
      return EmitReverse(node, inputs[0]->AsIndexableExpressionOrDie(), file);
    case Op::kSel: {
      Select* sel = node->As<Select>();
      auto cases = inputs;
      cases.remove_prefix(1);
      return EmitSel(sel, inputs[0], cases, /*caseno=*/0, file);
    }
    case Op::kShll:
    case Op::kShra:
    case Op::kShrl:
      return EmitShift(node, inputs[0], inputs[1], file);
    case Op::kSignExt: {
      if (node->operand(0)->BitCountOrDie() == 1) {
        // A sign extension of a single-bit value is just replication.
        return file->Concat(
            /*replication=*/node->BitCountOrDie(), {inputs[0]}, node->loc());
      }
      int64_t bits_added =
          node->BitCountOrDie() - node->operand(0)->BitCountOrDie();
      return file->Concat(
          {file->Concat(
               /*replication=*/bits_added,
               {file->Index(inputs[0]->AsIndexableExpressionOrDie(),
                            node->operand(0)->BitCountOrDie() - 1,
                            node->loc())},
               node->loc()),
           inputs[0]},
          node->loc());
    }
    case Op::kSDiv:
      return unimplemented();
    case Op::kSMod:
      // Wrap the expression in $unsigned to prevent the signed property from
      // leaking out into the rest of the expression.
      return file->Make<UnsignedCast>(
          node->loc(), file->Mod(file->Make<SignedCast>(node->loc(), inputs[0]),
                                 file->Make<SignedCast>(node->loc(), inputs[1]),
                                 node->loc()));
    case Op::kSGt:
      return file->GreaterThan(file->Make<SignedCast>(node->loc(), inputs[0]),
                               file->Make<SignedCast>(node->loc(), inputs[1]),
                               node->loc());
    case Op::kSGe:
      return file->GreaterThanEquals(
          file->Make<SignedCast>(node->loc(), inputs[0]),
          file->Make<SignedCast>(node->loc(), inputs[1]), node->loc());
    case Op::kSLe:
      return file->LessThanEquals(
          file->Make<SignedCast>(node->loc(), inputs[0]),
          file->Make<SignedCast>(node->loc(), inputs[1]), node->loc());
    case Op::kSLt:
      return file->LessThan(file->Make<SignedCast>(node->loc(), inputs[0]),
                            file->Make<SignedCast>(node->loc(), inputs[1]),
                            node->loc());
    case Op::kSub:
      return file->Sub(inputs[0], inputs[1], node->loc());
    case Op::kTupleIndex: {
      if (node->GetType()->IsArray()) {
        return UnflattenArrayShapedTupleElement(
            inputs[0]->AsIndexableExpressionOrDie(),
            node->operand(0)->GetType()->AsTupleOrDie(),
            node->As<TupleIndex>()->index(), file, node->loc());
      }
      const int64_t start =
          GetFlatBitIndexOfElement(node->operand(0)->GetType()->AsTupleOrDie(),
                                   node->As<TupleIndex>()->index());
      const int64_t width = node->GetType()->GetFlatBitCount();
      if (start == 0 && width == 1 &&
          node->operand(0)->GetType()->GetFlatBitCount() == 1) {
        // The operand is a single-bit type. Single-bit types are represented as
        // scalars in VAST and scalars cannot be indexed so just return the
        // operand.
        return inputs[0];
      }
      return file->Slice(inputs[0]->AsIndexableExpressionOrDie(),
                         start + width - 1, start, node->loc());
    }
    case Op::kTuple:
      return FlattenTuple(inputs, node->GetType()->AsTupleOrDie(), file,
                          node->loc());
    case Op::kXor:
      return do_nary_op([file, node](Expression* lhs, Expression* rhs) {
        return file->BitwiseXor(lhs, rhs, node->loc());
      });
    case Op::kXorReduce:
      return file->XorReduce(inputs[0], node->loc());
    case Op::kZeroExt: {
      int64_t bits_added =
          node->BitCountOrDie() - node->operand(0)->BitCountOrDie();

      return file->Concat(
          {file->Literal(0, bits_added, node->loc()), inputs[0]}, node->loc());
    }
    case Op::kInputPort:
    case Op::kOutputPort:
    case Op::kInstantiationInput:
    case Op::kInstantiationOutput:
    case Op::kGate:
      return unimplemented();
  }
  LOG(FATAL) << "Invalid op: " << static_cast<int64_t>(node->op());
}

bool ShouldInlineExpressionIntoMultipleUses(Node* node) {
  return node->Is<BitSlice>() || node->op() == Op::kNot ||
         node->op() == Op::kNeg;
}

absl::StatusOr<IndexableExpression*> ArrayIndexExpression(
    IndexableExpression* array, absl::Span<Expression* const> indices,
    ArrayIndex* array_index, const CodegenOptions& options) {
  VerilogFile* file = array->file();
  IndexableExpression* value = array;
  Type* type = array_index->array()->GetType();
  XLS_RET_CHECK_EQ(indices.size(), array_index->indices().size());
  for (int64_t i = 0; i < indices.size(); ++i) {
    Expression* index = indices[i];
    BitsType* index_type = array_index->indices()[i]->GetType()->AsBitsOrDie();
    ArrayType* array_type = type->AsArrayOrDie();
    Expression* clamped_index;
    // Out-of-bounds accesses return the final element of the array. Clamp the
    // index to the maximum index value. In some cases, clamping is not
    // necessary (index is a literal or not wide enough to express an OOB
    // index). This testing about whether a bounds check would be better handled
    // via another mechanism (e.g., an annotation on the array operation
    // indicating that the access is inbounds).
    // TODO(meheff) 2021-03-25 Simplify this when we have a better way of
    // handling OOB accesses.
    if (!options.array_index_bounds_checking() ||
        Bits::MinBitCountUnsigned(array_type->size()) >
            index_type->bit_count()) {
      // Index cannot be out-of-bounds because it is not wide enough to express
      // an out-of-bounds value.
      clamped_index = index;
    } else if (index->IsLiteral() &&
               bits_ops::ULessThan(index->AsLiteralOrDie()->bits(),
                                   array_type->size())) {
      // Index is an in-bounds literal.
      const int64_t short_index_width = std::max(
          int64_t{1}, Bits::MinBitCountUnsigned(array_type->size() - 1));
      if (index_type->bit_count() <= short_index_width) {
        clamped_index = index;
      } else {
        XLS_RET_CHECK(index->AsLiteralOrDie()->bits().FitsInUint64());
        clamped_index =
            file->Literal(*index->AsLiteralOrDie()->bits().ToUint64(),
                          short_index_width, array_index->loc());
      }
    } else {
      Expression* max_index =
          file->Literal(UBits(array_type->size() - 1, index_type->bit_count()),
                        array_index->loc());

      Expression* short_index = index;
      Expression* short_max_index = max_index;
      const int64_t short_index_width = std::max(
          int64_t{1}, Bits::MinBitCountUnsigned(array_type->size() - 1));
      if (index_type->bit_count() > short_index_width) {
        if (!index->IsIndexableExpression()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Index %d of array_index %s is not indexable, "
              "and is too wide for array %s; was %s",
              i, array_index->GetName(), array_index->array()->GetName(),
              array_index->indices()[i]->GetType()->ToString()));
        }
        short_index = file->Slice(index->AsIndexableExpressionOrDie(),
                                  short_index_width - 1, 0, array_index->loc());
        short_max_index =
            file->Literal(UBits(array_type->size() - 1, short_index_width),
                          array_index->loc());
      }

      clamped_index =
          file->Ternary(file->GreaterThan(index, max_index, array_index->loc()),
                        short_max_index, short_index, array_index->loc());
    }
    value = file->Index(value, clamped_index, array_index->loc());
    type = array_type->element_type();
  }
  return value;
}

}  // namespace verilog
}  // namespace xls
