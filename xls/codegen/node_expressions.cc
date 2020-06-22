// Copyright 2020 Google LLC
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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/flattening.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"

namespace xls {
namespace verilog {

bool OperandMustBeNamedReference(Node* node, int64 operand_no) {
  // Returns true if the emitted expression for the specified operand is
  // necessarily indexable. Generally, if the expression emitted for a node is
  // an indexing operation and the operand is emitted as an indexable expression
  // then there is no need to make the operand a declared expression because
  // indexing/slicing can be chained.
  //
  // For example a kArrayIndex of a kArrayIndex can be emitted as a chained VAST
  // Index expression like so:
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
      XLS_CHECK_EQ(operand_no, 0);
      return !operand_is_indexable();
    case Op::kArrayIndex:
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
      XLS_CHECK_EQ(operand_no, 0);
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

// Returns given Value as a VAST Literal created in the given file. must_flatten
// indicates if the value must be emitted as a flat bit vector. This argument is
// used when invoked recursively for when a tuple includes nested array
// elements. In this case, the nested array elements must be flattened rather
// than emitted as an unpacked array.
Expression* ValueToVastLiteral(const Value& value, VerilogFile* file,
                               bool must_flatten = false) {
  if (value.IsBits()) {
    return file->Literal(value.bits());
  } else if (value.IsTuple()) {
    std::vector<Expression*> elements;
    for (const Value& element : value.elements()) {
      elements.push_back(
          ValueToVastLiteral(element, file, /*must_flatten=*/true));
    }
    return file->Concat(elements);
  } else {
    XLS_CHECK(value.IsArray());
    std::vector<Expression*> elements;
    for (const Value& element : value.elements()) {
      elements.push_back(ValueToVastLiteral(element, file, must_flatten));
    }
    if (must_flatten) {
      return file->Concat(elements);
    }
    return file->ArrayAssignmentPattern(elements);
  }
}

xabsl::StatusOr<Expression*> EmitSel(Select* sel, Expression* selector,
                                     absl::Span<Expression* const> cases,
                                     int64 caseno, VerilogFile* file) {
  if (caseno + 1 == cases.size()) {
    return cases[caseno];
  }
  XLS_ASSIGN_OR_RETURN(Expression * rhs,
                       EmitSel(sel, selector, cases, caseno + 1, file));
  return file->Ternary(
      file->Equals(
          selector,
          file->Literal(caseno,
                        /*bit_count=*/sel->selector()->BitCountOrDie())),
      cases[caseno], rhs);
}

xabsl::StatusOr<Expression*> EmitOneHot(OneHot* one_hot,
                                        IndexableExpression* input,
                                        VerilogFile* file) {
  const int64 input_width = one_hot->operand(0)->BitCountOrDie();
  const int64 output_width = one_hot->BitCountOrDie();
  auto do_index_input = [&](int64 i) {
    int64 index =
        one_hot->priority() == LsbOrMsb::kLsb ? i : input_width - i - 1;
    return file->Index(input, index);
  };
  // When LSb has priority, does: x[hi_offset:0]
  // When MSb has priority, does: x[bit_count-1:bit_count-1-hi_offset]
  auto do_slice_input = [&](int64 hi_offset) {
    if (one_hot->priority() == LsbOrMsb::kLsb) {
      return file->Slice(input, hi_offset, 0);
    }
    return file->Slice(input, input_width - 1, input_width - 1 - hi_offset);
  };
  std::vector<Expression*> one_hot_bits;
  Expression* all_zero;
  for (int64 i = 0; i < output_width; ++i) {
    if (i == 0) {
      one_hot_bits.push_back(do_index_input(i));
      continue;
    }

    Expression* higher_priority_bits_zero;
    if (i == 1) {
      higher_priority_bits_zero = file->LogicalNot(do_index_input(0));
    } else {
      higher_priority_bits_zero = file->Equals(
          do_slice_input(i - 1), file->Literal(UBits(0, /*bit_count=*/i)));
    }

    if (i < output_width - 1) {
      one_hot_bits.push_back(
          file->LogicalAnd(do_index_input(i), higher_priority_bits_zero));
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
  return file->Concat(one_hot_bits);
}

xabsl::StatusOr<Expression*> EmitOneHotSelect(
    OneHotSelect* sel, IndexableExpression* selector,
    absl::Span<Expression* const> inputs, VerilogFile* file) {
  if (!sel->GetType()->IsBits()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Only bits-typed one-hot select supported:: %s", sel->ToString()));
  }
  int64 sel_width = sel->BitCountOrDie();
  Expression* sum = nullptr;
  for (int64 i = 0; i < inputs.size(); ++i) {
    Expression* masked_input =
        sel_width == 1
            ? file->BitwiseAnd(inputs[i], file->Index(selector, i))
            : file->BitwiseAnd(inputs[i], file->Concat(
                                              /*replication=*/sel_width,
                                              {file->Index(selector, i)}));
    if (sum == nullptr) {
      sum = masked_input;
    } else {
      sum = file->BitwiseOr(sum, masked_input);
    }
  }
  return sum;
}

// Returns an OR reduction of the given expressions. That is:
//   exprs[0] | exprs[1] | ... | exprs[n]
xabsl::StatusOr<Expression*> OrReduction(absl::Span<Expression* const> exprs,
                                         VerilogFile* file) {
  XLS_RET_CHECK(!exprs.empty());
  Expression* reduction = exprs[0];
  for (int i = 1; i < exprs.size(); ++i) {
    reduction = file->BitwiseOr(reduction, exprs[i]);
  }
  return reduction;
}

// Emits the given encode op and returns the Verilog expression.
xabsl::StatusOr<Expression*> EmitEncode(Encode* encode,
                                        IndexableExpression* operand,
                                        VerilogFile* file) {
  std::vector<Expression*> output_bits(encode->BitCountOrDie());
  // Encode produces the OR reduction of the ordinal positions of the set bits
  // of the input. For example, if bit 5 and bit 42 are set in the input, the
  // result is 5 | 42.
  for (int64 i = 0; i < encode->BitCountOrDie(); ++i) {
    std::vector<Expression*> elements;
    for (int64 j = 0; j < encode->operand(0)->BitCountOrDie(); ++j) {
      if (j & (1 << i)) {
        elements.push_back(file->Index(operand, j));
      }
    }
    XLS_RET_CHECK(!elements.empty());
    XLS_ASSIGN_OR_RETURN(output_bits[i], OrReduction(elements, file));
  }
  std::reverse(output_bits.begin(), output_bits.end());
  return file->Concat(output_bits);
}

// Reverses the order of the bits of the operand.
xabsl::StatusOr<Expression*> EmitReverse(Node* reverse,
                                         IndexableExpression* operand,
                                         VerilogFile* file) {
  const int64 width = reverse->BitCountOrDie();
  std::vector<Expression*> output_bits(width);
  for (int64 i = 0; i < width; ++i) {
    // The concat below naturally reverse bit indices so lhs and rhs can use the
    // same index.
    output_bits[i] = file->Index(operand, i);
  }
  return file->Concat(output_bits);
}

// Emits a shift (shll, shrl, shra).
xabsl::StatusOr<Expression*> EmitShift(Node* shift, Expression* operand,
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
        file->Shra(file->Make<SignedCast>(operand), shift_amount));
  } else if (shift->op() == Op::kShrl) {
    shifted_operand = file->Shrl(operand, shift_amount);
  } else {
    XLS_CHECK_EQ(shift->op(), Op::kShll);
    shifted_operand = file->Shll(operand, shift_amount);
  }

  // Verilog semantics are not defined for shifting by more than or equal to
  // the operand width so guard the shift with a test for overshifting (if
  // the width of the shift-amount is wide enough to overshift).
  const int64 width = shift->BitCountOrDie();
  if (shift->operand(1)->BitCountOrDie() < Bits::MinBitCountUnsigned(width)) {
    // Shift amount is not wide enough to overshift.
    return shifted_operand;
  }

  Expression* width_expr =
      file->Literal(width, /*bit_count=*/shift->operand(1)->BitCountOrDie());
  Expression* overshift_value;
  if (shift->op() == Op::kShra) {
    // Shrl: overshift value is all sign bits.
    overshift_value = file->Concat(
        /*replication=*/width,
        {file->Index(operand->AsIndexableExpressionOrDie(), width - 1)});
  } else {
    // Shll or shrl: overshift value is zero.
    overshift_value = file->Literal(0, shift->BitCountOrDie());
  }
  return file->Ternary(file->GreaterThanEquals(shift_amount, width_expr),
                       overshift_value, shifted_operand);
}

// Emits a decode instruction.
xabsl::StatusOr<Expression*> EmitDecode(Decode* decode, Expression* operand,
                                        VerilogFile* file) {
  Expression* result =
      file->Shll(file->Literal(1, decode->BitCountOrDie()), operand);
  if (Bits::MinBitCountUnsigned(decode->BitCountOrDie()) >
      decode->operand(0)->BitCountOrDie()) {
    // Output is wide enough to accommodate every possible input value. No need
    // to guard the input to avoid overshifting.
    return result;
  }
  // If operand value is greater than the width of the output, zero shoud be
  // emitted.
  return file->Ternary(
      file->GreaterThanEquals(
          operand,
          file->Literal(/*value=*/decode->BitCountOrDie(),
                        /*bit_count=*/decode->operand(0)->BitCountOrDie())),

      file->Literal(/*value=*/0, /*bit_count=*/decode->BitCountOrDie()),
      file->Shll(file->Literal(1, decode->BitCountOrDie()), operand));
}

// Emits an multiply with potentially mixed bit-widths.
xabsl::StatusOr<Expression*> EmitMultiply(Node* mul, Expression* lhs,
                                          Expression* rhs, VerilogFile* file) {
  // TODO(meheff): Arbitrary widths are supported via functions in module
  // builder. Unify the expression generation in this file with module builder
  // some how.
  XLS_RET_CHECK_EQ(mul->BitCountOrDie(), mul->operand(0)->BitCountOrDie());
  XLS_RET_CHECK_EQ(mul->BitCountOrDie(), mul->operand(1)->BitCountOrDie());
  XLS_RET_CHECK(mul->op() == Op::kUMul || mul->op() == Op::kSMul);
  if (mul->op() == Op::kSMul) {
    return file->Make<UnsignedCast>(
        file->Mul(file->Make<SignedCast>(lhs), file->Make<SignedCast>(rhs)));
  } else {
    return file->Mul(lhs, rhs);
  }
}

}  // namespace

xabsl::StatusOr<Expression*> NodeToExpression(
    Node* node, absl::Span<Expression* const> inputs, VerilogFile* file) {
  auto unimplemented = [&]() {
    return absl::UnimplementedError(
        absl::StrFormat("Node cannot be emitted as a Verilog expression: %s",
                        node->ToString()));
  };
  auto do_nary_op =
      [&](std::function<Expression*(Expression*, Expression*)> f) {
        Expression* accum = inputs[0];
        for (int64 i = 1; i < inputs.size(); ++i) {
          accum = f(accum, inputs[i]);
        }
        return accum;
      };
  switch (node->op()) {
    case Op::kAdd:
      return file->Add(inputs[0], inputs[1]);
    case Op::kAnd:
      return do_nary_op([file](Expression* lhs, Expression* rhs) {
        return file->BitwiseAnd(lhs, rhs);
      });
    case Op::kAndReduce:
      return file->AndReduce(inputs[0]);
    case Op::kNand:
      return file->BitwiseNot(
          do_nary_op([file](Expression* lhs, Expression* rhs) {
            return file->BitwiseAnd(lhs, rhs);
          }));
    case Op::kNor:
      return file->BitwiseNot(
          do_nary_op([file](Expression* lhs, Expression* rhs) {
            return file->BitwiseOr(lhs, rhs);
          }));
    case Op::kArray: {
      std::vector<Expression*> elements(inputs.begin(), inputs.end());
      return file->ArrayAssignmentPattern(elements);
    }
    case Op::kArrayIndex: {
      // Hack to avoid indexing scalar registers, this can be removed when we
      // support querying types of definitions in the VAST AST.
      if (node->operand(1)->Is<xls::Literal>() &&
          node->operand(1)->As<xls::Literal>()->IsZero()) {
        return file->Index(inputs[0]->AsIndexableExpressionOrDie(),
                           file->PlainLiteral(0));
      }
      return file->Index(inputs[0]->AsIndexableExpressionOrDie(), inputs[1]);
    }
    case Op::kArrayUpdate: {
      return absl::UnimplementedError("ArrayUpdate not yet implemented");
    }
    case Op::kBitSlice: {
      BitSlice* slice = node->As<BitSlice>();
      if (slice->width() == 1) {
        return file->Index(inputs[0]->AsIndexableExpressionOrDie(),
                           slice->start());
      } else {
        return file->Slice(inputs[0]->AsIndexableExpressionOrDie(),
                           slice->start() + slice->width() - 1, slice->start());
      }
    }
    case Op::kDynamicBitSlice: {
      return absl::UnimplementedError("DynamicBitSlice not yet implemented");
    }
    case Op::kConcat:
      return file->Concat(inputs);
    case Op::kUDiv:
      return file->Div(inputs[0], inputs[1]);
    case Op::kEq:
      return file->Equals(inputs[0], inputs[1]);
    case Op::kUGe:
      return file->GreaterThanEquals(inputs[0], inputs[1]);
    case Op::kUGt:
      return file->GreaterThan(inputs[0], inputs[1]);
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
    case Op::kLiteral:
      return ValueToVastLiteral(node->As<xls::Literal>()->value(), file);
    case Op::kULe:
      return file->LessThanEquals(inputs[0], inputs[1]);
    case Op::kULt:
      return file->LessThan(inputs[0], inputs[1]);
    case Op::kMap:
      return unimplemented();
    case Op::kUMul:
    case Op::kSMul:
      return EmitMultiply(node, inputs[0], inputs[1], file);
    case Op::kNe:
      return file->NotEquals(inputs[0], inputs[1]);
    case Op::kNeg:
      return file->Negate(inputs[0]);
    case Op::kNot:
      return file->BitwiseNot(inputs[0]);
    case Op::kOneHot:
      return EmitOneHot(node->As<OneHot>(),
                        inputs[0]->AsIndexableExpressionOrDie(), file);
    case Op::kOneHotSel:
      return EmitOneHotSelect(node->As<OneHotSelect>(),
                              inputs[0]->AsIndexableExpressionOrDie(),
                              inputs.subspan(1), file);
    case Op::kOr:
      return do_nary_op([file](Expression* lhs, Expression* rhs) {
        return file->BitwiseOr(lhs, rhs);
      });
    case Op::kOrReduce:
      return file->OrReduce(inputs[0]);
    case Op::kParam:
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
            /*replication=*/node->BitCountOrDie(), {inputs[0]});
      } else {
        int64 bits_added =
            node->BitCountOrDie() - node->operand(0)->BitCountOrDie();
        return file->Concat(
            {file->Concat(
                 /*replication=*/bits_added,
                 {file->Index(inputs[0]->AsIndexableExpressionOrDie(),
                              node->operand(0)->BitCountOrDie() - 1)}),
             inputs[0]});
      }
    }
    case Op::kSDiv:
      // Wrap the expression in $unsigned to prevent the signed property from
      // leaking out into the rest of the expression.
      return file->Make<UnsignedCast>(
          file->Div(file->Make<SignedCast>(inputs[0]),
                    file->Make<SignedCast>(inputs[1])));
    case Op::kSGt:
      return file->GreaterThan(file->Make<SignedCast>(inputs[0]),
                               file->Make<SignedCast>(inputs[1]));
    case Op::kSGe:
      return file->GreaterThanEquals(file->Make<SignedCast>(inputs[0]),
                                     file->Make<SignedCast>(inputs[1]));
    case Op::kSLe:
      return file->LessThanEquals(file->Make<SignedCast>(inputs[0]),
                                  file->Make<SignedCast>(inputs[1]));
    case Op::kSLt:
      return file->LessThan(file->Make<SignedCast>(inputs[0]),
                            file->Make<SignedCast>(inputs[1]));
    case Op::kSub:
      return file->Sub(inputs[0], inputs[1]);
    case Op::kTupleIndex: {
      if (node->GetType()->IsArray()) {
        return UnflattenArrayShapedTupleElement(
            inputs[0]->AsIndexableExpressionOrDie(),
            node->operand(0)->GetType()->AsTupleOrDie(),
            node->As<TupleIndex>()->index(), file);
      }
      const int64 start =
          GetFlatBitIndexOfElement(node->operand(0)->GetType()->AsTupleOrDie(),
                                   node->As<TupleIndex>()->index());
      const int64 width = node->GetType()->GetFlatBitCount();
      return file->Slice(inputs[0]->AsIndexableExpressionOrDie(),
                         start + width - 1, start);
    }
    case Op::kTuple: {
      std::vector<Expression*> flattened_inputs;
      // Tuples are represented as a flat vector of bits. Flatten and
      // concatenate all operands.
      for (int64 i = 0; i < node->operand_count(); ++i) {
        Expression* input = inputs[i];
        if (node->operand(i)->GetType()->IsArray()) {
          flattened_inputs.push_back(
              FlattenArray(input->AsIndexableExpressionOrDie(),
                           node->operand(i)->GetType()->AsArrayOrDie(), file));
        } else {
          flattened_inputs.push_back(input);
        }
      }
      return file->Concat(flattened_inputs);
    }
    case Op::kXor:
      return do_nary_op([file](Expression* lhs, Expression* rhs) {
        return file->BitwiseXor(lhs, rhs);
      });
    case Op::kXorReduce:
      return file->XorReduce(inputs[0]);
    case Op::kZeroExt: {
      int64 bits_added =
          node->BitCountOrDie() - node->operand(0)->BitCountOrDie();

      return file->Concat({file->Literal(0, bits_added), inputs[0]});
    }
  }
}

bool ShouldInlineExpressionIntoMultipleUses(Node* node) {
  return node->Is<BitSlice>() || node->op() == Op::kNot ||
         node->op() == Op::kNeg;
}

}  // namespace verilog
}  // namespace xls
