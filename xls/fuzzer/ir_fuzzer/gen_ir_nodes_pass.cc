// Copyright 2025 The XLS Authors
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

#include "xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "google/protobuf/repeated_field.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Loops through all of the FuzzOpProtos in the FuzzProgramProto. Each
// FuzzOpProto is a randomly generated object that is used to
// instantiate/generate an IR node/BValue. Add these BValues to the stack. Some
// FuzzOpProtos may require retrieving previous BValues from the stack.
void GenIrNodesPass::GenIrNodes() {
  for (FuzzOpProto& fuzz_op : *fuzz_program_->mutable_fuzz_ops()) {
    VisitFuzzOp(&fuzz_op);
  }
}

void GenIrNodesPass::HandleParam(FuzzParamProto* param) {
  param->set_bit_width(BoundedWidth(param->bit_width()));
  // Params are named as "p" followed by the stack index of the param.
  stack_.push_back(fb_->Param("p" + std::to_string(stack_.size()),
                              p_->GetBitsType(param->bit_width())));
}

void GenIrNodesPass::HandleShra(FuzzShraProto* shra) {
  BValue operand = GetOperand(shra->operand_idx());
  BValue amount = GetOperand(shra->amount_idx());
  stack_.push_back(fb_->Shra(operand, amount));
}

void GenIrNodesPass::HandleShrl(FuzzShrlProto* shrl) {
  BValue operand = GetOperand(shrl->operand_idx());
  BValue amount = GetOperand(shrl->amount_idx());
  stack_.push_back(fb_->Shrl(operand, amount));
}

void GenIrNodesPass::HandleShll(FuzzShllProto* shll) {
  BValue operand = GetOperand(shll->operand_idx());
  BValue amount = GetOperand(shll->amount_idx());
  stack_.push_back(fb_->Shll(operand, amount));
}

void GenIrNodesPass::HandleOr(FuzzOrProto* or_op) {
  or_op->set_bit_width(BoundedWidth(or_op->bit_width()));
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetWidthFittedOperands(or_op->mutable_operand_idxs(), or_op->bit_width(),
                             /*min_operand_count=*/1);
  stack_.push_back(fb_->Or(operands));
}

void GenIrNodesPass::HandleNor(FuzzNorProto* nor) {
  nor->set_bit_width(BoundedWidth(nor->bit_width()));
  std::vector<BValue> operands = GetWidthFittedOperands(
      nor->mutable_operand_idxs(), nor->bit_width(), /*min_operand_count=*/1);
  stack_.push_back(fb_->Nor(operands));
}

void GenIrNodesPass::HandleXor(FuzzXorProto* xor_op) {
  xor_op->set_bit_width(BoundedWidth(xor_op->bit_width()));
  std::vector<BValue> operands =
      GetWidthFittedOperands(xor_op->mutable_operand_idxs(),
                             xor_op->bit_width(), /*min_operand_count=*/1);
  stack_.push_back(fb_->Xor(operands));
}

void GenIrNodesPass::HandleAnd(FuzzAndProto* and_op) {
  and_op->set_bit_width(BoundedWidth(and_op->bit_width()));
  std::vector<BValue> operands =
      GetWidthFittedOperands(and_op->mutable_operand_idxs(),
                             and_op->bit_width(), /*min_operand_count=*/1);
  stack_.push_back(fb_->And(operands));
}

void GenIrNodesPass::HandleNand(FuzzNandProto* nand) {
  nand->set_bit_width(BoundedWidth(nand->bit_width()));
  std::vector<BValue> operands = GetWidthFittedOperands(
      nand->mutable_operand_idxs(), nand->bit_width(), /*min_operand_count=*/1);
  stack_.push_back(fb_->Nand(operands));
}

void GenIrNodesPass::HandleAndReduce(FuzzAndReduceProto* and_reduce) {
  BValue operand = GetOperand(and_reduce->operand_idx());
  stack_.push_back(fb_->AndReduce(operand));
}

void GenIrNodesPass::HandleOrReduce(FuzzOrReduceProto* or_reduce) {
  BValue operand = GetOperand(or_reduce->operand_idx());
  stack_.push_back(fb_->OrReduce(operand));
}

void GenIrNodesPass::HandleXorReduce(FuzzXorReduceProto* xor_reduce) {
  BValue operand = GetOperand(xor_reduce->operand_idx());
  stack_.push_back(fb_->XorReduce(operand));
}

void GenIrNodesPass::HandleUMul(FuzzUMulProto* umul) {
  BValue lhs = GetOperand(umul->lhs_idx());
  BValue rhs = GetOperand(umul->rhs_idx());
  umul->set_bit_width(BoundedWidth(
      umul->bit_width(), 1,
      std::max<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie())));
  stack_.push_back(fb_->UMul(lhs, rhs, umul->bit_width()));
}

void GenIrNodesPass::HandleSMul(FuzzSMulProto* smul) {
  BValue lhs = GetOperand(smul->lhs_idx());
  BValue rhs = GetOperand(smul->rhs_idx());
  smul->set_bit_width(BoundedWidth(
      smul->bit_width(), 1,
      std::max<int64_t>(1000, lhs.BitCountOrDie() + rhs.BitCountOrDie())));
  stack_.push_back(fb_->SMul(lhs, rhs, smul->bit_width()));
}

void GenIrNodesPass::HandleUDiv(FuzzUDivProto* udiv) {
  udiv->set_bit_width(BoundedWidth(udiv->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(udiv->mutable_lhs_idx(), udiv->bit_width());
  BValue rhs =
      GetWidthFittedOperand(udiv->mutable_rhs_idx(), udiv->bit_width());
  stack_.push_back(fb_->UDiv(lhs, rhs));
}

void GenIrNodesPass::HandleSDiv(FuzzSDivProto* sdiv) {
  sdiv->set_bit_width(BoundedWidth(sdiv->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(sdiv->mutable_lhs_idx(), sdiv->bit_width());
  BValue rhs =
      GetWidthFittedOperand(sdiv->mutable_rhs_idx(), sdiv->bit_width());
  stack_.push_back(fb_->SDiv(lhs, rhs));
}

void GenIrNodesPass::HandleUMod(FuzzUModProto* umod) {
  umod->set_bit_width(BoundedWidth(umod->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(umod->mutable_lhs_idx(), umod->bit_width());
  BValue rhs =
      GetWidthFittedOperand(umod->mutable_rhs_idx(), umod->bit_width());
  stack_.push_back(fb_->UMod(lhs, rhs));
}

void GenIrNodesPass::HandleSMod(FuzzSModProto* smod) {
  smod->set_bit_width(BoundedWidth(smod->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(smod->mutable_lhs_idx(), smod->bit_width());
  BValue rhs =
      GetWidthFittedOperand(smod->mutable_rhs_idx(), smod->bit_width());
  stack_.push_back(fb_->SMod(lhs, rhs));
}

void GenIrNodesPass::HandleSubtract(FuzzSubtractProto* subtract) {
  subtract->set_bit_width(BoundedWidth(subtract->bit_width()));
  BValue lhs =
      GetWidthFittedOperand(subtract->mutable_lhs_idx(), subtract->bit_width());
  BValue rhs =
      GetWidthFittedOperand(subtract->mutable_rhs_idx(), subtract->bit_width());
  stack_.push_back(fb_->Subtract(lhs, rhs));
}

void GenIrNodesPass::HandleAdd(FuzzAddProto* add) {
  add->set_bit_width(BoundedWidth(add->bit_width()));
  BValue lhs = GetWidthFittedOperand(add->mutable_lhs_idx(), add->bit_width());
  BValue rhs = GetWidthFittedOperand(add->mutable_rhs_idx(), add->bit_width());
  stack_.push_back(fb_->Add(lhs, rhs));
}

void GenIrNodesPass::HandleConcat(FuzzConcatProto* concat) {
  // Requires at least one operand.
  std::vector<BValue> operands =
      GetOperands(concat->mutable_operand_idxs(), /*min_operand_count=*/1);
  stack_.push_back(fb_->Concat(operands));
}

void GenIrNodesPass::HandleULe(FuzzULeProto* ule) {
  ule->set_bit_width(BoundedWidth(ule->bit_width()));
  BValue lhs = GetWidthFittedOperand(ule->mutable_lhs_idx(), ule->bit_width());
  BValue rhs = GetWidthFittedOperand(ule->mutable_rhs_idx(), ule->bit_width());
  stack_.push_back(fb_->ULe(lhs, rhs));
}

void GenIrNodesPass::HandleULt(FuzzULtProto* ult) {
  ult->set_bit_width(BoundedWidth(ult->bit_width()));
  BValue lhs = GetWidthFittedOperand(ult->mutable_lhs_idx(), ult->bit_width());
  BValue rhs = GetWidthFittedOperand(ult->mutable_rhs_idx(), ult->bit_width());
  stack_.push_back(fb_->ULt(lhs, rhs));
}

void GenIrNodesPass::HandleUGe(FuzzUGeProto* uge) {
  uge->set_bit_width(BoundedWidth(uge->bit_width()));
  BValue lhs = GetWidthFittedOperand(uge->mutable_lhs_idx(), uge->bit_width());
  BValue rhs = GetWidthFittedOperand(uge->mutable_rhs_idx(), uge->bit_width());
  stack_.push_back(fb_->UGe(lhs, rhs));
}

void GenIrNodesPass::HandleUGt(FuzzUGtProto* ugt) {
  ugt->set_bit_width(BoundedWidth(ugt->bit_width()));
  BValue lhs = GetWidthFittedOperand(ugt->mutable_lhs_idx(), ugt->bit_width());
  BValue rhs = GetWidthFittedOperand(ugt->mutable_rhs_idx(), ugt->bit_width());
  stack_.push_back(fb_->UGt(lhs, rhs));
}

void GenIrNodesPass::HandleSLe(FuzzSLeProto* sle) {
  sle->set_bit_width(BoundedWidth(sle->bit_width()));
  BValue lhs = GetWidthFittedOperand(sle->mutable_lhs_idx(), sle->bit_width());
  BValue rhs = GetWidthFittedOperand(sle->mutable_rhs_idx(), sle->bit_width());
  stack_.push_back(fb_->SLe(lhs, rhs));
}

void GenIrNodesPass::HandleSLt(FuzzSLtProto* slt) {
  slt->set_bit_width(BoundedWidth(slt->bit_width()));
  BValue lhs = GetWidthFittedOperand(slt->mutable_lhs_idx(), slt->bit_width());
  BValue rhs = GetWidthFittedOperand(slt->mutable_rhs_idx(), slt->bit_width());
  stack_.push_back(fb_->SLt(lhs, rhs));
}

void GenIrNodesPass::HandleSGe(FuzzSGeProto* sge) {
  sge->set_bit_width(BoundedWidth(sge->bit_width()));
  BValue lhs = GetWidthFittedOperand(sge->mutable_lhs_idx(), sge->bit_width());
  BValue rhs = GetWidthFittedOperand(sge->mutable_rhs_idx(), sge->bit_width());
  stack_.push_back(fb_->SGe(lhs, rhs));
}

void GenIrNodesPass::HandleSGt(FuzzSGtProto* sgt) {
  sgt->set_bit_width(BoundedWidth(sgt->bit_width()));
  BValue lhs = GetWidthFittedOperand(sgt->mutable_lhs_idx(), sgt->bit_width());
  BValue rhs = GetWidthFittedOperand(sgt->mutable_rhs_idx(), sgt->bit_width());
  stack_.push_back(fb_->SGt(lhs, rhs));
}

void GenIrNodesPass::HandleEq(FuzzEqProto* eq) {
  eq->set_bit_width(BoundedWidth(eq->bit_width()));
  BValue lhs = GetWidthFittedOperand(eq->mutable_lhs_idx(), eq->bit_width());
  BValue rhs = GetWidthFittedOperand(eq->mutable_rhs_idx(), eq->bit_width());
  stack_.push_back(fb_->Eq(lhs, rhs));
}

void GenIrNodesPass::HandleNe(FuzzNeProto* ne) {
  ne->set_bit_width(BoundedWidth(ne->bit_width()));
  BValue lhs = GetWidthFittedOperand(ne->mutable_lhs_idx(), ne->bit_width());
  BValue rhs = GetWidthFittedOperand(ne->mutable_rhs_idx(), ne->bit_width());
  stack_.push_back(fb_->Ne(lhs, rhs));
}

void GenIrNodesPass::HandleNegate(FuzzNegateProto* negate) {
  BValue operand = GetOperand(negate->operand_idx());
  stack_.push_back(fb_->Negate(operand));
}

void GenIrNodesPass::HandleNot(FuzzNotProto* not_op) {
  BValue operand = GetOperand(not_op->operand_idx());
  stack_.push_back(fb_->Not(operand));
}

void GenIrNodesPass::HandleLiteral(FuzzLiteralProto* literal) {
  literal->set_bit_width(BoundedWidth(literal->bit_width()));
  // Take the bytes protobuf datatype and convert it to a Bits object by making
  // a const uint8_t span. Any bytes that exceed the bit width of the literal
  // will be dropped.
  Bits value_bits =
      ChangeBytesBitWidth(literal->value_bytes(), literal->bit_width());
  stack_.push_back(fb_->Literal(value_bits));
}

// Retrieves an operand from the stack based off of a stack index.
BValue GenIrNodesPass::GetOperand(int64_t stack_idx) {
  if (stack_.empty()) {
    // If the stack is empty, return a default value.
    return fb_->Literal(UBits(0, 64));
  } else {
    // Retrieve the operand from the stack based off of the
    // randomly generated stack index.
    return stack_[stack_idx % stack_.size()];
  }
}

// Retrieves multiple operands from the stack based off of stack indices. If
// min_operand_count and max_operand_count are specified, the number of returned
// operands will be of the specified minimum or maximum constraint.
std::vector<BValue> GenIrNodesPass::GetOperands(
    google::protobuf::RepeatedField<int64_t>* operand_idxs, int64_t min_operand_count,
    int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetOperand(operand_idxs->at(i)));
  }
  // Fill in any remaining operands with default values.
  while (operands.size() < min_operand_count) {
    operands.push_back(fb_->Literal(UBits(0, 64)));
  }
  return operands;
}

// Retrieves an operand from the stack based off of a FittedOperandIdxProto. The
// width of the operand will be changed to match the specified bit width.
BValue GenIrNodesPass::GetWidthFittedOperand(FittedOperandIdxProto* operand_idx,
                                             int64_t bit_width) {
  BValue operand = GetOperand(operand_idx->stack_idx());
  // Change the width using the specified increase/decrease methods.
  return ChangeBitWidth(fb_, operand, bit_width,
                        operand_idx->mutable_width_fitting_method());
}

// Retrieves multiple operands from the stack based off of
// FittedOperandIdxProtos. If min_operand_count and max_operand_count are
// specified, the number of returned operands will be of the specified minimum
// or maximum constraint.
std::vector<BValue> GenIrNodesPass::GetWidthFittedOperands(
    google::protobuf::RepeatedPtrField<FittedOperandIdxProto>* operand_idxs,
    int64_t bit_width, int64_t min_operand_count, int64_t max_operand_count) {
  // If the max operand count is less than the min operand count, assume invalid
  // or default inputs, so just ignore max_operand_count.
  if (max_operand_count < min_operand_count) {
    max_operand_count = operand_idxs->size();
  }
  std::vector<BValue> operands;
  // Add operands up to the max operand count or up to the end of the vector.
  for (int64_t i = 0; i < operand_idxs->size() && i < max_operand_count;
       i += 1) {
    operands.push_back(GetWidthFittedOperand(&operand_idxs->at(i), bit_width));
  }
  // Fill in any remaining operands with default values of the same bit width.
  while (operands.size() < min_operand_count) {
    operands.push_back(fb_->Literal(UBits(0, bit_width)));
  }
  return operands;
}

}  // namespace xls
