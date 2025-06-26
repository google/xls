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

#include <string>
#include <vector>

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

void GenIrNodesPass::HandleAdd(FuzzAddProto* add) {
  FittedOperandIdxProto* lhs_idx = add->mutable_lhs_idx();
  FittedOperandIdxProto* rhs_idx = add->mutable_rhs_idx();
  BValue lhs = GetOperand(lhs_idx);
  BValue rhs = GetOperand(rhs_idx);
  // The operands need to be have the sane bit width in order to be added, so
  // update the lhs and rhs to be of the specified bit width of the
  // FuzzAddProto.
  lhs = ChangeBitWidth(fb_, lhs, add->bit_width(),
                       lhs_idx->mutable_width_fitting_method());
  rhs = ChangeBitWidth(fb_, rhs, add->bit_width(),
                       rhs_idx->mutable_width_fitting_method());
  stack_.push_back(fb_->Add(lhs, rhs));
}

void GenIrNodesPass::HandleLiteral(FuzzLiteralProto* literal) {
  // Take the bytes protobuf datatype and convert it to a Bits object by making
  // a const uint8_t span. Any bytes that exceed the bit width of the literal
  // will be dropped.
  Bits value_bits =
      ChangeBytesBitWidth(literal->value_bytes(), literal->bit_width());
  stack_.push_back(fb_->Literal(value_bits));
}

void GenIrNodesPass::HandleParam(FuzzParamProto* param) {
  // Params are named as "p" followed by the stack index of the param.
  stack_.push_back(fb_->Param("p" + std::to_string(stack_.size()),
                              p_->GetBitsType(param->bit_width())));
}

// Takes in a FittedOperandIdxProto and returns a BValue that is either a
// default value or a BValue from the stack based off of the stack index.
BValue GenIrNodesPass::GetOperand(FittedOperandIdxProto* operand_idx) {
  if (stack_.empty()) {
    // If the stack is empty, initialize the lhs and rhs with default values.
    return fb_->Literal(UBits(0, 64));
  } else {
    // Retrieve the lhs and rhs operands from the stack based off of the
    // randomly generated stack indices.
    return stack_[operand_idx->stack_idx() % stack_.size()];
  }
}

}  // namespace xls
