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

#include "xls/fuzzer/ir_fuzzer/stringify_program_pass.h"

#include <iostream>
#include <sstream>
#include <string>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"

namespace xls {

// Returns a string representing the FuzzProgramProto attribute in a
// human-readable format. Useful for debugging/visualizing the randomly
// generated FuzzProgramProto. Will not print unset fields.
std::string StringifyProgramPass::StringifyProgram() {
  ss_ << "FuzzProgram:" << "\n";
  ss_ << "  combine_stack_method: "
      << CombineStackMethod_Name(fuzz_program_->combine_stack_method()) << "\n";
  ss_ << "  fuzz_ops:" << "\n";
  for (FuzzOpProto& fuzz_op : *fuzz_program_->mutable_fuzz_ops()) {
    VisitFuzzOp(&fuzz_op);
  }
  return ss_.str();
}

void StringifyProgramPass::HandleAdd(FuzzAddProto* add) {
  ss_ << "    FuzzAddOp:" << "\n";
  ss_ << "      lhs_ref:" << "\n";
  StringifyOperandRef(add->mutable_lhs_ref());
  ss_ << "      rhs_ref:" << "\n";
  StringifyOperandRef(add->mutable_rhs_ref());
}

void StringifyProgramPass::HandleLiteral(FuzzLiteralProto* literal) {
  ss_ << "    FuzzLiteral:" << "\n";
  ss_ << "      value: " << literal->value() << "\n";
}

void StringifyProgramPass::HandleParam(FuzzParamProto* param) {
  ss_ << "    FuzzParam:" << "\n";
}

void StringifyProgramPass::StringifyOperandRef(
    FuzzOperandRefProto* operand_ref) {
  ss_ << "        stack_idx: " << operand_ref->stack_idx() << "\n";
}

}  // namespace xls
