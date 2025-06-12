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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

IrFuzzBuilder::IrFuzzBuilder(Package* package,
                             FunctionBuilder* function_builder,
                             const FuzzProgramProto& fuzz_program)
    : p_(package), fb_(function_builder), fuzz_program_(fuzz_program) {}

// High level function that processes the randomly generated FuzzProgramProto
// and returns a valid IR object/BValue.
BValue IrFuzzBuilder::BuildIr() {
  InstFuzzOps();
  return CombineStack();
}

// Returns a string representing the FuzzProgramProto attribute in a
// human-readable format. Useful for debugging/visualizing the randomly
// generated FuzzProgramProto.
std::string IrFuzzBuilder::PrintProgram() {
  std::stringstream ss;
  ss << std::endl << "Program:" << std::endl;
  ss << "  CombineStackMethod: " << fuzz_program_.combine_stack_method()
     << std::endl;
  for (const auto& fuzz_op : fuzz_program_.fuzz_ops()) {
    switch (fuzz_op.fuzz_op_case()) {
      case FuzzOpProto::kFuzzLiteral: {
        ss << "  Literal: " << std::endl;
        if (fuzz_op.fuzz_literal().has_value()) {
          ss << "    value: " << fuzz_op.fuzz_literal().value() << std::endl;
        }
        break;
      }
      case FuzzOpProto::kFuzzParam: {
        ss << "  Param:" << std::endl;
        break;
      }
      case FuzzOpProto::kFuzzAddOp: {
        ss << "  AddOp: " << std::endl;
        if (fuzz_op.fuzz_add_op().has_lhs_stack_idx()) {
          ss << "    lhs_stack_idx: " << fuzz_op.fuzz_add_op().lhs_stack_idx()
             << std::endl;
        }
        if (fuzz_op.fuzz_add_op().has_rhs_stack_idx()) {
          ss << "    rhs_stack_idx: " << fuzz_op.fuzz_add_op().rhs_stack_idx()
             << std::endl;
        }
        break;
      }
      case FuzzOpProto::FUZZ_OP_NOT_SET: {
        break;
      }
    }
  }
  return ss.str();
}

// Loops through all of the FuzzOpProtos in the FuzzProgramProto. Each
// FuzzOpProto is a randomly generated object that is used to instantiate/create
// a IR node/BValue. Add these BValues to the stack. Some FuzzOpProtos may
// require retrieving previous BValues from the stack.
void IrFuzzBuilder::InstFuzzOps() {
  for (const auto& fuzz_op : fuzz_program_.fuzz_ops()) {
    switch (fuzz_op.fuzz_op_case()) {
      case FuzzOpProto::kFuzzLiteral: {
        int64_t value = 0;
        if (fuzz_op.fuzz_literal().has_value()) {
          value = fuzz_op.fuzz_literal().value();
        }
        stack_.push_back(fb_->Literal(UBits(value, 64)));
        break;
      }
      case FuzzOpProto::kFuzzParam: {
        stack_.push_back(fb_->Param("p" + std::to_string(stack_.size()),
                                    p_->GetBitsType(64)));
        break;
      }
      case FuzzOpProto::kFuzzAddOp: {
        if (stack_.empty()) {
          stack_.push_back(
              fb_->Add(fb_->Literal(UBits(0, 64)), fb_->Literal(UBits(0, 64))));
          break;
        }
        int64_t lhs_stack_idx = 0;
        if (fuzz_op.fuzz_add_op().has_lhs_stack_idx()) {
          lhs_stack_idx = fuzz_op.fuzz_add_op().lhs_stack_idx() % stack_.size();
        }
        int64_t rhs_stack_idx = 0;
        if (fuzz_op.fuzz_add_op().has_rhs_stack_idx()) {
          rhs_stack_idx = fuzz_op.fuzz_add_op().rhs_stack_idx() % stack_.size();
        }
        // Retrieve the lhs and rhs operands from the stack based off of the
        // randomly generated stack idxs.
        stack_.push_back(
            fb_->Add(stack_[lhs_stack_idx], stack_[rhs_stack_idx]));
        break;
      }
      case FuzzOpProto::FUZZ_OP_NOT_SET: {
        break;
      }
    }
  }
}

// Combines the stack of BValues into a single IR object/BValue. There are
// multiple ways to combine the stack, based off of the CombineStackMethod
// specified in the FuzzProgramProto.
BValue IrFuzzBuilder::CombineStack() {
  if (stack_.empty()) {
    return fb_->Literal(UBits(0, 64));
  }
  switch (fuzz_program_.combine_stack_method()) {
    case CombineStackMethod::LAST_ELEMENT_METHOD: {
      return stack_[stack_.size() - 1];
    }
    case CombineStackMethod::ADD_STACK_METHOD: {
      BValue result = stack_[0];
      for (int64_t i = 1; i < stack_.size(); i += 1) {
        result = fb_->Add(result, stack_[i]);
      }
      return result;
    }
    default: {
      return fb_->Literal(UBits(0, 64));
    }
  }
}

}  // namespace xls
