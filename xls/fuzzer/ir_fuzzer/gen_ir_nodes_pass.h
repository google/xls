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

#ifndef XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_
#define XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_

#include <vector>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_visitor.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"

namespace xls {

// Pass that iterates over the FuzzOpProtos within the FuzzProgramProto. Each
// FuzzOpProto gets instantiated into an IR node/BValue. This value is placed on
// the stack. BValues on the stack may be used as operands by future
// instantiated FuzzOps. We use a stack because it is a simple way to reference
// previous FuzzOps through indices.
class GenIrNodesPass : public IrFuzzVisitor {
 public:
  GenIrNodesPass(FuzzProgramProto* fuzz_program, Package* p,
                 FunctionBuilder* fb, std::vector<BValue>& stack)
      : fuzz_program_(fuzz_program), p_(p), fb_(fb), stack_(stack) {}

  void GenIrNodes();

  void HandleAdd(FuzzAddProto* add) override;
  void HandleLiteral(FuzzLiteralProto* literal) override;
  void HandleParam(FuzzParamProto* param) override;

 private:
  BValue GetOperand(FittedOperandIdxProto* operand_idx);

  FuzzProgramProto* fuzz_program_;
  Package* p_;
  FunctionBuilder* fb_;
  std::vector<BValue>& stack_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_GEN_IR_NODES_PASS_H_
