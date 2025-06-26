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

#ifndef XLS_FUZZER_IR_FUZZER_STRINGIFY_PROGRAM_PASS_H_
#define XLS_FUZZER_IR_FUZZER_STRINGIFY_PROGRAM_PASS_H_

#include <sstream>
#include <string>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_visitor.h"

namespace xls {

// Pass that iterates over the FuzzProgramProto and returns a string
// representation of the FuzzProgramProto in a human readable format. This class
// will likely be deleted in the future as printing the proto object is almost
// as effective for debugging.
class StringifyProgramPass : public IrFuzzVisitor {
 public:
  explicit StringifyProgramPass(FuzzProgramProto* fuzz_program)
      : fuzz_program_(fuzz_program) {}

  std::string StringifyProgram();

  void HandleAdd(FuzzAddProto* add) override;
  void HandleLiteral(FuzzLiteralProto* literal) override;
  void HandleParam(FuzzParamProto* param) override;

 private:
  void StringifyOperandRef(FuzzOperandRefProto* operand_ref);

  FuzzProgramProto* fuzz_program_;
  // Stringstream used to build the string throughout runtime, which is returned
  // as a string at the end.
  std::stringstream ss_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_STRINGIFY_PROGRAM_PASS_H_
