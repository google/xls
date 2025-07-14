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

#include "absl/log/log.h"
#include "xls/fuzzer/ir_fuzzer/combine_stack.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.h"
#include "xls/ir/function_builder.h"

namespace xls {

// High level function that processes the randomly generated FuzzProgramProto
// and returns a valid IR object/BValue.
BValue IrFuzzBuilder::BuildIr() {
  // Logs the FuzzProgramProto for debugging.
  VLOG(3) << "IR Fuzzer-1: Fuzz Program Proto:" << "\n"
          << fuzz_program_->DebugString() << "\n";
  // Converts the FuzzProgramProto instructions into a stack of BValues IR
  // nodes.
  GenIrNodesPass gen_ir_nodes_pass(fuzz_program_, p_, fb_, stack_);
  gen_ir_nodes_pass.GenIrNodes();
  // Combines the stack of BValues into a single BValue.
  BValue combined_stack = CombineStack(fuzz_program_, fb_, stack_);
  return combined_stack;
}

}  // namespace xls
