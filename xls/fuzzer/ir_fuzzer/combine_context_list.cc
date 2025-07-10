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

#include "xls/fuzzer/ir_fuzzer/combine_context_list.h"

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Combines the list of BValues into a single IR object/BValue. There are
// multiple ways to combine the list, based off of the CombineListMethod
// specified in the FuzzProgramProto. The combined_list_ attribute is set as
// the final result.
BValue CombineContextList(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                          const IrNodeContextList& context_list) {
  switch (fuzz_program->combine_list_method()) {
    case CombineListMethod::LAST_ELEMENT_METHOD:
    default:
      return LastElement(fuzz_program, fb, context_list);
  }
}

// Returns the last element of the list.
BValue LastElement(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                   const IrNodeContextList& context_list) {
  return context_list.GetLastElement();
}

}  // namespace xls
