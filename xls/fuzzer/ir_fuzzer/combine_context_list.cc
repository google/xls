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

#include <cstdint>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Combines the context list of BValues into a single IR object/BValue. There
// are multiple ways to combine the context list, based off of the
// CombineListMethod specified in the FuzzProgramProto.
BValue CombineContextList(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                          const IrNodeContextList& context_list) {
  switch (fuzz_program->combine_list_method()) {
    case CombineListMethod::ADD_LIST_METHOD:
      return AddList(fb, context_list);
    case CombineListMethod::LAST_ELEMENT_METHOD:
    default:
      return LastElement(fb, context_list);
  }
}

// Returns the last element of the combined context list.
BValue LastElement(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  return context_list.GetElementAt(context_list.GetListSize() - 1);
}

// Adds everything in the list together.
BValue AddList(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  BValue combined_list = context_list.GetElementAt(0);
  for (int64_t i = 1; i < context_list.GetListSize(); i += 1) {
    // Change the bit width of the combined list to the bit width of the next
    // element in the list.
    combined_list = ChangeBitWidth(
        fb, combined_list, context_list.GetElementAt(i).BitCountOrDie());
    combined_list = fb->Add(combined_list, context_list.GetElementAt(i));
  }
  return combined_list;
}

}  // namespace xls
