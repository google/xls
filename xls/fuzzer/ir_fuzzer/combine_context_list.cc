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
#include <vector>

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Combines the context list of BValues into a single IR object/BValue. There
// are multiple ways to combine the context list, based off of the
// CombineListMethod specified in the FuzzProgramProto.
BValue CombineContextList(const FuzzProgramProto& fuzz_program,
                          FunctionBuilder* fb,
                          const IrNodeContextList& context_list) {
  switch (fuzz_program.combine_list_method()) {
    case CombineListMethod::TUPLE_LIST_METHOD:
      return TupleList(fb, context_list);
    case CombineListMethod::LAST_ELEMENT_METHOD:
    default:
      return LastElement(fb, context_list);
  }
}

// Returns the last element of the combined context list.
BValue LastElement(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  return context_list.GetElementAt(context_list.GetListSize() - 1);
}

// Tuples everything in the combined context list together.
BValue TupleList(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  std::vector<BValue> elements;
  for (int64_t i = 0; i < context_list.GetListSize(); i += 1) {
    elements.push_back(context_list.GetElementAt(i));
  }
  return fb->Tuple(elements);
}

}  // namespace xls
