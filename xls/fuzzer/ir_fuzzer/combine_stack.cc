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

#include "xls/fuzzer/ir_fuzzer/combine_stack.h"

#include <cstdint>

#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/function_builder.h"

namespace xls {

// Combines the stack of BValues into a single IR object/BValue. There are
// multiple ways to combine the stack, based off of the CombineStackMethod
// specified in the FuzzProgramProto. The combined_stack_ attribute is set as
// the final result.
BValue CombineStack(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                    absl::Span<const BValue> stack) {
  switch (fuzz_program->combine_stack_method()) {
    case CombineStackMethod::ADD_STACK_METHOD:
      return AddStack(fuzz_program, fb, stack);
    case CombineStackMethod::LAST_ELEMENT_METHOD:
    default:
      return LastElement(fuzz_program, fb, stack);
  }
}

// Returns the last element of the stack.
BValue LastElement(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                   absl::Span<const BValue> stack) {
  return stack[stack.size() - 1];
}

// Adds everything in the stack together.
BValue AddStack(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                absl::Span<const BValue> stack) {
  BValue combined_stack = stack[0];
  for (int64_t i = 1; i < stack.size(); i += 1) {
    // Change the bit width of the combined stack to the bit width of the next
    // element in the stack.
    combined_stack =
        ChangeBitWidth(fb, combined_stack, stack[i].BitCountOrDie());
    combined_stack = fb->Add(combined_stack, stack[i]);
  }
  return combined_stack;
}

}  // namespace xls
