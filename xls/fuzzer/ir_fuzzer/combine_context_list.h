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

#ifndef XLS_FUZZER_IR_FUZZER_COMBINE_CONTEXT_LIST_H_
#define XLS_FUZZER_IR_FUZZER_COMBINE_CONTEXT_LIST_H_

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"

namespace xls {

BValue CombineContextList(FuzzProgramProto* fuzz_program, FunctionBuilder* fb,
                          const IrNodeContextList& context_list);

BValue LastElement(FunctionBuilder* fb, const IrNodeContextList& context_list);
BValue AddList(FunctionBuilder* fb, const IrNodeContextList& context_list);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_COMBINE_STACK_H_
