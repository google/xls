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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_BUILDER_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_BUILDER_H_

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/fuzzer/ir_fuzzer/ir_node_context_list.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"

namespace xls {

// Class used to convert randomly generated objects into a valid IR object. More
// specifically, the conversion from a FuzzProgramProto to an IR BValue. It uses
// a context list to maintain a list of BValues, where each element represents a
// node in the IR that was instantiated/created from a FuzzOpProto. A context
// list is used to allow new FuzzOpProtos to retrieve previous BValues from the
// context list without using explicit references like the IR does.
class IrFuzzBuilder {
 public:
  IrFuzzBuilder(const FuzzProgramProto& fuzz_program, Package* p,
                FunctionBuilder* fb)
      : fuzz_program_(fuzz_program),
        p_(p),
        fb_(fb),
        context_list_(p_, fb_, IrFuzzHelpers(fuzz_program.version())) {}

  // Main function that returns the final IR BValue.
  BValue BuildIr();

 private:
  // FuzzProgramProto is randomly generated instructions based off a protobuf
  // structure used to instantiate IR BValues.
  const FuzzProgramProto& fuzz_program_;
  // Package and FunctionBuilder are used to create new IR BValues. They are not
  // owned by the IrFuzzBuilder, just referenced.
  Package* p_;
  FunctionBuilder* fb_;
  // The context list is used to maintain a list of BValues that are created
  // from the FuzzProgramProto. It contains multiple lists, including a combined
  // list and a list for each individual type.
  IrNodeContextList context_list_;
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_BUILDER_H_
