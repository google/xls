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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_

#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"

namespace xls {

// Interface visitor class inherited by classes which perform a pass over the
// FuzzProgramProto. This class is almost identical to the dfs_visitor.h class,
// but handles FuzzOpProtos instead of IR nodes.
class IrFuzzVisitor {
 public:
  virtual ~IrFuzzVisitor() = default;

  // These functions correlate to an IR Node.
  virtual void HandleAdd(FuzzAddProto* add) = 0;
  virtual void HandleLiteral(FuzzLiteralProto* literal) = 0;
  virtual void HandleParam(FuzzParamProto* param) = 0;

  void VisitFuzzOp(FuzzOpProto* fuzz_op);
};

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_VISITOR_H_
