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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_visitor.h"

namespace xls {

// Given a FuzzOpProto, call the corresponding Handle* function based on the
// FuzzOpProto type. This is a common visitor selection function.
void IrFuzzVisitor::VisitFuzzOp(FuzzOpProto* fuzz_op) {
  switch (fuzz_op->fuzz_op_case()) {
    case FuzzOpProto::kAdd:
      HandleAdd(fuzz_op->mutable_add());
      break;
    case FuzzOpProto::kLiteral:
      HandleLiteral(fuzz_op->mutable_literal());
      break;
    case FuzzOpProto::kParam:
      HandleParam(fuzz_op->mutable_param());
      break;
    case FuzzOpProto::FUZZ_OP_NOT_SET:
      break;
  }
}

}  // namespace xls
