// Copyright 2022 The XLS Authors
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

#ifndef XLS_CODEGEN_MULP_COMBINING_PASS_H_
#define XLS_CODEGEN_MULP_COMBINING_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Combines partial product operations followed by an add into a single multiply
// operation. Example:
//
//     tmp = umulp(a, b)
//     y = add(tuple-index(tmp, 0), tuple-index(tmp, 1)
//
//   =>
//
//     y = umul(a, b)
//
// Partial product multiplies are used to increase scheduling flexibility by
// splitting potentially expensive multiplies into two operations. After
// scheduling these split operations can be recombined if the partial product
// and add are scheduled in the same cycle.
class MulpCombiningPass : public CodegenPass {
 public:
  MulpCombiningPass()
      : CodegenPass("mulp_combining", "Combine mulp operations") {}
  ~MulpCombiningPass() override = default;

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const override;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_MULP_COMBINING_PASS_H_
