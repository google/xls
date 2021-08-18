// Copyright 2021 The XLS Authors
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

#ifndef XLS_CODEGEN_CODEGEN_WRAPPER_PASS_H_
#define XLS_CODEGEN_CODEGEN_WRAPPER_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/passes/passes.h"

namespace xls::verilog {

// A codegen pass wrapper which wraps a FunctionBasePass. This is useful for
// adding an optimization or transformation pass (most passes in xls/passes are
// FunctionBasePasses) to the codegen pipeline. The wrapped pass is run on the
// block being lowered to Verilog.
class CodegenWrapperPass : public CodegenPass {
 public:
  CodegenWrapperPass(std::unique_ptr<FunctionBasePass> wrapped_pass)
      : CodegenPass(absl::StrFormat("codegen_%s", wrapped_pass->short_name()),
                    absl::StrFormat("%s (codegen)", wrapped_pass->long_name())),
        wrapped_pass_(std::move(wrapped_pass)) {}
  ~CodegenWrapperPass() override {}

  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   PassResults* results) const override;

 private:
  std::unique_ptr<FunctionBasePass> wrapped_pass_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CODEGEN_WRAPPER_PASS_H_
