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

#ifndef XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_WRAPPER_PASS_H_
#define XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_WRAPPER_PASS_H_

#include <concepts>
#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

// A codegen pass wrapper which wraps a OptimizationPass. This is
// useful for adding an optimization or transformation pass (most passes in
// xls/passes are OptimizationPasses) to the codegen pipeline. The
// wrapped pass is run on the block being lowered to Verilog.
class BlockConversionWrapperPass : public BlockConversionPass {
 public:
  explicit BlockConversionWrapperPass(
      std::unique_ptr<OptimizationPass> wrapped_pass)
      : BlockConversionPass(
            absl::StrFormat("opt_pass<%s>", wrapped_pass->short_name()),
            absl::StrFormat("%s (codegen)", wrapped_pass->long_name())),
        wrapped_pass_(std::move(wrapped_pass)) {}
  ~BlockConversionWrapperPass() override = default;

  template <std::derived_from<OptimizationPass> OptPassT, typename... Args>
  static BlockConversionWrapperPass Create(Args... args) {
    return BlockConversionWrapperPass(std::make_unique<OptPassT>(args...));
  }

  bool IsCompound() const override { return wrapped_pass_->IsCompound(); }

  absl::StatusOr<bool> RunInternal(
      Package* package, const BlockConversionPassOptions& options,
      PassResults* results, BlockConversionContext& context) const final {
    return wrapped_pass_->Run(package, OptimizationPassOptions(options),
                              results, context.opt_context);
  }

 private:
  std::unique_ptr<OptimizationPass> wrapped_pass_;
};

}  // namespace xls::codegen

#endif  // XLS_CODEGEN_V_1_5_BLOCK_CONVERSION_WRAPPER_PASS_H_
