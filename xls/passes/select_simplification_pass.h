// Copyright 2020 The XLS Authors
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

#ifndef XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Pass which simplifies selects and one-hot-selects. Example optimizations
// include removing dead arms and eliminating selects with constant selectors.
class SelectSimplificationPass : public FunctionBasePass {
 public:
  // 'split_ops' indicates whether to perform optimizations which split
  // operations into smaller operations. Typically splitting optimizations
  // should be performed later in the optimization pipeline.
  explicit SelectSimplificationPass(bool split_ops)
      : FunctionBasePass("select_simp", "Select Simplification"),
        split_ops_(split_ops) {}
  ~SelectSimplificationPass() override {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override;

 private:
  bool split_ops_;
};

}  // namespace xls

#endif  // XLS_PASSES_SELECT_SIMPLIFICATION_PASS_H_
