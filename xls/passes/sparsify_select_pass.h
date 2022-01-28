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

#ifndef XLS_PASSES_SPARSIFY_SELECT_PASS_H_
#define XLS_PASSES_SPARSIFY_SELECT_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// The SparsifySelectPass is a type of range analysis-informed dead code
// elimination that removes cases from selects when range analysis proves that
// they can never occur. It does this by splitting a select into many selects,
// each of which covers a single interval from the selector interval set.
class SparsifySelectPass : public FunctionBasePass {
 public:
  SparsifySelectPass()
      : FunctionBasePass("sparsify_select", "Sparsify Select") {}
  ~SparsifySelectPass() override {}

 protected:
  // Sparsify selects using range analysis.
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_SPARSIFY_SELECT_PASS_H_
