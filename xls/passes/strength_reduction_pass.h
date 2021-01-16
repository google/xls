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

#ifndef XLS_PASSES_STRENGTH_REDUCTION_PASS_H_
#define XLS_PASSES_STRENGTH_REDUCTION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Replaces operations with equivalent cheaper operations. For example, multiply
// by a power-of-two constant may be replaced with a shift left.
class StrengthReductionPass : public FunctionBasePass {
 public:
  explicit StrengthReductionPass(bool split_ops)
      : FunctionBasePass("strength_red", "Strength Reduction"),
        split_ops_(split_ops) {}
  ~StrengthReductionPass() override {}

 protected:
  // Run all registered passes in order of registration.
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override;

 private:
  bool split_ops_;
};

}  // namespace xls

#endif  // XLS_PASSES_STRENGTH_REDUCTION_PASS_H_
