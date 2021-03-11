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

#ifndef XLS_PASSES_ARRAY_SIMPLIFICATION_H_
#define XLS_PASSES_ARRAY_SIMPLIFICATION_H_

#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/passes/passes.h"

namespace xls {

// Pass which simplifies or eliminates some array-type operations such as
// ArrayIndex.
class ArraySimplificationPass : public FunctionBasePass {
 public:
  ArraySimplificationPass(int64_t opt_level = kMaxOptLevel)
      : FunctionBasePass("array_simp", "Array Simplification"),
        opt_level_(opt_level) {}

 protected:
  int64_t opt_level_;
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_ARRAY_SIMPLIFICATION_H_
