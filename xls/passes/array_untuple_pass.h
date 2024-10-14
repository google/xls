// Copyright 2024 The XLS Authors
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

#ifndef XLS_PASSES_ARRAY_UNTUPLE_PASS_H_
#define XLS_PASSES_ARRAY_UNTUPLE_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which changes any (non-external) array-of-tuple into a tuple-of-arrays.
// We can see through tuples quite well but can't see through arrays to anywhere
// near the same extent. Therefore the struct-of-array representation is
// always superior.
//
// Note that this pass makes no attempt to unpack or repack arrays which escape
// the function-base. This means that anything which comes in through a function
// param, or a procs recv or escapes through a function return or a proc send is
// not untuple'd.
// TODO(allight): We could do this at the cost of a significant number of ir
// nodes. We should experiment to see if this is worth doing.
class ArrayUntuplePass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "array_untuple";
  explicit ArrayUntuplePass()
      : OptimizationFunctionBasePass(kName, "Array UnTuple") {}
  ~ArrayUntuplePass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};
}  // namespace xls

#endif  // XLS_PASSES_ARRAY_UNTUPLE_PASS_H_
