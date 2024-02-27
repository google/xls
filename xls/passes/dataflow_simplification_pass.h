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

#ifndef XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_

#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// An optimization which uses a lattice-based dataflow analysis to find
// equivalent nodes in the graph and replace them with a simpler form. The
// analysis traces through tuples, arrays, and select operations. Optimizations
// which can be performed by this pass:
//
//    tuple_index(tuple(x, y), index=1)  =>  y
//
//    select(selector, {z, z})  =>  z
//
//    array_index(array_update(A, x, index={42}), index={42})  =>  x
class DataflowSimplificationPass : public OptimizationFunctionBasePass {
 public:
  explicit DataflowSimplificationPass()
      : OptimizationFunctionBasePass("dataflow", "Dataflow Optimization") {}
  ~DataflowSimplificationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_SIMPLIFICATION_PASS_H_
