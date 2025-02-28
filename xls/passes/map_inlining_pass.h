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

#ifndef XLS_PASSES_MAP_INLINING_PASS_H_
#define XLS_PASSES_MAP_INLINING_PASS_H_

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// A pass to convert map nodes to in-line Invoke nodes. We don't directly lower
// maps to Verilog.
class MapInliningPass : public OptimizationFunctionBasePass {
 public:
  static constexpr std::string_view kName = "map_inlining";
  MapInliningPass();

  // Inline a single Map instruction. Provided for test and utility
  // (ir_minimizer) use.
  static absl::Status InlineOneMap(Map* map);

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* function, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override;

  // Replaces a single Map node with a CountedFor operation.
  absl::Status ReplaceMap(Map* map) const;
};

}  // namespace xls

#endif  // XLS_PASSES_MAP_INLINING_PASS_H_
