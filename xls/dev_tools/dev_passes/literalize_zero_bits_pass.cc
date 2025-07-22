// Copyright 2025 The XLS Authors
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

#include "xls/dev_tools/dev_passes/literalize_zero_bits_pass.h"

#include <vector>

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
absl::StatusOr<bool> LiteralizeZeroBits::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  bool changes = false;
  std::vector<Node*> orig_nodes(f->nodes().begin(), f->nodes().end());
  for (Node* n : orig_nodes) {
    if (n->GetType()->IsBits() && !n->Is<Literal>() && !n->Is<Param>() &&
        n->GetType()->AsBitsOrDie()->bit_count() == 0) {
      changes = true;
      XLS_RETURN_IF_ERROR(
          n->ReplaceUsesWithNew<Literal>(Value(UBits(0, 0))).status());
    }
  }
  return changes;
}

}  // namespace xls
