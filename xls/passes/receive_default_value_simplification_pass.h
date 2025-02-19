// Copyright 2023 The XLS Authors
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

#ifndef XLS_PASSES_RECEIVE_DEFAULT_VALUE_SIMPLIFICATION_PASS_H_
#define XLS_PASSES_RECEIVE_DEFAULT_VALUE_SIMPLIFICATION_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Optimization which removes useless selects between the data value of a
// conditional or non-blocking receive and the default value of the receive (all
// zeros).
class ReceiveDefaultValueSimplificationPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "recv_default";

  ReceiveDefaultValueSimplificationPass()
      : OptimizationProcPass(kName, "Receive default value simplification") {}
  ~ReceiveDefaultValueSimplificationPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(
      Proc* proc, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext* context) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_RECEIVE_DEFAULT_VALUE_SIMPLIFICATION_PASS_H_
