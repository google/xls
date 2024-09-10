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

#ifndef XLS_PASSES_PROC_STATE_FLATTENING_PASS_H_
#define XLS_PASSES_PROC_STATE_FLATTENING_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which flattens tuple elements of the proc state into their constituent
// components. Array elements are flattened in a different pass. Flattening
// improves optimizability because each state element can be considered and
// transformed in isolation. Flattening also gives the scheduler more
// flexibility; without flattening, each element in the aggregate must have the
// same lifetime.
class ProcStateTupleFlatteningPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "proc_state_tuple_flat";
  ProcStateTupleFlatteningPass()
      : OptimizationProcPass(kName, "Proc State Tuple Flattening") {}
  ~ProcStateTupleFlatteningPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_PROC_STATE_FLATTENING_PASS_H_
