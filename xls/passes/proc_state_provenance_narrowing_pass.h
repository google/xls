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

#ifndef XLS_PASSES_PROC_STATE_PROVENANCE_NARROWING_PASS_H_
#define XLS_PASSES_PROC_STATE_PROVENANCE_NARROWING_PASS_H_

#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

// Pass which tries to minimize the size and total number of elements of the
// proc state. This pass works by examining the provenance of the bits making up
// the next value to determine which (if any) bits are never actually modified.
//
// NB This is a separate pass from ProcStateNarrowing for simplicity of
// implementation. That pass mostly assumes we'll have a range-analysis which
// this does not need.
class ProcStateProvenanceNarrowingPass : public OptimizationProcPass {
 public:
  static constexpr std::string_view kName = "proc_state_provenance_narrow";
  ProcStateProvenanceNarrowingPass()
      : OptimizationProcPass(kName, "Proc State Provenance Narrowing") {}
  ~ProcStateProvenanceNarrowingPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnProcInternal(Proc* proc,
                                         const OptimizationPassOptions& options,
                                         PassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_PASSES_PROC_STATE_PROVENANCE_NARROWING_PASS_H_
