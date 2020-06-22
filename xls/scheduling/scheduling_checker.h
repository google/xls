// Copyright 2020 Google LLC
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

#ifndef XLS_SCHEDULING_SCHEDULING_CHECKER_H_
#define XLS_SCHEDULING_SCHEDULING_CHECKER_H_

#include "xls/passes/passes.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {

// Invariant checker for the IR and schedule. Verifies the package, and various
// invariants of the schedule.
class SchedulingChecker : public SchedulingInvariantChecker {
 public:
  absl::Status Run(SchedulingUnit* unit, const SchedulingPassOptions& options,
                   SchedulingPassResults* results) const override;
};

}  // namespace xls

#endif  // XLS_SCHEDULING_SCHEDULING_CHECKER_H_
