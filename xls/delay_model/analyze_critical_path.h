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

#ifndef XLS_DELAY_MODEL_ANALYZE_CRITICAL_PATH_H_
#define XLS_DELAY_MODEL_ANALYZE_CRITICAL_PATH_H_

#include <vector>

#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"

namespace xls {

struct CriticalPathEntry {
  Node* node;

  // Delay for producing the output of "node" from its inputs.
  int64 node_delay_ps;

  // Delay for producing the output of "node", as measured from the start of the
  // critical path (the path delay).
  int64 path_delay_ps;

  // Whether this node was "pushed out" by a cycle boundary; i.e. could have
  // executed earlier if there were no cycle boundaries. This will only be true
  // if clock_period_ps was provided to AnalyzeCriticalPath.
  bool delayed_by_cycle_boundary;
};

// Returns the critical path, decorated with the delay to produce the output of
// that node on the critical path.
//
// clock_ps optionally provides the clock period in picoseconds.. When this is
// provided delays can be elongated, because nodes will source values from the
// clock output delay instead of immediately when they're produced.
// TODO(meheff): This function should take a PipelineSchedule rather than a
// target clock period. As is, this function is doing scheduling again and it
// likely does not match what schedule is used in codegen.
//
// The return value for the function is at the front of the returned vector.
xabsl::StatusOr<std::vector<CriticalPathEntry>> AnalyzeCriticalPath(
    Function* f, absl::optional<int64> clock_period_ps,
    const DelayEstimator& delay_estimator);

}  // namespace xls

#endif  // XLS_DELAY_MODEL_ANALYZE_CRITICAL_PATH_H_
