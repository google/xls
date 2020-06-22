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

// Helpers for calculating "method of logical effort" based delay, as reflected
// in the blue Logical Effort book by Ivan Sutherland, et al.

#ifndef XLS_NETLIST_LOGICAL_EFFORT_H_
#define XLS_NETLIST_LOGICAL_EFFORT_H_

#include "xls/common/integral_types.h"
#include "xls/common/status/statusor.h"
#include "xls/netlist/netlist.h"

namespace xls {
namespace netlist {
namespace logical_effort {

// Also referred to as "g".
xabsl::StatusOr<double> GetLogicalEffort(CellKind kind, int64 input_count);

// Also referred to as "p".
//
// Returns a value in (coefficient) units of $p_{inv}$, the parasitic delay of
// an inverter. Note a typical value convenient for analysis is 1.0 (in units of
// \tao) for $p_{inv}$.
xabsl::StatusOr<double> GetParasiticDelay(CellKind kind, int64 input_count);

// Also referred to as "h".
xabsl::StatusOr<double> ComputeElectricalEffort(const rtl::Cell& cell);

// Computes $d = gh + p$.
xabsl::StatusOr<double> ComputeDelay(rtl::Cell* cell);

// Computes $G = \Pi g$ (product of the cell logical efforts in the path).
xabsl::StatusOr<double> ComputePathLogicalEffort(
    absl::Span<rtl::Cell* const> path);

// Computes $P = \Sigma p$ (sum of the cell parasitic delays in the path).
xabsl::StatusOr<double> ComputePathParasiticDelay(
    absl::Span<rtl::Cell* const> path);

// Computes $B = \Pi b_i$ (product of branching efforts for each stage in the
// path, where $b = \frac{C_{total}}{C_{useful}}$).
xabsl::StatusOr<double> ComputePathBranchingEffort(
    absl::Span<rtl::Cell* const> path);

// Computes $F = G B H$ (incorporates logical path effort, branching effort, and
// path electrical effort).
xabsl::StatusOr<double> ComputePathEffort(absl::Span<rtl::Cell* const> path,
                                          double input_pin_capacitance,
                                          double output_pin_capacitance);

// Computes $D \hat = N F^{1/N} + P$, the minimal delay achievable along the
// path.
xabsl::StatusOr<double> ComputePathLeastDelayAchievable(
    absl::Span<rtl::Cell* const> path, double input_pin_capacitance,
    double output_pin_capacitance);

}  // namespace logical_effort
}  // namespace netlist
}  // namespace xls

#endif  // XLS_NETLIST_LOGICAL_EFFORT_H_
