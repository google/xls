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

#include "xls/netlist/logical_effort.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace netlist {
namespace logical_effort {

absl::StatusOr<double> GetLogicalEffort(CellKind kind, int64_t input_count) {
  // Table 1.1 in the Logical Effort book.
  switch (kind) {
    case CellKind::kInverter:
      XLS_RET_CHECK_EQ(input_count, 1);
      return 1;
    case CellKind::kNand:
      XLS_RET_CHECK_GE(input_count, 1);
      return (input_count + 2.0) / 3.0;
    case CellKind::kNor:
      XLS_RET_CHECK_GE(input_count, 1);
      return (2.0 * input_count + 1.0) / 3.0;
    case CellKind::kMultiplexer:
      XLS_RET_CHECK_GE(input_count, 2);
      return 2;
    case CellKind::kXor:
      XLS_RET_CHECK_GE(input_count, 1);
      // Section 4.5.4 provides the formula for XOR: N * 2**(N-1). So a
      // two-input XOR has logical effort 4. For XORs with more inputs we
      // compute it as a tree of two-input XORs.
      return CeilOfLog2(input_count) * 4;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled cell kind for logical effort: %s",
                          CellKindToString(kind)));
  }
}

absl::StatusOr<double> GetParasiticDelay(CellKind kind, int64_t input_count) {
  // Table 1.2 in the Logical Effort book.
  switch (kind) {
    case CellKind::kInverter:
      XLS_RET_CHECK_EQ(input_count, 1);
      return 1.0;
    case CellKind::kNand:
      XLS_RET_CHECK_GE(input_count, 1);
      return input_count;
    case CellKind::kNor:
      XLS_RET_CHECK_GE(input_count, 1);
      return input_count;
    case CellKind::kMultiplexer:
      XLS_RET_CHECK_GE(input_count, 2);
      return 2.0 * input_count;
    case CellKind::kXor:
      // case CellKind::kXnor:
      XLS_RET_CHECK_GE(input_count, 1);
      return 4.0;
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Unhandled cell kind for parasitic delay: %s",
                          CellKindToString(kind)));
  }
}

absl::StatusOr<double> ComputeElectricalEffort(const rtl::Cell& cell) {
  int64_t output_load_same_kind = 0;
  for (const auto& iter : cell.outputs()) {
    rtl::NetRef output = iter.netref;
    for (const rtl::Cell* connected_cell : output->connected_cells()) {
      if (connected_cell == &cell) {
        continue;
      }
      if (connected_cell->kind() == cell.kind()) {
        output_load_same_kind++;
      } else {
        return absl::FailedPreconditionError(
            "Cannot compute eletric effort for cell, driving a cell of a "
            "different kind (absolute capacitance values are required).");
      }
    }
  }
  // TODO(leary): 2019-08-16 Need to have output pin capacitances.
  XLS_RET_CHECK_NE(output_load_same_kind, 0)
      << "Output of cell " << cell.name() << " appears to be unconnected.";
  return output_load_same_kind;
}

absl::StatusOr<double> ComputeDelay(rtl::Cell* cell) {
  XLS_ASSIGN_OR_RETURN(double g,
                       GetLogicalEffort(cell->kind(), cell->inputs().size()));
  XLS_ASSIGN_OR_RETURN(double h, ComputeElectricalEffort(*cell));
  XLS_ASSIGN_OR_RETURN(double p,
                       GetParasiticDelay(cell->kind(), cell->inputs().size()));
  double d = g * h + p;
  VLOG(4) << absl::StreamFormat("g: %f h: %f p: %f d=gh+p: %f", g, h, p, d);
  return d;
}

absl::StatusOr<double> ComputePathLogicalEffort(
    absl::Span<rtl::Cell* const> path) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        "Cannot compute logical effort of empty path.");
  }
  double effort = 1.0;
  for (const rtl::Cell* cell : path) {
    XLS_ASSIGN_OR_RETURN(double cell_effort,
                         GetLogicalEffort(cell->kind(), cell->inputs().size()));
    effort *= cell_effort;
  }
  return effort;
}

absl::StatusOr<double> ComputePathParasiticDelay(
    absl::Span<rtl::Cell* const> path) {
  double sum = 0.0;
  for (const rtl::Cell* cell : path) {
    XLS_ASSIGN_OR_RETURN(
        double p, GetParasiticDelay(cell->kind(), cell->inputs().size()));
    sum += p;
  }
  return sum;
}

absl::StatusOr<double> ComputePathBranchingEffort(
    absl::Span<rtl::Cell* const> path) {
  double branching_effort = 1.0;
  for (int64_t i = 0; i < path.size() - 1; ++i) {
    if (path[i]->outputs().size() != 1) {
      return absl::UnimplementedError(
          "More than one output net, cannot compute branching effort.");
    }
    rtl::NetRef out = path[i]->outputs().begin()->netref;
    XLS_ASSIGN_OR_RETURN(std::vector<rtl::Cell*> driven_cells,
                         out->GetConnectedCellsSans(path[i]));
    if (driven_cells.size() != 1) {
      return absl::UnimplementedError("Compute C_total / C_useful.");
    }
    XLS_RET_CHECK_EQ(*driven_cells.begin(), path[i + 1]);
    // Only a single path, so branching effort is multiplied by 1.0.
  }
  return branching_effort;
}

absl::StatusOr<double> ComputePathEffort(absl::Span<rtl::Cell* const> path,
                                         double input_pin_capacitance,
                                         double output_pin_capacitance) {
  XLS_ASSIGN_OR_RETURN(double G, ComputePathLogicalEffort(path));
  XLS_ASSIGN_OR_RETURN(double B, ComputePathBranchingEffort(path));
  double H = output_pin_capacitance / input_pin_capacitance;
  return G * B * H;
}

absl::StatusOr<double> ComputePathLeastDelayAchievable(
    absl::Span<rtl::Cell* const> path, double input_pin_capacitance,
    double output_pin_capacitance) {
  double N = path.size();
  XLS_ASSIGN_OR_RETURN(double F, ComputePathEffort(path, input_pin_capacitance,
                                                   output_pin_capacitance));
  XLS_ASSIGN_OR_RETURN(double P, ComputePathParasiticDelay(path));
  return N * std::pow(F, 1.0 / N) + P;
}

}  // namespace logical_effort
}  // namespace netlist
}  // namespace xls
