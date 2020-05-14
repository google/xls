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

#include "xls/delay_model/delay_estimator.h"

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/nodes.h"
#include "xls/netlist/logical_effort.h"

namespace xls {

DelayEstimatorManager& GetDelayEstimatorManagerSingleton() {
  static DelayEstimatorManager* manager = new DelayEstimatorManager;
  return *manager;
}

xabsl::StatusOr<DelayEstimator*> DelayEstimatorManager::GetDelayEstimator(
    absl::string_view name) const {
  if (!estimators_.contains(name)) {
    if (estimator_names_.empty()) {
      return absl::NotFoundError(
          absl::StrFormat("No delay estimator found named \"%s\". No "
                          "estimators are registered. Was InitXls called?",
                          name));
    } else {
      return absl::NotFoundError(absl::StrFormat(
          "No delay estimator found named \"%s\". Available estimators: %s",
          name, absl::StrJoin(estimator_names_, ", ")));
    }
  }
  return estimators_.at(name).get();
}

absl::Status DelayEstimatorManager::RegisterDelayEstimator(
    absl::string_view name, std::unique_ptr<DelayEstimator> delay_estimator) {
  if (estimators_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("Delay estimator named %s already exists", name));
  }
  estimators_[name] = std::move(delay_estimator);
  estimator_names_.push_back(std::string(name));
  std::sort(estimator_names_.begin(), estimator_names_.end());

  return absl::OkStatus();
}

namespace {

// TODO(leary): 2019-08-19 Read all of the curve-fit values from a
// characterization file for easier reference / recomputing if necessary.
/* static */ xabsl::StatusOr<int64> GetLogicalEffortDelayInTau(Node* node) {
  auto get_logical_effort = [node](netlist::CellKind kind,
                                   bool invert) -> xabsl::StatusOr<int64> {
    XLS_ASSIGN_OR_RETURN(double base_effort,
                         netlist::logical_effort::GetLogicalEffort(
                             kind, node->operands().size()));
    return std::ceil(invert ? base_effort + 1LL : base_effort);
  };
  auto get_reduction_logical_effort =
      [node](netlist::CellKind kind, bool invert) -> xabsl::StatusOr<int64> {
    int bit_count = node->BitCountOrDie();
    if (bit_count < 2) {
      return 0;
    }
    XLS_ASSIGN_OR_RETURN(
        double base_effort,
        netlist::logical_effort::GetLogicalEffort(kind, bit_count));
    return std::ceil(invert ? base_effort + 1LL : base_effort);
  };
  switch (node->op()) {
    // TODO(leary): 2019-09-24 Collect real numbers for these.
    case Op::kAnd:
      return get_logical_effort(netlist::CellKind::kNand, /*invert=*/true);
    case Op::kNand:
      return get_logical_effort(netlist::CellKind::kNand, /*invert=*/false);
    case Op::kNor:
      return get_logical_effort(netlist::CellKind::kNor, /*invert=*/false);
    case Op::kOr:
      return get_logical_effort(netlist::CellKind::kNor, /*invert=*/true);
    case Op::kXor:
      return get_logical_effort(netlist::CellKind::kXor, /*invert=*/false);
    case Op::kNot:
      return get_logical_effort(netlist::CellKind::kInverter, /*invert=*/false);
    case Op::kAndReduce:
      return get_reduction_logical_effort(netlist::CellKind::kNand,
                                          /*invert=*/true);
    case Op::kOrReduce:
      return get_reduction_logical_effort(netlist::CellKind::kNor,
                                          /*invert=*/true);
    case Op::kXorReduce:
      return get_reduction_logical_effort(netlist::CellKind::kXor,
                                          /*invert=*/false);
    case Op::kEncode: {
      // Each output bit is the OR reduction of half of the input
      // bits. Equivalently the NOR reduction delay plus an inverter delay.
      // TODO(meheff): Characterize this properly.
      int64 operand_width = node->operand(0)->BitCountOrDie();
      if (operand_width <= 2) {
        // A 2-bit or less encode simply passes through the MSB.
        return 0;
      } else {
        XLS_ASSIGN_OR_RETURN(
            int64 nor_delay,
            netlist::logical_effort::GetLogicalEffort(netlist::CellKind::kNor,
                                                      (operand_width + 1) / 2));
        return std::ceil(nor_delay + 1);
      }
    }
    default:
      break;
  }
  return absl::UnimplementedError(
      "Cannot compute delay of node using logical effort:" +
      node->ToStringWithOperandTypes());
}

}  // namespace

/* static */ xabsl::StatusOr<int64> DelayEstimator::GetLogicalEffortDelayInPs(
    Node* node, int64 tau_in_ps) {
  XLS_ASSIGN_OR_RETURN(int64 delay_in_tau, GetLogicalEffortDelayInTau(node));
  return delay_in_tau * tau_in_ps;
}

}  // namespace xls
