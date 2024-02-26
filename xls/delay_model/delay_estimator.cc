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

#include "xls/delay_model/delay_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/logical_effort.h"

namespace xls {

DelayEstimatorManager& GetDelayEstimatorManagerSingleton() {
  static absl::NoDestructor<DelayEstimatorManager> manager;
  return *manager;
}

absl::StatusOr<DelayEstimator*> DelayEstimatorManager::GetDelayEstimator(
    std::string_view name) const {
  if (!estimators_.contains(name)) {
    if (estimator_names_.empty()) {
      return absl::NotFoundError(
          absl::StrFormat("No delay estimator found named \"%s\". No "
                          "estimators are registered. Was InitXls called?",
                          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        "No delay estimator found named \"%s\". Available estimators: %s", name,
        absl::StrJoin(estimator_names_, ", ")));
  }
  return estimators_.at(name).second.get();
}

absl::StatusOr<DelayEstimator*>
DelayEstimatorManager::GetDefaultDelayEstimator() const {
  if (estimators_.empty()) {
    return absl::NotFoundError(
        "No delay estimator has been registered. Did the build "
        "target forget to link a plugin?");
  }
  int highest_precedence = 0;
  DelayEstimator* highest = nullptr;
  for (const std::string& name : estimator_names_) {
    const std::pair<DelayEstimatorPrecedence, std::unique_ptr<DelayEstimator>>&
        pair = estimators_.at(name);
    DelayEstimatorPrecedence precedence = pair.first;
    int precedence_value = static_cast<int>(precedence);
    if (precedence_value > highest_precedence) {
      highest_precedence = precedence_value;
      highest = pair.second.get();
    }
  }
  return highest;
}

absl::Status DelayEstimatorManager::RegisterDelayEstimator(
    std::unique_ptr<DelayEstimator> delay_estimator,
    DelayEstimatorPrecedence precedence) {
  std::string name = delay_estimator->name();
  if (estimators_.contains(name)) {
    return absl::InternalError(
        absl::StrFormat("Delay estimator named %s already exists", name));
  }
  estimators_[name] = {precedence, std::move(delay_estimator)};
  estimator_names_.push_back(name);
  std::sort(estimator_names_.begin(), estimator_names_.end());

  return absl::OkStatus();
}

namespace {

// TODO(leary): 2019-08-19 Read all of the curve-fit values from a
// characterization file for easier reference / recomputing if necessary.
/* static */ absl::StatusOr<int64_t> GetLogicalEffortDelayInTau(Node* node) {
  auto get_logical_effort = [node](netlist::CellKind kind,
                                   bool invert) -> absl::StatusOr<int64_t> {
    XLS_ASSIGN_OR_RETURN(double base_effort,
                         netlist::logical_effort::GetLogicalEffort(
                             kind, node->operands().size()));
    return std::ceil(invert ? base_effort + 1LL : base_effort);
  };
  auto get_reduction_logical_effort =
      [node](netlist::CellKind kind, bool invert) -> absl::StatusOr<int64_t> {
    const int64_t bit_count = node->operand(0)->BitCountOrDie();
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
    case Op::kGate:
      return get_logical_effort(netlist::CellKind::kNand, /*invert=*/true);
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
      int64_t operand_width = node->operand(0)->BitCountOrDie();
      if (operand_width <= 2) {
        // A 2-bit or less encode simply passes through the MSB.
        return 0;
      }
      XLS_ASSIGN_OR_RETURN(
          int64_t nor_delay,
          netlist::logical_effort::GetLogicalEffort(netlist::CellKind::kNor,
                                                    (operand_width + 1) / 2));
      return std::ceil(nor_delay + 1);
    }
    default:
      break;
  }
  return absl::UnimplementedError(
      "Cannot compute delay of node using logical effort:" +
      node->ToStringWithOperandTypes());
}

}  // namespace

DecoratingDelayEstimator::DecoratingDelayEstimator(
    std::string_view name, const DelayEstimator& decorated,
    std::function<int64_t(Node*, int64_t)> modifier)
    : DelayEstimator(name),
      decorated_(decorated),
      modifier_(ABSL_DIE_IF_NULL(std::move(modifier))) {}

absl::StatusOr<int64_t> DecoratingDelayEstimator::GetOperationDelayInPs(
    Node* node) const {
  XLS_ASSIGN_OR_RETURN(int64_t original,
                       decorated_.GetOperationDelayInPs(node));
  return modifier_(node, original);
}

FirstMatchDelayEstimator::FirstMatchDelayEstimator(
    std::string_view name, std::vector<const DelayEstimator*> estimators)
    : DelayEstimator(name), estimators_(std::move(estimators)) {}

absl::StatusOr<int64_t> FirstMatchDelayEstimator::GetOperationDelayInPs(
    Node* node) const {
  absl::StatusOr<int64_t> result;
  for (const DelayEstimator* estimator : estimators_) {
    result = estimator->GetOperationDelayInPs(node);
    if (result.ok()) {
      return result;
    }
  }
  return result;
}

CachingDelayEstimator::CachingDelayEstimator(std::string_view name,
                                             const DelayEstimator& cached)
    : DelayEstimator(name), cached_(cached) {}

absl::StatusOr<int64_t> CachingDelayEstimator::GetOperationDelayInPs(
    Node* node) const {
  if (ContainsNodeDelay(node)) {
    return GetNodeDelay(node);
  }

  XLS_ASSIGN_OR_RETURN(int64_t delay, cached_.GetOperationDelayInPs(node));
  AddNodeDelay(node, delay);
  return delay;
}

/* static */ absl::StatusOr<int64_t> DelayEstimator::GetLogicalEffortDelayInPs(
    Node* node, int64_t tau_in_ps) {
  XLS_ASSIGN_OR_RETURN(int64_t delay_in_tau, GetLogicalEffortDelayInTau(node));
  return delay_in_tau * tau_in_ps;
}

}  // namespace xls
