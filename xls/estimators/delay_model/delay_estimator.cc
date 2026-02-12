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

#include "xls/estimators/delay_model/delay_estimator.h"

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
#include "absl/container/flat_hash_map.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
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
      return absl::NotFoundError(absl::StrFormat(
          R"TXT(No delay estimator found named "%s".

No estimators are registered.

This can be caused by a few different issues.

1) Was InitXls called?

   This needs be called early during binary startup or before running other xls
   code.

2) Were estimators linked into the binary?

   Estimators need to be linked in order to be available. The standard
   estimators are linked from '//xls/estimators/delay_model/models'
   and are also exported as '//xls/estimators' and
   '//xls/public:passes_and_estimators'. At least one of these must
   be in the 'deps' tree to ensure that estimators are available.)TXT",
          name));
    }
    return absl::NotFoundError(absl::StrFormat(
        R"TXT(No delay estimator found named \"%s\".

Available estimators: %s

It is possible you didn't link the estimator into the binary.

Do you have '//xls/estimators/delay_model/models' in your dep tree?

Is the estimator library a dependency of '//.../models'?)TXT",
        name, absl::StrJoin(estimator_names_, ", ")));
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
  auto get_linear_logical_effort_for_n_operands =
      [](int64_t operand_count, netlist::CellKind kind,
         bool invert) -> absl::StatusOr<int64_t> {
    XLS_ASSIGN_OR_RETURN(
        double base_effort,
        netlist::logical_effort::GetLogicalEffort(kind, operand_count));
    return std::ceil(invert ? base_effort + int64_t{1} : base_effort);
  };
  auto get_logical_effort = [node](netlist::CellKind kind,
                                   bool invert) -> absl::StatusOr<int64_t> {
    XLS_ASSIGN_OR_RETURN(double base_effort,
                         netlist::logical_effort::GetLogicalEffort(
                             kind, node->operands().size()));
    return std::ceil(invert ? base_effort + int64_t{1} : base_effort);
  };
  // Calculate the reduction logical effort. We use the lower of a binary-tree
  // reduction and a linear reduction.
  //
  // TODO(allight): A binary tree is not actually always the best architecture
  // for a reduction and the correspondence between effort and delay is not
  // always simple. For now a binary-tree is a good approximation however. See
  // chapter 11 of "Sutherland, I., et al., Logical Effort: Designing Fast CMOS
  // Circuits" (and frankly the rest of the book) for more information. For a
  // follow up we can find the optimal branching factors and use them. See
  // chapter 11.2 for more information.
  auto get_reduction_logical_effort_of =
      [](netlist::CellKind kind, bool invert,
         int64_t bit_count) -> absl::StatusOr<int64_t> {
    if (bit_count < 2) {
      return 0;
    }
    XLS_ASSIGN_OR_RETURN(
        double base_effort,
        netlist::logical_effort::GetLogicalEffort(kind, bit_count));
    int64_t effort_raw =
        std::ceil(invert ? base_effort + int64_t{1} : base_effort);
    uint64_t tree_depth = CeilOfLog2(bit_count);
    XLS_ASSIGN_OR_RETURN(double base_bin_tree_effort,
                         netlist::logical_effort::GetLogicalEffort(kind, 2));
    int64_t tree_node_effort_raw = std::ceil(
        invert ? base_bin_tree_effort + int64_t{1} : base_bin_tree_effort);
    int64_t effort_tree = tree_node_effort_raw * tree_depth;
    return std::min(effort_raw, effort_tree);
  };
  auto get_reduction_logical_effort =
      [&](netlist::CellKind kind, bool invert) -> absl::StatusOr<int64_t> {
    const int64_t bit_count = node->operand(0)->BitCountOrDie();
    return get_reduction_logical_effort_of(kind, invert, bit_count);
  };
  switch (node->op()) {
    // TODO(leary): 2019-09-24 Collect real numbers for these.
    case Op::kGate:
      return get_logical_effort(netlist::CellKind::kNand, /*invert=*/true);
    // TODO(allight): The nary ops (And, Nand, Nor, Or, Xor) should all be
    // modeled using a tree structure too. Note that the current tree
    // implementation won't work with nand/nor since they are not associative
    // and care will need to be taken to ensure they are modeled accurately.
    // Currently this models it as though a single cell is always used but this
    // is simply not true (especially as the synthesizer trades off area and
    // delay).
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
    // NB By modeling the reduction ops as a binary tree we are assuming the
    // synthsizer takes the area/delay tradeoff inherent in which cells it
    // chooses in a particular direction. This choice is likely not the optimal
    // one for either area or delay in real process nodes however it is a
    // reasonable approximation.
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
      int64_t bit_count = (operand_width + 1) / 2;
      XLS_ASSIGN_OR_RETURN(int64_t nor_delay, get_reduction_logical_effort_of(
                                                  netlist::CellKind::kNor,
                                                  /*invert=*/true,
                                                  /*bit_count=*/bit_count));
      // The last not isn't needed so remove it.
      return std::ceil(nor_delay - 1);
    }
    case Op::kOneHotSel: {
      // Each bit of the OHS can be computed in parallel without interference.
      // That means the delay of an N-bit OHS is equivalent to the delay of a
      // 1-bit OHS.
      //
      // We model the 1 bit OHS as an OR-reduce of an AND between a bit of the
      // selector and the corresponding case.
      OneHotSelect* ohs = node->As<OneHotSelect>();
      if (ohs->cases().empty()) {
        // How does this even happen? Anyway this is just a constant 0.
        return 0;
      }

      if (ohs->selector()->Is<Literal>()) {
        // This should synthesize down to an OR of the selected inputs.
        int64_t selected_inputs =
            ohs->selector()->As<Literal>()->value().bits().PopCount();
        if (selected_inputs <= 1) {
          return 0;
        }
        XLS_ASSIGN_OR_RETURN(int64_t red, get_reduction_logical_effort_of(
                                              netlist::CellKind::kNor,
                                              /*invert=*/true,
                                              /*bit_count=*/selected_inputs));
        XLS_ASSIGN_OR_RETURN(
            int64_t nor,
            get_linear_logical_effort_for_n_operands(
                /*operand_count=*/selected_inputs, netlist::CellKind::kNor,
                /*invert=*/true));
        return std::min(red, nor);
      }

      if (ohs->cases().size() == 1) {
        // This should just be an AND with the selector.
        return get_linear_logical_effort_for_n_operands(
            /*operand_count=*/2, netlist::CellKind::kNand,
            /*invert=*/true);
      }

      // Can perform 1 bit OHS as
      //
      // Using de Morgan's Law:
      //
      //     OR(AND(C, X), AND(C, Y), ...) =
      //     NOT(AND(NOT(AND(C, X)), NOT(AND(C, Y)), ...)) =
      //     NAND(NAND(C, X), NAND(C, Y), ...)
      //
      // Inner Nand.
      XLS_ASSIGN_OR_RETURN(int64_t inner_nand_delay,
                           get_linear_logical_effort_for_n_operands(
                               2, netlist::CellKind::kNand, /*invert=*/false));
      // Outer nand
      // NB reduction_logical_effort_of returns the minimum of linear and tree
      // logical effort.
      XLS_ASSIGN_OR_RETURN(int64_t outer_nand_red,
                           get_reduction_logical_effort_of(
                               netlist::CellKind::kNand, /*invert=*/false,
                               /*bit_count=*/ohs->cases().size()));
      return outer_nand_red + inner_nand_delay;
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

absl::StatusOr<DelayAnnotator> DelayAnnotator::Create(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  absl::flat_hash_map<Node*, Entry> entries;
  for (Node* node : TopoSort(f)) {
    int64_t max_path_delay = 0;
    for (Node* operand : node->operands()) {
      max_path_delay =
          std::max(max_path_delay, entries.at(operand).path_delay_ps);
    }
    XLS_ASSIGN_OR_RETURN(int64_t node_delay,
                         delay_estimator.GetOperationDelayInPs(node));
    entries[node] = {
        .node_delay_ps = node_delay,
        .path_delay_ps = max_path_delay + node_delay,
    };
  }
  return DelayAnnotator(std::move(entries));
}

Annotation DelayAnnotator::NodeAnnotation(Node* node) const {
  auto it = entries_.find(node);
  if (it == entries_.end()) {
    return Annotation();
  }
  return Annotation{.suffix = absl::StrFormat("[%dps (+%dps)]",
                                              it->second.path_delay_ps,
                                              it->second.node_delay_ps)};
}

}  // namespace xls
