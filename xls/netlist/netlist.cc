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

#include "xls/netlist/netlist.h"

#include <variant>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/cell_library.h"

namespace xls {
namespace netlist {
namespace rtl {

const CellLibraryEntry* Module::AsCellLibraryEntry() const {
  if (!cell_library_entry_.has_value()) {
    std::vector<std::string> input_names;
    input_names.reserve(inputs_.size());
    for (const auto& input : inputs_) {
      input_names.push_back(input->name());
    }
    CellLibraryEntry::SimplePins output_pins;
    output_pins.reserve(outputs_.size());
    for (const auto& output : outputs_) {
      output_pins[output->name()] = "";
    }
    cell_library_entry_.emplace(CellLibraryEntry(
        CellKind::kOther, name_, input_names, output_pins, absl::nullopt));
  }
  return &cell_library_entry_.value();
}

xabsl::StatusOr<NetRef> Module::AddOrResolveNumber(int64 number) {
  auto status_or_ref = ResolveNumber(number);
  if (status_or_ref.ok()) {
    return status_or_ref.value();
  }

  std::string wire_name = absl::StrFormat("<constant_%d>", number);
  XLS_RETURN_IF_ERROR(AddNetDecl(NetDeclKind::kWire, wire_name));
  return ResolveNet(wire_name);
}

xabsl::StatusOr<NetRef> Module::ResolveNumber(int64 number) const {
  std::string wire_name = absl::StrFormat("<constant_%d>", number);
  return ResolveNet(wire_name);
}

xabsl::StatusOr<NetRef> Module::ResolveNet(absl::string_view name) const {
  for (const auto& net : nets_) {
    if (net->name() == name) {
      return net.get();
    }
  }

  return absl::NotFoundError(absl::StrCat("Could not find net: ", name));
}

xabsl::StatusOr<Cell*> Module::ResolveCell(absl::string_view name) const {
  for (const auto& cell : cells_) {
    if (cell->name() == name) {
      return cell.get();
    }
  }
  return absl::NotFoundError(
      absl::StrCat("Could not find cell with name: ", name));
}

xabsl::StatusOr<Cell*> Module::AddCell(Cell cell) {
  auto status_or_cell = ResolveCell(cell.name());
  if (status_or_cell.status().ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module already has a cell with name: ", cell.name()));
  }

  cells_.push_back(absl::make_unique<Cell>(cell));
  return cells_.back().get();
}

absl::Status Module::AddNetDecl(NetDeclKind kind, absl::string_view name) {
  auto status_or_net = ResolveNet(name);
  if (status_or_net.status().ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module already has a net/wire decl with name: ", name));
  }

  nets_.emplace_back(absl::make_unique<NetDef>(name));
  NetRef ref = nets_.back().get();
  switch (kind) {
    case NetDeclKind::kInput:
      inputs_.push_back(ref);
      break;
    case NetDeclKind::kOutput:
      outputs_.push_back(ref);
      break;
    case NetDeclKind::kWire:
      wires_.push_back(ref);
      break;
  }
  return absl::OkStatus();
}

xabsl::StatusOr<std::vector<Cell*>> NetDef::GetConnectedCellsSans(
    Cell* to_remove) const {
  std::vector<Cell*> new_cells;
  new_cells.reserve(connected_cells_.size() - 1);
  bool found = false;
  for (int i = 0; i < connected_cells_.size(); i++) {
    if (connected_cells_[i] == to_remove) {
      found = true;
    } else {
      new_cells.push_back(connected_cells_[i]);
    }
  }

  if (!found) {
    return absl::NotFoundError("Could not find cell in connected cell set: " +
                               to_remove->name());
  }
  return new_cells;
}

/* static */ xabsl::StatusOr<Cell> Cell::Create(
    const CellLibraryEntry* cell_library_entry, absl::string_view name,
    const absl::flat_hash_map<std::string, NetRef>& named_parameter_assignments,
    absl::optional<NetRef> clock, const NetRef dummy_net) {
  auto sorted_key_str = [named_parameter_assignments]() -> std::string {
    std::vector<std::string> keys;
    for (const auto& item : named_parameter_assignments) {
      keys.push_back(item.first);
    }
    std::sort(keys.begin(), keys.end());
    return "[" + absl::StrJoin(keys, ", ") + "]";
  };

  std::vector<Pin> cell_inputs;
  for (const std::string& input : cell_library_entry->input_names()) {
    auto it = named_parameter_assignments.find(input);
    if (it == named_parameter_assignments.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Missing named input parameter in instantiation: %s; got: %s", input,
          sorted_key_str()));
    }
    Pin cell_input;
    cell_input.name = input;
    cell_input.netref = it->second;
    cell_inputs.push_back(cell_input);
  }

  const CellLibraryEntry::OperationT& operation =
      cell_library_entry->operation();
  std::vector<Pin> cell_outputs;
  if (std::holds_alternative<CellLibraryEntry::SimplePins>(operation)) {
    const auto& output_pins = std::get<CellLibraryEntry::SimplePins>(operation);
    for (const auto& kv : output_pins) {
      Pin cell_output;
      cell_output.name = kv.first;
      auto it = named_parameter_assignments.find(cell_output.name);
      if (it == named_parameter_assignments.end()) {
        cell_output.netref = dummy_net;
      } else {
        cell_output.netref = it->second;
      }
      cell_outputs.push_back(cell_output);
    }
  } else {
    const StateTable& state_table = std::get<StateTable>(operation);
    for (const std::string& signal_name : state_table.output_signals()) {
      Pin cell_output;
      cell_output.name = signal_name;
      auto it = named_parameter_assignments.find(signal_name);
      if (it == named_parameter_assignments.end()) {
        cell_output.netref = dummy_net;
      } else {
        cell_output.netref = it->second;
      }
      cell_outputs.push_back(cell_output);
    }
  }

  if (cell_library_entry->clock_name().has_value() && !clock.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Missing clock parameter %s in instantiation; got inputs: %s.",
        cell_library_entry->clock_name().value(), sorted_key_str()));
  }

  return Cell(cell_library_entry, name, std::move(cell_inputs),
              std::move(cell_outputs), clock);
}

void Netlist::AddModule(std::unique_ptr<Module> module) {
  modules_.emplace_back(std::move(module));
}

xabsl::StatusOr<const Module*> Netlist::GetModule(
    const std::string& module_name) const {
  for (const auto& module : modules_) {
    if (module->name() == module_name) {
      return module.get();
    }
  }
  return absl::NotFoundError(
      absl::StrFormat("Module %s not found in netlist.", module_name));
}

}  // namespace rtl
}  // namespace netlist
}  // namespace xls
