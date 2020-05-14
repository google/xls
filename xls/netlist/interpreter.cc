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
#include "xls/netlist/interpreter.h"

#include "absl/strings/str_format.h"
#include "xls/codegen/flattening.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls {
namespace netlist {

Interpreter::Interpreter(rtl::Netlist* netlist) : netlist_(netlist) {}

xabsl::StatusOr<Bits> Interpreter::InterpretModule(
    const std::string& module_name, const Bits& inputs) {
  // Do a topological sort through all cells, evaluating each as its inputs are
  // fully satisfied, and store those results with each output wire.
  const rtl::Module* module = netlist_->GetModule(module_name);
  XLS_RET_CHECK(module != nullptr);

  XLS_RET_CHECK(module->inputs().size() == inputs.bit_count())
      << absl::StrFormat("Module inputs size: %d, input bit count: %d",
                         module->inputs().size(), inputs.bit_count());

  // Consider each input as a processed wire.
  absl::flat_hash_map<const rtl::NetRef, bool> processed_wires;
  for (int i = 0; i < module->inputs().size(); i++) {
    processed_wires[module->inputs()[i]] = inputs.Get(i);
  }

  using OutputsT = absl::flat_hash_map<const rtl::NetRef, bool>;
  XLS_ASSIGN_OR_RETURN(OutputsT outputs,
                       InterpretModule(module, processed_wires));

  // Finally, collect up all module outputs.
  BitsRope rope(module->outputs().size());
  for (const rtl::NetRef output : module->outputs()) {
    rope.push_back(outputs[output]);
  }

  return rope.Build();
}

xabsl::StatusOr<absl::flat_hash_map<const rtl::NetRef, bool>>
Interpreter::InterpretModule(
    const rtl::Module* module,
    const absl::flat_hash_map<const rtl::NetRef, bool>& inputs) {
  // Do a topological sort through all cells, evaluating each as its inputs are
  // fully satisfied, and store those results with each output wire.

  // First, build up the list of "unsatisfied" cells.
  absl::flat_hash_map<rtl::Cell*, absl::flat_hash_set<rtl::NetRef>> cell_inputs;
  for (const auto& cell : module->cells()) {
    absl::flat_hash_set<rtl::NetRef> inputs;
    for (const auto& input : cell->inputs()) {
      inputs.insert(input.netref);
    }
    cell_inputs[cell.get()] = std::move(inputs);
  }

  // Set all inputs as "active".
  std::deque<rtl::NetRef> active_wires;
  for (const rtl::NetRef ref : module->inputs()) {
    active_wires.push_back(ref);
  }

  absl::flat_hash_map<const rtl::NetRef, bool> processed_wires;
  for (const auto& input : inputs) {
    processed_wires[input.first] = input.second;
  }

  // Process all active wires : see if this wire satisfies all of a cell's
  // inputs. If so, interpret the cell, and place its outputs on the active wire
  // list.
  while (!active_wires.empty()) {
    rtl::NetRef wire = active_wires.front();
    active_wires.pop_front();
    XLS_VLOG(2) << "Processing wire: " << wire->name();

    for (const auto& cell : wire->connected_cells()) {
      if (IsCellOutput(*cell, wire)) {
        continue;
      }

      cell_inputs[cell].erase(wire);
      if (cell_inputs[cell].empty()) {
        XLS_VLOG(2) << "Processing cell: " << cell->name();
        XLS_RETURN_IF_ERROR(InterpretCell(*cell, &processed_wires));
        for (const auto& output : cell->outputs()) {
          active_wires.push_back(output.netref);
        }
      }
    }
  }

  // Sanity check that we've processed all cells (i.e., that there aren't
  // unsatisfiable cells).
  for (const auto& cell : module->cells()) {
    for (const auto& output : cell->outputs()) {
      if (!processed_wires.contains(output.netref)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Netlist contains unconnected subgraphs and cannot be translated. "
            "Example: cell %s, output %s.",
            cell->name(), output.netref->name()));
      }
    }
  }

  absl::flat_hash_map<const rtl::NetRef, bool> outputs;
  outputs.reserve(module->outputs().size());
  for (const rtl::NetRef output : module->outputs()) {
    outputs[output] = processed_wires.at(output);
  }

  return outputs;
}

absl::Status Interpreter::InterpretCell(
    const rtl::Cell& cell,
    absl::flat_hash_map<const rtl::NetRef, bool>* processed_wires) {
  const rtl::Module* module =
      netlist_->GetModule(cell.cell_library_entry()->name());
  if (module != nullptr) {
    // If this "cell" is actually a module defined in the netlist,
    // then recursively evaluate it.
    absl::flat_hash_map<const rtl::NetRef, bool> inputs;
    // who's input/output name - needs to be internal
    // need to map cell inputs to module inputs?
    const std::vector<rtl::NetRef>& module_input_refs = module->inputs();
    const absl::Span<const std::string> module_input_names =
        module->AsCellLibraryEntry()->input_names();

    for (const auto& input : cell.inputs()) {
      // We need to match the inputs - from the NetRefs in this module to the
      // NetRefs in the child module. In Module, the order of inputs
      // (as NetRefs) is the same as the input names in its CellLibraryEntry.
      // That means, for each input (in this module):
      //  - Find the child module input pin/NetRef with the same name.
      //  - Assign the corresponding child module input NetRef to have the value
      //    of the wire in this module.
      // If ever an input isn't found, that's bad. Abort.
      bool input_found = false;
      for (int i = 0; i < module_input_names.size(); i++) {
        if (module_input_names[i] == input.pin_name) {
          inputs[module_input_refs[i]] = processed_wires->at(input.netref);
          input_found = true;
          break;
        }
      }

      XLS_RET_CHECK(input_found) << absl::StrFormat(
          "Could not find input pin \"%s\" in module \"%s\", referenced in "
          "cell \"%s\"!",
          input.pin_name, module->name(), cell.name());
    }

    using ChildOutputsT = absl::flat_hash_map<const rtl::NetRef, bool>;
    XLS_ASSIGN_OR_RETURN(ChildOutputsT child_outputs,
                         InterpretModule(module, inputs));
    // We need to do the same here - map the NetRefs in the module's output
    // to the NetRefs in this module, using pin names as the matching keys.
    for (const auto& child_output : child_outputs) {
      bool output_found = false;
      for (const auto& cell_output : cell.outputs()) {
        if (child_output.first->name() == cell_output.pin.name) {
          (*processed_wires)[cell_output.netref] = child_output.second;
          output_found = true;
          break;
        }
      }
      XLS_RET_CHECK(output_found);
      XLS_RET_CHECK(output_found) << absl::StrFormat(
          "Could not find cell output pin \"%s\" in cell \"%s\", referenced in "
          "child module \"%s\"!",
          child_output.first->name(), cell.name(), module->name());
    }

    return absl::OkStatus();
  }

  for (const rtl::Cell::Output& output : cell.outputs()) {
    XLS_ASSIGN_OR_RETURN(
        function::Ast ast,
        function::Parser::ParseFunction(
            cell.cell_library_entry()->output_pins()[0].function));
    XLS_ASSIGN_OR_RETURN(bool result,
                         InterpretFunction(cell, ast, *processed_wires));
    (*processed_wires)[output.netref] = result;
  }

  return absl::OkStatus();
}

xabsl::StatusOr<bool> Interpreter::InterpretFunction(
    const rtl::Cell& cell, const function::Ast& ast,
    const absl::flat_hash_map<const rtl::NetRef, bool>& processed_wires) {
  switch (ast.kind()) {
    case function::Ast::Kind::kAnd: {
      XLS_ASSIGN_OR_RETURN(bool lhs, InterpretFunction(cell, ast.children()[0],
                                                       processed_wires));
      XLS_ASSIGN_OR_RETURN(bool rhs, InterpretFunction(cell, ast.children()[1],
                                                       processed_wires));
      return lhs & rhs;
    }
    case function::Ast::Kind::kIdentifier: {
      rtl::NetRef ref = nullptr;
      for (const auto& input : cell.inputs()) {
        if (input.pin_name == ast.name()) {
          ref = input.netref;
        }
      }

      if (ref == nullptr) {
        return absl::NotFoundError(
            absl::StrFormat("Identifier \"%s\" not found in cell %s's inputs.",
                            ast.name(), cell.name()));
      }

      return processed_wires.at(ref);
    }
    case function::Ast::Kind::kLiteralOne:
      return 1;
    case function::Ast::Kind::kLiteralZero:
      return 0;
    case function::Ast::Kind::kNot: {
      XLS_ASSIGN_OR_RETURN(
          bool value,
          InterpretFunction(cell, ast.children()[0], processed_wires));
      return !value;
    }
    case function::Ast::Kind::kOr: {
      XLS_ASSIGN_OR_RETURN(bool lhs, InterpretFunction(cell, ast.children()[0],
                                                       processed_wires));
      XLS_ASSIGN_OR_RETURN(bool rhs, InterpretFunction(cell, ast.children()[1],
                                                       processed_wires));
      return lhs | rhs;
    }
    case function::Ast::Kind::kXor: {
      XLS_ASSIGN_OR_RETURN(bool lhs, InterpretFunction(cell, ast.children()[0],
                                                       processed_wires));
      XLS_ASSIGN_OR_RETURN(bool rhs, InterpretFunction(cell, ast.children()[1],
                                                       processed_wires));
      return lhs ^ rhs;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown AST element type: ", ast.kind()));
  }
}

bool Interpreter::IsCellOutput(const rtl::Cell& cell, const rtl::NetRef ref) {
  for (const auto& output : cell.outputs()) {
    if (ref == output.netref) {
      return true;
    }
  }

  return false;
}

}  // namespace netlist
}  // namespace xls
