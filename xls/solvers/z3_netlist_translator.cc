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

#include "xls/solvers/z3_netlist_translator.h"

#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/solvers/z3_utils.h"
#include "external/z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

using netlist::CellLibraryEntry;
using netlist::StateTable;
using netlist::StateTableSignal;
using netlist::function::Ast;
using netlist::rtl::Cell;
using netlist::rtl::Module;
using netlist::rtl::NetRef;

absl::StatusOr<std::unique_ptr<NetlistTranslator>>
NetlistTranslator::CreateAndTranslate(
    Z3_context ctx, const Module* module,
    const absl::flat_hash_map<std::string, const Module*>& module_refs) {
  auto translator =
      absl::WrapUnique(new NetlistTranslator(ctx, module, module_refs));
  XLS_RETURN_IF_ERROR(translator->Init());
  XLS_RETURN_IF_ERROR(translator->Translate());
  return translator;
}

absl::StatusOr<Z3_ast> NetlistTranslator::GetTranslation(NetRef ref) {
  XLS_RET_CHECK(translated_.contains(ref)) << ref->name();
  return translated_.at(ref);
}

NetlistTranslator::NetlistTranslator(
    Z3_context ctx, const Module* module,
    const absl::flat_hash_map<std::string, const Module*>& module_refs)
    : ctx_(ctx), module_(module), module_refs_(module_refs) {}

absl::Status NetlistTranslator::Init() {
  // Create a symbolic constant for each module input and make it available for
  // downstream nodes.
  for (const NetRef& input : module_->inputs()) {
    translated_[input] =
        Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, input->name().c_str()),
                    Z3_mk_bv_sort(ctx_, 1));
  }

  // We only have to syntheize "clk" and "input_valid" symbols for XLS-generated
  // modules; they're automatically added in the translation between DSLX and
  // the netlist.
  Z3_sort sort = Z3_mk_bv_sort(ctx_, 1);
  auto status_or_clk = module_->ResolveNet("clk");
  if (status_or_clk.ok()) {
    translated_[status_or_clk.value()] = Z3_mk_int(ctx_, 1, sort);
  }
  auto status_or_input_valid = module_->ResolveNet("input_valid");
  if (status_or_input_valid.ok()) {
    translated_[status_or_input_valid.value()] = Z3_mk_int(ctx_, 1, sort);
  }

  translated_[module_->ResolveNumber(0).value()] = Z3_mk_int(ctx_, 0, sort);
  translated_[module_->ResolveNumber(1).value()] = Z3_mk_int(ctx_, 1, sort);

  return absl::OkStatus();
}

absl::Status NetlistTranslator::Retranslate(
    const absl::flat_hash_map<std::string, Z3_ast>& new_inputs) {
  translated_.clear();
  for (const auto& pair : new_inputs) {
    XLS_ASSIGN_OR_RETURN(const NetRef ref, module_->ResolveNet(pair.first));
    translated_[ref] = pair.second;
  }
  Z3_sort bit_sort = Z3_mk_bv_sort(ctx_, 1);
  auto status_or_clk = module_->ResolveNet("clk");
  if (status_or_clk.ok()) {
    translated_[status_or_clk.value()] = Z3_mk_int(ctx_, 1, bit_sort);
  }
  auto status_or_input_valid = module_->ResolveNet("input_valid");
  if (status_or_input_valid.ok()) {
    translated_[status_or_input_valid.value()] = Z3_mk_int(ctx_, 1, bit_sort);
  }
  translated_[module_->ResolveNumber(0).value()] = Z3_mk_int(ctx_, 0, bit_sort);
  translated_[module_->ResolveNumber(1).value()] = Z3_mk_int(ctx_, 1, bit_sort);
  return Translate();
}

// General idea: construct an AST by iterating over all the Cells in the module.
//  1. First, collect all input wires and put them on an "active" list.
//  2. Iterate through the active wire list and examine all cells they're
//     connected to, removing each as examined.
//  3. For any cell for which all inputs have been seen (are "active"),
//     translate that cell into Z3 space, and move its output wires
//     (the resulting Z3 nodes) onto the back of the active wire list.
//  4. Repeat until the active wire list is empty.
absl::Status NetlistTranslator::Translate() {
  // Utility structure so we don't have to iterate through a cell's inputs and
  // outputs every time it's examined.
  absl::flat_hash_map<Cell*, absl::flat_hash_set<NetRef>> cell_inputs;
  std::deque<NetRef> active_wires;
  for (const auto& cell : module_->cells()) {
    // If any cells have _no_ inputs, then their outputs should be made
    // immediately available.
    if (cell->inputs().empty()) {
      XLS_RETURN_IF_ERROR(TranslateCell(*cell));
      for (const auto& output : cell->outputs()) {
        active_wires.push_back(output.netref);
      }
    } else {
      absl::flat_hash_set<NetRef> inputs;
      for (const auto& input : cell->inputs()) {
        inputs.insert(input.netref);
      }
      cell_inputs[cell.get()] = std::move(inputs);
    }
  }

  // Remember - we pre-populated translated_ with the set of module inputs.
  for (const auto& pair : translated_) {
    active_wires.push_back(pair.first);
  }

  // For every active wire, check to see if all of its inputs are satisfied.
  // If so, then that cell's outputs are now active and should be considered
  // newly active on the next pass.
  while (!active_wires.empty()) {
    NetRef ref = active_wires.front();
    active_wires.pop_front();
    VLOG(2) << "Processing wire " << ref->name();

    // Check every connected cell to see if all of its inputs are now
    // available.
    for (auto& cell : ref->connected_cells()) {
      // Skip if this cell if the wire is its output!
      bool is_output = false;
      for (const auto& output : cell->outputs()) {
        if (output.netref == ref) {
          is_output = true;
          break;
        }
      }
      if (is_output) {
        continue;
      }

      cell_inputs[cell].erase(ref);
      if (cell_inputs[cell].empty()) {
        VLOG(2) << "Processing cell " << cell->name();
        XLS_RETURN_IF_ERROR(TranslateCell(*cell));

        for (const auto& output : cell->outputs()) {
          active_wires.push_back(output.netref);
        }
      }
    }
  }

  return absl::Status();
}

absl::Status NetlistTranslator::TranslateCell(const Cell& cell) {
  using netlist::function::Ast;

  // If this cell is actually a reference to a module defined in this netlist,
  // then translate it into Z3-space here and grab its output nodes.
  std::string entry_name = cell.cell_library_entry()->name();
  if (module_refs_.contains(entry_name)) {
    absl::flat_hash_map<std::string, Z3_ast> inputs;
    for (const auto& input : cell.inputs()) {
      inputs[input.name] = translated_[input.netref];
    }

    const Module* module_ref = module_refs_.at(entry_name);
    XLS_ASSIGN_OR_RETURN(
        auto subtranslator,
        NetlistTranslator::CreateAndTranslate(ctx_, module_ref, module_refs_));

    // Now match the module outputs to the corresponding netref in this module's
    // corresponding cell.
    for (const auto& module_output : module_ref->outputs()) {
      XLS_ASSIGN_OR_RETURN(Z3_ast translation,
                           subtranslator->GetTranslation(module_output));
      for (const auto& cell_output : cell.outputs()) {
        if (cell_output.name == module_output->name()) {
          if (translated_.contains(cell_output.netref)) {
            LOG(INFO) << "Skipping translation of "
                      << cell_output.netref->name() << "; already translated.";
          } else {
            translated_[cell_output.netref] = translation;
          }
          break;
        }
      }
    }
    return absl::OkStatus();
  }

  const CellLibraryEntry* entry = cell.cell_library_entry();
  absl::flat_hash_map<std::string, Z3_ast> state_table_values;
  if (entry->state_table()) {
    XLS_ASSIGN_OR_RETURN(state_table_values, TranslateStateTable(cell));
  }

  const CellLibraryEntry::OutputPinToFunction& pins =
      entry->output_pin_to_function();
  for (const auto& output : cell.outputs()) {
    XLS_ASSIGN_OR_RETURN(Ast ast, netlist::function::Parser::ParseFunction(
                                      pins.at(output.name)));
    XLS_ASSIGN_OR_RETURN(Z3_ast result,
                         TranslateFunction(cell, ast, state_table_values));
    translated_[output.netref] = result;
  }

  return absl::OkStatus();
}

// After all the above, this is the spot where any _ACTUAL_ translation happens.
absl::StatusOr<Z3_ast> NetlistTranslator::TranslateFunction(
    const Cell& cell, netlist::function::Ast ast,
    const absl::flat_hash_map<std::string, Z3_ast>& state_table_values) {
  switch (ast.kind()) {
    case Ast::Kind::kAnd: {
      XLS_ASSIGN_OR_RETURN(
          Z3_ast lhs,
          TranslateFunction(cell, ast.children()[0], state_table_values));
      XLS_ASSIGN_OR_RETURN(
          Z3_ast rhs,
          TranslateFunction(cell, ast.children()[1], state_table_values));
      return Z3_mk_bvand(ctx_, lhs, rhs);
    }
    case Ast::Kind::kIdentifier: {
      for (const auto& input : cell.inputs()) {
        if (input.name == ast.name()) {
          return translated_.at(input.netref);
        }
      }

      if (state_table_values.contains(ast.name())) {
        return state_table_values.at(ast.name());
      }

      return absl::NotFoundError(absl::StrFormat(
          "Identifier \"%s\", was not found in cell %s's inputs.", ast.name(),
          cell.name()));
    }
    case Ast::Kind::kLiteralOne: {
      return Z3_mk_int(ctx_, 1, Z3_mk_bv_sort(ctx_, 1));
    }
    case Ast::Kind::kLiteralZero: {
      return Z3_mk_int(ctx_, 0, Z3_mk_bv_sort(ctx_, 1));
    }
    case Ast::Kind::kNot: {
      XLS_ASSIGN_OR_RETURN(
          Z3_ast child,
          TranslateFunction(cell, ast.children()[0], state_table_values));
      return Z3_mk_bvnot(ctx_, child);
    }
    case Ast::Kind::kOr: {
      XLS_ASSIGN_OR_RETURN(
          Z3_ast lhs,
          TranslateFunction(cell, ast.children()[0], state_table_values));
      XLS_ASSIGN_OR_RETURN(
          Z3_ast rhs,
          TranslateFunction(cell, ast.children()[1], state_table_values));
      return Z3_mk_bvor(ctx_, lhs, rhs);
    }
    case Ast::Kind::kXor: {
      XLS_ASSIGN_OR_RETURN(
          Z3_ast lhs,
          TranslateFunction(cell, ast.children()[0], state_table_values));
      XLS_ASSIGN_OR_RETURN(
          Z3_ast rhs,
          TranslateFunction(cell, ast.children()[1], state_table_values));
      return Z3_mk_bvxor(ctx_, lhs, rhs);
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unknown AST kind: %d", static_cast<int>(ast.kind())));
  }
}

absl::StatusOr<absl::flat_hash_map<std::string, Z3_ast>>
NetlistTranslator::TranslateStateTable(const Cell& cell) {
  const StateTable& table = cell.cell_library_entry()->state_table().value();

  auto get_pin_netref =
      [](const absl::Span<const Cell::Pin>& pins,
         const std::string& pin_name) -> absl::StatusOr<NetRef> {
    for (const auto& pin : pins) {
      if (pin.name == pin_name) {
        return pin.netref;
      }
    }
    return absl::NotFoundError(absl::StrCat("Couldn't find pin: ", pin_name));
  };

  Z3_ast one = Z3_mk_int(ctx_, 1, Z3_mk_bv_sort(ctx_, 1));
  Z3_ast zero = Z3_mk_int(ctx_, 0, Z3_mk_bv_sort(ctx_, 1));
  // Simple pair of a combined stimulus value to an output value; essentially
  // one case in the state table for a given output pin and stimulus.
  struct OutputCase {
    Z3_ast combined_stimulus;
    Z3_ast output_value;
  };
  absl::flat_hash_map<std::string, std::vector<OutputCase>> output_table;

  for (const StateTable::Row& row : table.rows()) {
    // For each row, create a single Z3_ast combining all stimulus values.
    std::vector<Z3_ast> stimulus;
    for (const auto& kv : row.stimulus) {
      const std::string& input_name = kv.first;
      const StateTableSignal signal = kv.second;
      if (signal != StateTableSignal::kHigh &&
          signal != StateTableSignal::kLow) {
        VLOG(1) << "Non-high or -low input signal encountered: " << cell.name()
                << ":" << input_name << ": " << static_cast<int>(signal);
        continue;
      }

      XLS_ASSIGN_OR_RETURN(NetRef input_ref,
                           get_pin_netref(cell.inputs(), input_name));
      XLS_RET_CHECK(translated_.contains(input_ref));
      stimulus.push_back(
          Z3_mk_eq(ctx_, translated_[input_ref],
                   signal == StateTableSignal::kHigh ? one : zero));
    }

    Z3_ast combined_stimulus =
        Z3_mk_and(ctx_, stimulus.size(), stimulus.data());

    for (const auto& kv : row.response) {
      // Then assign the value of the signals in the output row.
      const std::string& output_name = kv.first;
      const StateTableSignal& signal = kv.second;

      if (signal != StateTableSignal::kHigh &&
          signal != StateTableSignal::kLow) {
        LOG(WARNING) << "Non-high or -low output signal encountered: "
                     << cell.name() << ":" << output_name << ": "
                     << static_cast<int>(signal);
        continue;
      }

      output_table[output_name].push_back(OutputCase{
          combined_stimulus, signal == StateTableSignal::kHigh ? one : zero});
    }
  }

  // Iterate through each output ref, assigning the final value as an
  // if-then-else chain.
  absl::flat_hash_map<std::string, Z3_ast> final_values;
  for (const auto& kv : output_table) {
    const std::string& pin_name = kv.first;
    const std::vector<OutputCase>& output_table = kv.second;

    // Need to go backwards for proper handling of "else".
    Z3_ast prev_case = output_table.back().output_value;
    for (int i = output_table.size() - 2; i >= 0; i--) {
      prev_case = Z3_mk_ite(ctx_, output_table[i].combined_stimulus,
                            output_table[i].output_value, prev_case);
    }

    // Finally, assign the root if-then-else to the ref.
    final_values[pin_name] = prev_case;
  }

  return final_values;
}

NetlistTranslator::ValueCone NetlistTranslator::GetValueCone(
    NetRef ref, const absl::flat_hash_set<Z3_ast>& terminals) {
  // Could also check for strings?
  if (terminals.contains(translated_[ref])) {
    return {translated_[ref], ref, {}};
  }

  const Cell* parent_cell = nullptr;
  for (const Cell* cell : ref->connected_cells()) {
    for (const auto& output : cell->outputs()) {
      if (output.netref == ref) {
        parent_cell = cell;
        break;
      }
    }
    if (parent_cell != nullptr) {
      break;
    }
  }

  ValueCone value_cone;
  value_cone.node = translated_[ref];
  value_cone.ref = ref;
  value_cone.parent_cell = parent_cell;
  CHECK(parent_cell != nullptr) << ref->name() << " has no parent?!";
  for (const auto& input : parent_cell->inputs()) {
    // Ick
    if (input.netref->name() == "input_valid" ||
        input.netref->name() == "clk" ||
        input.netref->name() == "<constant_0>" ||
        input.netref->name() == "<constant_1>") {
      continue;
    }

    value_cone.parents.push_back(GetValueCone(input.netref, terminals));
  }

  return value_cone;
}

void NetlistTranslator::PrintValueCone(const ValueCone& value_cone,
                                       Z3_model model, int level) {
  std::string prefix(level * 2, ' ');
  std::cerr << prefix << value_cone.ref->name() << ": " << '\n';
  std::cerr << prefix << "Parent: "
            << (value_cone.parent_cell == nullptr
                    ? "<null>"
                    : value_cone.parent_cell->name())
            << '\n';
  std::cerr << prefix << QueryNode(ctx_, model, value_cone.node, true) << '\n';
  for (const NetlistTranslator::ValueCone& parent : value_cone.parents) {
    PrintValueCone(parent, model, level + 1);
  }
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
