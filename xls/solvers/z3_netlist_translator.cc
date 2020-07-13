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

#include "xls/solvers/z3_netlist_translator.h"

#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/function_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/solvers/z3_propagate_updates.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

using netlist::function::Ast;
using netlist::rtl::Cell;
using netlist::rtl::Module;
using netlist::rtl::NetRef;

xabsl::StatusOr<std::unique_ptr<NetlistTranslator>>
NetlistTranslator::CreateAndTranslate(
    Z3_context ctx, const Module* module,
    const absl::flat_hash_map<std::string, const Module*>& module_refs,
    const absl::flat_hash_set<std::string>& high_cells) {
  auto translator = absl::WrapUnique(
      new NetlistTranslator(ctx, module, module_refs, high_cells));
  XLS_RETURN_IF_ERROR(translator->Init());
  XLS_RETURN_IF_ERROR(translator->Translate());
  return translator;
}

xabsl::StatusOr<Z3_ast> NetlistTranslator::GetTranslation(NetRef ref) {
  XLS_RET_CHECK(translated_.contains(ref)) << ref->name();
  return translated_.at(ref);
}

NetlistTranslator::NetlistTranslator(
    Z3_context ctx, const Module* module,
    const absl::flat_hash_map<std::string, const Module*>& module_refs,
    const absl::flat_hash_set<std::string>& high_cells)
    : ctx_(ctx),
      module_(module),
      module_refs_(module_refs),
      high_cells_(high_cells) {}

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

bool IsInputToCell(const NetRef netref, const Cell* cell) {
  for (const auto& input : cell->inputs()) {
    if (input.netref == netref) {
      return true;
    }
  }
  return false;
}

bool IsOutputFromCell(const NetRef netref, const Cell* cell) {
  for (const auto& input : cell->outputs()) {
    if (input.netref == netref) {
      return true;
    }
  }
  return false;
}

absl::Status NetlistTranslator::RebindInputNets(
    const absl::flat_hash_map<std::string, Z3_ast>& inputs) {
  std::vector<UpdatedNode<NetRef>> updated_refs;
  for (const auto& input : inputs) {
    XLS_ASSIGN_OR_RETURN(NetRef ref, module_->ResolveNet(input.first));
    Z3_ast old_ast = translated_[ref];
    Z3_ast new_ast = input.second;
    updated_refs.push_back({ref, old_ast, new_ast});
    translated_[ref] = new_ast;
  }

  auto get_inputs = [](NetRef ref) {
    // Find the parent cell of this ref, and collect its inputs as the
    // "inputs" to this ref.
    std::vector<NetRef> inputs;
    for (Cell* cell : ref->connected_cells()) {
      if (!IsOutputFromCell(ref, cell)) {
        continue;
      }

      inputs.reserve(cell->inputs().size());
      for (auto& input : cell->inputs()) {
        inputs.push_back(input.netref);
      }
    }
    return inputs;
  };

  auto get_outputs = [](NetRef ref) {
    // Find all cells that use this node as an output, and collect their
    // outputs.
    std::vector<NetRef> outputs;
    for (Cell* cell : ref->connected_cells()) {
      if (!IsInputToCell(ref, cell)) {
        continue;
      }

      // This cell depends on this ref. Collect all its outputs.
      for (auto& output : cell->outputs()) {
        outputs.push_back(output.netref);
      }
    }
    return outputs;
  };

  PropagateAstUpdates<NetRef>(ctx_, translated_, get_inputs, get_outputs,
                              updated_refs);

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
    XLS_VLOG(2) << "Processing wire " << ref->name();

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
        XLS_VLOG(2) << "Processing cell " << cell->name();
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
      inputs[input.pin_name] = translated_[input.netref];
    }

    const Module* module_ref = module_refs_.at(entry_name);
    XLS_ASSIGN_OR_RETURN(auto subtranslator,
                         NetlistTranslator::CreateAndTranslate(
                             ctx_, module_ref, module_refs_, high_cells_));

    // Now match the module outputs to the corresponding netref in this module's
    // corresponding cell.
    for (const auto& module_output : module_ref->outputs()) {
      XLS_ASSIGN_OR_RETURN(Z3_ast translation,
                           subtranslator->GetTranslation(module_output));
      for (const auto& cell_output : cell.outputs()) {
        if (cell_output.pin.name == module_output->name()) {
          if (translated_.contains(cell_output.netref)) {
            // TODO REMOVE
            XLS_LOG(INFO) << "Skipping translation of "
                          << cell_output.netref->name()
                          << "; already translated.";
          } else {
            translated_[cell_output.netref] = translation;
          }
          break;
        }
      }
    }
    return absl::OkStatus();
  }

  if (high_cells_.contains(cell.cell_library_entry()->name())) {
    // Set each output for the fixed-high cells to 1.
    for (const auto& output : cell.outputs()) {
      translated_[output.netref] = Z3_mk_int(ctx_, 1, Z3_mk_bv_sort(ctx_, 1));
    }
  } else {
    for (const auto& output : cell.outputs()) {
      XLS_ASSIGN_OR_RETURN(Ast ast, netlist::function::Parser::ParseFunction(
                                        output.pin.function));
      XLS_ASSIGN_OR_RETURN(Z3_ast result, TranslateFunction(cell, ast));
      translated_[output.netref] = result;
    }
  }

  return absl::OkStatus();
}

// After all the above, this is the spot where any _ACTUAL_ translation happens.
xabsl::StatusOr<Z3_ast> NetlistTranslator::TranslateFunction(
    const Cell& cell, netlist::function::Ast ast) {
  switch (ast.kind()) {
    case Ast::Kind::kAnd: {
      XLS_ASSIGN_OR_RETURN(Z3_ast lhs,
                           TranslateFunction(cell, ast.children()[0]));
      XLS_ASSIGN_OR_RETURN(Z3_ast rhs,
                           TranslateFunction(cell, ast.children()[1]));
      return Z3_mk_bvand(ctx_, lhs, rhs);
    }
    case Ast::Kind::kIdentifier: {
      NetRef ref = nullptr;
      for (const auto& input : cell.inputs()) {
        if (input.pin_name == ast.name()) {
          ref = input.netref;
          break;
        }
      }
      if (ref == nullptr) {
        return absl::NotFoundError(absl::StrFormat(
            "Identifier \"%s\", was not found in cell %s's inputs.", ast.name(),
            cell.name()));
      }
      return translated_.at(ref);
    }
    case Ast::Kind::kLiteralOne: {
      return Z3_mk_int(ctx_, 1, Z3_mk_bv_sort(ctx_, 1));
    }
    case Ast::Kind::kLiteralZero: {
      return Z3_mk_int(ctx_, 0, Z3_mk_bv_sort(ctx_, 1));
    }
    case Ast::Kind::kNot: {
      XLS_ASSIGN_OR_RETURN(Z3_ast child,
                           TranslateFunction(cell, ast.children()[0]));
      return Z3_mk_bvnot(ctx_, child);
    }
    case Ast::Kind::kOr: {
      XLS_ASSIGN_OR_RETURN(Z3_ast lhs,
                           TranslateFunction(cell, ast.children()[0]));
      XLS_ASSIGN_OR_RETURN(Z3_ast rhs,
                           TranslateFunction(cell, ast.children()[1]));
      return Z3_mk_bvor(ctx_, lhs, rhs);
    }
    case Ast::Kind::kXor: {
      XLS_ASSIGN_OR_RETURN(Z3_ast lhs,
                           TranslateFunction(cell, ast.children()[0]));
      XLS_ASSIGN_OR_RETURN(Z3_ast rhs,
                           TranslateFunction(cell, ast.children()[1]));
      return Z3_mk_bvxor(ctx_, lhs, rhs);
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unknown AST kind: %d", static_cast<int>(ast.kind())));
  }
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
  XLS_CHECK(parent_cell != nullptr) << ref->name() << " has no parent?!";
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
  std::cerr << prefix << value_cone.ref->name() << ": " << std::endl;
  std::cerr << prefix << "Parent: "
            << (value_cone.parent_cell == nullptr
                    ? "<null>"
                    : value_cone.parent_cell->name())
            << std::endl;
  std::cerr << prefix << QueryNode(ctx_, model, value_cone.node, true)
            << std::endl;
  for (const NetlistTranslator::ValueCone& parent : value_cone.parents) {
    PrintValueCone(parent, model, level + 1);
  }
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
