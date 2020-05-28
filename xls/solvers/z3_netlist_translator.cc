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
  XLS_RET_CHECK(translated_.contains(ref));
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

bool IsCellInput(const netlist::rtl::NetRef netref, const Cell* cell) {
  for (const auto& input : cell->inputs()) {
    if (input.netref == netref) {
      return true;
    }
  }
  return false;
}

absl::Status NetlistTranslator::RebindInputNet(
    const std::string& ref_name, Z3_ast dst,
    absl::flat_hash_set<Cell*> cells_to_consider) {
  // Overall approach:
  //  1. Find the NetRef/Def matching "src".
  //  2. For every cell with that as an input, replace that input with "dst".
  //  3. Fame & fortune.
  XLS_ASSIGN_OR_RETURN(netlist::rtl::NetRef src_ref,
                       module_->ResolveNet(ref_name));
  Z3_ast src = translated_[src_ref];
  translated_[src_ref] = dst;

  // For every cell that uses src_ref as an input, update its output wires to
  // use the new Z3 node instead.
  std::vector<UpdatedRef> updated_refs;
  for (Cell* cell : src_ref->connected_cells()) {
    if (!IsCellInput(src_ref, cell)) {
      continue;
    }

    if (!cells_to_consider.empty() && !cells_to_consider.contains(cell)) {
      continue;
    }

    // With a new input, we now need to update all using cells. We need to hold
    // the result of that update (in updated_refs), so we can propagate that
    // change down the rest of the tree.
    for (const auto& output : cell->outputs()) {
      translated_[output.netref] =
          Z3_substitute(ctx_, translated_[output.netref], 1, &src, &dst);
      updated_refs.push_back({output.netref, src, dst});
    }
  }

  PropagateAstUpdate(updated_refs);

  return absl::OkStatus();
}

NetlistTranslator::AffectedCells NetlistTranslator::GetAffectedCells(
    const std::vector<UpdatedRef>& input_refs) {
  AffectedCells affected_cells;

  // The NetRefs that need to be updated - those downstream of the input refs.
  std::deque<netlist::rtl::NetRef> affected_refs;
  for (const auto& ref : input_refs) {
    affected_refs.push_back(ref.netref);
  }

  absl::flat_hash_set<const Cell*> seen_cells;
  while (!affected_refs.empty()) {
    netlist::rtl::NetRef ref = affected_refs.front();
    affected_refs.pop_front();

    // Each cell with an updated ref as its input will need all of its outputs
    // to be updated.
    for (const Cell* cell : ref->connected_cells()) {
      if (!IsCellInput(ref, cell)) {
        continue;
      }

      affected_cells[cell].insert(ref);
      if (!seen_cells.contains(cell)) {
        seen_cells.insert(cell);
        for (const auto& output : cell->outputs()) {
          affected_refs.push_back(output.netref);
        }
      }
    }
  }

  return affected_cells;
}

void NetlistTranslator::PropagateAstUpdate(
    const std::vector<UpdatedRef>& input_refs) {
  // Get the list of affected cells and the refs that need to be updated.
  AffectedCells affected_cells = GetAffectedCells(input_refs);

  // At this point, we have the list of cells needing updating, along with the
  // updated refs they're waiting on. Now "activate" the top-level updated refs,
  // and let the updates propagate.
  // We can't combine this with the loop in GetAffectedCells() because we don't
  // know, in any particular iteration, the entire set of affected wires.
  std::deque<netlist::rtl::NetRef> active_refs;
  absl::flat_hash_map<netlist::rtl::NetRef, UpdatedRef> updated_refs;
  for (UpdatedRef input_ref : input_refs) {
    active_refs.push_back(input_ref.netref);
    updated_refs.insert({input_ref.netref, {input_ref}});
  }

  while (!active_refs.empty()) {
    netlist::rtl::NetRef ref = active_refs.front();
    active_refs.pop_front();
    for (auto& pair : affected_cells) {
      if (!pair.second.contains(ref)) {
        continue;
      }

      const Cell* cell = pair.first;
      pair.second.erase(ref);
      if (pair.second.empty()) {
        // We replace all updated references at once - that means we need to
        // collect all old/new Z3_ast pairs.
        std::vector<Z3_ast> old_inputs;
        std::vector<Z3_ast> new_inputs;
        for (const auto& input : cell->inputs()) {
          if (updated_refs.contains(input.netref)) {
            old_inputs.push_back(updated_refs[input.netref].old_ast);
            new_inputs.push_back(updated_refs[input.netref].new_ast);
          }
        }

        for (const auto& output : cell->outputs()) {
          Z3_ast old_ast = translated_[output.netref];
          Z3_ast new_ast = Z3_substitute(ctx_, old_ast, old_inputs.size(),
                                         old_inputs.data(), new_inputs.data());
          updated_refs[output.netref] = {output.netref, old_ast, new_ast};
          translated_[output.netref] = new_ast;
          active_refs.push_back(output.netref);
        }
      }
    }
  }
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

  // Sanity check that we've processed all cells (i.e., that there aren't
  // unsatisfiable cells).
  for (const auto& cell : module_->cells()) {
    for (const auto& output : cell->outputs()) {
      if (!translated_.contains(output.netref)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Netlist contains unconnected subgraphs and cannot be translated. "
            "Example: cell %s, output %s.",
            cell->name(), output.netref->name()));
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
          translated_[cell_output.netref] = translation;
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

}  // namespace z3
}  // namespace solvers
}  // namespace xls
