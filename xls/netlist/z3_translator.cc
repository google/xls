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

#include "xls/netlist/z3_translator.h"

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
namespace netlist {

using function::Ast;

xabsl::StatusOr<std::unique_ptr<Z3Translator>> Z3Translator::CreateAndTranslate(
    Z3_context ctx, const rtl::Module* module,
    const absl::flat_hash_map<std::string, const rtl::Module*>& module_refs,
    const absl::flat_hash_map<std::string, Z3_ast>& inputs,
    const absl::flat_hash_set<std::string>& high_cells) {
  auto translator =
      absl::WrapUnique(new Z3Translator(ctx, module, module_refs, high_cells));
  XLS_RETURN_IF_ERROR(translator->Init(inputs));
  XLS_RETURN_IF_ERROR(translator->Translate());
  return translator;
}

xabsl::StatusOr<Z3_ast> Z3Translator::GetTranslation(rtl::NetRef ref) {
  XLS_RET_CHECK(translated_.contains(ref));
  return translated_.at(ref);
}

Z3Translator::Z3Translator(
    Z3_context ctx, const rtl::Module* module,
    const absl::flat_hash_map<std::string, const rtl::Module*>& module_refs,
    const absl::flat_hash_set<std::string>& high_cells)
    : ctx_(ctx),
      module_(module),
      module_refs_(module_refs),
      high_cells_(high_cells) {}

absl::Status Z3Translator::Init(
    const absl::flat_hash_map<std::string, Z3_ast>& inputs) {
  // Associate each input with its NetRef and make it available for lookup.
  for (const auto& pair : inputs) {
    XLS_VLOG(2) << "Processing input : " << pair.first;
    XLS_ASSIGN_OR_RETURN(rtl::NetRef ref, module_->ResolveNet(pair.first));
    translated_[ref] = pair.second;
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

// General idea: construct an AST by iterating over all the Cells in the module.
//  1. First, collect all input wires and put them on an "active" list.
//  2. Iterate through the active wire list and examine all cells they're
//     connected to, removing each as examined.
//  3. For any cell for which all inputs have been seen (are "active"),
//     translate that cell into Z3 space, and move its output wires
//     (the resulting Z3 nodes) onto the back of the active wire list.
//  4. Repeat until the active wire list is empty.
absl::Status Z3Translator::Translate() {
  // Utility structure so we don't have to iterate through a cell's inputs and
  // outputs every time it's examined.
  absl::flat_hash_map<rtl::Cell*, absl::flat_hash_set<rtl::NetRef>> cell_inputs;
  for (const auto& cell : module_->cells()) {
    absl::flat_hash_set<rtl::NetRef> inputs;
    for (const auto& input : cell->inputs()) {
      inputs.insert(input.netref);
    }
    cell_inputs[cell.get()] = std::move(inputs);
  }

  // Double-buffer the active/next active wire lists.
  // Remember - we pre-populated translated_ with the set of module inputs.
  std::deque<rtl::NetRef> active_wires;
  for (const auto& pair : translated_) {
    active_wires.push_back(pair.first);
  }

  // For every active wire, check to see if all of its inputs are satisfied.
  // If so, then that cell's outputs are now active and should be considered
  // newly active on the next pass.
  while (!active_wires.empty()) {
    rtl::NetRef ref = active_wires.front();
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

absl::Status Z3Translator::TranslateCell(const rtl::Cell& cell) {
  using function::Ast;

  // If this cell is actually a reference to a module defined in this netlist,
  // then translate it into Z3-space here and grab its output nodes.
  std::string entry_name = cell.cell_library_entry()->name();
  if (module_refs_.contains(entry_name)) {
    absl::flat_hash_map<std::string, Z3_ast> inputs;
    for (const auto& input : cell.inputs()) {
      inputs[input.pin_name] = translated_[input.netref];
    }

    const rtl::Module* module_ref = module_refs_.at(entry_name);
    XLS_ASSIGN_OR_RETURN(auto subtranslator, Z3Translator::CreateAndTranslate(
                                                 ctx_, module_ref, module_refs_,
                                                 inputs, high_cells_));

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
      XLS_ASSIGN_OR_RETURN(function::Ast ast, function::Parser::ParseFunction(
                                                  output.pin.function));
      XLS_ASSIGN_OR_RETURN(Z3_ast result, TranslateFunction(cell, ast));
      translated_[output.netref] = result;
    }
  }

  return absl::OkStatus();
}

// After all the above, this is the spot where any _ACTUAL_ translation happens.
xabsl::StatusOr<Z3_ast> Z3Translator::TranslateFunction(const rtl::Cell& cell,
                                                        function::Ast ast) {
  switch (ast.kind()) {
    case Ast::Kind::kAnd: {
      XLS_ASSIGN_OR_RETURN(Z3_ast lhs,
                           TranslateFunction(cell, ast.children()[0]));
      XLS_ASSIGN_OR_RETURN(Z3_ast rhs,
                           TranslateFunction(cell, ast.children()[1]));
      return Z3_mk_bvand(ctx_, lhs, rhs);
    }
    case Ast::Kind::kIdentifier: {
      rtl::NetRef ref = nullptr;
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
      return Z3_mk_true(ctx_);
    }
    case Ast::Kind::kLiteralZero: {
      return Z3_mk_false(ctx_);
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

}  // namespace netlist
}  // namespace xls
