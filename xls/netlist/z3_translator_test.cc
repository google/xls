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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
#include "absl/types/optional.h"
#include "xls/common/cleanup.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"
#include "../z3/src/api/z3_api.h"

namespace xls {
namespace netlist {
namespace {

absl::flat_hash_map<std::string, Z3_ast> CreateInputs(Z3_context ctx,
                                                      int num_inputs) {
  absl::flat_hash_map<std::string, Z3_ast> inputs;
  Z3_sort input_sort = Z3_mk_bv_sort(ctx, /*sz=*/1);
  for (int i = 0; i < num_inputs; i++) {
    std::string input_name = absl::StrCat("input_", i);
    inputs[input_name] = Z3_mk_const(
        ctx, Z3_mk_string_symbol(ctx, input_name.c_str()), input_sort);
  }
  return inputs;
}

// Creates a netlist for testing purposes.
// This function creates a fake cell library and creates a "random" netlist from
// it, using the provided Z3_ast elements as potential inputs.
absl::Status CreateNetList(
    absl::BitGen* bitgen,
    const absl::flat_hash_map<std::string, Z3_ast>& inputs, int num_cells,
    const CellLibrary& cell_library, rtl::Module* module) {
  // Maintain a set of nets that can be used as inputs for new cells. Initially,
  // this is just the set of inputs, but will be expanded as cells are defined.
  // We could maintain a vector of strings and resolve them, or store a vector
  // of already-resolved string -> NetRefs. Turns out to be about the same in
  // practice.
  std::vector<std::string> available_inputs;
  for (const auto& input : inputs) {
    XLS_RETURN_IF_ERROR(
        module->AddNetDecl(rtl::NetDeclKind::kInput, input.first));
    available_inputs.push_back(input.first);
  }

  std::string clk_name = "wow_what_an_amazing_clock";
  XLS_RETURN_IF_ERROR(module->AddNetDecl(rtl::NetDeclKind::kWire, clk_name));
  XLS_ASSIGN_OR_RETURN(rtl::NetRef clk, module->ResolveNet(clk_name));

  // Names of the fake cells:
  //  - INV
  //  - DFF
  //  - AND
  //  - NAND
  //  - NOR4
  //  - AOI21
  // I'm currently skipping DFF, since they don't have formulae in the cell
  // library I've examined. I'll cross that bridge in time.
  std::vector<std::string> cell_names = {"INV", "AND", "NAND", "NOR4", "AOI21"};

  for (int cell_index = 0; cell_index < num_cells; cell_index++) {
    // Pick a cell at random.
    std::string cell_name =
        cell_names[absl::Uniform(*bitgen, 0u, cell_names.size())];
    XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* entry,
                         cell_library.GetEntry(cell_name));
    absl::StrAppend(&cell_name, "_", cell_index);

    // Assign a random available net to each cell input.
    absl::flat_hash_map<std::string, rtl::NetRef> param_assignments;
    absl::Span<const std::string> input_names = entry->input_names();
    for (int input_index = 0; input_index < input_names.size(); input_index++) {
      int available_index = absl::Uniform(*bitgen, 0u, available_inputs.size());
      XLS_ASSIGN_OR_RETURN(
          rtl::NetRef input_ref,
          module->ResolveNet(available_inputs[available_index]));
      param_assignments[input_names[input_index]] = input_ref;
    }

    // And associate the output with a new NetRef.
    absl::Span<const OutputPin> output_pins = entry->output_pins();
    for (int output_index = 0; output_index < output_pins.size();
         output_index++) {
      std::string output_net_name = absl::StrCat(cell_name, "_out");
      XLS_RETURN_IF_ERROR(
          module->AddNetDecl(rtl::NetDeclKind::kOutput, output_net_name));

      XLS_ASSIGN_OR_RETURN(rtl::NetRef output_ref,
                           module->ResolveNet(output_net_name));
      param_assignments[output_pins[output_index].name] = output_ref;
      available_inputs.push_back(output_net_name);
    }

    XLS_ASSIGN_OR_RETURN(rtl::Cell cell,
                         rtl::Cell::Create(entry, cell_name, param_assignments,
                                           clk, /*dummy_net=*/nullptr));
    XLS_ASSIGN_OR_RETURN(rtl::Cell * module_cell, module->AddCell(cell));
    XLS_VLOG(2) << "Added cell: " << module_cell->name();
    XLS_VLOG(2) << " - Inputs";
    for (const auto& input : module_cell->inputs()) {
      XLS_VLOG(2) << "   - " << input.netref->name();
    }
    XLS_VLOG(2) << " - Outputs";
    for (const auto& output : module_cell->outputs()) {
      XLS_VLOG(2) << "   - " << output.netref->name();
    }
    for (auto& pair : param_assignments) {
      pair.second->NoteConnectedCell(module_cell);
    }
  }

  return absl::OkStatus();
}

absl::flat_hash_map<std::string, Z3_ast> CreateInputs(
    const Z3_context& ctx, const rtl::Module& module) {
  absl::flat_hash_map<std::string, Z3_ast> inputs;
  Z3_sort input_sort = Z3_mk_bv_sort(ctx, 1);
  for (const auto& input : module.inputs()) {
    inputs[input->name()] = Z3_mk_const(
        ctx, Z3_mk_string_symbol(ctx, input->name().c_str()), input_sort);
  }
  return inputs;
}

// Simple test to make sure we can translate anything at all.
TEST(Z3TranslatorTest, BasicFunctionality) {
  constexpr int kNumCells = 16;
  constexpr int kNumInputs = 4;

  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = xabsl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });
  rtl::Module module("the_module");
  CellLibrary cell_library = MakeFakeCellLibrary();
  absl::BitGen bitgen;
  absl::flat_hash_map<std::string, Z3_ast> inputs =
      CreateInputs(ctx, kNumInputs);
  XLS_ASSERT_OK(
      CreateNetList(&bitgen, inputs, kNumCells, cell_library, &module));

  XLS_ASSERT_OK_AND_ASSIGN(auto translator, Z3Translator::CreateAndTranslate(
                                                ctx, &module, {}, inputs, {}));
}

// Tests that a simple (single-cell) netlist is translated correctly.
TEST(Z3TranslatorTest, SimpleNet) {
  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = xabsl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });

  rtl::Module module("the_module");

  CellLibrary cell_library = MakeFakeCellLibrary();
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* and_entry,
                           cell_library.GetEntry("AND"));
  absl::flat_hash_map<std::string, rtl::NetRef> param_assignments;
  absl::flat_hash_map<std::string, Z3_ast> inputs;
  Z3_sort input_sort = Z3_mk_bv_sort(ctx, /*sz=*/1);
  for (const auto& input_name : and_entry->input_names()) {
    XLS_ASSERT_OK(module.AddNetDecl(rtl::NetDeclKind::kInput, input_name));
    XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef ref, module.ResolveNet(input_name));
    param_assignments[input_name] = ref;

    Z3_ast z3_input = Z3_mk_const(
        ctx, Z3_mk_string_symbol(ctx, input_name.c_str()), input_sort);
    inputs[input_name] = z3_input;
  }

  std::string output_name = and_entry->output_pins().begin()->name;
  XLS_ASSERT_OK(module.AddNetDecl(rtl::NetDeclKind::kOutput, output_name));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef output_ref,
                           module.ResolveNet(output_name));
  param_assignments[output_name] = output_ref;

  std::string clk_name = "wow_what_an_amazing_clock";
  XLS_ASSERT_OK(module.AddNetDecl(rtl::NetDeclKind::kWire, clk_name));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef clk, module.ResolveNet(clk_name));

  XLS_ASSERT_OK_AND_ASSIGN(rtl::Cell tmp_cell,
                           rtl::Cell::Create(and_entry, "Rob's magic cell",
                                             param_assignments, clk, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Cell * cell, module.AddCell(tmp_cell));
  for (auto& pair : param_assignments) {
    pair.second->NoteConnectedCell(cell);
  }
  XLS_ASSERT_OK_AND_ASSIGN(auto translator, Z3Translator::CreateAndTranslate(
                                                ctx, &module, {}, inputs, {}));

  rtl::NetRef cell_output = cell->outputs().begin()->netref;
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast z3_output,
                           translator->GetTranslation(cell_output));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx, Z3_get_sort(ctx, z3_output)), 1);

  // I promise I tried, but I've not found a way in the Z3 API to extract the
  // operation that an AST node represents, so we're left with string
  // examination.
  std::string ast_text = Z3_ast_to_string(ctx, z3_output);
  EXPECT_NE(ast_text.find("bvand A B"), std::string::npos);
}

// Create a module that instantiates all child cells and combines them with a
// cell with the specified name.
xabsl::StatusOr<rtl::Module> CreateModule(
    const CellLibrary& cell_library, const std::string& module_name,
    const std::string& cell_name, const std::vector<std::string>& child_cells) {
  // Instantiate all child cells, collect their inputs and outputs
  // Synthesize any missing parent cell inputs (if we have too many, abort)
  //  - Each child cell output is a parent cell input
  //  - Any missing are new module inputs
  // Apply inputs to cell in a left-to-right order
  rtl::Module module(module_name);

  absl::flat_hash_map<std::string, rtl::NetRef> parent_params;
  std::vector<std::string> child_outputs;
  for (int i = 0; i < child_cells.size(); i++) {
    const std::string& child_name = child_cells[i];
    XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* entry,
                         cell_library.GetEntry(child_name));

    absl::flat_hash_map<std::string, rtl::NetRef> child_params;
    absl::Span<const std::string> entry_input_names = entry->input_names();
    for (const std::string& entry_input_name : entry_input_names) {
      std::string module_input_name =
          absl::StrCat(module_name, "_c", i, "_", entry_input_name);
      XLS_VLOG(2) << "Creating module input : " << module_input_name;
      XLS_RET_CHECK_OK(
          module.AddNetDecl(rtl::NetDeclKind::kInput, module_input_name));
      child_params[entry_input_name] =
          module.ResolveNet(module_input_name).value();
    }

    std::string child_output_name = absl::StrCat(module_name, "_c", i, "_o");
    XLS_VLOG(2) << "Creating child output : " << child_output_name;
    XLS_RET_CHECK_OK(
        module.AddNetDecl(rtl::NetDeclKind::kWire, child_output_name));
    child_params[entry->output_pins()[0].name] =
        module.ResolveNet(child_output_name).value();
    child_outputs.push_back(child_output_name);

    XLS_ASSIGN_OR_RETURN(
        rtl::Cell temp_cell,
        rtl::Cell::Create(entry, absl::StrCat(module_name, "_", i),
                          child_params, absl::nullopt, /*dummy_net=*/nullptr));
    XLS_ASSIGN_OR_RETURN(rtl::Cell * cell, module.AddCell(temp_cell));
    for (auto& pair : child_params) {
      pair.second->NoteConnectedCell(cell);
    }
  }

  // If we're short parent cell inputs, then synthesize some.
  XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* entry,
                       cell_library.GetEntry(cell_name));
  absl::Span<const std::string> entry_input_names = entry->input_names();
  XLS_RET_CHECK(child_outputs.size() <= entry_input_names.size())
      << "Too many inputs for cell: " << cell_name << "! "
      << entry_input_names.size() << " vs. " << child_outputs.size() << ".";
  while (child_outputs.size() < entry_input_names.size()) {
    std::string new_input_name =
        absl::StrCat(module_name, "_",
                     entry_input_names.size() - child_outputs.size(), "_o");
    XLS_VLOG(2) << "Synthesizing module input: " << new_input_name;
    XLS_RET_CHECK_OK(
        module.AddNetDecl(rtl::NetDeclKind::kInput, new_input_name));
    child_outputs.push_back(new_input_name);
  }

  for (int i = 0; i < entry_input_names.size(); i++) {
    parent_params[entry_input_names[i]] =
        module.ResolveNet(child_outputs[i]).value();
  }

  for (int i = 0; i < entry->output_pins().size(); i++) {
    std::string output_name = absl::StrCat(module_name, "_o", i);
    XLS_RETURN_IF_ERROR(
        module.AddNetDecl(rtl::NetDeclKind::kOutput, output_name));
    parent_params[entry->output_pins()[i].name] =
        module.ResolveNet(output_name).value();
  }

  XLS_ASSIGN_OR_RETURN(
      rtl::Cell temp_cell,
      rtl::Cell::Create(entry, absl::StrCat(module_name, "_", cell_name),
                        parent_params, absl::nullopt, /*dummy_net=*/nullptr));
  XLS_ASSIGN_OR_RETURN(rtl::Cell * cell, module.AddCell(temp_cell));
  for (auto& pair : parent_params) {
    pair.second->NoteConnectedCell(cell);
  }

  return std::move(module);
}

TEST(Z3TranslatorTest, HandlesSubmodules) {
  // Create four modules:
  //  - module_0: self-contained AND cell.
  //  - module_1: self-contained OR cell.
  //  - module_2: Takes four inputs, AND of module_0 and module_1 outputs.
  //  - module_3: references module_2.
  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = xabsl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });

  CellLibrary cell_library = MakeFakeCellLibrary();
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Module module_0,
                           CreateModule(cell_library, "m0", "AND", {}));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Module module_1,
                           CreateModule(cell_library, "m1", "OR", {}));

  XLS_ASSERT_OK(cell_library.AddEntry(*module_0.AsCellLibraryEntry()));
  XLS_ASSERT_OK(cell_library.AddEntry(*module_1.AsCellLibraryEntry()));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Module module_2,
                           CreateModule(cell_library, "m2", "AND",
                                        {module_0.name(), module_1.name()}));
  XLS_ASSERT_OK(cell_library.AddEntry(*module_2.AsCellLibraryEntry()));

  XLS_ASSERT_OK_AND_ASSIGN(
      rtl::Module module_3,
      CreateModule(cell_library, "m3", "INV", {module_2.name()}));

  // Create inputs for the top-level cell/module.
  absl::flat_hash_map<std::string, Z3_ast> inputs = CreateInputs(ctx, module_3);
  absl::flat_hash_map<std::string, const rtl::Module*> module_refs({
      {"m0", &module_0},
      {"m1", &module_1},
      {"m2", &module_2},
  });
  XLS_ASSERT_OK_AND_ASSIGN(auto translator,
                           Z3Translator::CreateAndTranslate(
                               ctx, &module_3, module_refs, inputs, {}));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast z3_output,
                           translator->GetTranslation(module_3.outputs()[0]));
  std::string ast_text = Z3_ast_to_string(ctx, z3_output);
  XLS_VLOG(1) << "Z3 AST:" << std::endl << ast_text;
  int not_pos = ast_text.find("bvnot");
  int and_pos = ast_text.find("bvand");
  int or_pos = ast_text.find("bvor");
  EXPECT_NE(not_pos, std::string::npos);
  EXPECT_NE(and_pos, std::string::npos);
  EXPECT_NE(or_pos, std::string::npos);
  EXPECT_LT(not_pos, and_pos);
  EXPECT_LT(and_pos, or_pos);
}

// This test verifies that cells with no inputs immediately fire.
TEST(Z3TranslatorTest, NoInputCells) {
  std::string module_text = R"(
module main (i0, o0);
  input i0;
  output o0;
  wire res0;

  LOGIC_ONE fixed_one( .O(res0) );
  AND and0( .A(i0), .B(res0), .Z(o0) );
endmodule)";
  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = xabsl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });

  CellLibrary cell_library = MakeFakeCellLibrary();
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Netlist netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist.GetModule("main"));
  absl::flat_hash_map<std::string, Z3_ast> inputs = CreateInputs(ctx, *module);

  // If this call succeeds, then we were able to translate the module, which
  // means that cell fixed_one was correctly activated.
  XLS_ASSERT_OK(
      Z3Translator::CreateAndTranslate(ctx, module, {}, inputs, {}).status());
}

}  // namespace
}  // namespace netlist
}  // namespace xls
