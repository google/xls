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

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"

namespace xls {
namespace solvers {
namespace z3 {

using netlist::CellLibrary;
using netlist::CellLibraryEntry;
using netlist::rtl::Cell;
using netlist::rtl::Module;
using netlist::rtl::NetDeclKind;
using netlist::rtl::NetRef;

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
    const CellLibrary& cell_library, Module* module) {
  // Maintain a set of nets that can be used as inputs for new cells. Initially,
  // this is just the set of inputs, but will be expanded as cells are defined.
  // We could maintain a vector of strings and resolve them, or store a vector
  // of already-resolved string -> NetRefs. Turns out to be about the same in
  // practice.
  std::vector<std::string> available_inputs;
  for (const auto& input : inputs) {
    XLS_RETURN_IF_ERROR(module->AddNetDecl(NetDeclKind::kInput, input.first));
    available_inputs.push_back(input.first);
  }

  std::string clk_name = "wow_what_an_amazing_clock";
  XLS_RETURN_IF_ERROR(module->AddNetDecl(NetDeclKind::kWire, clk_name));
  XLS_ASSIGN_OR_RETURN(NetRef clk, module->ResolveNet(clk_name));

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
    absl::flat_hash_map<std::string, NetRef> param_assignments;
    absl::Span<const std::string> input_names = entry->input_names();
    for (const auto& input_name : input_names) {
      int available_index = absl::Uniform(*bitgen, 0u, available_inputs.size());
      XLS_ASSIGN_OR_RETURN(
          NetRef input_ref,
          module->ResolveNet(available_inputs[available_index]));
      param_assignments[input_name] = input_ref;
    }

    // And associate the output with a new NetRef.
    const auto& pins = entry->output_pin_to_function();

    for (const auto& kv : pins) {
      std::string output_net_name = absl::StrCat(cell_name, "_out");
      XLS_RETURN_IF_ERROR(
          module->AddNetDecl(NetDeclKind::kOutput, output_net_name));

      XLS_ASSIGN_OR_RETURN(NetRef output_ref,
                           module->ResolveNet(output_net_name));
      param_assignments[kv.first] = output_ref;
      available_inputs.push_back(output_net_name);
    }

    XLS_ASSIGN_OR_RETURN(Cell cell,
                         Cell::Create(entry, cell_name, param_assignments, clk,
                                      /*dummy_net=*/nullptr));
    XLS_ASSIGN_OR_RETURN(Cell * module_cell, module->AddCell(cell));
    VLOG(2) << "Added cell: " << module_cell->name();
    VLOG(2) << " - Inputs";
    for (const auto& input : module_cell->inputs()) {
      VLOG(2) << "   - " << input.netref->name();
    }
    VLOG(2) << " - Outputs";
    for (const auto& output : module_cell->outputs()) {
      VLOG(2) << "   - " << output.netref->name();
    }
    for (auto& pair : param_assignments) {
      pair.second->NoteConnectedCell(module_cell);
    }
  }

  return absl::OkStatus();
}

// Simple test to make sure we can translate anything at all.
// TODO(rspringer): This test should really be updated to specify the module as
// a string.
TEST(NetlistTranslatorTest_Standalone, BasicFunctionality) {
  constexpr int kNumCells = 16;
  constexpr int kNumInputs = 4;

  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = absl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });
  Module module("the_module");
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  absl::BitGen bitgen;
  absl::flat_hash_map<std::string, Z3_ast> inputs =
      CreateInputs(ctx, kNumInputs);
  XLS_ASSERT_OK(
      CreateNetList(&bitgen, inputs, kNumCells, cell_library, &module));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto translator, NetlistTranslator::CreateAndTranslate(ctx, &module, {}));
}

// Tests that a simple (single-cell) netlist is translated correctly.
// TODO(rspringer): This test should really be updated to specify the module as
// a string.
TEST(NetlistTranslatorTest_Standalone, SimpleNet) {
  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = absl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });

  Module module("the_module");

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* and_entry,
                           cell_library.GetEntry("AND"));
  absl::flat_hash_map<std::string, NetRef> param_assignments;
  for (const auto& input_name : and_entry->input_names()) {
    XLS_ASSERT_OK(module.AddNetDecl(NetDeclKind::kInput, input_name));
    XLS_ASSERT_OK_AND_ASSIGN(NetRef ref, module.ResolveNet(input_name));
    param_assignments[input_name] = ref;
  }

  const auto& pins = and_entry->output_pin_to_function();
  std::string output_name = pins.begin()->first;
  XLS_ASSERT_OK(module.AddNetDecl(NetDeclKind::kOutput, output_name));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef output_ref, module.ResolveNet(output_name));
  param_assignments[output_name] = output_ref;

  std::string clk_name = "wow_what_an_amazing_clock";
  XLS_ASSERT_OK(module.AddNetDecl(NetDeclKind::kWire, clk_name));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef clk, module.ResolveNet(clk_name));

  XLS_ASSERT_OK_AND_ASSIGN(Cell tmp_cell,
                           Cell::Create(and_entry, "Rob's magic cell",
                                        param_assignments, clk, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(Cell * cell, module.AddCell(tmp_cell));
  for (auto& pair : param_assignments) {
    pair.second->NoteConnectedCell(cell);
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      auto translator, NetlistTranslator::CreateAndTranslate(ctx, &module, {}));

  NetRef cell_output = cell->outputs().begin()->netref;
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
absl::StatusOr<Module> CreateModule(
    const CellLibrary& cell_library, const std::string& module_name,
    const std::string& cell_name, const std::vector<std::string>& child_cells) {
  // Instantiate all child cells, collect their inputs and outputs
  // Synthesize any missing parent cell inputs (if we have too many, abort)
  //  - Each child cell output is a parent cell input
  //  - Any missing are new module inputs
  // Apply inputs to cell in a left-to-right order
  Module module(module_name);

  absl::flat_hash_map<std::string, NetRef> parent_params;
  std::vector<std::string> child_outputs;
  for (int i = 0; i < child_cells.size(); i++) {
    const std::string& child_name = child_cells[i];
    XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* entry,
                         cell_library.GetEntry(child_name));
    const auto& pins = entry->output_pin_to_function();

    absl::flat_hash_map<std::string, NetRef> child_params;
    absl::Span<const std::string> entry_input_names = entry->input_names();
    for (const std::string& entry_input_name : entry_input_names) {
      std::string module_input_name =
          absl::StrCat(module_name, "_c", i, "_", entry_input_name);
      VLOG(2) << "Creating module input : " << module_input_name;
      XLS_RET_CHECK_OK(
          module.AddNetDecl(NetDeclKind::kInput, module_input_name));
      child_params[entry_input_name] =
          module.ResolveNet(module_input_name).value();
    }

    std::string child_output_name = absl::StrCat(module_name, "_c", i, "_o");
    VLOG(2) << "Creating child output : " << child_output_name;
    XLS_RET_CHECK_OK(module.AddNetDecl(NetDeclKind::kWire, child_output_name));
    child_params[pins.begin()->first] =
        module.ResolveNet(child_output_name).value();
    child_outputs.push_back(child_output_name);

    XLS_ASSIGN_OR_RETURN(
        Cell temp_cell,
        Cell::Create(entry, absl::StrCat(module_name, "_", i), child_params,
                     std::nullopt, /*dummy_net=*/nullptr));
    XLS_ASSIGN_OR_RETURN(Cell * cell, module.AddCell(temp_cell));
    for (auto& pair : child_params) {
      pair.second->NoteConnectedCell(cell);
    }
  }

  // If we're short parent cell inputs, then synthesize some.
  XLS_ASSIGN_OR_RETURN(const CellLibraryEntry* entry,
                       cell_library.GetEntry(cell_name));
  const auto& pins = entry->output_pin_to_function();
  absl::Span<const std::string> entry_input_names = entry->input_names();
  XLS_RET_CHECK(child_outputs.size() <= entry_input_names.size())
      << "Too many inputs for cell: " << cell_name << "! "
      << entry_input_names.size() << " vs. " << child_outputs.size() << ".";
  while (child_outputs.size() < entry_input_names.size()) {
    std::string new_input_name =
        absl::StrCat(module_name, "_",
                     entry_input_names.size() - child_outputs.size(), "_o");
    VLOG(2) << "Synthesizing module input: " << new_input_name;
    XLS_RET_CHECK_OK(module.AddNetDecl(NetDeclKind::kInput, new_input_name));
    child_outputs.push_back(new_input_name);
  }

  for (int i = 0; i < entry_input_names.size(); i++) {
    parent_params[entry_input_names[i]] =
        module.ResolveNet(child_outputs[i]).value();
  }

  // Only support a single output.
  auto iter = pins.begin();
  for (int i = 0; i < pins.size(); i++) {
    std::string output_name = absl::StrCat(module_name, "_o", i);
    XLS_RETURN_IF_ERROR(module.AddNetDecl(NetDeclKind::kOutput, output_name));
    parent_params[iter->first] = module.ResolveNet(output_name).value();
    iter++;
  }

  XLS_ASSIGN_OR_RETURN(
      Cell temp_cell,
      Cell::Create(entry, absl::StrCat(module_name, "_", cell_name),
                   parent_params, std::nullopt, /*dummy_net=*/nullptr));
  XLS_ASSIGN_OR_RETURN(Cell * cell, module.AddCell(temp_cell));
  for (auto& pair : parent_params) {
    pair.second->NoteConnectedCell(cell);
  }

  return std::move(module);
}

// TODO(rspringer): This test should really be updated to specify the module as
// a string.
TEST(NetlistTranslatorTest_Standalone, HandlesSubmodules) {
  // Create four modules:
  //  - module_0: self-contained AND cell.
  //  - module_1: self-contained OR cell.
  //  - module_2: Takes four inputs, AND of module_0 and module_1 outputs.
  //  - module_3: references module_2.
  Z3_config config = Z3_mk_config();
  Z3_context ctx = Z3_mk_context(config);
  auto cleanup = absl::MakeCleanup([config, ctx] {
    Z3_del_context(ctx);
    Z3_del_config(config);
  });

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(Module module_0,
                           CreateModule(cell_library, "m0", "AND", {}));
  XLS_ASSERT_OK_AND_ASSIGN(Module module_1,
                           CreateModule(cell_library, "m1", "OR", {}));

  XLS_ASSERT_OK(cell_library.AddEntry(*module_0.AsCellLibraryEntry()));
  XLS_ASSERT_OK(cell_library.AddEntry(*module_1.AsCellLibraryEntry()));
  XLS_ASSERT_OK_AND_ASSIGN(Module module_2,
                           CreateModule(cell_library, "m2", "AND",
                                        {module_0.name(), module_1.name()}));
  XLS_ASSERT_OK(cell_library.AddEntry(*module_2.AsCellLibraryEntry()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Module module_3,
      CreateModule(cell_library, "m3", "INV", {module_2.name()}));

  absl::flat_hash_map<std::string, const Module*> module_refs({
      {"m0", &module_0},
      {"m1", &module_1},
      {"m2", &module_2},
  });
  XLS_ASSERT_OK_AND_ASSIGN(
      auto translator,
      NetlistTranslator::CreateAndTranslate(ctx, &module_3, module_refs));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast z3_output,
                           translator->GetTranslation(module_3.outputs()[0]));
  std::string ast_text = Z3_ast_to_string(ctx, z3_output);
  VLOG(1) << "Z3 AST:" << '\n' << ast_text;
  int not_pos = ast_text.find("bvnot");
  int and_pos = ast_text.find("bvand");
  int or_pos = ast_text.find("bvor");
  EXPECT_NE(not_pos, std::string::npos);
  EXPECT_NE(and_pos, std::string::npos);
  EXPECT_NE(or_pos, std::string::npos);
  EXPECT_LT(not_pos, and_pos);
  EXPECT_LT(and_pos, or_pos);
}

class NetlistTranslatorTest : public ::testing::Test {
 public:
  NetlistTranslatorTest()
      : config_(nullptr),
        ctx_(nullptr),
        cell_library_(netlist::MakeFakeCellLibrary().value()) {}
  ~NetlistTranslatorTest() override {
    if (ctx_) {
      Z3_del_context(ctx_);
    }

    if (config_) {
      Z3_del_config(config_);
    }
  }

  absl::Status Init(const std::string& module_text) {
    netlist::rtl::Scanner scanner(module_text);
    XLS_ASSIGN_OR_RETURN(
        netlist_, netlist::rtl::Parser::ParseNetlist(&cell_library_, &scanner));
    XLS_ASSIGN_OR_RETURN(module_, netlist_->GetModule("main"));

    // Verify that the input node i0 is a constant [but not w/a fixed value].
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);

    XLS_ASSIGN_OR_RETURN(
        translator_, NetlistTranslator::CreateAndTranslate(ctx_, module_, {}));
    return absl::OkStatus();
  }

  bool IsSatisfiable(Z3_ast condition) {
    Z3_solver solver = CreateSolver(ctx_, 1);
    Z3_solver_assert(ctx_, solver, condition);
    Z3_lbool satisfiable = Z3_solver_check(ctx_, solver);
    Z3_solver_dec_ref(ctx_, solver);
    return satisfiable == Z3_L_TRUE;
  }

 protected:
  Z3_config config_;
  Z3_context ctx_;
  std::unique_ptr<netlist::rtl::Netlist> netlist_;
  std::unique_ptr<NetlistTranslator> translator_;
  CellLibrary cell_library_;
  const Module* module_;
};

// This test verifies that cells with no inputs immediately fire.
TEST_F(NetlistTranslatorTest, NoInputCells) {
  std::string module_text = R"(
module main (i0, o0);
  input i0;
  output o0;
  wire res0;

  LOGIC_ONE fixed_one( .O(res0) );
  AND and0( .A(i0), .B(res0), .Z(o0) );
endmodule)";
  // If this call succeeds, then we were able to translate the module, which
  // means that cell fixed_one was correctly activated.
  XLS_ASSERT_OK(Init(module_text));
}

// This test verifies that nodes can be "swapped out" for other Z3 nodes.
TEST_F(NetlistTranslatorTest, CanRetranslate) {
  std::string module_text = R"(
module main (i0, o0);
  input i0;
  output o0;
  wire res0;

  LOGIC_ONE fixed_one( .O(res0) );
  AND and0( .A(i0), .B(res0), .Z(o0) );
endmodule)";
  XLS_ASSERT_OK(Init(module_text));

  // First, verify that the initial module can have a value of one.
  Z3_ast value_0 = Z3_mk_int(ctx_, 0, Z3_mk_bv_sort(ctx_, 1));
  Z3_ast value_1 = Z3_mk_int(ctx_, 1, Z3_mk_bv_sort(ctx_, 1));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Next, bind a fixed 0 to the input of the module. This will result in a
  // solver being unable to find a 1-valued output.
  std::string src_ref_name = module_->inputs()[0]->name();
  XLS_ASSERT_OK(translator_->Retranslate({{src_ref_name, value_0}}));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Now, replace that fixed 0 with a fixed one. This time, the solver WILL be
  // able to find a 1-valued output.
  XLS_ASSERT_OK(translator_->Retranslate({{src_ref_name, value_1}}));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));
}

TEST_F(NetlistTranslatorTest, CanRebindNetsTree) {
  std::string module_text = R"(
module main (i0, o0);
  input i0;
  output o0;
  wire fixed_1, res0, res1, res2;

  LOGIC_ONE fixed_one( .O(fixed_1) );
  AND and0( .A(i0), .B(fixed_1), .Z(res0) );
  AND and1( .A(fixed_1), .B(fixed_1), .Z(res1) );
  AND and2( .A(res0), .B(res1), .Z(o0) );
endmodule)";

  XLS_ASSERT_OK(Init(module_text));
  Z3_sort bit_sort = Z3_mk_bv_sort(ctx_, 1);
  Z3_ast value_0 = Z3_mk_int(ctx_, 0, bit_sort);
  Z3_ast value_1 = Z3_mk_int(ctx_, 1, bit_sort);
  Z3_ast free_constant =
      Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, "foo"), bit_sort);
  std::string src_ref_name = module_->inputs()[0]->name();

  // First, verify that the initial module can have a value of one or zero.
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Next, replace i0 with 0 and verify that the net can NOT have a value of
  // one, but can have a value of zero.
  XLS_ASSERT_OK(translator_->Retranslate({{src_ref_name, value_0}}));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  ASSERT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Now replace i0 with 1 and verify that the net can have a value of zero, but
  // not one.
  XLS_ASSERT_OK(translator_->Retranslate({{src_ref_name, value_1}}));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Finally, replace i0 with a constant, and verify we're back to one or zero
  // being possible.
  XLS_ASSERT_OK(translator_->Retranslate({{src_ref_name, free_constant}}));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));
}

// This test verifies that a state-table-containing-cell-containing netlist is
// properly handled - i.e., that we properly translate into Z3 when a state
// table is present.
TEST_F(NetlistTranslatorTest, ProcessesStateTables) {
  std::string module_text = R"(
module main (i0, i1, o0);
  input i0, i1;
  output o0;

  STATETABLE_AND st_and ( .A(i0), .B(i1), .Z(o0) );
endmodule)";

  XLS_ASSERT_OK(Init(module_text));

  // Verify each case - A&B, !A&B, A&!B, !A&!B.
  Z3_sort bit_sort = Z3_mk_bv_sort(ctx_, 1);
  Z3_ast value_0 = Z3_mk_int(ctx_, 0, bit_sort);
  Z3_ast value_1 = Z3_mk_int(ctx_, 1, bit_sort);

  // First, make sure unbound inputs make all outputs possible.
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  ASSERT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // Now rebind, as above.
  std::string src_ref_0 = module_->inputs()[0]->name();
  std::string src_ref_1 = module_->inputs()[1]->name();

  // A&B
  XLS_ASSERT_OK(translator_->Retranslate({
      {src_ref_0, value_1},
      {src_ref_1, value_1},
  }));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  EXPECT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  EXPECT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // !A&B
  XLS_ASSERT_OK(translator_->Retranslate({
      {src_ref_0, value_0},
      {src_ref_1, value_1},
  }));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  EXPECT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  EXPECT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // A&!B
  XLS_ASSERT_OK(translator_->Retranslate({
      {src_ref_0, value_1},
      {src_ref_1, value_0},
  }));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  EXPECT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  EXPECT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));

  // !A&!B
  XLS_ASSERT_OK(translator_->Retranslate({
      {src_ref_0, value_0},
      {src_ref_1, value_0},
  }));
  XLS_ASSERT_OK_AND_ASSIGN(module_output,
                           translator_->GetTranslation(module_->outputs()[0]));
  EXPECT_TRUE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_0)));
  EXPECT_FALSE(IsSatisfiable(Z3_mk_eq(ctx_, module_output, value_1)));
}

}  // namespace
}  // namespace z3
}  // namespace solvers
}  // namespace xls
