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
#include "xls/netlist/interpreter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"

namespace xls {
namespace netlist {
namespace {

// Smoke test to make sure anything works.
TEST(InterpreterTest, BasicFunctionality) {
  // Make a very simple A * B module.
  auto module = std::make_unique<rtl::Module>("the_module");
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "A"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "B"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kOutput, "O"));

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* entry,
                           library.GetEntry("AND"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef a_ref, module->ResolveNet("A"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef b_ref, module->ResolveNet("B"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef o_ref, module->ResolveNet("O"));

  absl::flat_hash_map<std::string, rtl::NetRef> params;
  params["A"] = a_ref;
  params["B"] = b_ref;
  params["Z"] = o_ref;

  XLS_ASSERT_OK_AND_ASSIGN(
      rtl::Cell tmp_cell,
      rtl::Cell::Create(entry, "the_cell", params, absl::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto cell, module->AddCell(tmp_cell));
  a_ref->NoteConnectedCell(cell);
  b_ref->NoteConnectedCell(cell);
  o_ref->NoteConnectedCell(cell);

  rtl::Netlist netlist;
  netlist.AddModule(std::move(module));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module_ptr,
                           netlist.GetModule("the_module"));
  Interpreter interpreter(&netlist);

  absl::flat_hash_map<const rtl::NetRef, bool> inputs;
  inputs[module_ptr->inputs()[0]] = true;
  inputs[module_ptr->inputs()[1]] = false;
  using OutputT = absl::flat_hash_map<const rtl::NetRef, bool>;
  XLS_ASSERT_OK_AND_ASSIGN(OutputT outputs,
                           interpreter.InterpretModule(module_ptr, inputs));
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module_ptr->outputs()[0]], 0);
}

// Verifies that a simple XOR(AND(), OR()) tree is interpreted correctly.
TEST(InterpreterTest, Tree) {
  // Make a very simple A * B module.
  auto module = std::make_unique<rtl::Module>("the_module");
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i0"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i1"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i2"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i3"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kWire, "and_o"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kWire, "or_o"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kOutput, "xor_o"));

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* and_entry,
                           library.GetEntry("AND"));
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* or_entry,
                           library.GetEntry("OR"));
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* xor_entry,
                           library.GetEntry("XOR"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i0, module->ResolveNet("i0"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i1, module->ResolveNet("i1"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i2, module->ResolveNet("i2"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i3, module->ResolveNet("i3"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef and_o, module->ResolveNet("and_o"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef or_o, module->ResolveNet("or_o"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef xor_o, module->ResolveNet("xor_o"));

  absl::flat_hash_map<std::string, rtl::NetRef> and_params;
  and_params["A"] = i0;
  and_params["B"] = i1;
  and_params["Z"] = and_o;

  XLS_ASSERT_OK_AND_ASSIGN(
      rtl::Cell tmp_cell,
      rtl::Cell::Create(and_entry, "and", and_params, absl::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto and_cell, module->AddCell(tmp_cell));
  i0->NoteConnectedCell(and_cell);
  i1->NoteConnectedCell(and_cell);
  and_o->NoteConnectedCell(and_cell);

  absl::flat_hash_map<std::string, rtl::NetRef> or_params;
  or_params["A"] = i2;
  or_params["B"] = i3;
  or_params["Z"] = or_o;

  XLS_ASSERT_OK_AND_ASSIGN(
      tmp_cell,
      rtl::Cell::Create(or_entry, "or", or_params, absl::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto or_cell, module->AddCell(tmp_cell));
  i2->NoteConnectedCell(or_cell);
  i3->NoteConnectedCell(or_cell);
  or_o->NoteConnectedCell(or_cell);

  absl::flat_hash_map<std::string, rtl::NetRef> xor_params;
  xor_params["A"] = and_o;
  xor_params["B"] = or_o;
  xor_params["Z"] = xor_o;
  XLS_ASSERT_OK_AND_ASSIGN(
      tmp_cell,
      rtl::Cell::Create(xor_entry, "xor", xor_params, absl::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto xor_cell, module->AddCell(tmp_cell));
  and_o->NoteConnectedCell(xor_cell);
  or_o->NoteConnectedCell(xor_cell);
  xor_o->NoteConnectedCell(xor_cell);

  rtl::Netlist netlist;
  netlist.AddModule(std::move(module));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module_ptr,
                           netlist.GetModule("the_module"));
  Interpreter interpreter(&netlist);

  absl::flat_hash_map<const rtl::NetRef, bool> inputs;
  // AND inputs
  inputs[module_ptr->inputs()[0]] = true;
  inputs[module_ptr->inputs()[1]] = false;

  // OR inputs
  inputs[module_ptr->inputs()[2]] = true;
  inputs[module_ptr->inputs()[3]] = false;

  using OutputT = absl::flat_hash_map<const rtl::NetRef, bool>;
  XLS_ASSERT_OK_AND_ASSIGN(OutputT outputs,
                           interpreter.InterpretModule(module_ptr, inputs));

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module_ptr->outputs()[0]], 1);
}

TEST(InterpreterTest, Submodules) {
  std::string module_text = R"(
module submodule_0 (i2_0, i2_1, o2_0);
  input i2_0, i2_1;
  output o2_0;

  AND and0( .A(i2_0), .B(i2_1), .Z(o2_0) );
endmodule

module submodule_1 (i2_2, i2_3, o2_1);
  input i2_2, i2_3;
  output o2_1;

  OR or0( .A(i2_2), .B(i2_3), .Z(o2_1) );
endmodule

module submodule_2 (i1_0, i1_1, i1_2, i1_3, o1_0);
  input i1_0, i1_1, i1_2, i1_3;
  output o1_0;
  wire res0, res1;

  submodule_0 and0 ( .i2_0(i1_0), .i2_1(i1_1), .o2_0(res0) );
  submodule_1 or0 ( .i2_2(i1_2), .i2_3(i1_3), .o2_1(res1) );
  XOR xor0 ( .A(res0), .B(res1), .Z(o1_0) );
endmodule

module main (i0, i1, i2, i3, o0);
  input i0, i1, i2, i3;
  output o0;

  submodule_2 bleh( .i1_0(i0), .i1_1(i1), .i1_2(i2), .i1_3(i3), .o1_0(o0) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  absl::flat_hash_map<const rtl::NetRef, bool> inputs;
  inputs[module->inputs()[0]] = true;
  inputs[module->inputs()[1]] = false;
  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = false;

  using OutputT = absl::flat_hash_map<const rtl::NetRef, bool>;
  XLS_ASSERT_OK_AND_ASSIGN(OutputT outputs,
                           interpreter.InterpretModule(module, inputs));

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module->outputs()[0]], 1);
}

// Verifies that a [combinational] StateTable can be correctly interpreted in a
// design.
TEST(InterpreterTest, StateTables) {
  std::string module_text = R"(
module main(i0, i1, i2, i3, o0);
  input i0, i1, i2, i3;
  output o0;
  wire and0_out, and1_out;

  AND and0 ( .A(i0), .B(i1), .Z(and0_out) );
  STATETABLE_AND and1 (.A(i2), .B(i3), .Z(and1_out) );
  AND and2 ( .A(and0_out), .B(and1_out), .Z(o0) );
endmodule
  )";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  absl::flat_hash_map<const rtl::NetRef, bool> inputs;
  inputs[module->inputs()[0]] = true;
  inputs[module->inputs()[1]] = true;
  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = true;

  using OutputT = absl::flat_hash_map<const rtl::NetRef, bool>;
  XLS_ASSERT_OK_AND_ASSIGN(OutputT outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_TRUE(outputs.begin()->second);

  // Make sure that it works on the flip side, too.
  inputs[module->inputs()[2]] = false;
  inputs[module->inputs()[3]] = true;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);

  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = false;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);

  inputs[module->inputs()[2]] = false;
  inputs[module->inputs()[3]] = false;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);
}

}  // namespace
}  // namespace netlist
}  // namespace xls
