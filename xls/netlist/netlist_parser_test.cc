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

#include "xls/netlist/netlist_parser.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/fake_cell_library.h"

namespace xls {
namespace netlist {
namespace rtl {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(NetlistParserTest, EmptyModule) {
  std::string netlist = R"(module main(); endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
}

TEST(NetlistParserTest, EmptyModuleWithComment) {
  std::string netlist = R"(
// This is a module named main.
/* this is a
  multiline
       comment */
module main();
  // This area left intentionally blank.
  /* regular // comment */
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
}

TEST(NetlistParserTest, WireMultiDecl) {
  std::string netlist = R"(module main();
  wire foo, bar, baz;
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));

  XLS_ASSERT_OK_AND_ASSIGN(NetRef foo, m->ResolveNet("foo"));
  EXPECT_EQ("foo", foo->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef bar, m->ResolveNet("bar"));
  EXPECT_EQ("bar", bar->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef baz, m->ResolveNet("baz"));
  EXPECT_EQ("baz", baz->name());
}

TEST(NetlistParserTest, Attributes) {
  std::string netlist = R"((* on_module  = "foo" *)
module main(_a, z);
  (* on_net = "first" *)
  (* on_net = "second" *)
  wire foo, bar, baz;
  (* on_input = "baz" *)
  input _a;
  (* on_output = "baz" *)
  output z;
  (* on_instance = "bar" *)
  INV inv_0(.A(_a), .ZN(z));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
}

TEST(NetlistParserTest, InverterModule) {
  std::string netlist = R"(module main(a, z);
  input a;
  output z;
  INV inv_0(.A(a), .ZN(z));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a, m->ResolveNet("a"));
  EXPECT_EQ("a", a->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef z, m->ResolveNet("z"));
  EXPECT_EQ("z", z->name());

  XLS_ASSERT_OK_AND_ASSIGN(Cell * c, m->ResolveCell("inv_0"));
  EXPECT_EQ(cell_library.GetEntry("INV").value(), c->cell_library_entry());
  EXPECT_EQ("inv_0", c->name());
}

TEST(NetlistParserTest, AOI21WithMultiBitInput) {
  std::string netlist = R"(module main(i, o);
  input [2:0] i;
  output o;
  AOI21 aoi21_0(.A(i[2]), .B(i[1]), .C(i[0]), .ZN(o));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef i0, m->ResolveNet("i[0]"));
  EXPECT_EQ("i[0]", i0->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef i1, m->ResolveNet("i[1]"));
  EXPECT_EQ("i[1]", i1->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef i2, m->ResolveNet("i[2]"));
  EXPECT_EQ("i[2]", i2->name());
  EXPECT_THAT(m->ResolveNet("i[3]"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Could not find net: i[3]")));

  XLS_ASSERT_OK_AND_ASSIGN(Cell * c, m->ResolveCell("aoi21_0"));
  EXPECT_EQ(cell_library.GetEntry("AOI21").value(), c->cell_library_entry());
  EXPECT_EQ("aoi21_0", c->name());
}

TEST(NetlistParserTest, NumberFormats) {
  std::string netlist = R"(module main();
  wire z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11;
  INV inv_0(.A(10), .ZN(z0));
  INV inv_1(.A(1'b1), .ZN(z1));
  INV inv_2(.A(1'o1), .ZN(z2));
  INV inv_3(.A(1'd1), .ZN(z3));
  INV inv_4(.A(1'h1), .ZN(z4));
  INV inv_5(.A(1'B1), .ZN(z5));
  INV inv_6(.A(1'O1), .ZN(z6));
  INV inv_7(.A(1'D1), .ZN(z7));
  INV inv_8(.A(1'H1), .ZN(z8));
  INV inv_9(.A(10'o777), .ZN(z9));
  INV inv_10(.A(20'd100), .ZN(z10));
  INV inv_11(.A(30'hbeef), .ZN(z11));
endmodule)";

  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
}

TEST(NetlistParserTest, MoreNumberFormats) {
  auto make_module = [](const std::string& number) {
    std::string module_base = R"(module main();
wire z0;
INV inv_0(.A($0), .ZN(z0));
endmodule)";
    return absl::Substitute(module_base, number);
  };

  std::vector<std::pair<std::string, int64_t>> test_cases({
      {"1'b1", 1},
      {"1'o1", 1},
      {"8'd255", 255},
      {"8'sd127", 127},
      {"8'sd255", -1},
      {"8'sd253", -3},
  });

  // For each test case, make sure we can find a netlist for the given number
  // (matching the Verilog number string) in the module.
  for (const auto& test_case : test_cases) {
    std::string module_text = make_module(test_case.first);
    Scanner scanner(module_text);
    XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                             Parser::ParseNetlist(&cell_library, &scanner));
    XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
    XLS_ASSERT_OK(m->ResolveNumber(test_case.second).status());
  }
}

TEST(NetlistParserTest, Lut) {
  // Test that a "real world" netlist parses
  std::string netlist =
      R"(/* Generated by Yosys 0.9+3675 (git sha1 71ca9a82, gcc 10.2.0-9 -fPIC -Os) */

(* top =  1  *)
(* src = "and.v:1.1-3.10" *)
module main(a0, a1, a2, a3, q0);
  (* src = "and.v:1.19-1.21" *)
  input a0;
  (* src = "and.v:1.29-1.31" *)
  input a1;
  (* src = "and.v:1.39-1.41" *)
  input a2;
  (* src = "and.v:1.49-1.51" *)
  input a3;
  (* src = "and.v:1.60-1.62" *)
  output q0;
  (* module_not_derived = 32'd1 *)
  (* src = "/usr/local/google/home/fcz/opt/yosys/bin/../share/yosys/ice40/cells_map.v:26.33-27.52" *)
  SB_LUT4 #(
    .LUT_INIT(16'h8000)
  ) q0_SB_LUT4_O (
    .I0(a0),
    .I1(a1),
    .I2(a2),
    .I3(a3),
    .O(q0)
  );
endmodule
)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());

  XLS_ASSERT_OK_AND_ASSIGN(Cell * lut_cell, m->ResolveCell("q0_SB_LUT4_O"));
  EXPECT_EQ("<lut_0x8000>", lut_cell->cell_library_entry()->name());
}

TEST(NetlistParserTest, PortOrderDeclarationDuplicateError) {
  // Test that a "real world" netlist parses
  std::string netlist =
      R"(
module main(double_decl);
  input double_decl;
  input double_decl;
endmodule
)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  EXPECT_THAT(
      Parser::ParseNetlist(&cell_library, &scanner),
      StatusIs(absl::StatusCode::kAlreadyExists,
               HasSubstr("Duplicate declaration of port 'double_decl'.")));
}

TEST(NetlistParserTest, PortOrderDeclarationNotFoundError) {
  // Test that a "real world" netlist parses
  std::string netlist =
      R"(
module main(o);
  input missing;
  output o;
endmodule
)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  EXPECT_THAT(
      Parser::ParseNetlist(&cell_library, &scanner),
      StatusIs(absl::StatusCode::kNotFound,
               HasSubstr("No match for input 'missing' in parameter list.")));
}

static void TestAssignHelper(const Module* m) {
  // Walk over the outputs and make sure that each of them is covered by an
  // assignment, that the right-hand-side of the assignment is an input, and
  // that while walking the ouputs, we cover all the assignments.
  const std::vector<NetRef>& inputs = m->inputs();
  const std::vector<NetRef>& outputs = m->outputs();
  const absl::flat_hash_map<NetRef, NetRef>& assigns = m->assigns();
  absl::flat_hash_set<NetRef> visited_assigns;
  for (const NetRef& output : outputs) {
    EXPECT_TRUE(assigns.contains(output));
    const NetRef rhs = assigns.at(output);
    EXPECT_TRUE(std::find(inputs.begin(), inputs.end(), rhs) != inputs.end() ||
                rhs == m->zero() || rhs == m->one());
    EXPECT_FALSE(visited_assigns.contains(output));
    visited_assigns.insert(output);
  }
  EXPECT_EQ(visited_assigns.size(), assigns.size());
}

TEST(NetlistParserTest, SimpleAssign) {
  std::string netlist = R"(module main(i,o);
  input i;
  output o;
  assign o = i;
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());

  TestAssignHelper(m);
}

TEST(NetlistParserTest, SimpleRangeAssign) {
  std::string netlist = R"(module main(i,o);
  input [1:0] i;
  output [1:0] o;
  assign o[1:0] = i[1:0];
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());

  TestAssignHelper(m);
}

TEST(NetlistParserTest, SimpleRangeAssign2) {
  std::string netlist = R"(module main(i,j,o);
  input i;
  input j;
  output [1:0] o;
  assign o[1] = i;
  assign o[0] = j;
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());

  TestAssignHelper(m);
}

TEST(NetlistParserTest, ComplexAssign) {
  std::string netlist = R"(module main(a, b, c, o);
  input [3:0] a;
  input b;
  input c;
  output [5:0] o;

  assign o[5] = c;
  assign o[3:2] = a[3:2];
  assign { o[4], o[1:0] } = { b, a[1:0] };
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a0, m->ResolveNet("a[0]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a1, m->ResolveNet("a[1]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a2, m->ResolveNet("a[2]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a3, m->ResolveNet("a[3]"));
  EXPECT_EQ("a[0]", a0->name());
  EXPECT_EQ("a[1]", a1->name());
  EXPECT_EQ("a[2]", a2->name());
  EXPECT_EQ("a[3]", a3->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef b, m->ResolveNet("b"));
  EXPECT_EQ("b", b->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef c, m->ResolveNet("c"));
  EXPECT_EQ("c", c->name());
  EXPECT_THAT(m->ResolveNet("a[4]"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Could not find net: a[4]")));

  TestAssignHelper(m);
}

TEST(NetlistParserTest, ComplexMismatchedAssignA) {
  std::string netlist = R"(module main(a, b, c, o);
  input [3:0] a;
  input b;
  input c;
  output [5:0] o;
  assign o = { c, a[3:2], b, a[1:0] };
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a0, m->ResolveNet("a[0]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a1, m->ResolveNet("a[1]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a2, m->ResolveNet("a[2]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a3, m->ResolveNet("a[3]"));
  EXPECT_EQ("a[0]", a0->name());
  EXPECT_EQ("a[1]", a1->name());
  EXPECT_EQ("a[2]", a2->name());
  EXPECT_EQ("a[3]", a3->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef b, m->ResolveNet("b"));
  EXPECT_EQ("b", b->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef c, m->ResolveNet("c"));
  EXPECT_EQ("c", c->name());

  TestAssignHelper(m);
}

TEST(NetlistParserTest, ComplexMismatchedAssignB) {
  std::string netlist = R"(module main(a, o);
  input [3:0] a;
  output [5:0] o;
  assign o = { a[3:2], 2'b10, a[1:0] };
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  EXPECT_EQ("main", m->name());
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a0, m->ResolveNet("a[0]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a1, m->ResolveNet("a[1]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a2, m->ResolveNet("a[2]"));
  XLS_ASSERT_OK_AND_ASSIGN(NetRef a3, m->ResolveNet("a[3]"));
  EXPECT_EQ("a[0]", a0->name());
  EXPECT_EQ("a[1]", a1->name());
  EXPECT_EQ("a[2]", a2->name());
  EXPECT_EQ("a[3]", a3->name());

  TestAssignHelper(m);
}

}  // namespace
}  // namespace rtl
}  // namespace netlist
}  // namespace xls
