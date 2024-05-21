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

#include "xls/netlist/logical_effort.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"

namespace xls::netlist::logical_effort {
namespace {

TEST(LogicalEffortTest, FO4Delay) {
  std::string netlist = R"(module fo4(i, o);
  input i;
  wire i_n;
  output [3:0] o;

  INV inv_0(.A(i), .ZN(i_n));
  // Fanned-out-to inverters.
  INV inv_fo0(.A(i_n), .ZN(o[0]));
  INV inv_fo1(.A(i_n), .ZN(o[1]));
  INV inv_fo2(.A(i_n), .ZN(o[2]));
  INV inv_fo3(.A(i_n), .ZN(o[3]));
endmodule)";
  rtl::Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<rtl::Netlist> n,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(auto m, n->GetModule("fo4"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Cell * inv_0, m->ResolveCell("inv_0"));
  XLS_ASSERT_OK_AND_ASSIGN(double delay, ComputeDelay(inv_0));
  // Per Logical Effort book example 1.2.
  EXPECT_FLOAT_EQ(5.0, delay);
}

// Per Logical Effort book example 1.3.
TEST(LogicalEffortTest, FourInputNorDriving10Identical) {
  std::string netlist = R"(module test(i, i_aux, o);
  input [3:0] i;
  input i_aux;
  wire fo;
  output [9:0] o;

  // Nor4 that produces the fanned-out signal.
  NOR4 nor_with_fo(.A(i[3]), .B(i[2]), .C(i[1]), .D(i[3]), .ZN(fo));

  // Fanned-out-to nor4s.
  NOR4 fo0(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[0]));
  NOR4 fo1(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[1]));
  NOR4 fo2(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[2]));
  NOR4 fo3(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[3]));
  NOR4 fo4(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[4]));
  NOR4 fo5(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[5]));
  NOR4 fo6(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[6]));
  NOR4 fo7(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[7]));
  NOR4 fo8(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[8]));
  NOR4 fo9(.A(fo), .B(i_aux), .C(i_aux), .D(i_aux), .ZN(o[9]));
endmodule)";
  rtl::Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<rtl::Netlist> n,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(auto m, n->GetModule("test"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::Cell * nor_with_fo,
                           m->ResolveCell("nor_with_fo"));
  XLS_ASSERT_OK_AND_ASSIGN(double delay, ComputeDelay(nor_with_fo));
  EXPECT_FLOAT_EQ(34.0, delay);
}

TEST(LogicalEffortTest, PathDelay) {
  std::string netlist = R"(module test(ai, i_aux, bo);
  input ai;
  wire i_aux, y, z;
  output bo;

  NAND nand_0(.A(ai), .B(i_aux), .ZN(y));
  NAND nand_1(.A(y), .B(i_aux), .ZN(z));
  NAND nand_2(.A(z), .B(i_aux), .ZN(bo));
endmodule)";
  rtl::Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<rtl::Netlist> n,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(auto m, n->GetModule("test"));
  std::vector<rtl::Cell*> path = {
      m->ResolveCell("nand_0").value(),
      m->ResolveCell("nand_1").value(),
      m->ResolveCell("nand_2").value(),
  };
  XLS_ASSERT_OK_AND_ASSIGN(double path_logical_effort,
                           ComputePathLogicalEffort(path));
  EXPECT_NEAR(2.37, path_logical_effort, 1e-3);
  XLS_ASSERT_OK_AND_ASSIGN(double path_parasitic_delay,
                           ComputePathParasiticDelay(path));
  EXPECT_FLOAT_EQ(6.0, path_parasitic_delay);
  constexpr double kInputPinCapacitance = 1.0;
  constexpr double kOutputPinCapacitance = 1.0;
  // Logical Effort book example 1.4.
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        double least_delay,
        ComputePathLeastDelayAchievable(path, kInputPinCapacitance,
                                        kOutputPinCapacitance));
    EXPECT_NEAR(10.0, least_delay, 1e-3);
  }

  // Logical Effort book example 1.5.
  {
    constexpr double kOutputPinCapacitance = 8.0;
    XLS_ASSERT_OK_AND_ASSIGN(
        double least_delay,
        ComputePathLeastDelayAchievable(path, kInputPinCapacitance,
                                        kOutputPinCapacitance));
    EXPECT_NEAR(14.0, least_delay, 1e-3);
  }
}

TEST(LogicalEffortTest, ManyOperandXors) {
  XLS_ASSERT_OK_AND_ASSIGN(double two_input_xor_effort,
                           GetLogicalEffort(CellKind::kXor, 2));
  EXPECT_NEAR(two_input_xor_effort, 4.0, 1e-3);

  XLS_ASSERT_OK_AND_ASSIGN(double five_input_xor_effort,
                           GetLogicalEffort(CellKind::kXor, 5));
  EXPECT_NEAR(five_input_xor_effort, 12.0, 1e-3);

  XLS_ASSERT_OK_AND_ASSIGN(double many_input_xor_effort,
                           GetLogicalEffort(CellKind::kXor, 100));
  EXPECT_NEAR(many_input_xor_effort, 28.0, 1e-3);
}

TEST(LogicalEffortTest, SingleOperandOperations) {
  XLS_ASSERT_OK_AND_ASSIGN(double one_input_nor_effort,
                           GetLogicalEffort(CellKind::kNor, 1));
  EXPECT_NEAR(one_input_nor_effort, 1.0, 1e-3);

  XLS_ASSERT_OK_AND_ASSIGN(double one_input_nand_effort,
                           GetLogicalEffort(CellKind::kNand, 1));
  EXPECT_NEAR(one_input_nand_effort, 1.0, 1e-3);

  XLS_ASSERT_OK_AND_ASSIGN(double one_input_xor_effort,
                           GetLogicalEffort(CellKind::kXor, 1));
  EXPECT_NEAR(one_input_xor_effort, 0.0, 1e-3);
}

}  // namespace
}  // namespace xls::netlist::logical_effort
