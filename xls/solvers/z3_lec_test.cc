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

#include "xls/solvers/z3_lec.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace solvers {
namespace z3 {
namespace {

using netlist::rtl::Netlist;

absl::StatusOr<bool> Match(const std::string& ir_text,
                           const std::string& netlist_text, bool expect_equal) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       Parser::ParsePackage(ir_text));
  XLS_ASSIGN_OR_RETURN(Function * entry_function, package->GetTopAsFunction());

  XLS_ASSIGN_OR_RETURN(netlist::CellLibrary cell_library,
                       netlist::MakeFakeCellLibrary());
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Netlist> netlist,
      netlist::rtl::Parser::ParseNetlist(&cell_library, &scanner));

  LecParams params;
  params.ir_package = package.get();
  params.ir_function = entry_function;

  params.netlist = netlist.get();
  params.netlist_module_name = "main";

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Lec> lec, Lec::Create(params));
  return lec->Run();
}

// This test verifies that we can do a simple LEC.
TEST(Z3LecTest, SimpleLec) {
  std::string ir_text = R"(
package p

top fn main(input: bits[4]) -> bits[4] {
  ret not.2: bits[4] = not(input)
}
)";

  std::string netlist_text = R"(
module main ( clk, input_3_, input_2_, input_1_, input_0_, out_3_, out_2_, out_1_, out_0_);
  input clk, input_3_, input_2_, input_1_, input_0_;
  output out_3_, out_2_, out_1_, out_0_;
  wire p0_input_3_, p0_input_2_, p0_input_1_, p0_input_0_,
       p0_not_2_comb_3_, p0_not_2_comb_2_, p0_not_2_comb_1_, p0_not_2_comb_0_;

  // Inputs are always latched, so this is representative.
  DFF p0_input_reg_3_ ( .D(input_3_), .CLK(clk), .Q(p0_input_3_) );
  DFF p0_input_reg_2_ ( .D(input_2_), .CLK(clk), .Q(p0_input_2_) );
  DFF p0_input_reg_1_ ( .D(input_1_), .CLK(clk), .Q(p0_input_1_) );
  DFF p0_input_reg_0_ ( .D(input_0_), .CLK(clk), .Q(p0_input_0_) );

  INV p0_not_2_3_ ( .A(p0_input_3_), .ZN(p0_not_2_comb_3_) );
  INV p0_not_2_2_ ( .A(p0_input_2_), .ZN(p0_not_2_comb_2_) );
  INV p0_not_2_1_ ( .A(p0_input_1_), .ZN(p0_not_2_comb_1_) );
  INV p0_not_2_0_ ( .A(p0_input_0_), .ZN(p0_not_2_comb_0_) );

  DFF p0_not_2_reg_3_ (.D(p0_not_2_comb_3_), .CLK(clk), .Q(out_3_));
  DFF p0_not_2_reg_2_ (.D(p0_not_2_comb_2_), .CLK(clk), .Q(out_2_));
  DFF p0_not_2_reg_1_ (.D(p0_not_2_comb_1_), .CLK(clk), .Q(out_1_));
  DFF p0_not_2_reg_0_ (.D(p0_not_2_comb_0_), .CLK(clk), .Q(out_0_));
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(bool match,
                           Match(ir_text, netlist_text, /*expect_equal=*/true));
  ASSERT_TRUE(match);
}

TEST(Z3LecTest, AdderLec) {
  std::string ir_text = R"(
package p

top fn main(a: bits[2], b: bits[2]) -> bits[2] {
  ret add.3: bits[2] = add(a, b)
}
)";

  std::string netlist_text = R"(
module main(clk, a_1_, a_0_, b_1_, b_0_, out_1_, out_0_);
  input clk, a_1_, a_0_, b_1_, b_0_;
  output out_1_, out_0_;
  wire p0_a_1_, p0_a_0_, p0_b_1_, p0_b_0_, p0_add_3_comb_0_, p0_add_3_comb_1_, carry, high;

  DFF p0_a_reg_1_ ( .D(a_1_), .CLK(clk), .Q(p0_a_1_) );
  DFF p0_a_reg_0_ ( .D(a_0_), .CLK(clk), .Q(p0_a_0_) );
  DFF p0_b_reg_1_ ( .D(b_1_), .CLK(clk), .Q(p0_b_1_) );
  DFF p0_b_reg_0_ ( .D(b_0_), .CLK(clk), .Q(p0_b_0_) );

  XOR out_0_cell ( .A(p0_a_0_), .B(p0_b_0_), .Z(p0_add_3_comb_0_) );

  AND carry_cell ( .A(p0_a_0_), .B(p0_b_0_), .Z(carry) );
  XOR high_cell ( .A(p0_a_1_), .B(p0_b_1_), .Z(high) );
  XOR out_1_cell ( .A(high), .B(carry), .Z(p0_add_3_comb_1_) );

  DFF p0_add_3_reg_1_ ( .D(p0_add_3_comb_1_), .CLK(clk), .Q(out_1_) );
  DFF p0_add_3_reg_0_ ( .D(p0_add_3_comb_0_), .CLK(clk), .Q(out_0_) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(bool match,
                           Match(ir_text, netlist_text, /*expect_equal=*/true));
  ASSERT_TRUE(match);
}

// Test verifies that z3::Lec correctly reports a mismatch in cases where the
// IR and netlist disagree.
TEST(Z3LecTest, FailsBadComparison) {
  std::string ir_text = R"(
package p

top fn main(input: bits[4]) -> bits[4] {
  ret not.2: bits[4] = not(input)
}
)";

  std::string netlist_text = R"(
module main ( clk, input_3_, input_2_, input_1_, input_0_, out_3_, out_2_, out_1_, out_0_);
  input clk, input_3_, input_2_, input_1_, input_0_;
  output out_3_, out_2_, out_1_, out_0_;
  wire p0_input_3_, p0_input_2_, p0_input_1_, p0_input_0_,
       p0_not_2_comb_3_, p0_not_2_comb_2_, p0_not_2_comb_1_, p0_not_2_comb_0_;

  // Inputs are always latched, so this is representative.
  DFF p0_input_reg_3_ ( .D(input_3_), .CLK(clk), .Q(p0_input_3_) );
  DFF p0_input_reg_2_ ( .D(input_2_), .CLK(clk), .Q(p0_input_2_) );
  DFF p0_input_reg_1_ ( .D(input_1_), .CLK(clk), .Q(p0_input_1_) );
  DFF p0_input_reg_0_ ( .D(input_0_), .CLK(clk), .Q(p0_input_0_) );

  INV p0_not_2_3_ ( .A(p0_input_3_), .ZN(p0_not_2_comb_3_) );
  INV p0_not_2_2_ ( .A(p0_input_2_), .ZN(p0_not_2_comb_2_) );
  OR  p0_not_2_1_ ( .A(p0_input_1_), .B(p0_input_1_), .Z(p0_not_2_comb_1_) );
  INV p0_not_2_0_ ( .A(p0_input_0_), .ZN(p0_not_2_comb_0_) );

  DFF p0_not_2_reg_3_ (.D(p0_not_2_comb_3_), .CLK(clk), .Q(out_3_));
  DFF p0_not_2_reg_2_ (.D(p0_not_2_comb_2_), .CLK(clk), .Q(out_2_));
  DFF p0_not_2_reg_1_ (.D(p0_not_2_comb_1_), .CLK(clk), .Q(out_1_));
  DFF p0_not_2_reg_0_ (.D(p0_not_2_comb_0_), .CLK(clk), .Q(out_0_));
endmodule
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      bool match, Match(ir_text, netlist_text, /*expect_equal=*/false));
  ASSERT_FALSE(match);
}

// This test verifies that we can do a simple multi-stage LEC.
// There are three defined stages:
// [inputs] -> p0_AND -> p1_OR -> p2 NOT -> [outputs]
TEST(Z3LecTest, SimpleMultiStage) {
  std::string ir_text = R"(
package p

top fn main(i0: bits[1], i1: bits[1], i2: bits[1], i3: bits[1]) -> bits[1] {
  // Stage 1:
  and.1: bits[1] = and(i0, i1)
  and.2: bits[1] = and(i2, i3)

  // Stage 2:
  or.3: bits[1] = or(and.1, and.2)

  // Stage 3:
  ret not.4: bits[1] = not(or.3)
}
)";

  std::string netlist_text = R"(
module main ( clk, i3, i2, i1, i0, out_0);
  input clk, i3, i2, i1, i0;
  output out_0;
  wire p0_i3, p0_i2, p0_i1, p0_i0,
       p1_and_1_comb, p1_and_2_comb, p1_and_1, p1_and_2,
       p2_or_3_comb, p2_or_3;

  DFF p0_i3_reg ( .D(i3), .CLK(clk), .Q(p0_i3) );
  DFF p0_i2_reg ( .D(i2), .CLK(clk), .Q(p0_i2) );
  DFF p0_i1_reg ( .D(i1), .CLK(clk), .Q(p0_i1) );
  DFF p0_i0_reg ( .D(i0), .CLK(clk), .Q(p0_i0) );

  AND p1_and_1 ( .A(p0_i0), .B(p0_i1), .Z(p1_and_1_comb) );
  AND p1_and_2 ( .A(p0_i2), .B(p0_i3), .Z(p1_and_2_comb) );
  DFF p1_and_1_reg ( .D(p1_and_1_comb), .CLK(clk), .Q(p1_and_1) );
  DFF p1_and_2_reg ( .D(p1_and_2_comb), .CLK(clk), .Q(p1_and_2) );

  OR p2_or_3 ( .A(p1_and_1), .B(p1_and_2), .Z(p2_or_3_comb) );
  DFF p2_or_3_reg ( .D(p2_or_3_comb), .CLK(clk), .Q(p2_or_3) );

  INV p3_not_4 ( .A(p2_or_3), .ZN(out_0) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * entry_function,
                           package->GetTopAsFunction());
  XLS_ASSERT_OK_AND_ASSIGN(netlist::CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Netlist> netlist,
      netlist::rtl::Parser::ParseNetlist(&cell_library, &scanner));

  LecParams params;
  params.ir_package = package.get();
  params.ir_function = entry_function;

  params.netlist = netlist.get();
  params.netlist_module_name = "main";

  ScheduleCycleMap cycle_map;
  for (Node* node : entry_function->nodes()) {
    if (node->Is<Param>()) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("and") != std::string::npos) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("or") != std::string::npos) {
      cycle_map[node] = 1;
    } else {
      cycle_map[node] = 2;
    }
  }

  PipelineSchedule schedule(entry_function, cycle_map, /*length=*/3);

  for (int i = 0; i < schedule.length(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Lec> lec,
                             Lec::CreateForStage(params, schedule, i));
    ASSERT_TRUE(lec->Run());
    LOG(INFO) << "Pass stage " << i;
  }
}

// This test verifies that a non-matching set of inputs "correctly" fails. This
// uses the same IR as in SimpleMultiStage, but with one swapped cell in stage
// 0.
TEST(Z3LecTest, FailsBadMultiStage) {
  std::string ir_text = R"(
package p

top fn main(i0: bits[1], i1: bits[1], i2: bits[1], i3: bits[1]) -> bits[1] {
  // Stage 1:
  and.1: bits[1] = and(i0, i1)
  and.2: bits[1] = and(i2, i3)

  // Stage 2:
  or.3: bits[1] = or(and.1, and.2)

  // Stage 3:
  ret not.4: bits[1] = not(or.3)
}
)";

  std::string netlist_text = R"(
module main ( clk, i3, i2, i1, i0, out_0);
  input clk, i3, i2, i1, i0;
  output out_0;
  wire p0_i3, p0_i2, p0_i1, p0_i0,
       p1_and_1_comb, p1_and_2_comb, p1_and_1, p1_and_2,
       p2_or_3_comb, p2_or_3;

  DFF p0_i3_reg ( .D(i3), .CLK(clk), .Q(p0_i3) );
  DFF p0_i2_reg ( .D(i2), .CLK(clk), .Q(p0_i2) );
  DFF p0_i1_reg ( .D(i1), .CLK(clk), .Q(p0_i1) );
  DFF p0_i0_reg ( .D(i0), .CLK(clk), .Q(p0_i0) );

  // Stage 1:
  AND p1_and_1 ( .A(p0_i0), .B(p0_i1), .Z(p1_and_1_comb) );
  OR p1_and_2 ( .A(p0_i2), .B(p0_i3), .Z(p1_and_2_comb) );
  DFF p1_and_1_reg ( .D(p1_and_1_comb), .CLK(clk), .Q(p1_and_1) );
  DFF p1_and_2_reg ( .D(p1_and_2_comb), .CLK(clk), .Q(p1_and_2) );

  // Stage 2:
  OR p2_or_3 ( .A(p1_and_1), .B(p1_and_2), .Z(p2_or_3_comb) );
  DFF p2_or_3_reg ( .D(p2_or_3_comb), .CLK(clk), .Q(p2_or_3) );

  // Stage 3:
  INV p3_not_4 ( .A(p2_or_3), .ZN(out_0) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * entry_function,
                           package->GetTopAsFunction());
  XLS_ASSERT_OK_AND_ASSIGN(netlist::CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Netlist> netlist,
      netlist::rtl::Parser::ParseNetlist(&cell_library, &scanner));

  LecParams params;
  params.ir_package = package.get();
  params.ir_function = entry_function;

  params.netlist = netlist.get();
  params.netlist_module_name = "main";

  ScheduleCycleMap cycle_map;
  for (Node* node : entry_function->nodes()) {
    if (node->Is<Param>()) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("and") != std::string::npos) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("or") != std::string::npos) {
      cycle_map[node] = 1;
    } else {
      cycle_map[node] = 2;
    }
  }

  PipelineSchedule schedule(entry_function, cycle_map, /*length=*/3);
  for (int i = 0; i < schedule.length(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Lec> lec,
                             Lec::CreateForStage(params, schedule, i));
    if (i == 0) {
      ASSERT_FALSE(lec->Run());
    } else {
      ASSERT_TRUE(lec->Run());
    }
    LOG(INFO) << "Pass stage " << i;
  }
}

// This test verifies that we can do a multibit LEC with >1b inputs.
// There are three defined stages:
// [inputs] -> p0_AND -> p1_OR -> p2 NOT -> [outputs]
// This test also serves as a great example of why we don't like writing this
// stuff out by hand. Sheesh.
TEST(Z3LecTest, MultibitMultiStage) {
  std::string ir_text = R"(
package p

top fn main(i0: bits[4], i1: bits[4], i2: bits[4], i3: bits[4]) -> bits[4] {
  // Stage 1:
  and.1: bits[4] = and(i0, i1)
  and.2: bits[4] = and(i2, i3)

  // Stage 2:
  or.3: bits[4] = or(and.1, and.2)

  // Stage 3:
  ret not.4: bits[4] = not(or.3)
}
)";

  std::string netlist_text = R"(
module main ( clk,
              i3_3_, i3_2_, i3_1_, i3_0_,
              i2_3_, i2_2_, i2_1_, i2_0_,
              i1_3_, i1_2_, i1_1_, i1_0_,
              i0_3_, i0_2_, i0_1_, i0_0_,
              out_3_, out_2_, out_1_, out_0_);
  input clk,
        i3_3_, i3_2_, i3_1_, i3_0_,
        i2_3_, i2_2_, i2_1_, i2_0_,
        i1_3_, i1_2_, i1_1_, i1_0_,
        i0_3_, i0_2_, i0_1_, i0_0_;
  output out_3_, out_2_, out_1_, out_0_;
  wire p0_i3_3_, p0_i3_2_, p0_i3_1_, p0_i3_0_,
       p0_i2_3_, p0_i2_2_, p0_i2_1_, p0_i2_0_,
       p0_i1_3_, p0_i1_2_, p0_i1_1_, p0_i1_0_,
       p0_i0_3_, p0_i0_2_, p0_i0_1_, p0_i0_0_,
       p1_and_1_comb_3_, p1_and_1_comb_2_, p1_and_1_comb_1_, p1_and_1_comb_0_,
       p1_and_2_comb_3_, p1_and_2_comb_2_, p1_and_2_comb_1_, p1_and_2_comb_0_,
       p1_and_1_3_, p1_and_1_2_, p1_and_1_1_, p1_and_1_0_,
       p1_and_2_3_, p1_and_2_2_, p1_and_2_1_, p1_and_2_0_,
       p2_or_3_comb_3_, p2_or_3_comb_2_, p2_or_3_comb_1_, p2_or_3_comb_0_,
       p2_or_3_3_, p2_or_3_2_, p2_or_3_1_, p2_or_3_0_,
       p3_not_4_comb_3_, p3_not_4_comb_2_, p3_not_4_comb_1_, p3_not_4_comb_0_;

  DFF p0_i3_reg_3_ ( .D(i3_3_), .CLK(clk), .Q(p0_i3_3_) );
  DFF p0_i3_reg_2_ ( .D(i3_2_), .CLK(clk), .Q(p0_i3_2_) );
  DFF p0_i3_reg_1_ ( .D(i3_1_), .CLK(clk), .Q(p0_i3_1_) );
  DFF p0_i3_reg_0_ ( .D(i3_0_), .CLK(clk), .Q(p0_i3_0_) );

  DFF p0_i2_reg_3_ ( .D(i2_3_), .CLK(clk), .Q(p0_i2_3_) );
  DFF p0_i2_reg_2_ ( .D(i2_2_), .CLK(clk), .Q(p0_i2_2_) );
  DFF p0_i2_reg_1_ ( .D(i2_1_), .CLK(clk), .Q(p0_i2_1_) );
  DFF p0_i2_reg_0_ ( .D(i2_0_), .CLK(clk), .Q(p0_i2_0_) );

  DFF p0_i1_reg_3_ ( .D(i1_3_), .CLK(clk), .Q(p0_i1_3_) );
  DFF p0_i1_reg_2_ ( .D(i1_2_), .CLK(clk), .Q(p0_i1_2_) );
  DFF p0_i1_reg_1_ ( .D(i1_1_), .CLK(clk), .Q(p0_i1_1_) );
  DFF p0_i1_reg_0_ ( .D(i1_0_), .CLK(clk), .Q(p0_i1_0_) );

  DFF p0_i0_reg_3_ ( .D(i0_3_), .CLK(clk), .Q(p0_i0_3_) );
  DFF p0_i0_reg_2_ ( .D(i0_2_), .CLK(clk), .Q(p0_i0_2_) );
  DFF p0_i0_reg_1_ ( .D(i0_1_), .CLK(clk), .Q(p0_i0_1_) );
  DFF p0_i0_reg_0_ ( .D(i0_0_), .CLK(clk), .Q(p0_i0_0_) );

  AND p1_and_1_3_ ( .A(p0_i0_3_), .B(p0_i1_3_), .Z(p1_and_1_comb_3_) );
  AND p1_and_1_2_ ( .A(p0_i0_2_), .B(p0_i1_2_), .Z(p1_and_1_comb_2_) );
  AND p1_and_1_1_ ( .A(p0_i0_1_), .B(p0_i1_1_), .Z(p1_and_1_comb_1_) );
  AND p1_and_1_0_ ( .A(p0_i0_0_), .B(p0_i1_0_), .Z(p1_and_1_comb_0_) );

  AND p1_and_2_3_ ( .A(p0_i2_3_), .B(p0_i3_3_), .Z(p1_and_2_comb_3_) );
  AND p1_and_2_2_ ( .A(p0_i2_2_), .B(p0_i3_2_), .Z(p1_and_2_comb_2_) );
  AND p1_and_2_1_ ( .A(p0_i2_1_), .B(p0_i3_1_), .Z(p1_and_2_comb_1_) );
  AND p1_and_2_0_ ( .A(p0_i2_0_), .B(p0_i3_0_), .Z(p1_and_2_comb_0_) );

  DFF p1_and_1_reg_3_ ( .D(p1_and_1_comb_3_), .CLK(clk), .Q(p1_and_1_3_) );
  DFF p1_and_1_reg_2_ ( .D(p1_and_1_comb_2_), .CLK(clk), .Q(p1_and_1_2_) );
  DFF p1_and_1_reg_1_ ( .D(p1_and_1_comb_1_), .CLK(clk), .Q(p1_and_1_1_) );
  DFF p1_and_1_reg_0_ ( .D(p1_and_1_comb_0_), .CLK(clk), .Q(p1_and_1_0_) );

  DFF p1_and_2_reg_3_ ( .D(p1_and_2_comb_3_), .CLK(clk), .Q(p1_and_2_3_) );
  DFF p1_and_2_reg_2_ ( .D(p1_and_2_comb_2_), .CLK(clk), .Q(p1_and_2_2_) );
  DFF p1_and_2_reg_1_ ( .D(p1_and_2_comb_1_), .CLK(clk), .Q(p1_and_2_1_) );
  DFF p1_and_2_reg_0_ ( .D(p1_and_2_comb_0_), .CLK(clk), .Q(p1_and_2_0_) );

  OR p2_or_3_3_ ( .A(p1_and_1_3_), .B(p1_and_2_3_), .Z(p2_or_3_comb_3_) );
  OR p2_or_3_2_ ( .A(p1_and_1_2_), .B(p1_and_2_2_), .Z(p2_or_3_comb_2_) );
  OR p2_or_3_1_ ( .A(p1_and_1_1_), .B(p1_and_2_1_), .Z(p2_or_3_comb_1_) );
  OR p2_or_3_0_ ( .A(p1_and_1_0_), .B(p1_and_2_0_), .Z(p2_or_3_comb_0_) );

  DFF p2_or_2_reg_3_ ( .D(p2_or_3_comb_3_), .CLK(clk), .Q(p2_or_3_3_) );
  DFF p2_or_2_reg_2_ ( .D(p2_or_3_comb_2_), .CLK(clk), .Q(p2_or_3_2_) );
  DFF p2_or_2_reg_1_ ( .D(p2_or_3_comb_1_), .CLK(clk), .Q(p2_or_3_1_) );
  DFF p2_or_2_reg_0_ ( .D(p2_or_3_comb_0_), .CLK(clk), .Q(p2_or_3_0_) );

  INV p3_not_4_3_ ( .A(p2_or_3_3_), .ZN(p3_not_4_comb_3_) );
  INV p3_not_4_2_ ( .A(p2_or_3_2_), .ZN(p3_not_4_comb_2_) );
  INV p3_not_4_1_ ( .A(p2_or_3_1_), .ZN(p3_not_4_comb_1_) );
  INV p3_not_4_0_ ( .A(p2_or_3_0_), .ZN(p3_not_4_comb_0_) );

  DFF p3_not_4_reg_3_ ( .D(p3_not_4_comb_3_), .CLK(clk), .Q(out_3_) );
  DFF p3_not_4_reg_2_ ( .D(p3_not_4_comb_2_), .CLK(clk), .Q(out_2_) );
  DFF p3_not_4_reg_1_ ( .D(p3_not_4_comb_1_), .CLK(clk), .Q(out_1_) );
  DFF p3_not_4_reg_0_ ( .D(p3_not_4_comb_0_), .CLK(clk), .Q(out_0_) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * entry_function,
                           package->GetTopAsFunction());
  XLS_ASSERT_OK_AND_ASSIGN(netlist::CellLibrary cell_library,
                           netlist::MakeFakeCellLibrary());
  netlist::rtl::Scanner scanner(netlist_text);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Netlist> netlist,
      netlist::rtl::Parser::ParseNetlist(&cell_library, &scanner));

  LecParams params;
  params.ir_package = package.get();
  params.ir_function = entry_function;

  params.netlist = netlist.get();
  params.netlist_module_name = "main";

  ScheduleCycleMap cycle_map;
  for (Node* node : entry_function->nodes()) {
    if (node->Is<Param>()) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("and") != std::string::npos) {
      cycle_map[node] = 0;
    } else if (node->GetName().find("or") != std::string::npos) {
      cycle_map[node] = 1;
    } else {
      cycle_map[node] = 2;
    }
  }

  PipelineSchedule schedule(entry_function, cycle_map, /*length=*/3);
  for (int i = 0; i < schedule.length(); i++) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Lec> lec,
                             Lec::CreateForStage(params, schedule, i));
    bool foo = lec->Run();
    if (!foo) {
      LOG(INFO) << lec->ResultToString();
    }
    ASSERT_TRUE(foo);
    LOG(INFO) << "Pass stage " << i;
  }
}

}  // namespace
}  // namespace z3
}  // namespace solvers
}  // namespace xls
