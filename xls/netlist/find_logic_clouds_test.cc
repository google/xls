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

#include "xls/netlist/find_logic_clouds.h"

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"

namespace xls {
namespace netlist {
namespace rtl {
namespace {

TEST(ClusterTest, TwoSimpleClusters) {
  std::string netlist = R"(module main(clk, ai, ao);
  input clk;
  input ai;
  output ao;
  wire a1;

  DFF dff_0(.D(ai), .Q(a1), .CLK(clk));
  DFF dff_1(.D(a1), .Q(ao), .CLK(clk));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  std::vector<Cluster> clusters = FindLogicClouds(*m, /*include_vacuous=*/true);
  EXPECT_EQ(2, clusters.size());
  EXPECT_EQ(R"(cluster {
  terminating_flop: dff_0
}
cluster {
  terminating_flop: dff_1
}
)",
            ClustersToString(clusters));
}

TEST(ClusterTest, InputAndOutputGatesOnTwoFlops) {
  std::string netlist = R"(module main(clk, ai, ao);
  input clk;
  input ai;
  output ao;
  wire ain, a1, a1n, a2;

  INV inv_a(.A(ai), .ZN(ain));
  DFF dff_0(.D(ain), .Q(a1), .CLK(clk));
  INV inv_b(.A(a1), .ZN(a1n));
  DFF dff_1(.D(a1n), .Q(a2), .CLK(clk));
  INV inv_c(.A(a2), .ZN(ao));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  std::vector<Cluster> clusters = FindLogicClouds(*m, /*include_vacuous=*/true);
  EXPECT_EQ(3, clusters.size());
  EXPECT_EQ(R"(cluster {
  other_cell: inv_c
}
cluster {
  terminating_flop: dff_0
  other_cell: inv_a
}
cluster {
  terminating_flop: dff_1
  other_cell: inv_b
}
)",
            ClustersToString(clusters));
}

TEST(ClusterTest, TwoStagesWithMergeCloudInMiddle) {
  // -(a0)->|dff_a0_a1|-(a1)->|inv_a1_a1n|-(a1n)->|dff_a1n_ao|->ao
  //                     \--.
  // -(b0)->|dff_b0_b1|-(b1)+>|and_a1_b1|-(ab1)->|dff_ab1_bo|->bo
  std::string netlist = R"(module main(clk, a0, b0, ao, bo);
  input clk;
  input a0, b0;
  output ao, bo;
  wire a1, a1n, ab1, b1;

  DFF dff_a0_a1(.D(a0), .Q(a1), .CLK(clk));
  INV inv_a1_a1n(.A(a1), .ZN(a1n));
  DFF dff_a1n_ao(.D(a1n), .Q(ao), .CLK(clk));

  DFF dff_b0_b1(.D(b0), .Q(b1), .CLK(clk));
  AND and_a1_b1(.A(a1), .B(b1), .Z(ab1));
  DFF dff_ab1_bo(.D(ab1), .Q(bo), .CLK(clk));
endmodule)";
  Scanner scanner(netlist);
  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Netlist> n,
                           Parser::ParseNetlist(&cell_library, &scanner));
  XLS_ASSERT_OK_AND_ASSIGN(const Module* m, n->GetModule("main"));
  std::vector<Cluster> clusters = FindLogicClouds(*m, /*include_vacuous=*/true);
  EXPECT_EQ(3, clusters.size());
  EXPECT_EQ(R"(cluster {
  terminating_flop: dff_a0_a1
}
cluster {
  terminating_flop: dff_a1n_ao
  terminating_flop: dff_ab1_bo
  other_cell: and_a1_b1
  other_cell: inv_a1_a1n
}
cluster {
  terminating_flop: dff_b0_b1
}
)",
            ClustersToString(clusters));
}

}  // namespace
}  // namespace rtl
}  // namespace netlist
}  // namespace xls
