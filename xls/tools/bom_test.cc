// Copyright 2023 The XLS Authors
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

#include "xls/tools/bom.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"

namespace xls {

TEST(BomTest, TestBomSummary) {
  // Find the test protobuf
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path testfile_path,
      GetXlsRunfilePath("xls/tools/testdata/bom.sig.textproto"));
  std::filesystem::path root_path(testfile_path.parent_path());
  using MapT = absl::flat_hash_map<std::string, verilog::XlsMetricsProto>;
  XLS_ASSERT_OK_AND_ASSIGN(
      MapT metrics_protos,
      CollectMetricsProtos(root_path, ".*/bom\\.sig\\.textproto"));
  EXPECT_EQ(metrics_protos.size(), 1);

  // Calculate and check the summary
  XLS_ASSERT_OK_AND_ASSIGN(auto bom_summary,
                           BomCalculateSummary(metrics_protos));

  // Extract the items expected in the BOM
  EXPECT_EQ(bom_summary.size(), 3);
  absl::btree_map<struct BomItem, int64_t>::iterator it = bom_summary.begin();

  EXPECT_EQ(it->first.op_kind, "BOM_KIND_ADDER");
  EXPECT_EQ(it->first.op_name, "add");
  EXPECT_EQ(it->first.width_input, 8);
  EXPECT_EQ(it->first.width_output, 8);
  EXPECT_EQ(it->second, 1);
  it++;

  EXPECT_EQ(it->first.op_kind, "BOM_KIND_COMPARISON");
  EXPECT_EQ(it->first.op_name, "ne");
  EXPECT_EQ(it->first.width_input, 8);
  EXPECT_EQ(it->first.width_output, 1);
  EXPECT_EQ(it->second, 2);
  it++;

  EXPECT_EQ(it->first.op_kind, "BOM_KIND_COMPARISON");
  EXPECT_EQ(it->first.op_name, "ne");
  EXPECT_EQ(it->first.width_input, 16);
  EXPECT_EQ(it->first.width_output, 1);
  EXPECT_EQ(it->second, 1);
  it++;

  std::ostringstream out_table;
  BomPrint(out_table, bom_summary);
  std::string expected_table =
      " +---------------------+-----+-------------+--------------+-------+\n"
      " |                Kind |  Op | Input Width | Output Width | Count |\n"
      " +---------------------+-----+-------------+--------------+-------+\n"
      " |      BOM_KIND_ADDER | add |           8 |            8 |     1 |\n"
      " +---------------------+-----+-------------+--------------+-------+\n"
      " | BOM_KIND_COMPARISON |  ne |           8 |            1 |     2 |\n"
      " |                     |     |          16 |            1 |     1 |\n"
      " +---------------------+-----+-------------+--------------+-------+\n";
  EXPECT_EQ(out_table.str(), expected_table);

  std::ostringstream out_csv;
  BomToCsv(out_csv, bom_summary);
  std::string expected_csv =
      "Kind,Op,Input Width,Output Width,Count\n"
      "BOM_KIND_ADDER,add,8,8,1\n"
      "BOM_KIND_COMPARISON,ne,8,1,2\n"
      "BOM_KIND_COMPARISON,ne,16,1,1\n";
  EXPECT_EQ(out_csv.str(), expected_csv);
}

}  // namespace xls
