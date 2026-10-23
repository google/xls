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
#include <filesystem>
#include <sstream>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/op.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(BomTest, TestBomSummary) {
  // Find the test protobuf
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path testfile_path,
      GetXlsRunfilePath("xls/tools/testdata/bom.metrics.textproto"));
  std::filesystem::path root_path(testfile_path.parent_path());
  using MapT = absl::flat_hash_map<std::string, verilog::XlsMetricsProto>;
  XLS_ASSERT_OK_AND_ASSIGN(
      MapT metrics_protos,
      CollectMetricsProtos(root_path, ".*/bom\\.metrics\\.textproto"));
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

TEST(BomTest, CollectMetricsProtosBinaryProtoInSubdirectory) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::filesystem::path testfile_path,
      GetXlsRunfilePath("xls/tools/testdata/bom.metrics.textproto"));
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::XlsMetricsProto metrics,
      ParseTextProtoFile<verilog::XlsMetricsProto>(testfile_path));

  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());
  std::filesystem::path subdir = temp_dir.path() / "nested";
  XLS_ASSERT_OK(RecursivelyCreateDir(subdir));
  std::filesystem::path binproto_path = subdir / "bom.metrics.proto";
  XLS_ASSERT_OK(SetProtobinFile(binproto_path, metrics));

  using MapT = absl::flat_hash_map<std::string, verilog::XlsMetricsProto>;
  XLS_ASSERT_OK_AND_ASSIGN(
      MapT metrics_protos,
      CollectMetricsProtos(temp_dir.path(), ".*\\.metrics\\.(text)?proto"));
  ASSERT_EQ(metrics_protos.size(), 1);
  EXPECT_TRUE(metrics_protos.contains(binproto_path.string()));

  XLS_ASSERT_OK_AND_ASSIGN(auto bom_summary,
                           BomCalculateSummary(metrics_protos));
  EXPECT_EQ(bom_summary.size(), 3);
}

TEST(BomTest, BomCalculateSummaryErrors) {
  using MapT = absl::flat_hash_map<std::string, verilog::XlsMetricsProto>;

  {
    MapT empty_block_metrics = {{"empty.metrics.textproto", {}}};
    EXPECT_THAT(
        BomCalculateSummary(empty_block_metrics),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Proto XlsMetricsProto from empty.metrics.textproto "
                           "has empty block metrics.")));
  }

  {
    verilog::XlsMetricsProto metrics;
    metrics.mutable_block_metrics();
    MapT no_bom = {{"no_bom.metrics.textproto", metrics}};
    EXPECT_THAT(
        BomCalculateSummary(no_bom),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Proto XlsMetricsProto from no_bom.metrics.textproto has "
                      "no bill of materials.")));
  }

  {
    verilog::XlsMetricsProto metrics;
    metrics.mutable_block_metrics()->add_bill_of_materials()->set_kind(
        verilog::BOM_KIND_ADDER);
    MapT missing_op = {{"missing_op.metrics.textproto", metrics}};
    EXPECT_THAT(
        BomCalculateSummary(missing_op),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("XlsMetricsProto from missing_op.metrics.textproto has a "
                      "BomEntryProto missing op value.")));
  }

  {
    verilog::XlsMetricsProto metrics;
    metrics.mutable_block_metrics()->add_bill_of_materials()->set_op(
        OpProto::OP_ADD);
    MapT missing_kind = {{"missing_kind.metrics.textproto", metrics}};
    EXPECT_THAT(
        BomCalculateSummary(missing_kind),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("XlsMetricsProto from missing_kind.metrics.textproto has "
                      "a BomEntryProto missing kind value.")));
  }
}

}  // namespace
}  // namespace xls
