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

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config_builder_options.pb.h"

namespace xls::noc {

static const char* kExamplesPath = "xls/noc/config/examples/";

static absl::Status ParseTextProtoFileToNetworkConfigBuilderOptionsProto(
    std::string_view filename) {
  std::string filepath = absl::StrCat(kExamplesPath, filename);
  absl::StatusOr<std::filesystem::path> runfile_path =
      GetXlsRunfilePath(filepath);
  XLS_EXPECT_OK(runfile_path);
  NetworkConfigBuilderOptionsProto proto;
  return ParseTextProtoFile(runfile_path.value(), &proto);
}

// Test the format of a linear options textproto example by parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, Linear) {
  const char* kFilename = "linear_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a ring options textproto example by parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, Ring) {
  const char* kFilename = "ring_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a mesh options textproto example by parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, Mesh) {
  const char* kFilename = "mesh_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a torus options textproto example by parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, Torus) {
  const char* kFilename = "torus_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a grid with row loopback options textproto example by
// parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, GridWithRowLoopback) {
  const char* kFilename =
      "grid_with_row_loopback_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a grid with column loopback options textproto example by
// parsing the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, GridWithColumnLoopback) {
  const char* kFilename =
      "grid_with_column_loopback_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a distribution tree options textproto example by parsing
// the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, DistributionTree) {
  const char* kFilename =
      "distribution_tree_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of an aggregation tree options textproto example by parsing
// the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, AggregationTree) {
  const char* kFilename =
      "aggregation_tree_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a single switch options textproto example by parsing the
// file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, SingleSwitch) {
  const char* kFilename =
      "single_switch_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a bidirectioanl tree options textproto example by parsing
// the file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, BidirectionalTree) {
  const char* kFilename =
      "bidirectional_tree_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

// Test the format of a fully connected options textproto example by parsing the
// file.
TEST(CustomNetworkConfigBuilderOptionsTextProtoTest, FullyConnected) {
  const char* kFilename =
      "fully_connected_network_config_builder_options.textproto";
  XLS_EXPECT_OK(
      ParseTextProtoFileToNetworkConfigBuilderOptionsProto(kFilename));
}

}  // namespace xls::noc
