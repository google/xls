// Copyright 2021 The XLS Authors
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

#include "xls/noc/simulation/indexer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {
namespace {

TEST(SimIndexerTest, PortVirtualChannelIndexerTest) {
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(33);
  builder.WithVirtualChannel("VC1").WithFlitBitWidth(200).WithDepth(34);
  builder.WithVirtualChannel("VC2").WithFlitBitWidth(300).WithDepth(35);
  builder.WithVirtualChannel("VC3").WithFlitBitWidth(100).WithDepth(36);

  auto router = builder.WithRouter("Router0");
  router.WithInputPort("in0").WithVirtualChannel("VC0");
  router.WithOutputPort("out0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  router.WithOutputPort("out1")
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1")
      .WithVirtualChannel("VC2")
      .WithVirtualChannel("VC3");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());

  const RouterConfigProto& router_proto = nc_proto.routers(0);
  const PortConfigProto& in0_proto = router_proto.ports(0);
  const PortConfigProto& out0_proto = router_proto.ports(1);
  const PortConfigProto& out1_proto = router_proto.ports(2);

  PortParam in0_param(nc_proto, in0_proto);
  PortParam out0_param(nc_proto, out0_proto);
  PortParam out1_param(nc_proto, out1_proto);

  NetworkComponentId nc_id(0, 0);
  PortId in0_id(0, 0, 0);
  PortId out0_id(0, 0, 1);
  PortId out1_id(0, 0, 2);

  PortIndexMapBuilder port_index_builder;
  XLS_EXPECT_OK(
      port_index_builder.SetPortIndex(in0_id, PortDirection::kInput, 0));
  XLS_EXPECT_OK(
      port_index_builder.SetPortIndex(out0_id, PortDirection::kOutput, 1));
  XLS_EXPECT_OK(
      port_index_builder.SetPortIndex(out1_id, PortDirection::kOutput, 0));
  XLS_EXPECT_OK_AND_EQ(
      port_index_builder.GetPortIndex(in0_id, PortDirection::kInput), 0);
  XLS_EXPECT_OK_AND_EQ(
      port_index_builder.GetPortIndex(out1_id, PortDirection::kOutput), 0);
  XLS_EXPECT_OK_AND_EQ(
      port_index_builder.GetPortIndex(out0_id, PortDirection::kOutput), 1);

  XLS_ASSERT_OK_AND_ASSIGN(PortIndexMap port_indexer,
                           port_index_builder.BuildPortIndex());

  XLS_EXPECT_OK_AND_EQ(port_indexer.InputPortCount(nc_id), 1);
  XLS_EXPECT_OK_AND_EQ(port_indexer.OutputPortCount(nc_id), 2);
  XLS_EXPECT_OK_AND_EQ(
      port_indexer.GetPortByIndex(nc_id, PortDirection::kInput, 0), in0_id);
  XLS_EXPECT_OK_AND_EQ(
      port_indexer.GetPortByIndex(nc_id, PortDirection::kOutput, 1), out0_id);
  XLS_EXPECT_OK_AND_EQ(
      port_indexer.GetPortByIndex(nc_id, PortDirection::kOutput, 0), out1_id);

  VirtualChannelParam vc0_param = out0_param.GetVirtualChannels().at(0);
  VirtualChannelParam vc1_param = out0_param.GetVirtualChannels().at(1);
  VirtualChannelParam vc2_param = out1_param.GetVirtualChannels().at(2);
  VirtualChannelParam vc3_param = out1_param.GetVirtualChannels().at(3);

  VirtualChannelIndexMapBuilder index_builder;
  EXPECT_THAT(
      index_builder.VirtualChannelCount(in0_id),
      status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                               testing::HasSubstr("has not been indexed")));
  XLS_ASSERT_OK(index_builder.SetVirtualChannelIndex(in0_id, in0_param, 0, 0));
  XLS_EXPECT_OK_AND_EQ(index_builder.VirtualChannelCount(in0_id), 1);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetVirtualChannelIndex(in0_id, vc0_param),
                       0);

  XLS_ASSERT_OK_AND_ASSIGN(VirtualChannelIndexMap index0,
                           index_builder.BuildVirtualChannelIndex());
  EXPECT_EQ(index0.GetVirtualChannelByIndex(in0_id, 0)->GetName(), "VC0");

  EXPECT_THAT(
      index_builder.VirtualChannelCount(out0_id),
      status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                               testing::HasSubstr("has not been indexed")));
  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out0_id, out0_param, 0, 1));
  EXPECT_THAT(
      index_builder.SetVirtualChannelIndex(out0_id, out0_param, 2, 0),
      status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                               testing::HasSubstr("VC original index")));
  EXPECT_THAT(index_builder.SetVirtualChannelIndex(out0_id, out0_param, 0, 2),
              status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       testing::HasSubstr("VC index")));
  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out0_id, out0_param, 1, 0));

  XLS_ASSERT_OK_AND_ASSIGN(VirtualChannelIndexMap index1,
                           index_builder.BuildVirtualChannelIndex());
  EXPECT_EQ(index1.GetVirtualChannelByIndex(out0_id, 0)->GetName(), "VC1");
  EXPECT_EQ(index1.GetVirtualChannelByIndex(out0_id, 1)->GetName(), "VC0");
  XLS_EXPECT_OK_AND_EQ(index1.GetVirtualChannelIndexByName(out0_id, "VC1"), 0);
  XLS_EXPECT_OK_AND_EQ(index1.GetVirtualChannelIndexByName(out0_id, "VC0"), 1);
  XLS_EXPECT_OK_AND_EQ(index1.GetVirtualChannelIndex(out0_id, vc0_param), 1);
  XLS_EXPECT_OK_AND_EQ(index1.GetVirtualChannelIndex(out0_id, vc1_param), 0);
  EXPECT_THAT(
      index1.VirtualChannelCount(out1_id),
      status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                               testing::HasSubstr("has not been indexed")));

  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out1_id, out1_param, 0, 3));
  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out1_id, out1_param, 1, 1));
  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out1_id, out1_param, 2, 0));
  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out1_id, out1_param, 3, 0));
  XLS_EXPECT_OK_AND_EQ(index_builder.VirtualChannelCount(out1_id), 4);
  EXPECT_THAT(
      index_builder.BuildVirtualChannelIndex(),
      status_testing::StatusIs(absl::StatusCode::kInternal,
                               testing::HasSubstr("duplicate indices")));
  XLS_EXPECT_OK_AND_EQ(index_builder.GetVirtualChannelIndex(out1_id, vc0_param),
                       3);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetVirtualChannelIndex(out1_id, vc1_param),
                       1);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetVirtualChannelIndex(out1_id, vc2_param),
                       0);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetVirtualChannelIndex(out1_id, vc3_param),
                       0);

  XLS_ASSERT_OK(
      index_builder.SetVirtualChannelIndex(out1_id, out1_param, 3, 2));

  XLS_ASSERT_OK_AND_ASSIGN(VirtualChannelIndexMap index2,
                           index_builder.BuildVirtualChannelIndex());
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndex(out1_id, vc0_param), 3);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndex(out1_id, vc1_param), 1);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndex(out1_id, vc2_param), 0);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndex(out1_id, vc3_param), 2);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndexByName(out1_id, "VC0"), 3);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndexByName(out1_id, "VC1"), 1);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndexByName(out1_id, "VC2"), 0);
  XLS_EXPECT_OK_AND_EQ(index2.GetVirtualChannelIndexByName(out1_id, "VC3"), 2);
  EXPECT_EQ(index2.GetVirtualChannelByIndex(out1_id, 0)->GetName(), "VC2");
  EXPECT_EQ(index2.GetVirtualChannelByIndex(out1_id, 1)->GetName(), "VC1");
  EXPECT_EQ(index2.GetVirtualChannelByIndex(out1_id, 2)->GetName(), "VC3");
  EXPECT_EQ(index2.GetVirtualChannelByIndex(out1_id, 3)->GetName(), "VC0");
}

TEST(SimIndexerTest, NetworkComponentIndexerTest0) {
  NetworkComponentIndexMapBuilder index_builder;
  NetworkComponentId nc0(0, 0);

  EXPECT_EQ(index_builder.NetworkComponentCount(), 0);
  EXPECT_THAT(index_builder.GetNetworkComponentIndex(nc0),
              status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                       testing::HasSubstr("not been indexed")));
  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc0, 1));
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc0), 1);
  EXPECT_EQ(index_builder.NetworkComponentCount(), 1);

  EXPECT_THAT(
      index_builder.BuildNetworkComponentIndex(),
      status_testing::StatusIs(absl::StatusCode::kInternal,
                               testing::HasSubstr("Unable to add index")));

  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc0, 0));
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc0), 0);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkComponentIndexMap index,
                           index_builder.BuildNetworkComponentIndex());
  XLS_EXPECT_OK_AND_EQ(index.GetNetworkComponentByIndex(0), nc0);
  EXPECT_EQ(index.GetNetworkComponents().at(0), nc0);
}

TEST(SimIndexerTest, NetworkComponentIndexerTest1) {
  NetworkComponentIndexMapBuilder index_builder;
  NetworkComponentId nc0(0, 0);
  NetworkComponentId nc1(0, 1);
  NetworkComponentId nc2(0, 2);
  NetworkComponentId nc3(0, 3);

  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc0, 3));
  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc1, 2));
  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc2, 0));
  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc3, 0));
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc0), 3);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc1), 2);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc2), 0);
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc3), 0);
  EXPECT_EQ(index_builder.NetworkComponentCount(), 4);
  EXPECT_THAT(
      index_builder.BuildNetworkComponentIndex(),
      status_testing::StatusIs(absl::StatusCode::kInternal,
                               testing::HasSubstr("duplicate indices")));

  XLS_ASSERT_OK(index_builder.SetNetworkComponentIndex(nc3, 1));
  XLS_EXPECT_OK_AND_EQ(index_builder.GetNetworkComponentIndex(nc3), 1);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkComponentIndexMap index,
                           index_builder.BuildNetworkComponentIndex());
  XLS_EXPECT_OK_AND_EQ(index.GetNetworkComponentByIndex(0), nc2);
  XLS_EXPECT_OK_AND_EQ(index.GetNetworkComponentByIndex(1), nc3);
  XLS_EXPECT_OK_AND_EQ(index.GetNetworkComponentByIndex(2), nc1);
  XLS_EXPECT_OK_AND_EQ(index.GetNetworkComponentByIndex(3), nc0);

  ASSERT_EQ(index.GetNetworkComponents().size(), 4);
  EXPECT_EQ(index.GetNetworkComponents().at(0), nc2);
  EXPECT_EQ(index.GetNetworkComponents().at(1), nc3);
  EXPECT_EQ(index.GetNetworkComponents().at(2), nc1);
  EXPECT_EQ(index.GetNetworkComponents().at(3), nc0);
}

}  // namespace
}  // namespace noc
}  // namespace xls
