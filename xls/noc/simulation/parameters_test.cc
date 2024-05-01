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

#include "xls/noc/simulation/parameters.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder.h"

namespace xls {
namespace noc {
namespace {

TEST(SimParametersTest, NetworkAndLinkParam) {
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VCA").WithFlitBitWidth(128).WithDepth(13);
  builder.WithVirtualChannel("VCB").WithFlitBitWidth(256).WithDepth(34);
  builder.WithVirtualChannel("VCC").WithFlitBitWidth(64).WithDepth(5);

  LinkConfigProtoBuilder link = builder.WithLink("Link0");

  link.WithPhitBitWidth(10);
  link.WithSourceSinkPipelineStage(20);
  link.WithSinkSourcePipelineStage(30);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  const LinkConfigProto& link_proto = nc_proto.links(0);

  NetworkParam network_param(nc_proto);

  EXPECT_EQ(network_param.GetName(), "Test");
  EXPECT_EQ(&network_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(network_param.VirtualChannelCount(), 3);
  EXPECT_EQ(network_param.GetVirtualChannels().at(0).GetDepth(), 13);
  EXPECT_EQ(network_param.GetVirtualChannels().at(1).GetDepth(), 34);
  EXPECT_EQ(network_param.GetVirtualChannels().at(2).GetDepth(), 5);
  EXPECT_EQ(network_param.GetVirtualChannels().at(0).GetFlitDataBitWidth(),
            128);
  EXPECT_EQ(network_param.GetVirtualChannels().at(1).GetFlitDataBitWidth(),
            256);
  EXPECT_EQ(network_param.GetVirtualChannels().at(2).GetFlitDataBitWidth(), 64);

  LinkParam link_param(nc_proto, link_proto);

  EXPECT_EQ(link_param.GetName(), "Link0");
  EXPECT_EQ(link_param.GetPhitDataBitWidth(), 10);
  EXPECT_EQ(link_param.GetSourceToSinkPipelineStages(), 20);
  EXPECT_EQ(link_param.GetSinkToSourcePipelineStages(), 30);

  EXPECT_EQ(&link_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&link_param.GetLinkProto(), &link_proto);
}

TEST(SimParametersTest, NetworkInterfaceParam) {
  NetworkConfigProtoBuilder builder("Test");
  builder.WithPort("Ingress0").AsInputDirection();
  builder.WithPort("Egress0").AsOutputDirection();

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  const PortConfigProto& ingress_proto = nc_proto.ports(0);
  const PortConfigProto& egress_proto = nc_proto.ports(1);

  NetworkInterfaceSrcParam ni_src_param(nc_proto, ingress_proto);
  NetworkInterfaceSinkParam ni_sink_param(nc_proto, egress_proto);

  EXPECT_EQ(ni_src_param.GetName(), "Ingress0");
  EXPECT_EQ(&ni_src_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&ni_src_param.GetPortProto(), &ingress_proto);
  EXPECT_EQ(ni_src_param.GetPortProto().direction(), PortConfigProto::INPUT);

  EXPECT_EQ(ni_sink_param.GetName(), "Egress0");
  EXPECT_EQ(&ni_sink_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&ni_sink_param.GetPortProto(), &egress_proto);
  EXPECT_EQ(ni_sink_param.GetPortProto().direction(), PortConfigProto::OUTPUT);

  EXPECT_EQ(ni_sink_param.GetDepth(), 0);
  ni_sink_param.SetDepth(32);
  EXPECT_EQ(ni_sink_param.GetDepth(), 32);
}

TEST(SimParametersTest, RouterPortAndVCParam) {
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(33);

  builder.WithVirtualChannel("VC1").WithFlitBitWidth(200).WithDepth(35);

  auto router = builder.WithRouter("Router0");
  router.WithInputPort("in0");
  router.WithOutputPort("out0").WithVirtualChannel("VC1").WithVirtualChannel(
      "VC0");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());
  const RouterConfigProto& router_proto = nc_proto.routers(0);
  const PortConfigProto& in0_proto = router_proto.ports(0);
  const PortConfigProto& in1_proto = router_proto.ports(1);

  RouterParam router_param(nc_proto, router_proto);
  EXPECT_EQ(router_param.GetName(), "Router0");
  EXPECT_EQ(&router_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&router_param.GetRouterProto(), &router_proto);

  PortParam in0_param(nc_proto, in0_proto);
  EXPECT_EQ(in0_param.GetName(), "in0");
  EXPECT_EQ(in0_param.VirtualChannelCount(), 0);
  EXPECT_EQ(in0_param.GetVirtualChannels().size(), 0);
  EXPECT_EQ(&in0_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&in0_param.GetPortProto(), &in0_proto);

  PortParam in1_param(nc_proto, in1_proto);
  EXPECT_EQ(in1_param.GetName(), "out0");
  EXPECT_EQ(in1_param.VirtualChannelCount(), 2);
  EXPECT_EQ(in1_param.GetVirtualChannels().size(), 2);
  EXPECT_EQ(&in1_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&in1_param.GetPortProto(), &in1_proto);

  VirtualChannelParam vc1_param = in1_param.GetVirtualChannels()[0];
  EXPECT_EQ(vc1_param.GetName(), "VC1");
  EXPECT_EQ(vc1_param.GetDepth(), 35);
  EXPECT_EQ(vc1_param.GetFlitDataBitWidth(), 200);
  EXPECT_EQ(&vc1_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&vc1_param.GetVirtualChannelProto(), &nc_proto.virtual_channels(1));

  VirtualChannelParam vc0_param = in1_param.GetVirtualChannels()[1];
  EXPECT_EQ(vc0_param.GetName(), "VC0");
  EXPECT_EQ(vc0_param.GetDepth(), 33);
  EXPECT_EQ(vc0_param.GetFlitDataBitWidth(), 100);
  EXPECT_EQ(&vc0_param.GetNetworkProto(), &nc_proto);
  EXPECT_EQ(&vc0_param.GetVirtualChannelProto(), &nc_proto.virtual_channels(0));
}

TEST(SimParametersTest, NocParametersMap) {
  NetworkConfigProtoBuilder builder("Test");
  builder.WithPort("Ingress0").AsInputDirection();
  builder.WithPort("Egress0").AsOutputDirection();
  builder.WithLink("Link0");
  auto router = builder.WithRouter("Router0");
  router.WithInputPort("in0");
  router.WithOutputPort("out0");
  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto nc_proto, builder.Build());

  const PortConfigProto& ingress_proto = nc_proto.ports(0);
  const PortConfigProto& egress_proto = nc_proto.ports(1);
  const LinkConfigProto& link_proto = nc_proto.links(0);
  const RouterConfigProto& router_proto = nc_proto.routers(0);
  const PortConfigProto& port_proto_0 = router_proto.ports(0);
  const PortConfigProto& port_proto_1 = router_proto.ports(1);

  NetworkParam network_param(nc_proto);
  NetworkInterfaceSrcParam ni_src_param(nc_proto, ingress_proto);
  NetworkInterfaceSinkParam ni_sink_param(nc_proto, egress_proto);
  LinkParam link_param(nc_proto, link_proto);
  RouterParam router_param(nc_proto, router_proto);
  PortParam port_param_0(nc_proto, port_proto_0);
  PortParam port_param_1(nc_proto, port_proto_1);

  NetworkId network_id(0);
  NetworkComponentId ni_src_id(0, 2);
  NetworkComponentId ni_sink_id(0, 3);
  NetworkComponentId router_id(0, 0);
  PortId port_id_0(0, 0, 0);
  PortId port_id_1(0, 0, 1);
  NetworkComponentId link_id(0, 1);

  NocParameters param_map;
  param_map.SetNetworkParam(network_id, network_param);
  EXPECT_EQ(param_map.GetNetworkParam(network_id)->GetName(), "Test");
  EXPECT_EQ(&param_map.GetNetworkParam(network_id)->GetNetworkProto(),
            &nc_proto);

  param_map.SetNetworkComponentParam(ni_src_id, ni_src_param);
  param_map.SetNetworkComponentParam(ni_sink_id, ni_sink_param);
  param_map.SetNetworkComponentParam(router_id, router_param);
  param_map.SetNetworkComponentParam(link_id, link_param);

  EXPECT_EQ(std::get<NetworkInterfaceSrcParam>(
                *param_map.GetNetworkComponentParam(ni_src_id))
                .GetName(),
            "Ingress0");
  EXPECT_EQ(std::get<NetworkInterfaceSinkParam>(
                *param_map.GetNetworkComponentParam(ni_sink_id))
                .GetName(),
            "Egress0");
  EXPECT_EQ(
      std::get<RouterParam>(*param_map.GetNetworkComponentParam(router_id))
          .GetName(),
      "Router0");
  EXPECT_EQ(std::get<LinkParam>(*param_map.GetNetworkComponentParam(link_id))
                .GetName(),
            "Link0");

  param_map.SetPortParam(port_id_1, port_param_1);
  param_map.SetPortParam(port_id_0, port_param_0);
  EXPECT_EQ(param_map.GetPortParam(port_id_0)->GetName(), "in0");
  EXPECT_EQ(param_map.GetPortParam(port_id_1)->GetName(), "out0");
}

}  // namespace
}  // namespace noc
}  // namespace xls
