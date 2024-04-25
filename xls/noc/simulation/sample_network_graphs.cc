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

#include "xls/noc/simulation/sample_network_graphs.h"

#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls::noc {

absl::Status BuildNetworkGraphLinear000(NetworkConfigProto* nc_proto,
                                        NetworkManager* graph,
                                        NocParameters* params) {
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(3);

  builder.WithPort("SendPort0").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");

  builder.WithLink("Link0A")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(64)
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(64)
      .WithSourceSinkPipelineStage(2);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build network.
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphLinear001(NetworkConfigProto* nc_proto,
                                        NetworkManager* graph,
                                        NocParameters* params) {
  LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  builder.WithPort("SendPort0").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("SendPort1").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort1").AsOutputDirection().WithVirtualChannel("VC0");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(128).WithDepth(10);

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0");

  auto routerb = builder.WithRouter("RouterB");
  routerb.WithInputPort("Bin0").WithVirtualChannel("VC0");
  routerb.WithInputPort("Bin1").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout0").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout1").WithVirtualChannel("VC0");

  builder.WithLink("LinkAI0")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAI1")
      .WithSourcePort("SendPort1")
      .WithSinkPort("Ain1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAO0")
      .WithSourcePort("Aout0")
      .WithSinkPort("Bin0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAO1")
      .WithSourcePort("Aout1")
      .WithSinkPort("Bin1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBO0")
      .WithSourcePort("Bout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBO1")
      .WithSourcePort("Bout1")
      .WithSinkPort("RecvPort1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphTree000(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params) {
  LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(100).WithDepth(3);
  builder.WithVirtualChannel("VC1").WithFlitBitWidth(200).WithDepth(3);

  builder.WithPort("SendPort0")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("SendPort1")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("SendPort2")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort0")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort1")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort2")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort3")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  auto routerb = builder.WithRouter("RouterB");
  routerb.WithInputPort("Bin0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithInputPort("Bin1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routerb.WithOutputPort("Bout2").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  builder.WithLink("Link0A").WithSourcePort("SendPort0").WithSinkPort("Ain0");
  builder.WithLink("Link1A").WithSourcePort("SendPort1").WithSinkPort("Ain1");
  builder.WithLink("LinkAB").WithSourcePort("Aout1").WithSinkPort("Bin0");
  builder.WithLink("Link2A").WithSourcePort("SendPort2").WithSinkPort("Bin1");

  builder.WithLink("LinkA0").WithSourcePort("Aout0").WithSinkPort("RecvPort0");
  builder.WithLink("LinkB1")
      .WithSourcePort("Bout0")
      .WithSinkPort("RecvPort1")
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkB2").WithSourcePort("Bout1").WithSinkPort("RecvPort2");
  builder.WithLink("LinkB3").WithSourcePort("Bout2").WithSinkPort("RecvPort3");

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphTree001(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params) {
  LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  builder.WithVirtualChannel("VC0").WithDepth(3);
  builder.WithVirtualChannel("VC1").WithDepth(3);

  builder.WithPort("SendPort0")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("SendPort1")
      .AsInputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");

  builder.WithPort("RecvPort0")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");
  builder.WithPort("RecvPort1")
      .AsOutputDirection()
      .WithVirtualChannel("VC0")
      .WithVirtualChannel("VC1");

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0").WithVirtualChannel(
      "VC1");

  builder.WithLink("Link0A")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(128);
  builder.WithLink("Link1A")
      .WithSourcePort("SendPort1")
      .WithSinkPort("Ain1")
      .WithPhitBitWidth(128);

  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128);
  builder.WithLink("LinkA1")
      .WithSourcePort("Aout1")
      .WithSinkPort("RecvPort1")
      .WithPhitBitWidth(128);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphLoop000(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params) {
  LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  builder.WithPort("SendPort0").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("SendPort1").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort1").AsOutputDirection().WithVirtualChannel("VC0");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(128).WithDepth(10);

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0");

  auto routerb = builder.WithRouter("RouterB");
  routerb.WithInputPort("Bin0").WithVirtualChannel("VC0");
  routerb.WithInputPort("Bin1").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout0").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout1").WithVirtualChannel("VC0");

  builder.WithLink("LinkAI0")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAO0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAB1")
      .WithSourcePort("Aout1")
      .WithSinkPort("Bin1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAB2")
      .WithSourcePort("Bout1")
      .WithSinkPort("Ain1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBI0")
      .WithSourcePort("SendPort1")
      .WithSinkPort("Bin0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBO0")
      .WithSourcePort("Bout0")
      .WithSinkPort("RecvPort1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphLoop001(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params) {
  LOG(INFO) << "Setting up network ...";
  NetworkConfigProtoBuilder builder("Test");

  builder.WithPort("SendPort0").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("SendPort1").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("SendPort2").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("SendPort3").AsInputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort0").AsOutputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort1").AsOutputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort2").AsOutputDirection().WithVirtualChannel("VC0");
  builder.WithPort("RecvPort3").AsOutputDirection().WithVirtualChannel("VC0");

  builder.WithVirtualChannel("VC0").WithFlitBitWidth(128).WithDepth(10);

  auto routera = builder.WithRouter("RouterA");
  routera.WithInputPort("Ain0").WithVirtualChannel("VC0");
  routera.WithInputPort("Ain1").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout0").WithVirtualChannel("VC0");
  routera.WithOutputPort("Aout1").WithVirtualChannel("VC0");
  routera.WithInputPort("ABin0").WithVirtualChannel("VC0");
  routera.WithInputPort("ABin1").WithVirtualChannel("VC0");
  routera.WithOutputPort("ABout0").WithVirtualChannel("VC0");
  routera.WithOutputPort("ABout1").WithVirtualChannel("VC0");

  auto routerb = builder.WithRouter("RouterB");
  routerb.WithInputPort("Bin0").WithVirtualChannel("VC0");
  routerb.WithInputPort("Bin1").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout0").WithVirtualChannel("VC0");
  routerb.WithOutputPort("Bout1").WithVirtualChannel("VC0");
  routerb.WithInputPort("BAin0").WithVirtualChannel("VC0");
  routerb.WithInputPort("BAin1").WithVirtualChannel("VC0");
  routerb.WithOutputPort("BAout0").WithVirtualChannel("VC0");
  routerb.WithOutputPort("BAout1").WithVirtualChannel("VC0");

  builder.WithLink("LinkAI0")
      .WithSourcePort("SendPort0")
      .WithSinkPort("Ain0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAI1")
      .WithSourcePort("SendPort1")
      .WithSinkPort("Ain1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAO0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAO1")
      .WithSourcePort("Aout1")
      .WithSinkPort("RecvPort1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBA0")
      .WithSourcePort("BAout0")
      .WithSinkPort("ABin0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBA1")
      .WithSourcePort("BAout1")
      .WithSinkPort("ABin1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAB0")
      .WithSourcePort("ABout0")
      .WithSinkPort("BAin0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkAB1")
      .WithSourcePort("ABout1")
      .WithSinkPort("BAin1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBI0")
      .WithSourcePort("SendPort2")
      .WithSinkPort("Bin0")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBI1")
      .WithSourcePort("SendPort3")
      .WithSinkPort("Bin1")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBO0")
      .WithSourcePort("Bout0")
      .WithSinkPort("RecvPort2")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);
  builder.WithLink("LinkBO1")
      .WithSourcePort("Bout1")
      .WithSinkPort("RecvPort3")
      .WithPhitBitWidth(128)
      .WithSourceSinkPipelineStage(2)
      .WithSinkSourcePipelineStage(2);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  LOG(INFO) << *nc_proto;
  LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

}  // namespace xls::noc
