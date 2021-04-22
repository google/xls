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
      .WithSourceSinkPipelineStage(2);
  builder.WithLink("LinkA0")
      .WithSourcePort("Aout0")
      .WithSinkPort("RecvPort0")
      .WithSourceSinkPipelineStage(2);

  XLS_ASSIGN_OR_RETURN(*nc_proto, builder.Build());
  XLS_LOG(INFO) << nc_proto->DebugString();
  XLS_LOG(INFO) << "Done ...";

  // Build network.
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  XLS_LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

absl::Status BuildNetworkGraphTree000(NetworkConfigProto* nc_proto,
                                      NetworkManager* graph,
                                      NocParameters* params) {
  XLS_LOG(INFO) << "Setting up network ...";
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
  XLS_LOG(INFO) << nc_proto->DebugString();
  XLS_LOG(INFO) << "Done ...";

  // Build and assign simulation objects
  XLS_RETURN_IF_ERROR(BuildNetworkGraphFromProto(*nc_proto, graph, params));
  graph->Dump();
  XLS_LOG(INFO) << "Network Graph Complete ...";

  return absl::OkStatus();
}

}  // namespace xls::noc
