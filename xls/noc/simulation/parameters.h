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

#ifndef XLS_NOC_SIMULATION_PARAMETERS_H_
#define XLS_NOC_SIMULATION_PARAMETERS_H_

#include <cstdint>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/common.h"

namespace xls {
namespace noc {

// This file defines the objects used to describe the configuration/options
// of network objects.
//
// Two abstractions are provided:
//  1. Param objects which provide additional interfaces
//     and abstractions to the proto description of a network.  These objects
//     useful when specific concepts in the proto description are implicitly
//     defined.
//  2. NocParameters which associates NetworkGraph objects to Param objects.
//
// TODO(tedhong): 2020-12-23 - Add a third abstraction that maps proto names
//                             to network graph ids.

// Interface to protos describing a virtual channel
struct VirtualChannelParam {
 public:
  VirtualChannelParam(const NetworkConfigProto& network,
                      const VirtualChannelConfigProto& vc)
      : network_proto_(&network), vc_proto_(&vc) {}

  std::string_view GetName() const { return vc_proto_->name(); }

  // Get depth of this virtual channel.
  //
  // This is global for all uses of this virtual channel.
  //
  // Actual buffer depth used in hardware will be greater than this,
  // but not less.
  int64_t GetDepth() const { return vc_proto_->depth(); }

  // Get bits used for the data in a single flit.
  int64_t GetFlitDataBitWidth() const { return vc_proto_->flit_bit_width(); }

  // Get assoicated network proto for this router.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get associated router proto.
  const VirtualChannelConfigProto& GetVirtualChannelProto() const {
    return *vc_proto_;
  }

 private:
  const NetworkConfigProto* network_proto_;
  const VirtualChannelConfigProto* vc_proto_;
};

// Interface to protos describing a network.
class NetworkParam {
 public:
  explicit NetworkParam(const NetworkConfigProto& network)
      : network_proto_(&network) {}

  std::string_view GetName() const { return network_proto_->name(); }

  // Get source proto for the network this link is in.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Count of VCs associated with this port.
  //  Could be 0 if no VCs sare used.
  int64_t VirtualChannelCount() const {
    return network_proto_->virtual_channels_size();
  }

  // Get the virtual channels used in the network.
  std::vector<VirtualChannelParam> GetVirtualChannels() const {
    std::vector<VirtualChannelParam> ret;

    for (int64_t i = 0; i < VirtualChannelCount(); ++i) {
      const VirtualChannelConfigProto& vc_proto =
          network_proto_->virtual_channels().at(i);
      ret.emplace_back(*network_proto_, vc_proto);
    }

    return ret;
  }

 private:
  const NetworkConfigProto* network_proto_;
};

// Interface to protos describing a port.
struct PortParam {
 public:
  PortParam(const NetworkConfigProto& network, const PortConfigProto& port)
      : network_proto_(&network), port_proto_(&port) {}

  std::string_view GetName() const { return port_proto_->name(); }

  // Count of VCs associated with this port.
  //  Could be 0 if no VCs sare used.
  int64_t VirtualChannelCount() const {
    return port_proto_->virtual_channels_size();
  }

  // Returns vector of virtual channel parameters associated with this port.
  std::vector<VirtualChannelParam> GetVirtualChannels() const {
    std::vector<VirtualChannelParam> ret;
    for (int64_t i = 0; i < VirtualChannelCount(); ++i) {
      std::string_view vc_name = port_proto_->virtual_channels(i);

      const VirtualChannelConfigProto* vc_proto = nullptr;
      for (const VirtualChannelConfigProto& p :
           network_proto_->virtual_channels()) {
        if (p.name() == vc_name) {
          vc_proto = &p;
        }
      }
      CHECK(vc_proto != nullptr);  // The VC should be configured.
      ret.emplace_back(*network_proto_, *vc_proto);
    }
    return ret;
  }

  // Get assoicated network proto for this port.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get associated port proto.
  const PortConfigProto& GetPortProto() const { return *port_proto_; }

 private:
  const NetworkConfigProto* network_proto_;
  const PortConfigProto* port_proto_;
};

// Interface to protos describing a link.
struct LinkParam {
 public:
  LinkParam(const NetworkConfigProto& network, const LinkConfigProto& link)
      : network_proto_(&network), link_proto_(&link) {}

  std::string_view GetName() const { return link_proto_->name(); }

  // Get pipeline stages from source to sink.
  int64_t GetSourceToSinkPipelineStages() const {
    return link_proto_->source_sink_pipeline_stage();
  }

  // Get pipeline stages from sink to source (ex. for flow control).
  int64_t GetSinkToSourcePipelineStages() const {
    return link_proto_->sink_source_pipeline_stage();
  }

  // Get number of data (non-control) bits in a phit.
  //   i.e. for source-to-link path.
  int64_t GetPhitDataBitWidth() const { return link_proto_->phit_bit_width(); }

  // Get source proto for the network this link is in.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get link proto associated with this link.
  const LinkConfigProto& GetLinkProto() const { return *link_proto_; }

 private:
  const NetworkConfigProto* network_proto_;
  const LinkConfigProto* link_proto_;
};

// Interface to protos describing a source network interface.
class NetworkInterfaceSrcParam {
 public:
  NetworkInterfaceSrcParam(const NetworkConfigProto& network,
                           const PortConfigProto& port)
      : network_proto_(&network), port_proto_(&port) {}

  std::string_view GetName() const { return port_proto_->name(); }

  // Returns associated port param
  PortParam GetPortParam() const {
    return PortParam(*network_proto_, *port_proto_);
  }

  // Get assoicated network proto for this network interface.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get associated port proto.
  const PortConfigProto& GetPortProto() const { return *port_proto_; }

 private:
  const NetworkConfigProto* network_proto_;
  const PortConfigProto* port_proto_;
};

// Interface to protos describing a sink network interface.
class NetworkInterfaceSinkParam {
 public:
  NetworkInterfaceSinkParam(const NetworkConfigProto& network,
                            const PortConfigProto& port, int64_t depth = 0)
      : network_proto_(&network), port_proto_(&port), depth_(depth) {}

  // Sets depth (buffer/fifo size)of this network interface.
  void SetDepth(int64_t depth) { depth_ = depth; }

  // Returns number of flits this network interface can buffer.
  int64_t GetDepth() const { return depth_; }

  std::string_view GetName() const { return port_proto_->name(); }

  // Construct associated port param
  PortParam GetPortParam() const {
    return PortParam(*network_proto_, *port_proto_);
  }

  // Get assoicated network proto for this network interface.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get associated port proto.
  const PortConfigProto& GetPortProto() const { return *port_proto_; }

 private:
  const NetworkConfigProto* network_proto_;
  const PortConfigProto* port_proto_;

  // TODO(tedhong): 2020-12-14 support configuration via protos.
  int64_t depth_;
};

// Interface to protos describing a router.
class RouterParam {
 public:
  RouterParam(const NetworkConfigProto& network,
              const RouterConfigProto& router)
      : network_proto_(&network), router_proto_(&router) {}

  std::string_view GetName() const { return router_proto_->name(); }

  // Get assoicated network proto for this router.
  const NetworkConfigProto& GetNetworkProto() const { return *network_proto_; }

  // Get associated router proto.
  const RouterConfigProto& GetRouterProto() const { return *router_proto_; }

 private:
  const NetworkConfigProto* network_proto_;
  const RouterConfigProto* router_proto_;
};

// Variant used to store all possible param objects for
// each type of network component.
using NetworkComponentParam =
    std::variant<NetworkInterfaceSrcParam, NetworkInterfaceSinkParam,
                 RouterParam, LinkParam>;

// Associates Param objects with NetworkGraph objects.
class NocParameters {
 public:
  // Associates a Network's Id and Param.
  void SetNetworkParam(NetworkId id, NetworkParam p) {
    networks_.emplace(std::make_pair(id, p));
  }

  // Retrieves associated NetworkParam.
  absl::StatusOr<NetworkParam> GetNetworkParam(NetworkId id) const {
    auto i = networks_.find(id);

    if (i == networks_.end()) {
      return absl::InternalError(
          absl::StrFormat("NetworkId %d is missing associated "
                          "Param mapping",
                          id.AsUInt64()));
    } else {
      return i->second;
    }
  }

  // Associates a NetworkComponent's Id and Param.
  void SetNetworkComponentParam(NetworkComponentId id,
                                NetworkComponentParam p) {
    components_.emplace(std::make_pair(id, p));
  }

  // Retrieves associated NetworkComponentParam.
  absl::StatusOr<NetworkComponentParam> GetNetworkComponentParam(
      NetworkComponentId id) const {
    auto i = components_.find(id);

    if (i == components_.end()) {
      return absl::InternalError(
          absl::StrFormat("NetworkComponentId %d is missing associated "
                          "Param mapping",
                          id.AsUInt64()));
    } else {
      return i->second;
    }
  }

  // Associated a Port's Id and Param.
  void SetPortParam(PortId id, PortParam p) {
    ports_.emplace(std::make_pair(id, p));
  }

  // Retrieves associated PortParam.
  absl::StatusOr<PortParam> GetPortParam(PortId id) const {
    auto i = ports_.find(id);

    if (i == ports_.end()) {
      return absl::InternalError(
          absl::StrFormat("PortId %d is missing associated "
                          "Param mapping",
                          id.AsUInt64()));
    } else {
      return i->second;
    }
  }

 protected:
  // TODO(tedhong): 2020-12-14, use vectors based off of local ids vs maps.
  absl::flat_hash_map<NetworkId, NetworkParam> networks_;
  absl::flat_hash_map<PortId, PortParam> ports_;
  absl::flat_hash_map<NetworkComponentId, NetworkComponentParam> components_;
};

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_PARAMETERS_H_
