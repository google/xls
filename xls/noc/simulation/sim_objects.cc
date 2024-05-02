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

#include "xls/noc/simulation/sim_objects.h"

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/flit.h"
#include "xls/noc/simulation/network_graph.h"
#include "xls/noc/simulation/parameters.h"

namespace xls {
namespace noc {
namespace {

// Implements an simple pipeline between two connections.
//
// Template parameters are used to switch between the different types of
// flits we support -- either data (TimedDataFlit) or
// metadata (TimedMetadataFlit).
template <typename DataTimePhitT>
class SimplePipelineImpl {
 public:
  SimplePipelineImpl(int64_t stage_count, DataTimePhitT& from_channel,
                     DataTimePhitT& to_channel,
                     std::queue<DataTimePhitT>& state,
                     int64_t& internal_propagated_cycle)
      : stage_count_(stage_count),
        from_(from_channel),
        to_(to_channel),
        state_(state),
        internal_propagated_cycle_(internal_propagated_cycle) {}

  bool TryPropagation(NocSimulator& simulator);

 private:
  int64_t stage_count_;
  DataTimePhitT& from_;
  DataTimePhitT& to_;
  // TODO(vmirian) 09-07-21 Optimize to select flit data and its metadata
  std::queue<DataTimePhitT>& state_;
  int64_t& internal_propagated_cycle_;
};

template <typename DataTimePhitT>
bool SimplePipelineImpl<DataTimePhitT>::TryPropagation(
    NocSimulator& simulator) {
  int64_t current_cycle = simulator.GetCurrentCycle();

  if (internal_propagated_cycle_ == current_cycle) {
    return true;
  }

  if (stage_count_ == 0) {
    // No pipeline stages, so output is updated when input is ready.
    if (from_.cycle == current_cycle) {
      VLOG(2) << absl::StreamFormat("... link received data %v type %d",
                                    from_.flit.data, from_.flit.type);

      to_.flit = from_.flit;
      to_.cycle = current_cycle;
      to_.metadata = from_.metadata;

      VLOG(2) << absl::StreamFormat(
          "... link sending data %v type %d connection", to_.flit.data,
          to_.flit.type);

      internal_propagated_cycle_ = current_cycle;
    }
  } else {
    // There is one pipeline stage so output can be updated
    // immediately.
    if (to_.cycle != current_cycle) {
      if (state_.size() >= stage_count_) {
        to_.flit = state_.front().flit;
        to_.cycle = current_cycle;
        to_.metadata = state_.front().metadata;
        state_.pop();
      } else {
        to_.flit.type = FlitType::kInvalid;
        to_.flit.data = Bits(32);
        to_.cycle = current_cycle;
      }

      VLOG(2) << absl::StreamFormat(
          "... link sending data %v type %d connection", to_.flit.data,
          to_.flit.type);
    }

    if (from_.cycle == current_cycle) {
      state_.push(from_);
      VLOG(2) << absl::StreamFormat("... link received data %v type %d",
                                    from_.flit.data, from_.flit.type);

      internal_propagated_cycle_ = current_cycle;
    }
  }

  // Finished propagation if internal cycle has been updated.
  // The output port is either updated at the same time or was previously
  // updated.
  return internal_propagated_cycle_ == current_cycle;
}

}  // namespace

absl::Status NocSimulator::CreateSimulationObjects(NetworkId network) {
  Network& network_obj = mgr_->GetNetwork(network);

  // Create connection simulation objects.
  for (int64_t i = 0; i < network_obj.GetConnectionCount(); ++i) {
    ConnectionId id = network_obj.GetConnectionIdByIndex(i);
    XLS_RETURN_IF_ERROR(CreateConnection(id));
  }

  // Create component simulation objects.
  for (int64_t i = 0; i < network_obj.GetNetworkComponentCount(); ++i) {
    NetworkComponentId id = network_obj.GetNetworkComponentIdByIndex(i);
    XLS_RETURN_IF_ERROR(CreateNetworkComponent(id));
  }

  return absl::OkStatus();
}

absl::Status NocSimulator::CreateConnection(ConnectionId connection) {
  // Find number of vc's.
  Connection& connection_obj = mgr_->GetConnection(connection);
  XLS_ASSIGN_OR_RETURN(PortParam from_port_param,
                       params_->GetPortParam(connection_obj.src()));
  int64_t vc_count = from_port_param.VirtualChannelCount();

  // Construct new connection object.
  SimConnectionState& new_connection = NewConnection(connection);

  new_connection.id = connection_obj.id();
  new_connection.forward_channels.cycle = cycle_;
  XLS_ASSIGN_OR_RETURN(new_connection.forward_channels.flit,
                       DataFlitBuilder().Invalid().BuildFlit());

  if (vc_count == 0) {
    vc_count = 1;
  }

  new_connection.reverse_channels.resize(vc_count);
  for (int64_t i = 0; i < vc_count; ++i) {
    TimedMetadataFlit& flit = new_connection.reverse_channels[i];
    flit.cycle = cycle_;
    XLS_ASSIGN_OR_RETURN(flit.flit,
                         MetadataFlitBuilder().Invalid().BuildFlit());
  }

  return absl::OkStatus();
}

absl::Status NocSimulator::CreateNetworkComponent(NetworkComponentId nc_id) {
  NetworkComponent& network_component = mgr_->GetNetworkComponent(nc_id);

  switch (network_component.kind()) {
    case NetworkComponentKind::kNISrc:
      return CreateNetworkInterfaceSrc(nc_id);
    case NetworkComponentKind::kNISink:
      return CreateNetworkInterfaceSink(nc_id);
    case NetworkComponentKind::kLink:
      return CreateLink(nc_id);
    case NetworkComponentKind::kRouter:
      return CreateRouter(nc_id);
    case NetworkComponentKind::kNone:
      break;
  }

  return absl::InternalError(absl::StrFormat(
      "Unsupported network component kind %d", network_component.kind()));
}

absl::Status NocSimulator::CreateNetworkInterfaceSrc(NetworkComponentId nc_id) {
  int64_t index = network_interface_sources_.size();

  XLS_ASSIGN_OR_RETURN(SimNetworkInterfaceSrc sim_obj,
                       SimNetworkInterfaceSrc::Create(nc_id, *this));
  network_interface_sources_.push_back(std::move(sim_obj));
  src_index_map_.insert({nc_id, index});

  return absl::OkStatus();
}

absl::Status NocSimulator::CreateNetworkInterfaceSink(
    NetworkComponentId nc_id) {
  int64_t index = network_interface_sinks_.size();

  XLS_ASSIGN_OR_RETURN(SimNetworkInterfaceSink sim_obj,
                       SimNetworkInterfaceSink::Create(nc_id, *this));
  network_interface_sinks_.push_back(std::move(sim_obj));
  sink_index_map_.insert({nc_id, index});

  return absl::OkStatus();
}

absl::Status NocSimulator::CreateLink(NetworkComponentId nc_id) {
  XLS_ASSIGN_OR_RETURN(SimLink sim_obj, SimLink::Create(nc_id, *this));
  links_.push_back(std::move(sim_obj));
  return absl::OkStatus();
}

absl::Status NocSimulator::CreateRouter(NetworkComponentId nc_id) {
  XLS_ASSIGN_OR_RETURN(SimInputBufferedVCRouter sim_obj,
                       SimInputBufferedVCRouter::Create(nc_id, *this));
  routers_.push_back(std::move(sim_obj));
  return absl::OkStatus();
}

void NocSimulator::Dump() {
  Network& network_obj = mgr_->GetNetwork(network_);
  // Create connection simulation objects
  for (int64_t i = 0; i < network_obj.GetConnectionCount(); ++i) {
    ConnectionId id = network_obj.GetConnectionIdByIndex(i);
    int64_t index = GetConnectionIndex(id);
    SimConnectionState& connection = GetSimConnectionByIndex(index);

    VLOG(2) << absl::StreamFormat(
        "Simul Connection id %x data %s cycle %d", id.AsUInt64(),
        connection.forward_channels.flit, connection.forward_channels.cycle);
  }

  // Create connection simulation objects
  for (int64_t i = 0; i < network_obj.GetNetworkComponentCount(); ++i) {
    auto id = network_obj.GetNetworkComponentIdByIndex(i);

    VLOG(2) << absl::StreamFormat("Simul Component id %x", id.AsUInt64());
    mgr_->GetNetworkComponent(id).Dump();
  }
}

absl::Status NocSimulator::RunCycle(int64_t max_ticks) {
  ++cycle_;
  VLOG(2) << "";
  VLOG(2) << absl::StreamFormat("*** Simul Cycle %d", cycle_);

  for (NocSimulatorServiceShim* svc : pre_cycle_services_) {
    XLS_RET_CHECK_OK(svc->RunCycle());
  }

  bool converged = false;
  int64_t nticks = 0;
  while (!converged) {
    VLOG(2) << absl::StreamFormat("Tick %d", nticks);
    converged = Tick();
    ++nticks;
    if (nticks >= max_ticks) {
      return absl::InternalError(absl::StrFormat(
          "Simulator unable to converge after %d ticks for cycle %d", nticks,
          cycle_));
    }
  }

  for (int64_t i = 0; i < connections_.size(); ++i) {
    VLOG(2) << absl::StreamFormat("  Connection %d (%x)", i,
                                  connections_[i].id.AsUInt64());

    VLOG(2) << absl::StreamFormat("    FWD %s",
                                  connections_[i].forward_channels);

    for (int64_t vc = 0; vc < connections_[i].reverse_channels.size(); ++vc) {
      VLOG(2) << absl::StreamFormat("    REV %d %s", vc,
                                    connections_[i].reverse_channels[vc]);
    }
  }

  for (NocSimulatorServiceShim* svc : post_cycle_services_) {
    XLS_RET_CHECK_OK(svc->RunCycle());
  }

  return absl::OkStatus();
}

bool NocSimulator::Tick() {
  // Goes through each simulator object and run atick.
  // Converges when everyone returns True -- that determines new cycle

  bool converged = true;

  VLOG(2) << " Network Interfaces";
  for (SimNetworkInterfaceSrc& nc : network_interface_sources_) {
    NetworkComponentId id = nc.GetId();
    bool this_converged = nc.Tick(*this);
    converged &= this_converged;
    VLOG(2) << absl::StreamFormat(" NC %x Converged %d", id.AsUInt64(),
                                  this_converged);
  }

  VLOG(2) << " Links";
  for (SimLink& nc : links_) {
    NetworkComponentId id = nc.GetId();
    bool this_converged = nc.Tick(*this);
    converged &= this_converged;
    VLOG(2) << absl::StreamFormat(" NC %x Converged %d", id.AsUInt64(),
                                  this_converged);
  }

  VLOG(2) << " Routers";
  for (SimInputBufferedVCRouter& nc : routers_) {
    NetworkComponentId id = nc.GetId();
    bool this_converged = nc.Tick(*this);
    converged &= this_converged;
    VLOG(2) << absl::StreamFormat(" NC %x Converged %d", id.AsUInt64(),
                                  this_converged);
  }

  VLOG(2) << " Sinks";
  for (SimNetworkInterfaceSink& nc : network_interface_sinks_) {
    NetworkComponentId id = nc.GetId();
    bool this_converged = nc.Tick(*this);
    converged &= this_converged;
    VLOG(2) << absl::StreamFormat(" NC %x Converged %d", id.AsUInt64(),
                                  this_converged);
  }

  return converged;
}

bool SimNetworkComponentBase::Tick(NocSimulator& simulator) {
  int64_t cycle = simulator.GetCurrentCycle();

  bool converged = true;
  if (forward_propagated_cycle_ != cycle) {
    if (TryForwardPropagation(simulator)) {
      forward_propagated_cycle_ = cycle;
    } else {
      converged = false;
    }
  }
  if (reverse_propagated_cycle_ != cycle) {
    if (TryReversePropagation(simulator)) {
      reverse_propagated_cycle_ = cycle;
    } else {
      converged = false;
    }
  }
  return converged;
}

int64_t SimLink::GetSourceConnectionIndex() const {
  return src_connection_index_;
}

absl::Status SimLink::InitializeImpl(NocSimulator& simulator) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentParam nc_param,
      simulator.GetNocParameters()->GetNetworkComponentParam(id_));
  LinkParam& param = std::get<LinkParam>(nc_param);

  forward_pipeline_stages_ = param.GetSourceToSinkPipelineStages();
  reverse_pipeline_stages_ = param.GetSinkToSourcePipelineStages();
  phit_width_ = param.GetPhitDataBitWidth();

  NetworkManager* network_manager = simulator.GetNetworkManager();

  PortId src_port =
      network_manager->GetNetworkComponent(id_).GetPortIdByIndex(0);
  PortId sink_port =
      network_manager->GetNetworkComponent(id_).GetPortIdByIndex(1);
  if (network_manager->GetPort(src_port).direction() ==
      PortDirection::kOutput) {
    // Swap src/sink if the src port actually had index 0.
    PortId tmp = src_port;
    src_port = sink_port;
    sink_port = tmp;
  }

  ConnectionId src_connection = network_manager->GetPort(src_port).connection();
  ConnectionId sink_connection =
      network_manager->GetPort(sink_port).connection();

  src_connection_index_ = simulator.GetConnectionIndex(src_connection);
  sink_connection_index_ = simulator.GetConnectionIndex(sink_connection);
  internal_forward_propagated_cycle_ = simulator.GetCurrentCycle();

  // Create a reverse pipeline stage for each vc.
  SimConnectionState& sink =
      simulator.GetSimConnectionByIndex(sink_connection_index_);

  int64_t reverse_channel_count = sink.reverse_channels.size();
  reverse_credit_stages_.resize(reverse_channel_count);
  internal_reverse_propagated_cycle_ =
      std::vector(reverse_channel_count, simulator.GetCurrentCycle());

  return absl::OkStatus();
}

absl::Status SimNetworkInterfaceSrc::InitializeImpl(NocSimulator& simulator) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentParam nc_param,
      simulator.GetNocParameters()->GetNetworkComponentParam(id_));
  NetworkInterfaceSrcParam& param =
      std::get<NetworkInterfaceSrcParam>(nc_param);

  int64_t virtual_channel_count = param.GetPortParam().VirtualChannelCount();
  data_to_send_.resize(virtual_channel_count);
  credit_.resize(virtual_channel_count, 0);
  credit_update_.resize(virtual_channel_count,
                        CreditState{simulator.GetCurrentCycle(), 0});

  NetworkManager* network_manager = simulator.GetNetworkManager();
  PortId sink_port =
      network_manager->GetNetworkComponent(id_).GetPortIdByIndex(0);
  ConnectionId sink_connection =
      network_manager->GetPort(sink_port).connection();

  sink_connection_index_ = simulator.GetConnectionIndex(sink_connection);

  return absl::OkStatus();
}

absl::Status SimNetworkInterfaceSrc::SendFlitAtTime(TimedDataFlit flit) {
  int64_t vc_index = flit.flit.vc;

  if (vc_index < data_to_send_.size()) {
    data_to_send_[vc_index].push(flit);
    return absl::OkStatus();
  }
  return absl::OutOfRangeError(
      absl::StrFormat("Unable to send flit to vc index %d, max %d", vc_index,
                      data_to_send_.size()));
}

absl::Status SimNetworkInterfaceSink::InitializeImpl(NocSimulator& simulator) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentParam nc_param,
      simulator.GetNocParameters()->GetNetworkComponentParam(id_));
  NetworkInterfaceSinkParam& param =
      std::get<NetworkInterfaceSinkParam>(nc_param);

  PortParam port_param = param.GetPortParam();
  std::vector<VirtualChannelParam> vc_params = port_param.GetVirtualChannels();
  int64_t virtual_channel_count = port_param.VirtualChannelCount();

  input_buffers_.resize(virtual_channel_count);

  for (int64_t vc = 0; vc < virtual_channel_count; ++vc) {
    input_buffers_[vc].max_queue_size = vc_params[vc].GetDepth();
  }

  NetworkManager* network_manager = simulator.GetNetworkManager();
  PortId src_port =
      network_manager->GetNetworkComponent(id_).GetPortIdByIndex(0);
  ConnectionId src_connection = network_manager->GetPort(src_port).connection();

  src_connection_index_ = simulator.GetConnectionIndex(src_connection);

  return absl::OkStatus();
}

int64_t SimInputBufferedVCRouter::GetUtilizationCycleCount() const {
  return utilization_cycle_count_;
}
absl::Status SimInputBufferedVCRouter::InitializeImpl(NocSimulator& simulator) {
  NetworkManager* network_manager = simulator.GetNetworkManager();
  NetworkComponent& nc = network_manager->GetNetworkComponent(id_);
  const PortIndexMap& port_indexer =
      simulator.GetRoutingTable()->GetPortIndices();

  // Setup structures associated with the inputs.
  //  - input to SimConnectionState (input_connection_index_start_ and count_)
  //  - input_buffers_
  input_connection_count_ = nc.GetInputPortIds().size();
  input_connection_index_start_ =
      simulator.GetNewConnectionIndicesStore(input_connection_count_);
  absl::Span<int64_t> input_indices = simulator.GetConnectionIndicesStore(
      input_connection_index_start_, input_connection_count_);

  input_buffers_.resize(input_connection_count_);
  input_credit_to_send_.resize(input_connection_count_);
  max_vc_ = 0;
  for (int64_t i = 0; i < input_connection_count_; ++i) {
    XLS_ASSIGN_OR_RETURN(
        PortId port_id,
        port_indexer.GetPortByIndex(nc.id(), PortDirection::kInput, i));
    Port& port = network_manager->GetPort(port_id);
    input_indices[i] = simulator.GetConnectionIndex(port.connection());

    XLS_ASSIGN_OR_RETURN(PortParam port_param,
                         simulator.GetNocParameters()->GetPortParam(port_id));
    std::vector<VirtualChannelParam> vc_params =
        port_param.GetVirtualChannels();

    input_buffers_[i].resize(port_param.VirtualChannelCount());
    for (int64_t vc = 0; vc < port_param.VirtualChannelCount(); ++vc) {
      input_buffers_[i][vc].max_queue_size = vc_params[vc].GetDepth();
    }
    input_credit_to_send_[i].resize(port_param.VirtualChannelCount());
    if (max_vc_ < port_param.VirtualChannelCount()) {
      max_vc_ = port_param.VirtualChannelCount();
    }
  }

  // Setup structures associated with the outputs.
  //  - output to SimConnectionState (output_connection_index_start_ and count_)
  //  - credits associated with the outputs
  output_connection_count_ = nc.GetOutputPortIds().size();
  output_connection_index_start_ =
      simulator.GetNewConnectionIndicesStore(output_connection_count_);
  absl::Span<int64_t> output_indices = simulator.GetConnectionIndicesStore(
      output_connection_index_start_, output_connection_count_);
  credit_.resize(output_connection_count_);
  credit_update_.resize(output_connection_count_);
  for (int64_t i = 0; i < output_connection_count_; ++i) {
    XLS_ASSIGN_OR_RETURN(
        PortId port_id,
        port_indexer.GetPortByIndex(nc.id(), PortDirection::kOutput, i));
    Port& port = network_manager->GetPort(port_id);
    output_indices[i] = simulator.GetConnectionIndex(port.connection());

    XLS_ASSIGN_OR_RETURN(PortParam port_param,
                         simulator.GetNocParameters()->GetPortParam(port_id));
    credit_[i].resize(port_param.VirtualChannelCount(), 0);
    credit_update_[i].resize(port_param.VirtualChannelCount(),
                             CreditState{simulator.GetCurrentCycle(), 0});
  }

  internal_propagated_cycle_ = simulator.GetCurrentCycle();
  utilization_cycle_count_ = 0;

  return absl::OkStatus();
}

bool SimLink::TryForwardPropagation(NocSimulator& simulator) {
  SimConnectionState& src =
      simulator.GetSimConnectionByIndex(src_connection_index_);
  SimConnectionState& sink =
      simulator.GetSimConnectionByIndex(sink_connection_index_);

  bool did_propagate =
      SimplePipelineImpl<TimedDataFlit>(
          forward_pipeline_stages_, src.forward_channels, sink.forward_channels,
          forward_data_stages_, internal_forward_propagated_cycle_)
          .TryPropagation(simulator);

  if (did_propagate) {
    VLOG(2) << absl::StreamFormat("Forward propagated from connection %x to %x",
                                  src.id.AsUInt64(), sink.id.AsUInt64());
    forward_propagated_cycle_ = simulator.GetCurrentCycle();
  }

  return did_propagate;
}

bool SimLink::TryReversePropagation(NocSimulator& simulator) {
  SimConnectionState& src =
      simulator.GetSimConnectionByIndex(src_connection_index_);
  SimConnectionState& sink =
      simulator.GetSimConnectionByIndex(sink_connection_index_);

  int64_t vc_count = sink.reverse_channels.size();
  int64_t num_propagated = 0;
  for (int64_t vc = 0; vc < vc_count; ++vc) {
    if (SimplePipelineImpl<TimedMetadataFlit>(
            reverse_pipeline_stages_, sink.reverse_channels.at(vc),
            src.reverse_channels.at(vc), reverse_credit_stages_.at(vc),
            internal_reverse_propagated_cycle_.at(vc))
            .TryPropagation(simulator)) {
      ++num_propagated;
      VLOG(2) << absl::StreamFormat(
          "Reverse propagated from connection %x to %x", sink.id.AsUInt64(),
          src.id.AsUInt64());
    }
  }

  if (num_propagated == vc_count) {
    reverse_propagated_cycle_ = simulator.GetCurrentCycle();
    return true;
  }
  return false;
}

bool SimNetworkInterfaceSrc::TryForwardPropagation(NocSimulator& simulator) {
  int64_t current_cycle = simulator.GetCurrentCycle();
  SimConnectionState& sink =
      simulator.GetSimConnectionByIndex(sink_connection_index_);

  // Update credits.
  // No need to check for cycle here, because forward propagation
  // always succeeds and occurs before reverse propagation.
  // Sequence of operations is
  //  1. Credits are updated based off of prior cycle's received update
  //  2. Phits are sent va forward propagation.
  //  3. Reverse propagation updates the credit_update (for next cycle).
  for (int64_t vc = 0; vc < credit_.size(); ++vc) {
    if (credit_update_[vc].credit > 0) {
      credit_[vc] += credit_update_[vc].credit;
      VLOG(2) << absl::StrFormat("... ni-src vc %d added credits %d, now %d",
                                 vc, credit_update_[vc].credit, credit_[vc]);
    }
  }

  // Send data.
  bool flit_sent = false;

  for (int64_t vc = 0; vc < data_to_send_.size(); ++vc) {
    std::queue<TimedDataFlit>& send_queue = data_to_send_[vc];
    if (!send_queue.empty() && send_queue.front().cycle <= current_cycle) {
      if (credit_[vc] > 0) {
        sink.forward_channels.flit = send_queue.front().flit;
        sink.forward_channels.flit.vc = vc;
        sink.forward_channels.cycle = current_cycle;
        sink.forward_channels.metadata = send_queue.front().metadata;
        sink.forward_channels.metadata.timed_route_info.route.push_back(
            TimedRouteItem{id_, current_cycle});

        --credit_[vc];

        send_queue.pop();
        flit_sent = true;

        VLOG(2) << absl::StreamFormat(
            "... ni-src sending data %s vc %d credit now %d",
            sink.forward_channels.flit, vc, credit_[vc]);
        break;
      }
      VLOG(2) << absl::StreamFormat(
          "... ni-src unable to send data %s vc %d credit %d",
          sink.forward_channels.flit, vc, credit_[vc]);
    }
  }

  if (!flit_sent) {
    sink.forward_channels.flit =
        DataFlitBuilder().Invalid().BuildFlit().value();
    sink.forward_channels.cycle = current_cycle;
  }

  forward_propagated_cycle_ = current_cycle;

  return true;
}

bool SimNetworkInterfaceSrc::TryReversePropagation(NocSimulator& simulator) {
  int64_t current_cycle = simulator.GetCurrentCycle();
  SimConnectionState& sink =
      simulator.GetSimConnectionByIndex(sink_connection_index_);

  int64_t vc_count = credit_update_.size();
  int64_t num_propagated = 0;
  VLOG(2) << absl::StreamFormat("... ni-src vc %d", vc_count);
  for (int64_t vc = 0; vc < vc_count; ++vc) {
    TimedMetadataFlit possible_credit = sink.reverse_channels[vc];
    if (possible_credit.cycle == current_cycle) {
      if (credit_update_[vc].cycle != current_cycle) {
        credit_update_[vc].cycle = current_cycle;

        if (possible_credit.flit.type != FlitType::kInvalid) {
          int64_t credit = possible_credit.flit.data.ToInt64().value();
          credit_update_[vc].credit = credit;
        } else {
          credit_update_[vc].credit = 0;
        }

        VLOG(2) << absl::StreamFormat(
            "... ni-src received credit %d vc %d via connection %x",
            credit_update_[vc].credit, vc, sink.id.AsUInt64());
      }

      VLOG(2) << absl::StreamFormat("... ni-src credit update cycle %x vc %d",
                                    credit_update_[vc].cycle, vc);

      ++num_propagated;
    }
  }

  if (num_propagated == vc_count) {
    VLOG(2) << absl::StreamFormat(
        "... ni-src %x connected to %x finished reverse propagation",
        GetId().AsUInt64(), sink.id.AsUInt64());

    reverse_propagated_cycle_ = current_cycle;
    return true;
  }
  return false;
}

absl::StatusOr<SimInputBufferedVCRouter::PortIndexAndVCIndex>
SimInputBufferedVCRouter::GetDestinationPortIndexAndVcIndex(
    NocSimulator& simulator, PortIndexAndVCIndex input,
    int64_t destination_index) {
  DistributedRoutingTable* routes = simulator.GetRoutingTable();

  XLS_ASSIGN_OR_RETURN(PortId input_port,
                       routes->GetPortIndices().GetPortByIndex(
                           GetId(), PortDirection::kInput, input.port_index));

  PortAndVCIndex port_from{input_port, input.vc_index};

  XLS_ASSIGN_OR_RETURN(
      PortAndVCIndex port_to,
      routes->GetRouterOutputPortByIndex(port_from, destination_index));

  XLS_ASSIGN_OR_RETURN(int64_t output_port_index,
                       routes->GetPortIndices().GetPortIndex(
                           port_to.port_id_, PortDirection::kOutput));

  return PortIndexAndVCIndex{output_port_index, port_to.vc_index_};
}

bool SimInputBufferedVCRouter::TryForwardPropagation(NocSimulator& simulator) {
  // TODO(tedhong): 2020-02-16 Factor out with strategy pattern.

  int64_t current_cycle = simulator.GetCurrentCycle();
  absl::Span<int64_t> input_connection_index =
      simulator.GetConnectionIndicesStore(input_connection_index_start_,
                                          input_connection_count_);
  absl::Span<int64_t> output_connection_index =
      simulator.GetConnectionIndicesStore(output_connection_index_start_,
                                          output_connection_count_);

  // Update credits (for output ports)
  if (internal_propagated_cycle_ != current_cycle) {
    for (int64_t i = 0; i < credit_update_.size(); ++i) {
      for (int64_t vc = 0; vc < credit_update_[i].size(); ++vc) {
        if (credit_update_[i][vc].credit > 0) {
          credit_[i][vc] += credit_update_[i][vc].credit;
          VLOG(2) << absl::StrFormat(
              "... router %x output port %d vc %d added credits %d, now %d",
              GetId().AsUInt64(), i, vc, credit_update_[i][vc].credit,
              credit_[i][vc]);
        } else {
          VLOG(2) << absl::StrFormat(
              "... router %x output port %d vc %d did not add credits %d, now "
              "%d",
              GetId().AsUInt64(), i, vc, credit_update_[i][vc].credit,
              credit_[i][vc]);
        }
      }
    }

    internal_propagated_cycle_ = current_cycle;
  }

  // See if we can propagate forward.
  bool can_propagate_forward = true;
  for (int64_t i = 0; i < input_connection_count_; ++i) {
    SimConnectionState& input =
        simulator.GetSimConnectionByIndex(input_connection_index[i]);

    if (input.forward_channels.cycle != current_cycle) {
      can_propagate_forward = false;
      break;
    }
  }

  if (!can_propagate_forward) {
    return false;
  }

  // Reset credits to send on reverse channel to 0.
  for (int64_t i = 0; i < input_connection_count_; ++i) {
    for (int64_t vc = 0; vc < input_credit_to_send_[i].size(); ++vc) {
      input_credit_to_send_[i][vc] = 0;
    }
  }

  bool flit_sent = false;
  // This router supports bypass so a flit arriving at the
  // input can be routed to the output immediately.
  for (int64_t i = 0; i < input_connection_count_; ++i) {
    SimConnectionState& input =
        simulator.GetSimConnectionByIndex(input_connection_index[i]);

    if (input.forward_channels.flit.type != FlitType::kInvalid) {
      int64_t vc = input.forward_channels.flit.vc;
      input_buffers_[i][vc].queue.push(
          {input.forward_channels.flit, input.forward_channels.metadata});

      VLOG(2) << absl::StrFormat(
          "... router %x from %x received data %s port %d vc %d",
          GetId().AsUInt64(), input.id.AsUInt64(), input.forward_channels.flit,
          i, vc);
    }
  }

  // Use fixed priority to route to output ports.
  // Priority goes to the port with the least vc and the least port index.
  for (int64_t vc = 0; vc < max_vc_; ++vc) {
    for (int64_t i = 0; i < input_buffers_.size(); ++i) {
      if (vc >= input_buffers_[i].size()) {
        continue;
      }

      // See if we have a flit to route and can route it.
      if (input_buffers_[i][vc].queue.empty()) {
        continue;
      }

      DataFlit flit = input_buffers_[i][vc].queue.front().flit;
      TimedDataFlitInfo metadata = input_buffers_[i][vc].queue.front().metadata;
      int64_t destination_index = flit.destination_index;

      PortIndexAndVCIndex input{i, vc};
      absl::StatusOr<PortIndexAndVCIndex> output_status =
          GetDestinationPortIndexAndVcIndex(simulator, input,
                                            destination_index);
      CHECK_OK(output_status.status());
      PortIndexAndVCIndex output = output_status.value();

      // Now see if we have sufficient credits.
      if (credit_.at(output.port_index).at(output.vc_index) <= 0) {
        VLOG(2) << absl::StreamFormat(
            "... router unable to send data %s vc %d credit now %d"
            " from port index %d to port index %d.",
            flit, flit.vc, credit_.at(output.port_index).at(output.vc_index), i,
            output.port_index);
        continue;
      }

      // Check that no other port has already used the output port
      // (since this is a router without output buffers.
      SimConnectionState& output_state = simulator.GetSimConnectionByIndex(
          output_connection_index.at(output.port_index));
      if (output_state.forward_channels.cycle == current_cycle) {
        continue;
      }

      // Now send the flit along.
      output_state.forward_channels.flit = flit;
      output_state.forward_channels.flit.vc = output.vc_index;
      output_state.forward_channels.cycle = current_cycle;
      output_state.forward_channels.metadata = metadata;
      output_state.forward_channels.metadata.timed_route_info.route.push_back(
          TimedRouteItem{id_, current_cycle});

      // Update credit on output.
      --credit_.at(output.port_index).at(output.vc_index);

      // Update credit to send back to input.
      ++input_credit_to_send_[i][vc];
      input_buffers_[i][vc].queue.pop();

      flit_sent = true;

      VLOG(2) << absl::StreamFormat(
          "... router sending data %s vc %d credit now %d"
          " from port index %d to port index %d on %x.",
          output_state.forward_channels.flit,
          output_state.forward_channels.flit.vc,
          credit_.at(output.port_index).at(output.vc_index), i,
          output.port_index, output_state.id.AsUInt64());
    }
  }

  // Now put bubbles in output ports that couldn't send data.
  for (int64_t i = 0; i < output_connection_index.size(); ++i) {
    SimConnectionState& output =
        simulator.GetSimConnectionByIndex(output_connection_index[i]);
    if (output.forward_channels.cycle != current_cycle) {
      output.forward_channels.flit =
          DataFlitBuilder().Invalid().BuildFlit().value();
      output.forward_channels.cycle = current_cycle;
    }
  }

  // Collect some statistics
  if (flit_sent) {
    utilization_cycle_count_++;
  }

  forward_propagated_cycle_ = current_cycle;

  return true;
}

bool SimInputBufferedVCRouter::TryReversePropagation(NocSimulator& simulator) {
  int64_t current_cycle = simulator.GetCurrentCycle();

  // Reverse propagation occurs only after forward propagation.
  if (forward_propagated_cycle_ != current_cycle) {
    return false;
  }

  absl::Span<int64_t> input_connection_index =
      simulator.GetConnectionIndicesStore(input_connection_index_start_,
                                          input_connection_count_);

  // Send credit upstream.
  for (int64_t i = 0; i < input_connection_count_; ++i) {
    SimConnectionState& input =
        simulator.GetSimConnectionByIndex(input_connection_index[i]);

    for (int64_t vc = 0; vc < input.reverse_channels.size(); ++vc) {
      input.reverse_channels[vc].flit.type = FlitType::kTail;

      // Upon reset (cycle-0) a full update of credits is sent.
      if (current_cycle == 0) {
        input.reverse_channels[vc].flit.data =
            UBits(input_buffers_[i][vc].max_queue_size, 32);
      } else {
        input.reverse_channels[vc].flit.data =
            UBits(input_credit_to_send_[i][vc], 32);
      }
      input.reverse_channels[vc].cycle = current_cycle;

      VLOG(2) << absl::StreamFormat(
          "... router %x sending credit update %s"
          " input port %d vc %d connection %x",
          GetId().AsUInt64(), input.reverse_channels[vc].flit, i, vc,
          input.id.AsUInt64());
    }
  }

  // Recieve credit from downstream.
  absl::Span<int64_t> output_connection_index =
      simulator.GetConnectionIndicesStore(output_connection_index_start_,
                                          output_connection_count_);

  int64_t num_propagated = 0;
  int64_t possible_propagation = 0;
  for (int64_t i = 0; i < credit_update_.size(); ++i) {
    SimConnectionState& output =
        simulator.GetSimConnectionByIndex(output_connection_index.at(i));

    for (int64_t vc = 0; vc < credit_update_[i].size(); ++vc) {
      TimedMetadataFlit possible_credit = output.reverse_channels[vc];

      if (possible_credit.cycle == current_cycle) {
        if (credit_update_[i][vc].cycle != current_cycle) {
          credit_update_[i][vc].cycle = current_cycle;

          if (possible_credit.flit.type != FlitType::kInvalid) {
            int64_t credit = possible_credit.flit.data.ToInt64().value();
            credit_update_[i][vc].credit = credit;
          } else {
            credit_update_[i][vc].credit = 0;
          }

          VLOG(2) << absl::StreamFormat(
              "... router received credit %d output port %d vc %d via "
              "connection %x",
              credit_update_[i][vc].credit, i, vc, output.id.AsUInt64());
        }

        ++num_propagated;
      } else {
        VLOG(2) << absl::StreamFormat(
            "... router output port %d vc %d waiting for credits via "
            "connection %x",
            i, vc, output.id.AsUInt64());
      }

      ++possible_propagation;
    }
  }

  if (possible_propagation == num_propagated) {
    VLOG(2) << absl::StreamFormat("... router %x finished reverse propagation",
                                  GetId().AsUInt64());
    return true;
  }
  VLOG(2) << absl::StreamFormat(
      "... router %x did not finish reverse propagation", GetId().AsUInt64());
  return false;
}

bool SimNetworkInterfaceSink::TryForwardPropagation(NocSimulator& simulator) {
  int64_t current_cycle = simulator.GetCurrentCycle();

  SimConnectionState& src =
      simulator.GetSimConnectionByIndex(src_connection_index_);

  if (src.forward_channels.cycle != current_cycle) {
    return false;
  }

  if (src.forward_channels.flit.type != FlitType::kInvalid) {
    Bits data = src.forward_channels.flit.data;
    int64_t vc = src.forward_channels.flit.vc;

    // TODO(tedhong): 2021-01-31 Support blocking traffic at sink.
    // without blocking, the queue never gets empty so we don't
    // emplace into input_buffers_[vc].queue.
    TimedDataFlit received_flit;
    received_flit.cycle = current_cycle;
    received_flit.flit = src.forward_channels.flit;
    received_flit.metadata = src.forward_channels.metadata;
    received_flit.metadata.timed_route_info.route.push_back(
        TimedRouteItem{id_, current_cycle});
    received_traffic_.push_back(received_flit);

    // Send one credit back
    src.reverse_channels[vc].cycle = current_cycle;
    src.reverse_channels[vc].flit.type = FlitType::kTail;
    src.reverse_channels[vc].flit.data = UBits(1, 32);

    VLOG(2) << absl::StreamFormat(
        "... sink %x received data %v on vc %d cycle %d, sending 1 credit on "
        "%x",
        GetId().AsUInt64(), data, vc, current_cycle, src.id.AsUInt64());
  }

  // In cycle 0, a full credit update is sent
  if (current_cycle == 0) {
    for (int64_t vc = 0; vc < src.reverse_channels.size(); ++vc) {
      src.reverse_channels[vc].cycle = current_cycle;
      src.reverse_channels[vc].flit.type = FlitType::kTail;
      src.reverse_channels[vc].flit.data =
          UBits(input_buffers_[vc].max_queue_size, 32);

      VLOG(2) << absl::StreamFormat(
          "... sink %x sending %d credit vc %d on %x", GetId().AsUInt64(),
          input_buffers_[vc].max_queue_size, vc, src.id.AsUInt64());
    }
  } else {
    for (int64_t vc = 0; vc < src.reverse_channels.size(); ++vc) {
      if (src.reverse_channels[vc].cycle != current_cycle) {
        src.reverse_channels[vc].cycle = current_cycle;
        src.reverse_channels[vc].flit.type = FlitType::kInvalid;
        src.reverse_channels[vc].flit.data = Bits(32);
      }
    }
  }

  return true;
}

absl::StatusOr<SimNetworkInterfaceSrc*> NocSimulator::GetSimNetworkInterfaceSrc(
    NetworkComponentId src) {
  auto iter = src_index_map_.find(src);
  if (iter == src_index_map_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Unable to find sim object for"
                        " network interface src %x",
                        src.AsUInt64()));
  }

  return &network_interface_sources_[iter->second];
}

absl::StatusOr<SimNetworkInterfaceSink*>
NocSimulator::GetSimNetworkInterfaceSink(NetworkComponentId sink) {
  auto iter = sink_index_map_.find(sink);
  if (iter == sink_index_map_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("Unable to find sim object for"
                        " network interface src %x",
                        sink.AsUInt64()));
  }

  return &network_interface_sinks_[iter->second];
}

absl::Span<const SimInputBufferedVCRouter> NocSimulator::GetRouters() const {
  return routers_;
}

absl::Span<const SimLink> NocSimulator::GetLinks() const { return links_; }

}  // namespace noc
}  // namespace xls
