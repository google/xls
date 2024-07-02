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

#ifndef XLS_NOC_SIMULATION_SIM_OBJECTS_H_
#define XLS_NOC_SIMULATION_SIM_OBJECTS_H_

#include <cstdint>
#include <queue>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/flit.h"
#include "xls/noc/simulation/global_routing_table.h"
#include "xls/noc/simulation/parameters.h"
#include "xls/noc/simulation/simulator_shims.h"

// This file contains classes used to store, access, and define simulation
// objects.  Each network object (defined network_graph.h) is associated
// with a simulation object, depending on how said network object is
// configured via its parameters (see parameters.h).

namespace xls {
namespace noc {

// Used to store the state of phits in-flight for a network.
// It is associated with a ConnectionId which connects two ports.
struct SimConnectionState {
  ConnectionId id;
  TimedDataFlit forward_channels;
  std::vector<TimedMetadataFlit> reverse_channels;
};

// Used to store the valid credit available at a certain time.
struct CreditState {
  int64_t cycle;
  int64_t credit;
};

struct DataFlitQueueElement {
  DataFlit flit;
  TimedDataFlitInfo metadata;
};

// Represents a fifo/buffer used to store phits.
struct DataFlitQueue {
  std::queue<DataFlitQueueElement> queue;
  int64_t max_queue_size;
};

// Represents a fifo/buffer used to store metadata phits.

class NocSimulator;

// Common functionality and base class for all simulator objects.
class SimNetworkComponentBase {
 public:
  // Perform a single tick of simulator.
  // Returns true if the component has converged for the given cycle.
  //
  // A component has converged it both forward and reverse propgation
  // have completed.  This means that all SimConnectionState objects
  // attached to this component have state associated with the current.
  // cycle.
  //
  // See NocSimulator::Tick.
  bool Tick(NocSimulator& simulator);

  // Returns the associated NetworkComponentId.
  NetworkComponentId GetId() const { return id_; }

  virtual ~SimNetworkComponentBase() = default;

 protected:
  SimNetworkComponentBase() = default;

  // Initialize this simulator object.
  //
  // After initialization, the simulator object will be set up to
  // simulate the specific component as described in the protos.
  //
  // For example, buffer sizes and the number of virtual channels will
  // be read from NOC config protos to properly size the simulation object.
  absl::Status Initialize(NetworkComponentId nc_id, NocSimulator& simulator) {
    id_ = nc_id;
    forward_propagated_cycle_ = -1;
    reverse_propagated_cycle_ = -1;

    return InitializeImpl(simulator);
  }

  // Component specific initialization of a SimNetworkComponent.
  virtual absl::Status InitializeImpl(NocSimulator& simulator) {
    return absl::OkStatus();
  }

  // Propagates simulation state from source connections to sink.
  // Returns true if ready and simulation state was propagated.
  //
  // True can be returned if 1) all input ports are ready for forward
  // propagation (input port's connection forward_channel time stamp
  // equals current cycle), and 2) all output port state have been updated
  // (output port connection's forward_channel time stamp equals the current
  // cycle).
  //
  // A simulation cycle is complete once all component's Forward and
  // Reverse propagation methods return true.
  virtual bool TryForwardPropagation(NocSimulator& simulator) { return true; }

  // Propagates simulation state from sink connections to source.
  // Returns true if ready and simulation state was propagated.
  //
  // True can be returned if 1) all output ports are ready for reverse
  // propagation (output port's connection reverse_channel time stamp
  // equals current cycle) , and 2) all input port state have been updated
  // (input port connection's reverse channel time stamp equals the current
  // cycle).
  virtual bool TryReversePropagation(NocSimulator& simulator) { return true; }

  NetworkComponentId id_;
  int64_t forward_propagated_cycle_;
  int64_t reverse_propagated_cycle_;
};

// A pair of pipeline stages connecting two ports/network components.
//
// DataFlits are propagated forward, while MetaDataFlits are propagated
// backwards.
class SimLink : public SimNetworkComponentBase {
 public:
  static absl::StatusOr<SimLink> Create(NetworkComponentId nc_id,
                                        NocSimulator& simulator) {
    SimLink ret;
    XLS_RETURN_IF_ERROR(ret.Initialize(nc_id, simulator));
    return ret;
  }

  // Get the source connection index that in used in the simulator.
  int64_t GetSourceConnectionIndex() const;

  // Get the sink connection index that in used in the simulator.

 private:
  SimLink() = default;

  absl::Status InitializeImpl(NocSimulator& simulator) override;

  bool TryForwardPropagation(NocSimulator& simulator) override;
  bool TryReversePropagation(NocSimulator& simulator) override;

  int64_t forward_pipeline_stages_;
  int64_t reverse_pipeline_stages_;

  // TODO(tedhong): 2020-01-25 support phit_width, currently unused.
  int64_t phit_width_;

  int64_t src_connection_index_;
  int64_t sink_connection_index_;

  std::queue<TimedDataFlit> forward_data_stages_;
  int64_t internal_forward_propagated_cycle_;

  std::vector<std::queue<TimedMetadataFlit>> reverse_credit_stages_;
  std::vector<int64_t> internal_reverse_propagated_cycle_;
};

// Source - injects traffic into the network.
class SimNetworkInterfaceSrc : public SimNetworkComponentBase {
 public:
  static absl::StatusOr<SimNetworkInterfaceSrc> Create(
      NetworkComponentId nc_id, NocSimulator& simulator) {
    SimNetworkInterfaceSrc ret;
    XLS_RETURN_IF_ERROR(ret.Initialize(nc_id, simulator));
    return ret;
  }

  // Register a flit to be sent at a specific time.
  absl::Status SendFlitAtTime(TimedDataFlit flit);

 private:
  SimNetworkInterfaceSrc() = default;

  absl::Status InitializeImpl(NocSimulator& simulator) override;
  bool TryForwardPropagation(NocSimulator& simulator) override;
  bool TryReversePropagation(NocSimulator& simulator) override;

  int64_t sink_connection_index_;
  std::vector<int64_t> credit_;
  std::vector<CreditState> credit_update_;
  std::vector<std::queue<TimedDataFlit>> data_to_send_;
};

// Sink - traffic leaves the network via a sink.
class SimNetworkInterfaceSink : public SimNetworkComponentBase {
 public:
  static absl::StatusOr<SimNetworkInterfaceSink> Create(
      NetworkComponentId nc_id, NocSimulator& simulator) {
    SimNetworkInterfaceSink ret;
    XLS_RETURN_IF_ERROR(ret.Initialize(nc_id, simulator));
    return ret;
  }

  // Returns all traffic received by this sink from the beginning
  // of the simulation.
  absl::Span<const TimedDataFlit> GetReceivedTraffic() {
    return received_traffic_;
  }

  // Returns the observed rate of traffic in MebiBytes Per Second from the
  // beginning of simulation to the last flit processed by this sink.
  //
  // VC is used to filter out the traffic as received on a specific vc index.
  // Negative VC is used to match any vc.
  double MeasuredTrafficRateInMiBps(int64_t cycle_time_ps, int64_t vc = -1) {
    // TODO(tedhong): 2021-07-01 Factor this logic out into common library.
    int64_t num_bits = 0;
    int64_t max_cycle = 0;
    for (TimedDataFlit& f : received_traffic_) {
      if (vc < 0) {
        num_bits += f.flit.data_bit_count;
      } else {
        if (f.flit.vc == vc) {
          num_bits += f.flit.data_bit_count;
        }
      }

      if (max_cycle < f.cycle) {
        max_cycle = f.cycle;
      }
    }

    double total_sec = static_cast<double>(max_cycle + 1) *
                       static_cast<double>(cycle_time_ps) * 1.0e-12;
    double bits_per_sec = static_cast<double>(num_bits) / total_sec;
    return bits_per_sec / 1024.0 / 1024.0 / 8.0;
  }

 private:
  SimNetworkInterfaceSink() = default;

  absl::Status InitializeImpl(NocSimulator& simulator) override;

  bool TryForwardPropagation(NocSimulator& simulator) override;

  int64_t src_connection_index_;
  std::vector<DataFlitQueue> input_buffers_;
  std::vector<TimedDataFlit> received_traffic_;
};

// Represents an input-buffered, fixed priority, credit-based, virtual-channel
// router.
//
// This router implements a specific type of router used by the simulator.
// Additional routers are implemented either as a separate class or
// by configuring this class.
//
// Specific features include
//   - Input buffered - phits are buffered at the input.
//   - Input bypass - a flit can enter the router and leave on the same cycle.
//   - Credits - the router keeps track of the absolute credit count and
//               expects incremental updates from the components downstream.
//             - credits are registered so there is a one-cycle delay
//               from when the credit is received and the credit count
//               updated.
//             - the router likewise sends credit updates upstream.
//   - Dedicated credit channels - Each vc is associated with an independent
//                                 channel for credit updates.
//   - Output bufferless - once a flit is arbitrated for, the flit is
//                         immediately transferred downstream.
//   - Fixed priority - a fixed priority scheme is implemented.
// TODO(tedhong): 2021-01-31 - Add support for alternative priority scheme.
class SimInputBufferedVCRouter : public SimNetworkComponentBase {
 public:
  static absl::StatusOr<SimInputBufferedVCRouter> Create(
      NetworkComponentId nc_id, NocSimulator& simulator) {
    SimInputBufferedVCRouter ret;
    XLS_RETURN_IF_ERROR(ret.Initialize(nc_id, simulator));
    return ret;
  }

  int64_t GetUtilizationCycleCount() const;

 private:
  SimInputBufferedVCRouter() = default;

  // Represents a specific input or output location.
  struct PortIndexAndVCIndex {
    int64_t port_index;
    int64_t vc_index;
  };

  absl::Status InitializeImpl(NocSimulator& simulator) override;

  // Forward propagation
  //  1. Updates the credit count (internal propagation)
  //  2. Waits until all input ports are ready.
  //  3. Enqueues phits into input buffers and performs routing if able.
  bool TryForwardPropagation(NocSimulator& simulator) override;

  // Reverse propagation
  //  1. Sends credits back upstream (due to fwd propagation routing phits).
  //  2. Registers credits received from downstream.
  bool TryReversePropagation(NocSimulator& simulator) override;

  // Perform the routing function of this router.
  //
  // Returns a pair of <output_port_index, output_vc_index> -- the
  // output port and vc a flit should go out on given the input port and vc
  // along with the eventual flit destination.
  absl::StatusOr<PortIndexAndVCIndex> GetDestinationPortIndexAndVcIndex(
      NocSimulator& simulator, PortIndexAndVCIndex input,
      int64_t destination_index);

  // Index for the input connections associated with this router.
  // Each input port is associated with a single connection.
  int64_t input_connection_index_start_;
  int64_t input_connection_count_;

  // Index for the output connections associated with this router.
  // Each output port is associated with a single connection.
  int64_t output_connection_index_start_;
  int64_t output_connection_count_;

  // The router as finished internal propagation once it has
  // updated its credit count from the updates received in the previous cycle.
  int64_t internal_propagated_cycle_;

  // Stores the input buffers associated with each input port and vc.
  std::vector<std::vector<DataFlitQueue>> input_buffers_;

  // Stores the credit count associated with each output port and vc.
  // Each cycle, the router updates its credit count from credit_update_.
  std::vector<std::vector<int64_t>> credit_;

  // Stores the credit count received on cycle N-1.
  std::vector<std::vector<CreditState>> credit_update_;

  // The maximum number of vcs on for an input port.
  // Used for the priority scheme implementation.
  int64_t max_vc_;

  // Used by forward propagation to store the number of phits that left
  // the input buffers and hence credits that can be sent back upstream.
  std::vector<std::vector<int64_t>> input_credit_to_send_;

  // The number of cycles that a transfer from input to output occurred.
  int64_t utilization_cycle_count_;
};

// Main simulator class that drives the simulation and stores simulation
// state and objects.
class NocSimulator {
 public:
  NocSimulator()
      : mgr_(nullptr), params_(nullptr), routing_(nullptr), cycle_(-1) {}

  // Creates all simulation objects for a given network.
  // NetworkManager, NocParameters, and DistributedRoutingTable should
  // have aleady been setup.
  absl::Status Initialize(NetworkManager& mgr, NocParameters& params,
                          DistributedRoutingTable& routing, NetworkId network) {
    mgr_ = &mgr;
    params_ = &params;
    routing_ = &routing;
    network_ = network;
    cycle_ = -1;

    return CreateSimulationObjects(network);
  }

  NetworkManager* GetNetworkManager() { return mgr_; }
  NocParameters* GetNocParameters() { return params_; }
  DistributedRoutingTable* GetRoutingTable() { return routing_; }

  // Maps a given connection id to its index in the connection store.
  int64_t GetConnectionIndex(ConnectionId id) {
    return connection_index_map_[id];
  }

  // Returns a SimConnectionState given an index.
  SimConnectionState& GetSimConnectionByIndex(int64_t index) {
    return connections_[index];
  }

  // Allocates and returns a new SimConnectionState object.
  SimConnectionState& NewConnection(ConnectionId id) {
    int64_t index = connections_.size();
    connections_.resize(index + 1);
    connection_index_map_[id] = index;
    return GetSimConnectionByIndex(index);
  }

  // Returns a reference to the store previously reserved with
  // GetNewConnectionIndicesStore.
  absl::Span<int64_t> GetConnectionIndicesStore(int64_t start,
                                                int64_t size = 1) {
    return absl::Span<int64_t>(component_to_connection_index_.data() + start,
                               size);
  }

  // Allocates and returns an index that can then be used
  // with GetConnectionIndicesStore to retrieve an array of size.
  int64_t GetNewConnectionIndicesStore(int64_t size) {
    int64_t next_start = component_to_connection_index_.size();
    component_to_connection_index_.resize(next_start + size);
    return next_start;
  }

  // Allocates and returns an index that can be used with
  // GetPortIdStore to retreive an array of size)

  // Returns a reference to the store previously reserved with
  // GetNewConnectionIndicesStore.

  // Returns current/in-progress cycle;
  int64_t GetCurrentCycle() { return cycle_; }

  // Logs the current simulation state.
  void Dump();

  // Run a single cycle of the simulator.
  absl::Status RunCycle(int64_t max_ticks = 9999);

  // Runs a single tick of the simulator.
  bool Tick();

  // Register a service to run once at the beginning of each cycle.
  // TODO(tedhong): 2021-07-27 Add a scheme to provide a total order
  //                of services.
  void RegisterPreCycleService(NocSimulatorServiceShim& svc) {
    pre_cycle_services_.push_back(&svc);
  }

  // Register a service to run once at the end of each cycle.
  void RegisterPostCycleService(NocSimulatorServiceShim& svc) {
    post_cycle_services_.push_back(&svc);
  }

  // Returns corresponding simulation object for a src network component.
  absl::StatusOr<SimNetworkInterfaceSrc*> GetSimNetworkInterfaceSrc(
      NetworkComponentId src);

  // Returns corresponding simulation object for a sink network component.
  absl::StatusOr<SimNetworkInterfaceSink*> GetSimNetworkInterfaceSink(
      NetworkComponentId sink);

  // Returns the routers of the simulator.
  absl::Span<const SimInputBufferedVCRouter> GetRouters() const;

  // Returns the links of the simulator.
  absl::Span<const SimLink> GetLinks() const;

 private:
  absl::Status CreateSimulationObjects(NetworkId network);
  absl::Status CreateConnection(ConnectionId connection_id);
  absl::Status CreateNetworkComponent(NetworkComponentId nc_id);
  absl::Status CreateNetworkInterfaceSrc(NetworkComponentId nc_id);
  absl::Status CreateNetworkInterfaceSink(NetworkComponentId nc_id);
  absl::Status CreateLink(NetworkComponentId nc_id);
  absl::Status CreateRouter(NetworkComponentId nc_id);

  NetworkManager* mgr_;
  NocParameters* params_;
  DistributedRoutingTable* routing_;

  NetworkId network_;
  int64_t cycle_;

  // Map a specific ConnectionId to an index used to access
  // a specific SimConnectionState via the connections_ object.
  absl::flat_hash_map<ConnectionId, int64_t> connection_index_map_;

  // Map a network interface src to a SimNetworkInterfaceSrc.
  absl::flat_hash_map<NetworkComponentId, int64_t> src_index_map_;

  // Map a network interface sink to a SimNetworkInterfaceSink.
  absl::flat_hash_map<NetworkComponentId, int64_t> sink_index_map_;

  // Used by network components to store an array of indices.
  //
  // Those indices are used to index into the connection_ object to
  // access a SimConnectionState.
  //
  // For example, a router can reserve space so that for port x
  //  connections_[component_to_connection_index_[x]] is then the
  //  corresponding SimConnectionState for said port.
  std::vector<int64_t> component_to_connection_index_;
  std::vector<SimConnectionState> connections_;

  // Stores port ids for routers.
  std::vector<PortId> port_id_store_;

  std::vector<SimLink> links_;
  std::vector<SimNetworkInterfaceSrc> network_interface_sources_;
  std::vector<SimNetworkInterfaceSink> network_interface_sinks_;
  std::vector<SimInputBufferedVCRouter> routers_;

  // Shims to services to run at the beginning of each cycle.
  std::vector<NocSimulatorServiceShim*> pre_cycle_services_;

  // Shims to services to run at the end of each cycle.
  std::vector<NocSimulatorServiceShim*> post_cycle_services_;
};

}  // namespace noc
}  // namespace xls

#endif  // XLS_NOC_SIMULATION_SIM_OBJECTS_H_
