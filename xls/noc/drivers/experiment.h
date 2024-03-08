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

#ifndef XLS_NOC_DRIVERS_EXPERIMENT_H_
#define XLS_NOC_DRIVERS_EXPERIMENT_H_

#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/flit.h"
#include "xls/noc/simulation/global_routing_table.h"
#include "xls/noc/simulation/traffic_description.h"

// This file contains classes used to construct different
// experiments for simulating multiple related configs of a noc.

namespace xls::noc {

// A base configuration for a NOC.
class ExperimentConfig {
 public:
  NocTrafficManager& GetTrafficConfig() { return traffic_; }
  const NocTrafficManager& GetTrafficConfig() const { return traffic_; }

  NetworkConfigProto& GetNetworkConfig() { return network_; }
  const NetworkConfigProto& GetNetworkConfig() const { return network_; }

  void SetNetworkConfig(NetworkConfigProto network) { network_ = network; }
  void SetTrafficConfig(NocTrafficManager traffic) { traffic_ = traffic; }

 private:
  NocTrafficManager traffic_;
  NetworkConfigProto network_;
};

// A set of steps to describe a sequence of ExperimentConfigs.
//
// Said sequence consists of
//   1. A base config (step 0). The step applies the identity mutation.
//      The identity mutation is a mutation where the config is not modified.
//   2. Mutated configs based off of the base config (step 1..N) by
//      applying a SweepStep functor to the base config.
class ExperimentSweeps {
 public:
  // A step is any functor that will accept a ExperimentConfig
  // (as the base config).  It will then
  // modify said config for the current step.
  using SweepStep = std::function<absl::Status(ExperimentConfig& config)>;

  // Mutate the config according to step index.
  //
  // The zero step applies the identity mutation to the base config
  // so the base_config is returned unmodified.
  absl::Status ApplyMutationStep(int64_t index,
                                 ExperimentConfig& base_config) const {
    XLS_RET_CHECK(steps_.size() >= index || index < 0);
    return (index == 0) ? absl::OkStatus() : steps_[index - 1](base_config);
  }

  // Get the number of steps in the sweep.
  //
  // Note state as step 0 is the base config, there is no mutation functor
  // stored in steps_ so the number of steps is steps_.size() +1.
  int64_t GetStepCount() const { return steps_.size() + 1; }

  // Add a new step.
  void AddNewStep(SweepStep step) { steps_.push_back(step); }

 private:
  std::vector<SweepStep> steps_;
};

// Stores metrics obtained during simulation.
//
// TODO(tedhong): 2021-07-13 make it easier to add and find new metrics by
//                imposing some structure/hierarchy/schema to these.
class ExperimentMetrics {
 public:
  // Sets/overrides the integer metric.
  void SetIntegerMetric(std::string_view metric, int64_t value) {
    integer_metrics_[metric] = value;
  }

  // Retrieve the value of said integer point metric.
  absl::StatusOr<int64_t> GetIntegerMetric(std::string_view metric) const {
    XLS_RET_CHECK(integer_metrics_.contains(metric));
    return integer_metrics_.at(metric);
  }

  // Sets/overrides the floating point metric.
  void SetFloatMetric(std::string_view metric, double value) {
    float_metrics_[metric] = value;
  }

  // Retrieve the value of said floating point metric.
  absl::StatusOr<double> GetFloatMetric(std::string_view metric) const {
    XLS_RET_CHECK(float_metrics_.contains(metric));
    return float_metrics_.at(metric);
  }

  // Sets/overrides the integer metric.
  void SetIntegerIntegerMapMetric(
      std::string_view metric,
      const absl::flat_hash_map<int64_t, int64_t>&& value) {
    integer_integer_map_metrics_[metric] = std::move(value);
  }
  void SetIntegerIntegerMapMetric(
      std::string_view metric,
      const absl::flat_hash_map<int64_t, int64_t>& value) {
    integer_integer_map_metrics_[metric] = value;
  }

  // Retrieve the value of said integer point metric.
  absl::StatusOr<absl::flat_hash_map<int64_t, int64_t>>
  GetIntegerIntegerMapMetric(std::string_view metric) const {
    XLS_RET_CHECK(integer_integer_map_metrics_.contains(metric));
    return integer_integer_map_metrics_.at(metric);
  }

  // Prints out the metrics and values stored.
  absl::Status DebugDump() const;

 private:
  absl::btree_map<std::string, double> float_metrics_;
  absl::btree_map<std::string, absl::flat_hash_map<int64_t, int64_t>>
      integer_integer_map_metrics_;
  absl::btree_map<std::string, int64_t> integer_metrics_;
};

struct SinkVcPair {
  std::string sink;
  std::string vc;

  // Hash.
  template <typename H>
  friend H AbslHashValue(H h, SinkVcPair sink_vc_pair) {
    return H::combine(std::move(h), sink_vc_pair.sink, sink_vc_pair.vc);
  }
};

inline bool operator==(const SinkVcPair& lhs, const SinkVcPair& rhs) {
  return lhs.sink == rhs.sink && lhs.vc == rhs.vc;
}

// The key defines the sink-vc pair and the value contains the packet count
// destined for its respective key (sink-vc pair).
using SinkVcPairPacketCount = absl::flat_hash_map<SinkVcPair, int64_t>;

// Stores information obtained during simulation.
//
// TODO(vmirian): 2021-11-11 Create a hierarchical design to modify (e.g.
// expand) with ease.
class ExperimentInfo {
 public:
  // Append a timed route info.
  void AppendTimedRouteInfo(std::string_view metric,
                            TimedRouteInfo&& timed_route_info) {
    timed_route_info_[metric].emplace_back(std::move(timed_route_info));
  }
  void AppendTimedRouteInfo(std::string_view metric,
                            const TimedRouteInfo& timed_route_info) {
    timed_route_info_[metric].emplace_back(timed_route_info);
  }

  // Retrieve timed route infos.
  absl::StatusOr<std::vector<TimedRouteInfo>> GetTimedRouteInfo(
      std::string_view info) const {
    if (!timed_route_info_.contains(info)) {
      return absl::NotFoundError(absl::StrFormat("%s not found.", info));
    }
    return timed_route_info_.at(info);
  }

  const absl::btree_map<std::string, SinkVcPairPacketCount>&
  GetLinkToPacketCountMap() const {
    return link_to_packet_count_map_;
  }

 private:
  friend class ExperimentRunner;
  absl::btree_map<std::string, std::vector<TimedRouteInfo>> timed_route_info_;
  // The key defines the link name and the value contains
  // ::xls::noc::SinkVcPairPacketCount.
  absl::btree_map<std::string, SinkVcPairPacketCount> link_to_packet_count_map_;
};

struct ExperimentData {
  ExperimentMetrics metrics;
  ExperimentInfo info;
};

// Class to setup and run a single step of the experiment,
// including the setup and initialization of the traffic model.
class ExperimentRunner {
 public:
  ExperimentRunner()
      : total_simulation_cycle_count_(0), cycle_time_in_ps_(0), seed_(0) {}

  absl::StatusOr<ExperimentData> RunExperiment(
      const ExperimentConfig& experiment_config,
      DistributedRoutingTableBuilderBase&& distributed_routing_table_builder =
          DistributedRoutingTableBuilderForTrees()) const;

  ExperimentRunner& SetSimulationCycleCount(int64_t count) {
    CHECK_GE(count, 0);
    total_simulation_cycle_count_ = count;
    return *this;
  }

  ExperimentRunner& SetCycleTimeInPs(int64_t ps) {
    CHECK_GT(ps, 0);
    cycle_time_in_ps_ = ps;
    return *this;
  }

  ExperimentRunner& SetTrafficMode(std::string_view mode_name) {
    mode_name_ = mode_name;
    return *this;
  }

  ExperimentRunner& SetSimulationSeed(int16_t seed) {
    seed_ = seed;
    return *this;
  }

  int64_t GetSimulationCycleCount() const {
    return total_simulation_cycle_count_;
  }
  int64_t GetCycleTimeInPs() const { return cycle_time_in_ps_; }

  int16_t GetSeed() const { return seed_; }
  std::string_view GetTrafficMode() const { return mode_name_; }

 private:
  int64_t total_simulation_cycle_count_;
  int64_t cycle_time_in_ps_;
  int16_t seed_;

  std::string mode_name_;
};

class ExperimentBuilderBase;

// A description of an experiment.
//
// An experiment is a describes how to configure, run, and
// measure different metrics across a set of networks.
class Experiment {
 public:
  // Create the config and run the simulation for the given step.
  //
  // The config for step 0 is the base configuration as setup in the builder,
  // each subsequent step in independent and the config for step N is
  // created by applying the mutation for step N on top of the base
  // configuration as run in step 0.
  absl::StatusOr<ExperimentData> RunStep(
      int64_t step,
      DistributedRoutingTableBuilderBase&& distributed_routing_table_builder =
          DistributedRoutingTableBuilderForTrees()) const {
    XLS_RET_CHECK(step >= 0 && step < GetStepCount());

    XLS_ASSIGN_OR_RETURN(ExperimentConfig config, GetConfigForStep(step));

    // Make a copy of the runner to run the single step.
    ExperimentRunner runner = runner_;
    return runner.RunExperiment(config,
                                std::move(distributed_routing_table_builder));
  }

  // Get the configuration for step N.
  absl::StatusOr<ExperimentConfig> GetConfigForStep(int64_t step) const {
    XLS_RET_CHECK(step >= 0 && step < GetStepCount());

    // Make a copy of the base config and apply the mutation.
    ExperimentConfig config = GetBaseConfig();
    XLS_RET_CHECK_OK(sweeps_.ApplyMutationStep(step, config));
    return config;
  }

  // Get number of steps (including the base step).
  int64_t GetStepCount() const { return sweeps_.GetStepCount(); }

  // Return a reference to the base config.
  const ExperimentConfig& GetBaseConfig() const { return config_; }

  // Returns a reference to the various sweeps.
  const ExperimentSweeps& GetSweeps() const { return sweeps_; }

  // Returns a reference to the experiment runner.
  const ExperimentRunner& GetRunner() const { return runner_; }

 private:
  friend ExperimentBuilderBase;

  ExperimentConfig config_;
  ExperimentSweeps sweeps_;
  ExperimentRunner runner_;
};

// Interface to different builders that can construct an experiment.
//
// A hierarchy of builders is envisioned that allows to default different
// classes for different sources of experiments such as:
//   1. Predefined and built-in experiments for demonstration/testing.
//   2. Fully user configured builders via command line/config file.
//   3. NOC solver based configurations where part of the
//      experiment relies on the NOC solver to configure part of the system.
//
class ExperimentBuilderBase {
 public:
  // Delegate to the implementation to build an experiment.
  //
  // An experiment consists of a base config, a set of sweeps that modify
  // said config, and a configuration of the simulator to run those modified
  // configurations.
  //
  absl::StatusOr<Experiment> BuildExperiment() {
    Experiment experiment;

    XLS_ASSIGN_OR_RETURN(experiment.config_, BuildExperimentConfig());
    XLS_ASSIGN_OR_RETURN(experiment.sweeps_, BuildExperimentSweeps());
    XLS_ASSIGN_OR_RETURN(experiment.runner_, BuildExperimentRunner());

    return experiment;
  }

  virtual ~ExperimentBuilderBase() = default;

 protected:
  virtual absl::StatusOr<ExperimentConfig> BuildExperimentConfig() = 0;
  virtual absl::StatusOr<ExperimentSweeps> BuildExperimentSweeps() = 0;
  virtual absl::StatusOr<ExperimentRunner> BuildExperimentRunner() = 0;
};

namespace internal {

struct PacketInfo {
  int64_t injection_clock_cycle_time;
  int64_t arrival_clock_cycle_time;
};

struct Stats {
  int64_t min_injection_cycle_time = std::numeric_limits<int64_t>::max();
  int64_t max_injection_cycle_time = std::numeric_limits<int64_t>::min();
  int64_t min_arrival_cycle_time = std::numeric_limits<int64_t>::max();
  int64_t max_arrival_cycle_time = std::numeric_limits<int64_t>::min();
  int64_t min_latency = std::numeric_limits<int64_t>::max();
  int64_t max_latency = std::numeric_limits<int64_t>::min();
  double average_latency = 0.0;

  // The key represents the latency, the value represents the packet count.
  absl::flat_hash_map<int64_t, int64_t> latency_histogram;
};

// Get latency from source to sink for each packet.
// flits: a list from flits from the sink.
// vc_index: the vc_index to filter the flits. Only considers the flits
// corresponding to the vc_index. Packets composed of multiple flits retrieves
// the injection time from the HEAD flit and the arrival time from the TAIL
// flit. Packets composed of a single flit retrieves the injection time and the
// arrival time from the TAIL flit.
std::vector<PacketInfo> GetPacketInfo(absl::Span<const TimedDataFlit> flits,
                                      int64_t vc_index);

// Calculate the Stats entries from a list of PacketInfo.
// If list is empty, returns xls::noc::Stats.
Stats GetStats(absl::Span<const PacketInfo> packets);

}  // namespace internal

}  // namespace xls::noc

#endif  // XLS_NOC_DRIVERS_EXPERIMENT_H_
