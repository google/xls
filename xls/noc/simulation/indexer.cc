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

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"

namespace xls {
namespace noc {
namespace {

// Constructs an ordered_index of the same size as unordered_index, where
// ordered_index->at(i) = T iff (T, i) is within unordered_index.
//
// unordered_index should be a collection with an iterator
// that that points to a std::pair<T, int64_t>.
template <typename T, typename IndexTPairCollection>
absl::StatusOr<std::vector<T>> ConstructOrderedIndex(
    std::string_view id_name, const IndexTPairCollection& unordered_index) {
  int64_t count = unordered_index.size();
  if (count == 0) {
    return std::vector<T>();
  }

  std::vector<T> ordered_index(count, unordered_index.begin()->first);

  absl::flat_hash_set<int64_t> present_indices;

  for (auto [id, index] : unordered_index) {
    if (index < 0 || index >= count) {
      return absl::InternalError(
          absl::StrFormat("Unable to add index, %s has index "
                          "%d, should be in [0,%d)",
                          id_name, index, count));
    }

    present_indices.insert(index);
    ordered_index.at(index) = id;
  }

  if (present_indices.size() != count) {
    return absl::InternalError("Unable to add index, duplicate indices used");
  }

  return ordered_index;
}

}  // namespace

absl::Status PortIndexMapBuilder::SetPortIndex(PortId port_id,
                                               PortDirection dir,
                                               int64_t index) {
  if (!nc_to_ports_.contains(port_id.GetNetworkComponentId())) {
    // Add a new nc to the indexer.
    // Initialze VirtualChannelOrder with all present VCs and
    // their corresponding index to -1.
    nc_to_ports_.insert({port_id.GetNetworkComponentId(), PortOrder()});
  }

  if (index < 0) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Index %d for port %d should non-negative", index, port_id.AsUInt64()));
  }

  PortOrder& param_vc_index = nc_to_ports_[port_id.GetNetworkComponentId()];

  if (dir == PortDirection::kInput) {
    param_vc_index.input_port_index.push_back({port_id, index});
  } else {
    param_vc_index.output_port_index.push_back({port_id, index});
  }

  return absl::OkStatus();
}

absl::StatusOr<int64_t> PortIndexMap::GetPortIndex(PortId port_id,
                                                   PortDirection dir) const {
  NetworkComponentId nc_id = port_id.GetNetworkComponentId();
  if (!nc_to_ports_.contains(nc_id)) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Network Component %d has not been indexed", nc_id.AsUInt64()));
  }

  const std::vector<PortId>& port_index =
      dir == PortDirection::kInput
          ? nc_to_ports_.at(nc_id).ordered_input_ports
          : nc_to_ports_.at(nc_id).ordered_output_ports;

  for (int64_t i = 0; i < port_index.size(); ++i) {
    if (port_index[i] == port_id) {
      return i;
    }
  }

  return absl::OutOfRangeError(
      absl::StrFormat("Unable to find index for port %x", port_id.AsUInt64()));
}

absl::StatusOr<int64_t> PortIndexMapBuilder::GetPortIndex(
    PortId port_id, PortDirection dir) const {
  NetworkComponentId nc_id = port_id.GetNetworkComponentId();
  if (!nc_to_ports_.contains(nc_id)) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Network Component %d has not been indexed", nc_id.AsUInt64()));
  }

  const std::vector<std::pair<PortId, int64_t>>& port_index =
      dir == PortDirection::kInput ? nc_to_ports_.at(nc_id).input_port_index
                                   : nc_to_ports_.at(nc_id).output_port_index;

  for (auto [id, index] : port_index) {
    if (id == port_id) {
      return index;
    }
  }

  return absl::OutOfRangeError(
      absl::StrFormat("Unable to find index for port %x", port_id.AsUInt64()));
}

absl::StatusOr<PortIndexMap> PortIndexMapBuilder::BuildPortIndex() {
  PortIndexMap ret;

  for (auto& [nc_id, nc_index] : nc_to_ports_) {
    XLS_RET_CHECK_OK(ret.Add(nc_id, PortDirection::kInput,
                             absl::Span<const std::pair<PortId, int64_t>>(
                                 nc_index.input_port_index)));

    XLS_RET_CHECK_OK(ret.Add(nc_id, PortDirection::kOutput,
                             absl::Span<const std::pair<PortId, int64_t>>(
                                 nc_index.output_port_index)));
  }

  return ret;
}

absl::Status PortIndexMap::Add(
    NetworkComponentId nc_id, PortDirection dir,
    absl::Span<const std::pair<PortId, int64_t>> port_index) {
  std::vector<PortId>* ordered_ports =
      dir == PortDirection::kInput ? &nc_to_ports_[nc_id].ordered_input_ports
                                   : &nc_to_ports_[nc_id].ordered_output_ports;

  XLS_ASSIGN_OR_RETURN(*ordered_ports,
                       ConstructOrderedIndex<PortId>("port", port_index));
  return absl::OkStatus();
}

absl::StatusOr<int64_t> PortIndexMap::InputPortCount(
    NetworkComponentId nc_id) const {
  if (!nc_to_ports_.contains(nc_id)) {
    return absl::OutOfRangeError(absl::StrFormat(
        "PortIndexMap did not index network component %x", nc_id.AsUInt64()));
  }

  return nc_to_ports_.at(nc_id).ordered_input_ports.size();
}

absl::StatusOr<int64_t> PortIndexMap::OutputPortCount(
    NetworkComponentId nc_id) const {
  if (!nc_to_ports_.contains(nc_id)) {
    return absl::OutOfRangeError(absl::StrFormat(
        "PortIndexMap did not index network component %x", nc_id.AsUInt64()));
  }

  return nc_to_ports_.at(nc_id).ordered_output_ports.size();
}

absl::StatusOr<PortId> PortIndexMap::GetPortByIndex(NetworkComponentId nc_id,
                                                    PortDirection dir,
                                                    int64_t port_index) const {
  if (!nc_to_ports_.contains(nc_id)) {
    return absl::OutOfRangeError(absl::StrFormat(
        "PortIndexMap did not index network component %x", nc_id.AsUInt64()));
  }

  const std::vector<PortId>& ordered_ports =
      dir == PortDirection::kInput
          ? nc_to_ports_.at(nc_id).ordered_input_ports
          : nc_to_ports_.at(nc_id).ordered_output_ports;

  if (port_index >= ordered_ports.size() || port_index < 0) {
    return absl::OutOfRangeError(
        absl::StrFormat("Index %d out of range for component %d with size %d",
                        port_index, nc_id.AsUInt64(), ordered_ports.size()));
  }

  return ordered_ports[port_index];
}

absl::StatusOr<int64_t> VirtualChannelIndexMapBuilder::VirtualChannelCount(
    PortId port_id) const {
  if (!port_to_vcs_.contains(port_id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Port %x has not been indexed", port_id.AsUInt64()));
  }

  return port_to_vcs_.at(port_id).param_vc_index.size();
}

absl::StatusOr<int64_t> VirtualChannelIndexMap::VirtualChannelCount(
    PortId port_id) const {
  if (!port_to_vcs_.contains(port_id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Port %x has not been indexed", port_id.AsUInt64()));
  }

  return port_to_vcs_.at(port_id).size();
}

absl::Status VirtualChannelIndexMapBuilder::SetVirtualChannelIndex(
    PortId port_id, PortParam port_param, int64_t orig_index, int64_t index) {
  if (!port_to_vcs_.contains(port_id)) {
    // Add a new port to the indexer.
    // Initialze VirtualChannelOrder with all present VCs and
    // their corresponding index to -1.
    port_to_vcs_.insert({port_id, VirtualChannelOrder()});
    VirtualChannelOrder& vc_order = port_to_vcs_[port_id];

    std::vector<VirtualChannelParam> vc_params =
        port_param.GetVirtualChannels();
    for (int64_t i = 0; i < vc_params.size(); ++i) {
      vc_order.param_vc_index.push_back({vc_params[i], -1});
    }
  }

  std::vector<std::pair<VirtualChannelParam, int64_t>>& param_vc_index =
      port_to_vcs_[port_id].param_vc_index;
  if (orig_index < 0 || orig_index >= param_vc_index.size()) {
    return absl::OutOfRangeError(absl::StrFormat(
        "VC original index %d out of range for port %d with size %d",
        orig_index, port_id.AsUInt64(), param_vc_index.size()));
  }

  if (index < 0 || index >= param_vc_index.size()) {
    return absl::OutOfRangeError(
        absl::StrFormat("VC index %d out of range for port %d with size %d",
                        index, port_id.AsUInt64(), param_vc_index.size()));
  }

  param_vc_index[orig_index].second = index;

  return absl::OkStatus();
}

absl::StatusOr<int64_t> VirtualChannelIndexMapBuilder::GetVirtualChannelIndex(
    PortId port_id, VirtualChannelParam vc) const {
  if (!port_to_vcs_.contains(port_id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Port %d has not been indexed", port_id.AsUInt64()));
  }

  const std::vector<std::pair<VirtualChannelParam, int64_t>>&
      virtual_channel_index = port_to_vcs_.at(port_id).param_vc_index;
  for (auto& [vc_param, index] : virtual_channel_index) {
    if (vc_param.GetName() == vc.GetName()) {
      return index;
    }
  }

  return absl::OutOfRangeError(absl::StrFormat(
      "Unable to find virtual channel %s associated with port %d", vc.GetName(),
      port_id.AsUInt64()));
}

absl::StatusOr<int64_t> VirtualChannelIndexMap::GetVirtualChannelIndex(
    PortId port_id, VirtualChannelParam vc) const {
  return GetVirtualChannelIndexByName(port_id, vc.GetName());
}

absl::StatusOr<int64_t> VirtualChannelIndexMap::GetVirtualChannelIndexByName(
    PortId port_id, std::string_view vc_name) const {
  if (!port_to_vcs_.contains(port_id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Port %d has not been indexed", port_id.AsUInt64()));
  }

  const std::vector<VirtualChannelParam>& virtual_channel_index =
      port_to_vcs_.at(port_id);
  for (int64_t i = 0; i < virtual_channel_index.size(); ++i) {
    if (virtual_channel_index[i].GetName() == vc_name) {
      return i;
    }
  }

  return absl::OutOfRangeError(absl::StrFormat(
      "Unable to find virtual channel %s associated with port %d", vc_name,
      port_id.AsUInt64()));
}

absl::StatusOr<VirtualChannelIndexMap>
VirtualChannelIndexMapBuilder::BuildVirtualChannelIndex() const {
  VirtualChannelIndexMap ret;

  for (auto& vc_order_iter : port_to_vcs_) {
    PortId port_id = vc_order_iter.first;
    const VirtualChannelOrder& vc_order = vc_order_iter.second;

    XLS_RETURN_IF_ERROR(ret.Add(
        port_id, absl::Span<const std::pair<VirtualChannelParam, int64_t>>(
                     vc_order.param_vc_index)));
  }

  return ret;
}

absl::Status VirtualChannelIndexMap::Add(
    PortId port_id,
    absl::Span<const std::pair<VirtualChannelParam, int64_t>> vc_index) {
  std::vector<VirtualChannelParam>& ordered_vc = port_to_vcs_[port_id];

  XLS_ASSIGN_OR_RETURN(ordered_vc, ConstructOrderedIndex<VirtualChannelParam>(
                                       "virtual channel", vc_index));

  return absl::OkStatus();
}

absl::StatusOr<VirtualChannelParam>
VirtualChannelIndexMap::GetVirtualChannelByIndex(PortId port_id,
                                                 int64_t vc_index) const {
  if (!port_to_vcs_.contains(port_id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Port %d has not been indexed", port_id.AsUInt64()));
  }

  const std::vector<VirtualChannelParam>& ordered_virtual_channels =
      port_to_vcs_.at(port_id);

  if (vc_index >= ordered_virtual_channels.size() || vc_index < 0) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Index %d out of range for port %d with size %d", vc_index,
        port_id.AsUInt64(), ordered_virtual_channels.size()));
  }

  return ordered_virtual_channels[vc_index];
}

int64_t NetworkComponentIndexMap::NetworkComponentCount() const {
  return ordered_components_.size();
}

int64_t NetworkComponentIndexMapBuilder::NetworkComponentCount() const {
  return component_index_.size();
}

absl::Status NetworkComponentIndexMapBuilder::SetNetworkComponentIndex(
    NetworkComponentId component_id, int64_t index) {
  if (index < 0) {
    return absl::OutOfRangeError(
        absl::StrFormat("Index %d for component %d should non-negative", index,
                        component_id.AsUInt64()));
  }

  component_index_[component_id] = index;

  return absl::OkStatus();
}

absl::StatusOr<int64_t> NetworkComponentIndexMap::GetNetworkComponentIndex(
    NetworkComponentId id) const {
  if (!component_index_.contains(id)) {
    return absl::OutOfRangeError(
        absl::StrFormat("Component %d has not been indexed.", id.AsUInt64()));
  }

  return component_index_.at(id);
}

absl::StatusOr<int64_t>
NetworkComponentIndexMapBuilder::GetNetworkComponentIndex(
    NetworkComponentId id) const {
  for (auto [nc_id, index] : component_index_) {
    if (nc_id == id) {
      return index;
    }
  }

  return absl::OutOfRangeError(
      absl::StrFormat("Component %d has not been indexed.", id.AsUInt64()));
}

absl::Status NetworkComponentIndexMap::Add(
    const absl::flat_hash_map<NetworkComponentId, int64_t>& nc_index) {
  component_index_ = nc_index;

  XLS_ASSIGN_OR_RETURN(
      ordered_components_,
      ConstructOrderedIndex<NetworkComponentId>("network component", nc_index));

  return absl::OkStatus();
}

absl::StatusOr<NetworkComponentId>
NetworkComponentIndexMap::GetNetworkComponentByIndex(int64_t index) const {
  if (index < 0 || index >= ordered_components_.size()) {
    return absl::OutOfRangeError(
        absl::StrFormat("Index %d out of range, expected [0, %d).", index,
                        ordered_components_.size()));
  }

  return ordered_components_[index];
}

absl::StatusOr<NetworkComponentIndexMap>
NetworkComponentIndexMapBuilder::BuildNetworkComponentIndex() const {
  NetworkComponentIndexMap ret;
  XLS_RETURN_IF_ERROR(ret.Add(component_index_));
  return ret;
}

}  // namespace noc
}  // namespace xls
