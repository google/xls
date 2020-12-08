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

#include "xls/noc/simulation/common.h"

#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace noc {

absl::StatusOr<NetworkId> NetworkId::ValidateAndReturnId(int64 id) {
  if (id > MaxIndex()) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Cannot pack %d to a NetworkId, expected <= %d", id, MaxIndex()));
  }

  return NetworkId(static_cast<uint16>(id));
}

absl::StatusOr<NetworkComponentId> NetworkComponentId::ValidateAndReturnId(
    int64 network, int64 id) {
  XLS_ASSIGN_OR_RETURN(NetworkId network_id,
                       NetworkId::ValidateAndReturnId(network));

  if (id > MaxIndex()) {
    return absl::OutOfRangeError(
        absl::StrFormat("Cannot pack %d to a NetworkComonentId, expected <= %d",
                        id, MaxIndex()));
  }

  return NetworkComponentId(network_id.id(), static_cast<uint32>(id));
}

absl::StatusOr<ConnectionId> ConnectionId::ValidateAndReturnId(int64 network,
                                                               int64 id) {
  XLS_ASSIGN_OR_RETURN(NetworkId network_id,
                       NetworkId::ValidateAndReturnId(network));

  if (id > MaxIndex()) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Cannot pack %d to a ConnectionId, expected <= %d", id, MaxIndex()));
  }

  return ConnectionId(network_id.id(), static_cast<uint32>(id));
}

absl::StatusOr<PortId> PortId::ValidateAndReturnId(int64 network,
                                                   int64 component, int64 id) {
  XLS_ASSIGN_OR_RETURN(
      NetworkComponentId component_id,
      NetworkComponentId::ValidateAndReturnId(network, component));

  if (id > MaxIndex()) {
    return absl::OutOfRangeError(absl::StrFormat(
        "Cannot pack %d to a PortId, expected <= %d", id, MaxIndex()));
  }

  return PortId(component_id.GetNetworkId().id(), component_id.id(),
                static_cast<uint16>(id));
}

}  // namespace noc
}  // namespace xls
