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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_UTILS_H_
#define XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_UTILS_H_

#include "absl/status/status.h"

namespace xls::noc {

class NetworkConnection;

// Returns OK if the connection is valid. Otherwise, returns an error.
// A valid connection has: 1) a source port that is valid with an output
// direction, 2) a sink port that is valid with an input direction, and, 3) the
// source and sink port types are equivalent.
absl::Status ValidateNetworkConnection(const NetworkConnection& connection);

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_UTILS_H_
