// Copyright 2023 The XLS Authors
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

#ifndef XLS_SIMULATION_GENERIC_ICONNECTION_H_
#define XLS_SIMULATION_GENERIC_ICONNECTION_H_

#include <string_view>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls::simulation::generic {

// IConnection is used in conjunction with IPeripheral and provides methods for
// XLS-initiated communication with Renode:
//  - SendResponse() is used to send a response for a request received via
//  IPeripheral::HandleRequest().
//  - SendRequest() is used to send peripheral-initiated requests (e.g. IRQ).
//  - ReceiveResponse() is used to receive a response for a request transmitted
//  via SendRequest.
//  - Log() allows to push messages to the Renode's logging subsystem.
class IConnection {
 public:
  virtual absl::Status Log(absl::LogSeverity level, std::string_view msg) = 0;
  virtual ~IConnection() = default;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ICONNECTION_H_
