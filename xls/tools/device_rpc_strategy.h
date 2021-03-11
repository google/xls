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

#ifndef XLS_TOOLS_DEVICE_RPC_STRATEGY_H_
#define XLS_TOOLS_DEVICE_RPC_STRATEGY_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class DeviceRpcStrategy {
 public:
  virtual ~DeviceRpcStrategy() = default;

  // Opens a connection to the device numbered as "device_ordinal". E.g. if
  // there are multiple IceSticks plugged into a given system, their serial
  // endpoints will be ordered in some fashion by the USB subsystem.
  //
  // TODO(leary): 2019-04-06 Create host-level abstraction for enumerating /
  // describing the properties of accessible devices.
  virtual absl::Status Connect(int64_t device_ordinal) = 0;

  // Calls an unnamed function on the device.
  virtual absl::StatusOr<Value> CallUnnamed(
      const FunctionType& function_type, absl::Span<const Value> arguments) = 0;
};

}  // namespace xls

#endif  // XLS_TOOLS_DEVICE_RPC_STRATEGY_H_
