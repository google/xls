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

#ifndef XLS_SIMULATION_RENODE_RENODE_PROTOCOL_H_
#define XLS_SIMULATION_RENODE_RENODE_PROTOCOL_H_

#include <cstdint>

namespace renode {

// Must be in sync with Renode's
// Antmicro.Renode.Plugins.VerilatorPlugin.Connection.Protocols.ProtocolMessage
#pragma pack(push, 2)
struct ProtocolMessage {
  ProtocolMessage() = default;
  ProtocolMessage(int actionId, uint64_t addr, uint64_t value)
      : actionId(actionId), addr(addr), value(value) {}

  int actionId;
  uint64_t addr;
  uint64_t value;
};
#pragma pack(pop)

enum Action {
#include "plugins/VerilatorIntegrationLibrary/src/renode_action_enumerators.txt"
};

enum LogLevel {
  LOG_LEVEL_NOISY = -1,
  LOG_LEVEL_DEBUG = 0,
  LOG_LEVEL_INFO = 1,
  LOG_LEVEL_WARNING = 2,
  LOG_LEVEL_ERROR = 3
};

}  // namespace renode

#endif  // XLS_SIMULATION_RENODE_RENODE_PROTOCOL_H_
