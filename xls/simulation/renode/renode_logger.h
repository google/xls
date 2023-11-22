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

#ifndef XLS_SIMULATION_RENODE_RENODE_LOGGER_H_
#define XLS_SIMULATION_RENODE_RENODE_LOGGER_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xls/common/logging/log_sink_registry.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/renode/renode_protocol.h"

namespace xls::simulation::renode {

namespace generic = xls::simulation::generic;

class RenodeLogger : public LogSink {
 public:
  static absl::Status RegisterRenodeLogger(generic::IConnection&);
  static absl::Status UnRegisterRenodeLogger();
  void Send(const LogEntry& entry) override;

 private:
  static RenodeLogger* logger_;
  explicit RenodeLogger(generic::IConnection&);
  RenodeLogger(const RenodeLogger&) = delete;
  generic::IConnection& connection;
};

}  // namespace xls::simulation::renode

#endif  // XLS_SIMULATION_RENODE_RENODE_LOGGER_H_
