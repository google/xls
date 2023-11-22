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

#include "xls/simulation/renode/renode_logger.h"

#include "xls/common/logging/logging.h"

namespace xls::simulation::renode {

namespace generic = xls::simulation::generic;

RenodeLogger* RenodeLogger::logger_ = nullptr;

/* static */ absl::Status RenodeLogger::RegisterRenodeLogger(
    generic::IConnection& communicationChannel) {
  if (RenodeLogger::logger_ == nullptr) {
    RenodeLogger::logger_ = new RenodeLogger(communicationChannel);
    ::xls::AddLogSink(RenodeLogger::logger_);
    return absl::OkStatus();
  }
  return absl::AlreadyExistsError("RenodeLogger already registered");
}

/* static */ absl::Status RenodeLogger::UnRegisterRenodeLogger() {
  if (RenodeLogger::logger_ != nullptr) {
    ::xls::RemoveLogSink(RenodeLogger::logger_);
    delete RenodeLogger::logger_;
    RenodeLogger::logger_ = nullptr;
    return absl::OkStatus();
  }
  return absl::AlreadyExistsError("RenodeLogger not registered");
}

RenodeLogger::RenodeLogger(generic::IConnection& connection_)
    : connection(connection_) {}

void RenodeLogger::Send(const LogEntry& entry) {
  XLS_CHECK_OK(connection.Log(entry.log_severity(), entry.ToString()));
}

}  // namespace xls::simulation::renode
