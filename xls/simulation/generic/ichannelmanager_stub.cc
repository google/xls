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

#include "xls/simulation/generic/ichannelmanager_stub.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/simulation/generic/ichannel.h"

namespace xls::simulation::generic {

IChannelManagerStub::IChannelManagerStub(uint64_t base_address,
                                         uint64_t address_range_end)
    : IChannelManager(base_address, address_range_end) {}

IChannelManagerStub::~IChannelManagerStub() = default;

absl::Status IChannelManagerStub::WriteU8AtAddress(uint64_t address,
                                                   uint8_t value) {
  XLS_LOG(INFO) << "IChannelManagerStub::WriteU8AtAddress()";

  return absl::OkStatus();
}

absl::Status IChannelManagerStub::WriteU16AtAddress(uint64_t address,
                                                    uint16_t value) {
  XLS_LOG(INFO) << "IChannelManagerStub::WriteU16AtAddress()";

  return absl::OkStatus();
}

absl::Status IChannelManagerStub::WriteU32AtAddress(uint64_t address,
                                                    uint32_t value) {
  XLS_LOG(INFO) << "IChannelManagerStub::WriteU32AtAddress()";

  return absl::OkStatus();
}

absl::Status IChannelManagerStub::WriteU64AtAddress(uint64_t address,
                                                    uint64_t value) {
  XLS_LOG(INFO) << "IChannelManagerStub::WriteU64AtAddress()";

  return absl::OkStatus();
}

absl::StatusOr<uint8_t> IChannelManagerStub::ReadU8AtAddress(uint64_t address) {
  XLS_LOG(INFO) << "IChannelManagerStub::ReadU8AtAddress()";

  return 0xDE;
}

absl::StatusOr<uint16_t> IChannelManagerStub::ReadU16AtAddress(
    uint64_t address) {
  XLS_LOG(INFO) << "IChannelManagerStub::ReadU16AtAddress()";

  return 0xDEAD;
}

absl::StatusOr<uint32_t> IChannelManagerStub::ReadU32AtAddress(
    uint64_t address) {
  XLS_LOG(INFO) << "IChannelManagerStub::ReadU32AtAddress()";

  return 0xDEADBEEF;
}

absl::StatusOr<uint64_t> IChannelManagerStub::ReadU64AtAddress(
    uint64_t address) {
  XLS_LOG(INFO) << "IChannelManagerStub::ReadU64AtAddress()";

  return 0xDEADBEEFFEEBDAED;
}

}  // namespace xls::simulation::generic
