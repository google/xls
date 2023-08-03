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

#ifndef XLS_SIMULATION_GENERIC_ICHANNELMANAGER_STUB_H_
#define XLS_SIMULATION_GENERIC_ICHANNELMANAGER_STUB_H_

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/ichannel.h"
#include "xls/simulation/generic/ichannelmanager.h"

namespace xls::simulation::generic {

class IChannelManagerStub : public IChannelManager {
 public:
  absl::Status WriteU8AtAddress(uint64_t address, uint8_t value) override;
  absl::Status WriteU16AtAddress(uint64_t address, uint16_t value) override;
  absl::Status WriteU32AtAddress(uint64_t address, uint32_t value) override;
  absl::Status WriteU64AtAddress(uint64_t address, uint64_t value) override;
  absl::StatusOr<uint8_t> ReadU8AtAddress(uint64_t address) override;
  absl::StatusOr<uint16_t> ReadU16AtAddress(uint64_t address) override;
  absl::StatusOr<uint32_t> ReadU32AtAddress(uint64_t address) override;
  absl::StatusOr<uint64_t> ReadU64AtAddress(uint64_t address) override;

  IChannelManagerStub(uint64_t base_address, uint64_t max_offset);
  virtual ~IChannelManagerStub();
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ICHANNELMANAGER_STUB_H_
