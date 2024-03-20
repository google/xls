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

#ifndef XLS_SIMULATION_GENERIC_ICHANNELMANAGER_H_
#define XLS_SIMULATION_GENERIC_ICHANNELMANAGER_H_

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/ichannel.h"

namespace xls::simulation::generic {

class IChannelManager {
 public:
  using channel_addr_t = uint64_t;

  virtual absl::Status WriteU8AtAddress(channel_addr_t, uint8_t) = 0;
  virtual absl::Status WriteU16AtAddress(channel_addr_t, uint16_t) = 0;
  virtual absl::Status WriteU32AtAddress(channel_addr_t, uint32_t) = 0;
  virtual absl::Status WriteU64AtAddress(channel_addr_t, uint64_t) = 0;
  virtual absl::StatusOr<uint8_t> ReadU8AtAddress(channel_addr_t) = 0;
  virtual absl::StatusOr<uint16_t> ReadU16AtAddress(channel_addr_t) = 0;
  virtual absl::StatusOr<uint32_t> ReadU32AtAddress(channel_addr_t) = 0;
  virtual absl::StatusOr<uint64_t> ReadU64AtAddress(channel_addr_t) = 0;

  // range: [base_address, base_address+address_range_end_)
  explicit IChannelManager(uint64_t base_address)
      : base_address_(base_address), address_range_end_(0) {}
  IChannelManager(uint64_t base_address, uint64_t address_range_end)
      : base_address_(base_address), address_range_end_(address_range_end) {}

  uint64_t GetBaseAddress() const { return this->base_address_; }

  bool InRange(uint64_t address) const {
    return this->base_address_ <= address &&
           address < (this->base_address_ + this->address_range_end_);
  }

  virtual ~IChannelManager() = default;

 protected:
  uint64_t base_address_;
  uint64_t address_range_end_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ICHANNELMANAGER_H_
