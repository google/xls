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

#ifndef XLS_SIMULATION_GENERIC_ICHANNEL_H_
#define XLS_SIMULATION_GENERIC_ICHANNEL_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/common.h"

namespace xls::simulation::generic {

class IChannel {
 public:
  virtual absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const = 0;
  virtual absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const = 0;
  virtual absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const = 0;
  virtual absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const = 0;
  virtual absl::Status SetPayloadData8(uint64_t offset, uint8_t data) = 0;
  virtual absl::Status SetPayloadData16(uint64_t offset, uint16_t data) = 0;
  virtual absl::Status SetPayloadData32(uint64_t offset, uint32_t data) = 0;
  virtual absl::Status SetPayloadData64(uint64_t offset, uint64_t data) = 0;

  virtual uint64_t GetChannelWidth() const = 0;

  virtual ~IChannel() = default;

  uint64_t GetChannelWidthInBytes() { return BitsToBytes(GetChannelWidth()); }
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ICHANNEL_H_
