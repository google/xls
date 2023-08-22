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

#ifndef XLS_SIMULATION_GENERIC_SINGLEVALUEMANAGER_H_
#define XLS_SIMULATION_GENERIC_SINGLEVALUEMANAGER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/ichannel.h"
#include "xls/simulation/generic/ichannelmanager.h"
#include "xls/simulation/generic/iregister.h"

namespace xls::simulation::generic {

class SingleValueManager : public IChannelManager {
 public:
  explicit SingleValueManager(uint64_t base_address);
  SingleValueManager(SingleValueManager&& old) = default;
  SingleValueManager& operator=(SingleValueManager&& old) = default;

  absl::Status RegisterIRegister(std::unique_ptr<IRegister> reg,
                                 uint64_t offset);
  absl::Status WriteU8AtAddress(channel_addr_t address, uint8_t value) override;
  absl::Status WriteU16AtAddress(channel_addr_t address,
                                 uint16_t value) override;
  absl::Status WriteU32AtAddress(channel_addr_t address,
                                 uint32_t value) override;
  absl::Status WriteU64AtAddress(channel_addr_t address,
                                 uint64_t value) override;
  absl::StatusOr<uint8_t> ReadU8AtAddress(channel_addr_t address) override;
  absl::StatusOr<uint16_t> ReadU16AtAddress(channel_addr_t address) override;
  absl::StatusOr<uint32_t> ReadU32AtAddress(channel_addr_t address) override;
  absl::StatusOr<uint64_t> ReadU64AtAddress(channel_addr_t address) override;

  virtual ~SingleValueManager() = default;

 private:
  friend class SingleValueManagerTest;
  // 64-bit alignment
  static const uint64_t kChannelAlignment = 8;

  SingleValueManager();

  absl::Status WriteInternal(channel_addr_t address, uint64_t data,
                             uint64_t log_byte_count);
  absl::StatusOr<uint64_t> ReadInternal(channel_addr_t address,
                                        uint64_t log_byte_count);

  bool IsOffsetFree(uint64_t offset, uint64_t channel_width);
  absl::StatusOr<IRegister*> GetMappingOrStatus(uint64_t address,
                                                std::string&& err_msg);
  uint64_t GetMaxOffset() const { return this->address_range_end_; }

  std::map<uint64_t, std::unique_ptr<IRegister>> offset_to_channel_;
  std::unordered_map<IRegister*, uint64_t> channel_to_offset_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_SINGLEVALUEMANAGER_H_
