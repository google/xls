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

#ifndef XLS_SIMULATION_GENERIC_STREAMMANAGER_H_
#define XLS_SIMULATION_GENERIC_STREAMMANAGER_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/bits_util.h"
#include "xls/simulation/generic/ichannel.h"
#include "xls/simulation/generic/ichannelmanager.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

class StreamManager : public IChannelManager {
 public:
  // A stream along with its beginning address (relative to StreamManager's base
  // address)
  using StreamRange = std::pair<channel_addr_t, std::unique_ptr<IStream>>;
  // Registers a stream when using StreamManager::Build
  using StreamRegistrator =
      std::function<void(channel_addr_t offset, IStream* stream)>;
  using StreamCreator = std::function<absl::Status(const StreamRegistrator&)>;

  // All registered streams begin with a control register that has the following
  // format:
  //
  //     BIT:       ┌63──────4┬──────3┬──2┬─────1┬────0┐
  //                ┌─┬─┬─┬─┬─┬───────┬───┬──────┬─────┐
  //     CONTENT:   │(unused) │ERRXFER│DIR│DOXFER│READY│
  //                └─┴─┴─┴─┴─┴───────┴───┴──────┴─────┘
  //     ACCESS:    └────R────┴──R/W──┴─R─┴───W──┴──R──┘
  //
  // The address where the stream get registered will contain that register,
  // followed by the stream's payload data register.

  // Ctrl register offset relative to the registration offset
  static const uint64_t ctrl_reg_offset_bytes_ = 0x0;
  // Ctrl register width
  static const uint64_t ctrl_reg_width_bytes_ = 8;
  // Ctrl register bits
  static const uint64_t ctrl_reg_READY_ = 0x01;
  static const uint64_t ctrl_reg_DOXFER_ = 0x02;
  static const uint64_t ctrl_reg_DIR_ = 0x04;
  static const uint64_t ctrl_reg_ERRXFER_ = 0x08;

  StreamManager(StreamManager&& other) = default;
  StreamManager& operator=(StreamManager&& other) = default;
  ~StreamManager() override = default;

  // Use a stream-registering function to build StreamManager.
  // The function takes one argument which is a callback for registering a
  // stream.
  static absl::StatusOr<StreamManager> Build(uint64_t base_address,
                                             const StreamCreator& BuildStreams);

  absl::Status LastTransferStatus() { return this->tsfr_status_; }

  absl::Status WriteU8AtAddress(channel_addr_t addr, uint8_t value) override;
  absl::Status WriteU16AtAddress(channel_addr_t addr, uint16_t value) override;
  absl::Status WriteU32AtAddress(channel_addr_t addr, uint32_t value) override;
  absl::Status WriteU64AtAddress(channel_addr_t addr, uint64_t value) override;
  absl::StatusOr<uint8_t> ReadU8AtAddress(channel_addr_t addr) override;
  absl::StatusOr<uint16_t> ReadU16AtAddress(channel_addr_t addr) override;
  absl::StatusOr<uint32_t> ReadU32AtAddress(channel_addr_t addr) override;
  absl::StatusOr<uint64_t> ReadU64AtAddress(channel_addr_t addr) override;

  class AccessToStream {
   public:
    IStream* stream_;
    channel_addr_t offset_bytes_;
    uint8_t shift_;

    int64_t OffsetRelativeToPayload() const {
      return this->offset_bytes_ - StreamManager::ctrl_reg_width_bytes_;
    }

   protected:
    template <typename StatusT>
    absl::StatusOr<uint64_t> ShiftResult(StatusT s) const {
      if (s.ok()) {
        return *s << this->shift_;
      }
      return s;
    }
  };
  template <typename AccessType>
  class TypedAccessToStream : public AccessToStream {
   public:
    absl::StatusOr<uint64_t> GetData() const {
      return absl::InternalError("Unsupported access type");
    }
    absl::Status SetData(uint64_t data) const {
      return absl::InternalError("Unsupported access type");
    }
  };

 private:
  static constexpr uint64_t ctrl_reg_end_ =
      StreamManager::ctrl_reg_offset_bytes_ +
      StreamManager::ctrl_reg_width_bytes_;

  StreamManager(uint64_t base_address,
                std::vector<StreamRange>&& sorted_channels);

  static constexpr unsigned int floorlog2(unsigned int x) {
    return x == 1 ? 0 : 1 + floorlog2(x >> 1);
  }

  template <typename AccessType, typename F>
  absl::Status forEachStreamInRange(channel_addr_t offset_bytes, F action);

  template <typename AccessType>
  absl::StatusOr<AccessType> ReadUAtAddress(channel_addr_t addr);

  template <typename T>
  absl::Status WriteUAtAddress(channel_addr_t addr, T value);

  absl::StatusOr<std::pair<channel_addr_t, IStream*>> GetOffsetChannelPair(
      channel_addr_t offset_bytes);

  static channel_addr_t StreamEnd(channel_addr_t offset_bytes,
                                  const IStream& stream);

  template <typename AccessType>
  absl::Status WriteTail(TypedAccessToStream<AccessType>,
                         AccessType value) const;

  template <typename AccessType>
  absl::StatusOr<uint64_t> ReadTail(
      const TypedAccessToStream<AccessType>& access) const;
  void WriteCtrl(const AccessToStream& access, uint64_t value);
  uint64_t ReadCtrl(const AccessToStream& access);

  template <typename AccessType>
  absl::StatusOr<AccessType> ReadAccess(TypedAccessToStream<AccessType> access);

  template <typename AccessType>
  absl::Status WriteAccess(TypedAccessToStream<AccessType> access,
                           AccessType value);

  template <typename AccessType>
  static bool AccessExceedsCtrlArea(
      const TypedAccessToStream<AccessType>& access);

  std::vector<StreamRange> channels_;  // Streams sorted by their
                                       //  beginning addresses
  absl::Status tsfr_status_;           // Status of the last transfer
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_STREAMMANAGER_H_
