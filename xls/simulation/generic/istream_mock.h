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

#ifndef XLS_SIMULATION_GENERIC_ISTREAM_MOCK_H_
#define XLS_SIMULATION_GENERIC_ISTREAM_MOCK_H_

#include <array>
#include <cstdint>
#include <queue>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/simulation/generic/iregister_mock.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

// Mock implementation of IStream interface backed by std::deque queue.
// Provides usual IStream methods as well as access to the queues.
template <uint64_t WidthBits, bool IsReadStream_>
class IStreamMock : public IStream {
 public:
  using Register = IRegisterMock<WidthBits>;
  constexpr static uint64_t WidthBytes = Register::WidthBytes;
  using Bytes = std::array<uint8_t, WidthBytes>;

  absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const override {
    return holding_reg_.GetPayloadData8(offset);
  }

  absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const override {
    return holding_reg_.GetPayloadData16(offset);
  }

  absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const override {
    return holding_reg_.GetPayloadData32(offset);
  }

  absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const override {
    return holding_reg_.GetPayloadData64(offset);
  }

  absl::Status SetPayloadData8(uint64_t offset, uint8_t data) override {
    return holding_reg_.SetPayloadData8(offset, data);
  }

  absl::Status SetPayloadData16(uint64_t offset, uint16_t data) override {
    return holding_reg_.SetPayloadData16(offset, data);
  }

  absl::Status SetPayloadData32(uint64_t offset, uint32_t data) override {
    return holding_reg_.SetPayloadData32(offset, data);
  }

  absl::Status SetPayloadData64(uint64_t offset, uint64_t data) override {
    return holding_reg_.SetPayloadData64(offset, data);
  }

  uint64_t GetChannelWidth() const override { return WidthBits; }

  bool IsReadStream() const override { return IsReadStream_; }

  bool IsReady() const override {
    if (IsReadStream_)
      return !fifo_.empty();
    else
      return true;
  }

  absl::Status Transfer() override {
    if (IsReadStream_) {
      if (fifo_.empty()) return absl::InternalError("FIFO is empty");
      holding_reg_ = IRegisterMock<WidthBits>{fifo_.front()};
      fifo_.pop();
    } else {
      fifo_.push(holding_reg_.bytes());
    }
    return absl::OkStatus();
  }

  Register& holding_reg() { return holding_reg_; }
  const Register& holding_reg() const { return holding_reg_; }

  std::queue<Bytes>& fifo() { return fifo_; }
  const std::queue<Bytes>& fifo() const { return fifo_; }

 private:
  Register holding_reg_;
  std::queue<Bytes> fifo_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ISTREAM_MOCK_H_
