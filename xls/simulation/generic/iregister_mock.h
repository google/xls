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

#ifndef XLS_SIMULATION_GENERIC_IREGISTER_MOCK_H_
#define XLS_SIMULATION_GENERIC_IREGISTER_MOCK_H_

#include <array>
#include <cstdint>
#include <initializer_list>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/simulation/generic/byteops.h"
#include "xls/simulation/generic/iregister.h"

namespace xls::simulation::generic {

// Mock implementation of IRegister interface for tests
// Exposes normal IRegister interface, as well as test-only
// methods to access underlying data bytes as std::vector
template <uint64_t WidthBits>
class IRegisterMock : public IRegister {
 public:
  constexpr static uint64_t WidthBytes = (WidthBits + 7) / 8;
  constexpr static uint64_t LastByte = (WidthBits - 1) / 8;
  constexpr static uint64_t LastByteBitmask =
      (1 << (WidthBits - 8 * LastByte)) - 1;
  using ByteArray = std::array<uint8_t, WidthBytes>;

  IRegisterMock() = default;
  explicit IRegisterMock(ByteArray init_bytes) : bytes_{init_bytes} {
    bytes_[LastByte] &= LastByteBitmask;
  };

  absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const override {
    return byteops::bytes_read_word<uint8_t>(bytes_, offset);
  }

  absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const override {
    return byteops::bytes_read_word<uint16_t>(bytes_, offset);
  }

  absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const override {
    return byteops::bytes_read_word<uint32_t>(bytes_, offset);
  }

  absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const override {
    return byteops::bytes_read_word<uint64_t>(bytes_, offset);
  }

  absl::Status SetPayloadData8(uint64_t offset, uint8_t data) override {
    XLS_RETURN_IF_ERROR(
        byteops::bytes_write_word(absl::MakeSpan(bytes_), offset, data));
    bytes_[LastByte] &= LastByteBitmask;
    return absl::OkStatus();
  }

  absl::Status SetPayloadData16(uint64_t offset, uint16_t data) override {
    XLS_RETURN_IF_ERROR(
        byteops::bytes_write_word(absl::MakeSpan(bytes_), offset, data));
    bytes_[LastByte] &= LastByteBitmask;
    return absl::OkStatus();
  }

  absl::Status SetPayloadData32(uint64_t offset, uint32_t data) override {
    XLS_RETURN_IF_ERROR(
        byteops::bytes_write_word(absl::MakeSpan(bytes_), offset, data));
    bytes_[LastByte] &= LastByteBitmask;
    return absl::OkStatus();
  }

  absl::Status SetPayloadData64(uint64_t offset, uint64_t data) override {
    XLS_RETURN_IF_ERROR(
        byteops::bytes_write_word(absl::MakeSpan(bytes_), offset, data));
    bytes_[LastByte] &= LastByteBitmask;
    return absl::OkStatus();
  }

  uint64_t GetChannelWidth() const override { return WidthBits; }

  const ByteArray& bytes() const { return bytes_; }
  ByteArray& bytes() { return bytes_; }

 private:
  ByteArray bytes_{};
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IREGISTER_MOCK_H_
