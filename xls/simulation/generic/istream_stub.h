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

#ifndef XLS_SIMULATION_GENERIC_ISTREAM_STUB_H_
#define XLS_SIMULATION_GENERIC_ISTREAM_STUB_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

class IStreamStub : public IStream {
 public:
  // IStream
  bool IsReadStream() const override;
  bool IsReady() const override;
  absl::Status Transfer() override;
  // IChannel
  absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const override;
  absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const override;
  absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const override;
  absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const override;
  absl::Status SetPayloadData8(uint64_t offset, uint8_t data) override;
  absl::Status SetPayloadData16(uint64_t offset, uint16_t data) override;
  absl::Status SetPayloadData32(uint64_t offset, uint32_t data) override;
  absl::Status SetPayloadData64(uint64_t offset, uint64_t data) override;
  uint64_t GetChannelWidth() const override;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ISTREAM_STUB_H_
