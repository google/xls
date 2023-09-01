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

#ifndef XLS_SIMULATION_GENERIC_IR_STREAM_H_
#define XLS_SIMULATION_GENERIC_IR_STREAM_H_

#include <stdint.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/value.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {
class IRStream : public IStream {
 public:
  static absl::StatusOr<IRStream> MakeIRStream(xls::ChannelQueue*);
  IRStream(IRStream&& old)
      : read_channel_(old.read_channel_),
        holding_reg_(old.holding_reg_),
        ir_stream_queue_(old.ir_stream_queue_) {}

  IRStream& operator=(IRStream&& rvalue_) {
    this->read_channel_ = rvalue_.read_channel_;
    this->holding_reg_ = rvalue_.holding_reg_;
    this->ir_stream_queue_ = rvalue_.ir_stream_queue_;
    return *this;
  }

  absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const override;
  absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const override;
  absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const override;
  absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const override;

  absl::Status SetPayloadData8(uint64_t offset, uint8_t data) override;
  absl::Status SetPayloadData16(uint64_t offset, uint16_t data) override;
  absl::Status SetPayloadData32(uint64_t offset, uint32_t data) override;
  absl::Status SetPayloadData64(uint64_t offset, uint64_t data) override;

  virtual ~IRStream() = default;

  uint64_t GetChannelWidth() const override;

  bool IsReadStream() const override;
  bool IsReady() const override;
  absl::Status Transfer() override;

 private:
  // TODO(mtdudek): 2023-09-01 Add field to configuration file that will
  // set write FIFO limit. For now use 4 as limit.
  const int kWriteFifoMaxDepth = 4;

  IRStream();
  explicit IRStream(xls::ChannelQueue*);

  absl::StatusOr<uint64_t> InternalRead(uint64_t, int) const;
  absl::Status InternalWrite(uint64_t, uint64_t, int);

  bool read_channel_;
  xls::Value holding_reg_;
  xls::ChannelQueue* ir_stream_queue_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IR_STREAM_H_
