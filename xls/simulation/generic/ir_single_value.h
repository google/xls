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

#ifndef XLS_SIMULATION_GENERIC_IR_SINGLE_VALUE_H_
#define XLS_SIMULATION_GENERIC_IR_SINGLE_VALUE_H_

#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/simulation/generic/iregister.h"

namespace xls::simulation::generic {

class IRSingleValue : public IRegister {
 public:
  static absl::StatusOr<IRSingleValue> MakeIRSingleValue(xls::ChannelQueue*);

  IRSingleValue(IRSingleValue&& old)
      : ir_single_value_queue_(old.ir_single_value_queue_) {}

  IRSingleValue& operator=(IRSingleValue&& rvalue_) {
    this->ir_single_value_queue_ = rvalue_.ir_single_value_queue_;
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

  uint64_t GetChannelWidth() const override;

  ~IRSingleValue() = default;

 private:
  IRSingleValue();
  explicit IRSingleValue(xls::ChannelQueue*);

  absl::StatusOr<uint64_t> InternalRead(uint64_t, int) const;
  absl::Status InternalWrite(uint64_t, uint64_t, int);

  xls::ChannelQueue* ir_single_value_queue_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IR_SINGLE_VALUE_H_
