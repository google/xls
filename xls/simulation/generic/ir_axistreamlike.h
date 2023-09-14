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

#ifndef XLS_SIMULATION_GENERIC_IR_AXISTREAMLIKE_H_
#define XLS_SIMULATION_GENERIC_IR_AXISTREAMLIKE_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/value.h"
#include "xls/simulation/generic/iaxistreamlike.h"

namespace xls::simulation::generic {

class IrAxiStreamLike : public IAxiStreamLike {
 public:
  // TODO(rdobrodii): 2023-09-13 Add field to configuration file that will
  // set write FIFO limit. For now use 4 as limit.
  static const int kWriteFifoMaxDepth = 4;

  // - multisymbol controls how the data payload is treated:
  //   - if it is 'true', data element should be an array of size N. In this
  //   case AXI Stream-like protocol assumes that
  // data contains N symbols which can be individually set to be present or
  // absent with the help of TKEEP signal.
  //   - if it is 'false', data element can be of any time. In this case AXI
  //   Stream-like protocol assumes that the data
  // contains a single symbol (N=1), and if TKEEP is present, its length will
  // be 1.
  // - data_value_index specifies the index of xls::Value element used to convey
  // data.
  // - tlast_value_index should specify a bits[1] tuple member that is used to
  // convey TLAST AXI Stream value
  // - tkeep_value_index should specify a bits[N] tuple member that is used to
  // convey TKEEP AXI Stream value
  static absl::StatusOr<IrAxiStreamLike> Make(
      xls::ChannelQueue* queue, bool multisymbol, uint64_t data_value_index,
      std::optional<uint64_t> tlast_value_index,
      std::optional<uint64_t> tkeep_value_index);

  absl::StatusOr<uint8_t> GetPayloadData8(uint64_t offset) const override;
  absl::StatusOr<uint16_t> GetPayloadData16(uint64_t offset) const override;
  absl::StatusOr<uint32_t> GetPayloadData32(uint64_t offset) const override;
  absl::StatusOr<uint64_t> GetPayloadData64(uint64_t offset) const override;

  absl::Status SetPayloadData8(uint64_t offset, uint8_t data) override;
  absl::Status SetPayloadData16(uint64_t offset, uint16_t data) override;
  absl::Status SetPayloadData32(uint64_t offset, uint32_t data) override;
  absl::Status SetPayloadData64(uint64_t offset, uint64_t data) override;

  uint64_t GetChannelWidth() const override {
    return num_symbols_ * symbol_bytes_padded_ * 8;
  }
  uint64_t GetNumSymbols() const override { return num_symbols_; }
  uint64_t GetSymbolWidth() const override { return symbol_bits_; }
  uint64_t GetSymbolSize() const override { return symbol_bytes_padded_; }

  bool IsReadStream() const override { return read_channel_; }
  bool IsReady() const override;
  absl::Status Transfer() override;

  void SetDataValid(std::vector<bool> dataValid) override;
  std::vector<bool> GetDataValid() const override;

  void SetLast(bool last) override;
  bool GetLast() const override;

 private:
  explicit IrAxiStreamLike(xls::ChannelQueue* queue, bool multisymbol,
                           uint64_t data_index,
                           std::optional<uint64_t> tlast_index,
                           std::optional<uint64_t> tkeep_index);

  Value PackChannelPayload();
  void UnpackChannelPayload(const Value& value);
  absl::StatusOr<uint64_t> DataRead(uint64_t offset, uint64_t byte_count) const;
  absl::Status DataWrite(uint64_t offset, uint64_t data, uint64_t byte_count);

  xls::ChannelQueue* queue_{};
  bool multisymbol_;
  uint64_t data_index_;
  std::optional<uint64_t> tlast_index_;
  std::optional<uint64_t> tkeep_index_;
  bool read_channel_;
  uint64_t num_symbols_;
  uint64_t symbol_bits_;
  uint64_t symbol_bytes_padded_;
  xls::Value zero_payload_;
  xls::Value zero_padding_;
  // data_reg_ this has length of 2N: even elements are data, odd are padding
  std::vector<xls::Value> data_reg_;
  bool tlast_reg_;
  absl::InlinedVector<bool, 1> tkeep_reg_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IR_AXISTREAMLIKE_H_
