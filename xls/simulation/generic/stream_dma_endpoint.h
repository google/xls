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

#ifndef XLS_SIMULATION_GENERIC_STREAM_DMA_ENDPOINT_H_
#define XLS_SIMULATION_GENERIC_STREAM_DMA_ENDPOINT_H_

#include <memory>
#include <utility>

#include "xls/simulation/generic/idmaendpoint.h"
#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

class StreamDmaEndpoint : public IDmaEndpoint {
 public:
  explicit StreamDmaEndpoint(std::unique_ptr<IStream> stream)
      : stream_(std::move(stream)) {
    transfer_size_ = (stream_->GetChannelWidth() + 7) / 8;
  }

  uint64_t GetElementSize() const override { return transfer_size_; }
  uint64_t GetMaxElementsPerTransfer() const override { return 1; }
  bool IsReadStream() const override { return stream_->IsReadStream(); }
  bool IsReady() const override { return stream_->IsReady(); }
  absl::Status Write(Payload payload) override;
  absl::StatusOr<Payload> Read() override;

  IStream* GetStream() const { return stream_.get(); }

 private:
  std::unique_ptr<IStream> stream_;
  uint64_t transfer_size_;
};

};  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_STREAM_DMA_ENDPOINT_H_
