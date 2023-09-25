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

#ifndef XLS_SIMULATION_GENERIC_IDMAENDPOINT_H_
#define XLS_SIMULATION_GENERIC_IDMAENDPOINT_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls::simulation::generic {

// IDmaEndpoint represents a peripheral end of a 1D DMA channel.
//
// It allows to transfer a stream of fixed-size elements, and supports framing
// those elements into packets by flagging the last element of the transfer.
// User can query the size of a single element, as well as a maximum number of
// elements that a single transfer can fit. In general, transferring any number
// of elements between 0 and maximum should be supported by the implementation.
//
// Transfers with number of bytes not divisible by the element size, are deemed
// as erroneous. Empty transfers (zero bytes) are supported by the interface,
// however, an implemnentation is allowed to ignore them.
class IDmaEndpoint {
 public:
  struct Payload {
    std::vector<uint8_t> data;
    bool last;
  };

  virtual ~IDmaEndpoint() = default;

  virtual uint64_t GetElementSize() const = 0;
  virtual uint64_t GetMaxElementsPerTransfer() const = 0;
  virtual bool IsReadStream() const = 0;
  virtual bool IsReady() const = 0;
  virtual absl::Status Write(Payload payload) = 0;
  virtual absl::StatusOr<Payload> Read() = 0;
};

};  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IDMAENDPOINT_H_
