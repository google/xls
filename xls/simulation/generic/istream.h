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

#ifndef XLS_SIMULATION_GENERIC_ISTREAM_H_
#define XLS_SIMULATION_GENERIC_ISTREAM_H_

#include "absl/status/status.h"
#include "xls/simulation/generic/ichannel.h"

namespace xls::simulation::generic {

// IStream - interface to the FIFO-like XLS streams (see xls::StreamingChannel)
//
// Contains internal holding register which stores a single data item. Holding
// register can be accessed via methods provided by IChannel.
class IStream : public IChannel {
 public:
  // Returns 'true' if the device reads from this stream, 'false' if it writes
  // to it.
  virtual bool IsReadStream() const = 0;

  // Returns 'true' if the stream is ready to perform a single transfer:
  //  - For a read stream to be ready, the underlying stream implementation
  //    must have at least one data item available for reading, and must be
  //    ready to transfer it into the holding register when Transfer() is
  //    called.
  //  - For a write stream to be ready, the underlying stream implementation
  //    must be ready to accept at least a single data item whenever
  //    Transfer() is called.
  virtual bool IsReady() const = 0;

  // Performs a single transfer on the stream:
  //  - For read streams, this extracts a single data item from the stream
  //    and places it into a holding register.
  //  - For write streams, this takes a data item from the holding
  //    register and sends it to the stream.
  virtual absl::Status Transfer() = 0;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_ISTREAM_H_
