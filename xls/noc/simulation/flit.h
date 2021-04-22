// Copyright 2021 The XLS Authors
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

#ifndef XLS_NOC_SIMULATION_FLIT_H_
#define XLS_NOC_SIMULATION_FLIT_H_

#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

// This file contains classes used to store and build objects related to
// NOC flits.

namespace xls::noc {

// Universal encoding of Flit types.
enum class FlitType { kInvalid = 0x0, kHead = 0x2, kBody = 0x1, kTail = 0x3 };

// Represents a flit being sent from a source to a sink (forward).
struct DataFlit {
  // TODO(tedhong) : 2021-03-07 - Support parameterization of flit depending
  //                              on network requirements.

  // Type of a flit depends on the specifics of the network.  For example,
  // two bits can be used to represent the following whether the
  // flit is valid and/or whether the flit is the tail-end of a packet.
  //
  // Note that for this implementation, a single flit packet is a "tail flit"
  // and indices are global for a network (see NetworkComponentIndexMap).

  FlitType type;
  int16_t source_index;
  int16_t destination_index;
  int16_t vc;  // Virtual channel.
  int16_t data_bit_count;
  int64_t data;
};

// Associates a flit with a time (cycle).
struct TimedDataFlit {
  int64_t cycle;
  DataFlit flit;
};

// Represents a flit being used for metadata (i.e. credits).
struct MetadataFlit {
  // TODO(tedhong) : 2020-01-24 - Convert to use Bits/DSLX structs.
  FlitType type;
  int64_t data;
};

// Associates a metadata flit with a time (cycle).
struct TimedMetadataFlit {
  int64_t cycle;
  MetadataFlit flit;
};

// Fluent builder for a DataFlit.
class DataFlitBuilder {
 public:
  DataFlitBuilder& Invalid() {
    flit_.type = FlitType::kInvalid;
    return *this;
  }

  DataFlitBuilder& Type(FlitType type) {
    flit_.type = type;
    return *this;
  }

  DataFlitBuilder& SourceIndex(int16_t src_index) {
    flit_.source_index = src_index;
    return *this;
  }

  DataFlitBuilder& DestinationIndex(int16_t dest_index) {
    flit_.destination_index = dest_index;
    return *this;
  }

  DataFlitBuilder& VirtualChannel(int16_t vc) {
    flit_.vc = vc;
    return *this;
  }

  DataFlitBuilder& Data(Bits bits) {
    data_ = bits;
    return *this;
  }

  DataFlitBuilder& Cycle(int64_t cycle) {
    cycle_ = cycle;
    return *this;
  }

  // Builds a DataFlit flit.
  absl::StatusOr<DataFlit> BuildFlit() {
    flit_.data_bit_count = data_.bit_count();
    XLS_ASSIGN_OR_RETURN(flit_.data, data_.ToUint64());

    return flit_;
  }

  // Builds a flit with an associated time.
  absl::StatusOr<TimedDataFlit> BuildTimedFlit() {
    TimedDataFlit timed_flit;

    timed_flit.cycle = cycle_;
    XLS_ASSIGN_OR_RETURN(timed_flit.flit, BuildFlit());

    return timed_flit;
  }

 private:
  DataFlit flit_ = {.type = FlitType::kInvalid,
                    .source_index = 0,
                    .destination_index = 0,
                    .vc = 0};

  Bits data_;
  int64_t cycle_ = 0;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_FLIT_H_
