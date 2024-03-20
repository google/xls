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

#ifndef XLS_SIMULATION_GENERIC_IAXISTREAMLIKE_H_
#define XLS_SIMULATION_GENERIC_IAXISTREAMLIKE_H_

#include <cstdint>
#include <vector>

#include "xls/simulation/generic/istream.h"

namespace xls::simulation::generic {

// IAxiStreamLike - extension to IStream interface that provides methods
// for managing AXI-Stream-like bits of data payload.
class IAxiStreamLike : public IStream {
 public:
  virtual uint64_t GetNumSymbols() const = 0;
  // Get the width of the underlying XLS channel symbol in bits.
  // In the CPU-facing interface (IChannel), implementation must pad symbols
  // with zero MSB bits to align them on 8/16/32..-bit boundaries.
  virtual uint64_t GetSymbolWidth() const = 0;
  // Get number of bytes occupied by a single symbol in the IChannel interface.
  virtual uint64_t GetSymbolSize() const = 0;

  // Number of data valid (TKEEP) flags must match the number of symbols.
  virtual void SetDataValid(std::vector<bool> dataValid) = 0;
  virtual std::vector<bool> GetDataValid() const = 0;

  virtual void SetLast(bool last) = 0;
  virtual bool GetLast() const = 0;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IAXISTREAMLIKE_H_
