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

#ifndef XLS_SIMULATION_GENERIC_IMASTERPORT_H_
#define XLS_SIMULATION_GENERIC_IMASTERPORT_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/common.h"

namespace xls::simulation::generic {

class IMasterPort {
 public:
  virtual absl::Status RequestWrite(uint64_t address, uint64_t value,
                                    AccessWidth type) = 0;
  virtual absl::StatusOr<uint64_t> RequestRead(uint64_t address,
                                               AccessWidth type) = 0;

  virtual ~IMasterPort() = default;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_IMASTERPORT_H_
