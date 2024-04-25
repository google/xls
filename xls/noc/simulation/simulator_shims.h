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

#ifndef XLS_NOC_SIMULATION_SIMULATOR_SHIMS_H_
#define XLS_NOC_SIMULATION_SIMULATOR_SHIMS_H_

#include "absl/status/status.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/flit.h"

// This file contains classes used by the simulator to interface to
// other infrastructure components such the NocTrafficInjector.

namespace xls::noc {

// Shim to integrate additional services to NocSimulator's simulation loop.
// These servces run pre or post-cycle for each cycle simulated.
class NocSimulatorServiceShim {
 public:
  virtual absl::Status RunCycle() = 0;
  virtual ~NocSimulatorServiceShim() = default;
};

// Shim to inject traffic into the simulator.
class NocSimulatorTrafficServiceShim : public NocSimulatorServiceShim {
 public:
  // Register a flit to be sent via source.
  virtual absl::Status SendFlitAtTime(TimedDataFlit flit,
                                      NetworkComponentId source) = 0;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_SIMULATOR_SHIMS_H_
