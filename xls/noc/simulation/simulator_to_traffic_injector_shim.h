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

#ifndef XLS_NOC_SIMULATION_SIMULATOR_TO_TRAFFIC_INJECTOR_SHIM_H_
#define XLS_NOC_SIMULATION_SIMULATOR_TO_TRAFFIC_INJECTOR_SHIM_H_

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/simulation/common.h"
#include "xls/noc/simulation/flit.h"
#include "xls/noc/simulation/noc_traffic_injector.h"
#include "xls/noc/simulation/sim_objects.h"
#include "xls/noc/simulation/simulator_shims.h"

namespace xls::noc {

// Shim to inject traffic into the simulator.
class NocSimulatorToNocTrafficInjectorShim
    : public NocSimulatorTrafficServiceShim {
 public:
  NocSimulatorToNocTrafficInjectorShim(NocSimulator& simulator,
                                       NocTrafficInjector& traffic_injector)
      : simulator_(&simulator), traffic_injector_(&traffic_injector) {}

  // Called by the simulator each cycle to request for traffic.
  absl::Status RunCycle() override { return traffic_injector_->RunCycle(); }

  // Called by the traffic injector to inject traffic.
  absl::Status SendFlitAtTime(TimedDataFlit flit,
                              NetworkComponentId source) override {
    XLS_ASSIGN_OR_RETURN(SimNetworkInterfaceSrc * src,
                         simulator_->GetSimNetworkInterfaceSrc(source));
    return src->SendFlitAtTime(flit);
  }

 private:
  NocSimulator* simulator_;
  NocTrafficInjector* traffic_injector_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_SIMULATION_SIMULATOR_TO_TRAFFIC_INJECTOR_SHIM_H_
