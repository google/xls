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

#ifndef XLS_SIMULATION_GENERIC_XLSPERIPHERAL_H_
#define XLS_SIMULATION_GENERIC_XLSPERIPHERAL_H_

#include <memory>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/simulation/generic/config.h"
#include "xls/simulation/generic/ichannelmanager.h"
#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/iperipheral.h"
#include "xls/simulation/generic/runtime_manager.h"
#include "xls/simulation/generic/runtime_status.h"
#include "xls/simulation/generic/singlevaluemanager.h"

namespace xls::simulation::generic {

class XlsPeripheral : public IPeripheral {
 public:
  XlsPeripheral() = delete;
  XlsPeripheral(const XlsPeripheral&) = delete;
  XlsPeripheral& operator=(const XlsPeripheral&) = delete;
  XlsPeripheral(XlsPeripheral&&) = default;

  static absl::StatusOr<XlsPeripheral> Make(IConnection& connection,
                                            std::string_view context);

  absl::Status CheckRequest(uint64_t addr, AccessWidth width) override;
  absl::StatusOr<uint64_t> HandleRead(uint64_t addr,
                                      AccessWidth width) override;
  absl::Status HandleWrite(uint64_t addr, AccessWidth width,
                           uint64_t payload) override;
  absl::StatusOr<IRQEnum> HandleIRQ() override;
  absl::Status HandleTick() override;
  absl::Status Reset() override;

 private:
  XlsPeripheral(Config&& config, IConnection& connection,
                std::unique_ptr<Package> package,
                std::unique_ptr<RuntimeManager> runtime);

  bool last_irq_;
  Config config_;
  IConnection& connection_;
  std::unique_ptr<Package> package_;
  std::unique_ptr<RuntimeManager> runtime_;
  std::vector<std::unique_ptr<IChannelManager>> managers_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_XLSPERIPHERAL_H_
