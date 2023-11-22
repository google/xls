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

#include "xls/simulation/generic/peripheral_factory.h"

#include <functional>
#include <memory>
#include <string_view>
#include <utility>

#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/iperipheral.h"
#include "xls/simulation/generic/xlsperipheral.h"

namespace xls::simulation::generic {

PeripheralFactory& PeripheralFactory::Instance() {
  static PeripheralFactory inst{};
  return inst;
}

PeripheralFactory::PeripheralFactory() {
  // Default behavior - create XlsPeripheral, can be modified by calling
  // OverrideFactoryMethod.
  make_ = [](IConnection& connection,
             std::string_view context) -> std::unique_ptr<IPeripheral> {
    auto peripheral = XlsPeripheral::Make(connection, context);
    if (!peripheral.ok()) {
      XLS_LOG(FATAL) << "Failed to build XLS peripheral. Reason: "
                     << peripheral.status();
      return std::unique_ptr<XlsPeripheral>(nullptr);
    }
    return std::make_unique<XlsPeripheral>(std::move(peripheral.value()));
  };
}

std::unique_ptr<IPeripheral> PeripheralFactory::Make(IConnection& connection,
                                                     std::string_view context) {
  return make_(connection, context);
}

void PeripheralFactory::OverrideFactoryMethod(
    std::function<PeripheralFactory::FactoryMethod> func) {
  make_ = std::move(func);
}

}  // namespace xls::simulation::generic
