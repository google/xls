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

#ifndef XLS_SIMULATION_GENERIC_PERIPHERAL_FACTORY_H_
#define XLS_SIMULATION_GENERIC_PERIPHERAL_FACTORY_H_

#include <functional>
#include <memory>
#include <string_view>

#include "xls/simulation/generic/iconnection.h"
#include "xls/simulation/generic/iperipheral.h"

namespace xls::simulation::generic {

// This factory exists solely to make it possible to test low-level connection
// modules like 'SharedLibConnection', since those have static code (e.g.
// 'extern "C"' callbacks) which normally instantiates XlsPeripheral, but
// in the test we'd like to replace it with something like
// PeripheralMock. That is most easily done by factoring out
// XlsPeripheral instantiation into a separate factory method, which can
// be replaced by the testing code before the test actually provokes
// instantiation of the Peripheral.
class PeripheralFactory {
 public:
  using FactoryMethod = std::unique_ptr<IPeripheral>(IConnection& connection,
                                                     std::string_view context);

  static PeripheralFactory& Instance();

  // Creates a peripheral, by default XlsPeripheral
  std::unique_ptr<IPeripheral> Make(IConnection& connection,
                                    std::string_view context);

  // Allows to override factory method (intended to be used by tests)
  void OverrideFactoryMethod(std::function<FactoryMethod> func);

 private:
  PeripheralFactory();
  std::function<FactoryMethod> make_;
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_PERIPHERAL_FACTORY_H_
