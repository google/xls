// Copyright 2020 The XLS Authors
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

#include "xls/common/module_initializer.h"
#include "xls/tools/device_rpc_strategy_factory.h"
#include "xls/tools/ice40_device_rpc_strategy.h"

XLS_REGISTER_MODULE_INITIALIZER(xls_tools_ice40_device_rpc_strategy_registry, {
  xls::DeviceRpcStrategyFactory::GetSingleton().Add("ice40", []() {
    return std::make_unique<xls::Ice40DeviceRpcStrategy>();
  });
});
