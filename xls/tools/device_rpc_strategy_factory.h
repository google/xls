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

#ifndef XLS_TOOLS_DEVICE_RPC_STRATEGY_FACTORY_H_
#define XLS_TOOLS_DEVICE_RPC_STRATEGY_FACTORY_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/tools/device_rpc_strategy.h"

namespace xls {

// Factory / registry for device-specific device-RPC discovery / functionality.
class DeviceRpcStrategyFactory {
 public:
  static DeviceRpcStrategyFactory& GetSingleton() {
    static absl::NoDestructor<DeviceRpcStrategyFactory> singleton;
    return *singleton;
  }

  ~DeviceRpcStrategyFactory() = default;

  void Add(std::string_view target_device,
           std::function<std::unique_ptr<DeviceRpcStrategy>()> fcreate) {
    target_device_to_factory_.insert(
        {std::string(target_device), std::move(fcreate)});
  }

  absl::StatusOr<std::unique_ptr<DeviceRpcStrategy>> Create(
      std::string_view target_device);

 private:
  absl::flat_hash_map<std::string,
                      std::function<std::unique_ptr<DeviceRpcStrategy>()>>
      target_device_to_factory_;
};

}  // namespace xls

#endif  // XLS_TOOLS_DEVICE_RPC_STRATEGY_FACTORY_H_
