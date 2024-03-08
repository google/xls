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

#ifndef XLS_CONTRIB_ICE40_IO_STRATEGY_FACTORY_H_
#define XLS_CONTRIB_ICE40_IO_STRATEGY_FACTORY_H_

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/contrib/ice40/io_strategy.h"

namespace xls {
namespace verilog {

class IOStrategyFactory {
 public:
  static IOStrategyFactory& GetSingleton() {
    static absl::NoDestructor<IOStrategyFactory> singleton;
    return *singleton;
  }

  static absl::StatusOr<std::unique_ptr<IOStrategy>> CreateForDevice(
      std::string_view target_device, VerilogFile* f);

  void Add(std::string_view target_device,
           std::function<std::unique_ptr<IOStrategy>(VerilogFile*)> f) {
    strategies_.insert({std::string(target_device), std::move(f)});
  }

 private:
  absl::flat_hash_map<std::string,
                      std::function<std::unique_ptr<IOStrategy>(VerilogFile*)>>
      strategies_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CONTRIB_ICE40_IO_STRATEGY_FACTORY_H_
