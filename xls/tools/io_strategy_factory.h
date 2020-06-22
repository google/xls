// Copyright 2020 Google LLC
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

#ifndef XLS_TOOLS_IO_STRATEGY_FACTORY_H_
#define XLS_TOOLS_IO_STRATEGY_FACTORY_H_

#include <functional>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/statusor.h"
#include "xls/tools/io_strategy.h"

namespace xls {
namespace verilog {

class IoStrategyFactory {
 public:
  static IoStrategyFactory* GetSingleton() {
    static IoStrategyFactory* singleton = new IoStrategyFactory;
    return singleton;
  }

  static xabsl::StatusOr<std::unique_ptr<IoStrategy>> CreateForDevice(
      absl::string_view target_device, VerilogFile* f);

  void Add(absl::string_view target_device,
           std::function<std::unique_ptr<IoStrategy>(VerilogFile*)> f) {
    strategies_.insert({std::string(target_device), std::move(f)});
  }

 private:
  absl::flat_hash_map<std::string,
                      std::function<std::unique_ptr<IoStrategy>(VerilogFile*)>>
      strategies_;
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_TOOLS_IO_STRATEGY_FACTORY_H_
