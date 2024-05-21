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

#include "xls/contrib/ice40/io_strategy_factory.h"

#include <memory>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/vast.h"
#include "xls/contrib/ice40/io_strategy.h"

namespace xls {
namespace verilog {

absl::StatusOr<std::unique_ptr<IOStrategy>> IOStrategyFactory::CreateForDevice(
    std::string_view target_device, VerilogFile* f) {
  auto it = GetSingleton().strategies_.find(target_device);
  if (it == GetSingleton().strategies_.end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not find target device for I/O strategy creation: \"%s\"",
        target_device));
  }

  return it->second(f);
}

}  // namespace verilog
}  // namespace xls
