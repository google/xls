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

#ifndef XLS_TOOLS_ICE40_DEVICE_RPC_STRATEGY_H_
#define XLS_TOOLS_ICE40_DEVICE_RPC_STRATEGY_H_

#include <optional>

#include "absl/status/statusor.h"
#include "xls/contrib/ice40/device_rpc_strategy.h"

namespace xls {

class Ice40DeviceRpcStrategy : public DeviceRpcStrategy {
 public:
  ~Ice40DeviceRpcStrategy() override;

  absl::Status Connect(int64_t device_ordinal) override;

  absl::StatusOr<Value> CallUnnamed(const FunctionType& function_type,
                                    absl::Span<const Value> arguments) override;

 private:
  std::optional<int> tty_fd_;
};

}  // namespace xls

#endif  // XLS_TOOLS_ICE40_DEVICE_RPC_STRATEGY_H_
