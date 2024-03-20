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

#ifndef XLS_SIMULATION_GENERIC_MANAGED_SERIAL_PROC_RUNTIME_H_
#define XLS_SIMULATION_GENERIC_MANAGED_SERIAL_PROC_RUNTIME_H_

#include <cstdint>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/serial_proc_runtime.h"

namespace xls::simulation::generic {

class ManagedSerialProcRuntime : public SerialProcRuntime {
 private:
  ManagedSerialProcRuntime() = default;

  absl::Status Tick();
  absl::StatusOr<int64_t> TickUntilOutput(
      absl::flat_hash_map<Channel*, int64_t> output_counts,
      std::optional<int64_t> max_ticks = std::nullopt);
  absl::StatusOr<int64_t> TickUntilBlocked(
      std::optional<int64_t> max_ticks = std::nullopt);
};

}  // namespace xls::simulation::generic

#endif  // XLS_SIMULATION_GENERIC_MANAGED_SERIAL_PROC_RUNTIME_H_
