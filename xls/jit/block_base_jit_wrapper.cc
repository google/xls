// Copyright 2024 The XLS Authors
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

#include "xls/jit/block_base_jit_wrapper.h"

#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/ir/value.h"

namespace xls {

const absl::flat_hash_map<std::string, Value>&
BaseBlockJitWrapperContinuation::GetOutputPortsMap() const {
  if (saved_output_ports_.empty()) {
    saved_output_ports_ = inner_->GetOutputPortsMap();
  }
  return saved_output_ports_;
}

const absl::flat_hash_map<std::string, Value>&
BaseBlockJitWrapperContinuation::GetRegistersMap() const {
  if (saved_output_registers_.empty()) {
    saved_output_registers_ = inner_->GetRegistersMap();
  }
  return saved_output_registers_;
}

absl::StatusOr<Value> BaseBlockJitWrapperContinuation::GetOutputByName(
    std::string_view name) const {
  const auto& map = GetOutputPortsMap();
  if (auto it = map.find(name); it != map.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Output port ", name, " is not found."));
}

absl::Status BaseBlockJitWrapperContinuation::PrepareForCycle() {
  saved_output_ports_.clear();
  saved_output_registers_.clear();
  return absl::OkStatus();
}

}  // namespace xls
