// Copyright 2025 The XLS Authors
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

#ifndef XLS_FUZZER_IR_FUZZER_REPRODUCER_TO_IR_H_
#define XLS_FUZZER_IR_FUZZER_REPRODUCER_TO_IR_H_

#include <vector>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "absl/status/statusor.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
namespace xls {

absl::StatusOr<std::shared_ptr<Package>> FuzzerReproToIr(
    std::string_view data, std::optional<int64_t> num_args = 10);
absl::StatusOr<std::vector<std::vector<Value>>> FuzzerReproToValues(
    std::string_view data, std::optional<int64_t> num_args = 10);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_REPRODUCER_TO_IR_H_
