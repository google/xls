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

#include "xls/fuzzer/ir_fuzzer/reproducer_to_ir.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {

absl::StatusOr<std::vector<std::vector<Value>>> FuzzerReproToValues(
    std::string_view data, std::optional<int64_t> num_args) {
  if (!num_args) {
    return std::vector<std::vector<Value>>();
  }
  auto domain = IrFuzzDomainWithArgs(*num_args);
  XLS_ASSIGN_OR_RETURN(auto [fuzz_data],
                       fuzztest::unstable::ParseReproducerValue(data, domain));
  return std::move(fuzz_data.arg_sets);
}
absl::StatusOr<std::shared_ptr<Package>> FuzzerReproToIr(
    std::string_view data, std::optional<int64_t> num_args) {
  if (num_args) {
    auto domain = IrFuzzDomainWithArgs(*num_args);
    XLS_ASSIGN_OR_RETURN(
        auto [fuzz_data],
        fuzztest::unstable::ParseReproducerValue(data, domain));
    return std::shared_ptr<Package>(fuzz_data.fuzz_package.p.release());
  } else {
    auto domain = IrFuzzDomain();
    XLS_ASSIGN_OR_RETURN(
        auto [fuzz_data],
        fuzztest::unstable::ParseReproducerValue(data, domain));
    return fuzz_data;
  }
}

}  // namespace xls
