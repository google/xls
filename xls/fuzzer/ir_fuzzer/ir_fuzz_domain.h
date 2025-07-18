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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "xls/common/fuzzing/fuzztest.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

// Contains functions that return IR fuzz test domains.

namespace xls {

// Stores a Package object which contains a function. param_sets contains
// multiple sets of parameter values for the function.
struct PackageAndTestParams {
  std::shared_ptr<Package> p;
  std::vector<std::vector<Value>> param_sets;
};

// These functions return a shared_ptr instead of a unique_ptr because the
// IrFuzzDomainWithBytesParams() function in the ir_fuzz_domain.cc file uses
// fuzztest::Just() which refuses to deal with move-only types.
// TODO: Implement a clone_ptr class that makes a deep copy of a
// unique_ptr to avoid returning a shared_ptr.
fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain();
fuzztest::Domain<PackageAndTestParams> IrFuzzDomainWithParams(
    int64_t param_set_count = 1);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
