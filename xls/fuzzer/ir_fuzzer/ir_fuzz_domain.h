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
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/package.h"

// Contains functions that return IR fuzz test domains.

namespace xls {

fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain();
fuzztest::Domain<FuzzPackageWithArgs> IrFuzzDomainWithArgs(
    int64_t arg_set_count);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
