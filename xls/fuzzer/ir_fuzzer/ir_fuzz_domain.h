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

#include <memory>

#include "xls/common/fuzzing/fuzztest.h"
#include "xls/ir/package.h"

namespace xls {

// TODO: Implement a clone_ptr class that makes a deep copy of a
// unique_ptr to avoid returning a shared_ptr.
fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain();

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
