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

#include "xls/common/fuzzing/fuzztest.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

// Perform tests on the Package object which contains the IR. This is a general
// test that just verifies if the Package object is valid.
void IrFuzzTest(std::unique_ptr<Package> p) {
  XLS_ASSERT_OK(VerifyPackage(p.get()));
}

// Use of gtest FUZZ_TEST to randomly generate IR while being compatible with
// Google infrastructure. The IrFuzzTest function is called and represents the
// main test logic. A domain is specified to define the range of possible values
// that the FuzzProgram protobuf can have when generating random values.
FUZZ_TEST(IrFuzzTest, IrFuzzTest).WithDomains(IrFuzzDomain());

}  // namespace
}  // namespace xls
