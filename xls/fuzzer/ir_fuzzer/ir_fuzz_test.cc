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

#include <memory>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

// Perform tests on the Package object which contains the IR. This is a general
// test that just verifies if the Package object is valid.
void VerifyIrFuzzPackage(std::shared_ptr<Package> p) {
  XLS_ASSERT_OK(VerifyPackage(p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction(kFuzzTestName));
  VLOG(3) << "IR Fuzzer-2: IR:" << "\n" << f->DumpIr() << "\n";
}
// Use of gtest FUZZ_TEST to randomly generate IR while being compatible with
// Google infrastructure. The IrFuzzTest function is called and represents the
// main test logic. A domain is specified to define the range of possible values
// that the FuzzProgram protobuf can have when generating random values.
FUZZ_TEST(IrFuzzTest, VerifyIrFuzzPackage).WithDomains(IrFuzzDomain());

}  // namespace
}  // namespace xls
