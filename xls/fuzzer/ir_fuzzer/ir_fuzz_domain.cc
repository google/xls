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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"

#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {

// Returns a fuzztest domain, which is a range of possible values that an object
// can have. In this case, our object is a FuzzProgramProto protobuf. The
// Arbitrary domain allows all possible values and vector sizes for fields. Some
// fields may be unset. fuzztest::Map is used to convert the FuzzProgramProto
// domain into a Package domain for test creation readability.
fuzztest::Domain<PackageAndFuzzProgram> IrFuzzDomain() {
  return fuzztest::Map(
      [](FuzzProgramProto fuzz_program) {
        // Create the package.
        std::unique_ptr<Package> p = IrTestBase::CreatePackage();
        FunctionBuilder fb(IrTestBase::TestName(), p.get());
        // Build the IR from the FuzzProgramProto.
        IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
        BValue ir = ir_fuzz_builder.BuildIr();
        CHECK_OK(fb.BuildWithReturnValue(ir))
            << "Failed to build package from FuzzProgramProto: "
            << fuzz_program.DebugString();
        // Create the PackageAndFuzzProgram object as the domain export.
        return PackageAndFuzzProgram(std::move(p), fuzz_program);
      },
      // Specify the range of possible values for the FuzzProgramProto protobuf.
      fuzztest::Arbitrary<FuzzProgramProto>().WithRepeatedProtobufField(
          "fuzz_ops",
          fuzztest::VectorOf(fuzztest::Arbitrary<FuzzOpProto>()
                                 // We want all FuzzOps to be defined.
                                 .WithOneofAlwaysSet("fuzz_op"))
              // Generate at least one FuzzOp.
              .WithMinSize(1)));
}

}  // namespace xls
