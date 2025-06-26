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

#include <memory>
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

// Initializes the IrFuzzBuilder so that it can convert the FuzzProgramProto
// into a valid IR object, which exists inside a Function object, which exists
// inside a Package object.
absl::StatusOr<std::shared_ptr<Package>> BuildPackage(
    FuzzProgramProto& fuzz_program) {
  std::unique_ptr<Package> p = IrTestBase::CreatePackage();
  FunctionBuilder fb(IrTestBase::TestName(), p.get());
  IrFuzzBuilder ir_fuzz_builder(&fuzz_program, p.get(), &fb);
  BValue ir = ir_fuzz_builder.BuildIr();
  XLS_RET_CHECK_OK(fb.BuildWithReturnValue(ir));
  return std::move(p);
}

}  // namespace

// Returns a fuzztest domain, which is a range of possible values that an object
// can have. In this case, our object is a FuzzProgramProto protobuf. The
// Arbitrary domain allows all possible values and vector sizes for fields. Some
// fields may be unset. fuzztest::Map is used to convert the FuzzProgramProto
// domain into a Package domain for test creation readability.
fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain() {
  return fuzztest::Map(
      [](FuzzProgramProto fuzz_program) {
        absl::StatusOr<std::shared_ptr<Package>> p = BuildPackage(fuzz_program);
        CHECK_OK(p.status())
            << "Failed to build package from FuzzProgramProto: "
            << fuzz_program;
        return *std::move(p);
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
