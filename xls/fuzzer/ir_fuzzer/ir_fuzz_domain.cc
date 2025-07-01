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

#include <cstdint>
#include <sstream>
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
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

// A domain that expands upon the IrFuzzDomain by returning a vector of bytes
// parameters along with the Package. The strings in the vector correlate to a
// parameter input into the Function. A fuzztest::FlatMap is used in order to
// create the IrFuzzDomain first, then determine the number of parameters in the
// Function in order to randomly generate a vector of strings.
fuzztest::Domain<
    std::pair<std::shared_ptr<Package>, std::vector<std::vector<std::string>>>>
IrFuzzDomainWithBytesParams(int64_t param_set_count) {
  return fuzztest::FlatMap(
      [param_set_count](std::shared_ptr<Package> p) {
        Function* f = p->GetFunction(IrTestBase::TestName()).value();
        return fuzztest::PairOf(
            // fuzztest::Just does not like to deal with move-only types so we
            // are using a shared_ptr for now.
            fuzztest::Just(std::move(p)),
            // A string represents a byte array. The byte array can be
            // interpreted as a large integer.
            fuzztest::VectorOf(
                fuzztest::VectorOf(fuzztest::Arbitrary<std::string>())
                    // Retrieve the number of parameters in the Function.
                    .WithSize(f->params().size()))
                // Generate param_set_count number of param sets.
                .WithSize(param_set_count));
      },
      IrFuzzDomain());
}

// Returns a human-readable string representation of the param sets.
std::string StringifyParamSets(
    absl::Span<const std::vector<Value>> param_sets) {
  std::stringstream ss;
  ss << "[";
  // Iterate over the number of param sets.
  for (int64_t i = 0; i < param_sets.size(); i += 1) {
    // Iterate over the param set elements.
    for (int64_t j = 0; j < param_sets[i].size(); j += 1) {
      ss << (param_sets[i][j].ToHumanString())
         << (j != param_sets[i].size() - 1 ? ", " : "");
    }
    ss << (i != param_sets.size() - 1 ? "], [" : "]");
  }
  return ss.str();
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

// A domain that expands upon the IrFuzzDomainWithBytesParams by converting the
// string parameters into a vector of Value objects. This vector can be plugged
// into the Function directly for interpreting the IR. The byte arrays are
// truncated to the bit width of the parameters. The param_set_count parameter
// specifies the number of param sets to generate in case you need multiple sets
// of inputs for testing.
// TODO: Consider using randomly generated parameter values with something like
// absl::BitGen instead of FuzzTest because FuzzTest does not have much of an
// advantage over the simplicity of absl::BitGen in this case.
fuzztest::Domain<PackageAndTestParams> IrFuzzDomainWithParams(
    int64_t param_set_count) {
  return fuzztest::Map(
      [](std::pair<std::shared_ptr<Package>,
                   std::vector<std::vector<std::string>>>
             bytes_paramaterized_package) {
        auto [p, bytes_param_sets] = bytes_paramaterized_package;
        Function* f = p->GetFunction(IrTestBase::TestName()).value();
        std::vector<std::vector<Value>> param_sets(bytes_param_sets.size());
        // Iterate over the number of param sets.
        for (int64_t i = 0; i < bytes_param_sets.size(); i += 1) {
          // Iterate over the actual function parameters.
          for (int64_t j = 0; j < bytes_param_sets[i].size(); j += 1) {
            // Truncate the byte arrays to the bit width of the parameters.
            int64_t bit_width = f->param(j)->BitCountOrDie();
            Bits value_bits =
                ChangeBytesBitWidth(bytes_param_sets[i][j], bit_width);
            param_sets[i].push_back(Value(value_bits));
          }
        }
        VLOG(3) << "2. Param Sets: " << StringifyParamSets(param_sets);
        return PackageAndTestParams(p, param_sets);
      },
      IrFuzzDomainWithBytesParams(param_set_count));
}

}  // namespace xls
