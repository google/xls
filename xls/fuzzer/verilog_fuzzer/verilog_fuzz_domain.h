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

#ifndef XLS_FUZZER_IR_FUZZER_VERILOG_FUZZ_DOMAIN_H_
#define XLS_FUZZER_IR_FUZZER_VERILOG_FUZZ_DOMAIN_H_

#include <string>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/ir/package.h"
#include "xls/public/runtime_codegen_actions.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

struct VerilogGenerator {
  std::shared_ptr<Package> package;
  SchedulingOptionsFlagsProto scheduling_options;
  CodegenFlagsProto codegen_options;

  absl::StatusOr<ScheduleAndCodegenResult> GenerateVerilog();
};

// A domain like StatusOrVerilogFuzzDomain that filters out any inputs that are
// infeasible given the provided scheduling and codegen options.
fuzztest::Domain<std::string> VerilogFuzzDomain(
    SchedulingOptionsFlagsProto scheduling_options,
    CodegenFlagsProto codegen_options);

// Some codegen options are illegal and lead to CHECK-failures. We provide a
// domain that filters out such options.
fuzztest::Domain<CodegenFlagsProto> CodegenFlagsDomain();

// This domain ensures FDO and related flags are disabled. FDO is expected to be
// more fragile and is not appropriate for fuzzing.
fuzztest::Domain<SchedulingOptionsFlagsProto>
NoFdoSchedulingOptionsFlagsDomain();

// A domain that can be used to create verilog out of arbitrary xls packages.
//
// To get the actual verilog the user should call the 'GenerateVerilog()'
// function. This domain may not always be able to generate verilog because
// some combinations of XLS packages and scheduling/codegen options are
// infeasible.
fuzztest::Domain<VerilogGenerator> VerilogGeneratorDomain(
    fuzztest::Domain<std::shared_ptr<Package>>&& package_domain =
        IrFuzzDomain(),
    fuzztest::Domain<SchedulingOptionsFlagsProto>&& scheduling_options =
        NoFdoSchedulingOptionsFlagsDomain(),
    fuzztest::Domain<CodegenFlagsProto>&& codegen_options =
        CodegenFlagsDomain());
namespace internal {
std::string UnwrapStatusOrVerilog(
    const absl::StatusOr<ScheduleAndCodegenResult>& verilog);
absl::StatusOr<ScheduleAndCodegenResult> DoGenerateVerilog(
    VerilogGenerator verilog);
}  // namespace internal

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_VERILOG_FUZZ_DOMAIN_H_
