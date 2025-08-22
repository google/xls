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
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {

// A domain that generates standalone Verilog files.
// These files are created by taking the output of IrFuzzDomain() and
// running codegen on it with the provided options. The returned domain
// contains a StatusOr<> because codegen can fail for a variety of reasons,
// e.g. scheduling options that cannot be satisfied.
fuzztest::Domain<absl::StatusOr<std::string>> StatusOrVerilogFuzzDomain(
    SchedulingOptionsFlagsProto scheduling_options,
    CodegenFlagsProto codegen_options);

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

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_VERILOG_FUZZ_DOMAIN_H_
