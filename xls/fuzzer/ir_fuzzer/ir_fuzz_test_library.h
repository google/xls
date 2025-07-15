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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_TEST_LIBRARY_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_TEST_LIBRARY_H_

#include <cstdint>
#include <string>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/ir/node.h"
#include "xls/passes/optimization_pass.h"

// Contains functions that are used in IR fuzz tests.

namespace xls {

void OptimizationPassChangesOutputs(
    PackageAndFuzzProgram package_and_fuzz_program, int64_t arg_set_count,
    OptimizationPass& pass);
absl::Status PassChangesOutputsWithProto(std::string proto_string,
                                         int64_t arg_set_count,
                                         OptimizationPass& pass);
absl::Status EquateProtoToIrTest(
    std::string proto_string, testing::Matcher<const Node*> expected_ir_node);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_TEST_LIBRARY_H_
