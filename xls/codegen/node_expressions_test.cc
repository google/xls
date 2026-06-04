// Copyright 2026 The XLS Authors
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

#include "xls/codegen/node_expressions.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class NodeExpressionsTest : public VerilogTestBase {};

TEST_P(NodeExpressionsTest, RejectsZeroBitLiteralNode) {
  Package package(TestName());
  FunctionBuilder fb(TestName(), &package);
  BValue zero = fb.Literal(UBits(0, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(zero));

  VerilogFile file(GetFileType());
  EXPECT_THAT(
      NodeToExpression(f->return_value(), /*inputs=*/{}, &file,
                       CodegenOptions()),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("zero-bit")));
}

INSTANTIATE_TEST_SUITE_P(NodeExpressionsTestInstantiation, NodeExpressionsTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<NodeExpressionsTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
