// Copyright 2024 The XLS Authors
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

#include "xls/codegen/trace_verbosity_pass.h"

#include <cstdint>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using TraceVerbosityPassTest = IrTestBase;

using ::absl_testing::IsOkAndHolds;
using ::testing::Contains;
using ::testing::Eq;
using ::testing::Gt;

namespace m = ::xls::op_matchers;

absl::StatusOr<bool> RunTraceVerbosityPass(Block* block,
                                           int64_t max_trace_verbosity = 0) {
  PassResults results;
  const verilog::CodegenPassOptions options = {
      .codegen_options = verilog::CodegenOptions().set_max_trace_verbosity(
          max_trace_verbosity),
  };
  verilog::CodegenContext context(block);
  return verilog::TraceVerbosityPass().Run(block->package(), options, &results,
                                           context);
}

TEST_F(TraceVerbosityPassTest, NoTraceIsNoop) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  bb.OutputPort("b", a);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  EXPECT_THAT(RunTraceVerbosityPass(block), IsOkAndHolds(false));
}

TEST_F(TraceVerbosityPassTest, SimpleTraceRemoved) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  bb.OutputPort("b", a);
  BValue tok = bb.AfterAll({});
  bb.Trace(tok, bb.Eq(a, bb.Literal(UBits(0, 32))), {}, "a == 0",
           /*verbosity=*/1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  EXPECT_THAT(block->nodes(), Contains(m::Trace()));
  absl::StatusOr<bool> graph_changed =
      RunTraceVerbosityPass(block, /*max_trace_verbosity=*/1);
  EXPECT_THAT(graph_changed, IsOkAndHolds(false));

  graph_changed = RunTraceVerbosityPass(block, /*max_trace_verbosity=*/0);
  EXPECT_THAT(graph_changed, IsOkAndHolds(true));
  EXPECT_THAT(block->nodes(), Not(Contains(m::Trace())));
}

TEST_F(TraceVerbosityPassTest, RelatedTracesRemoved) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  bb.OutputPort("b", a);
  BValue tok = bb.AfterAll({});
  BValue trace0 =
      bb.Trace(tok, bb.Eq(a, bb.Literal(UBits(0, 32))), {}, "a == 0",
               /*verbosity=*/1);
  BValue trace1 =
      bb.Trace(trace0, bb.Eq(a, bb.Literal(UBits(1, 32))), {}, "a == 1",
               /*verbosity=*/0);
  bb.Trace(trace1, bb.Eq(a, bb.Literal(UBits(2, 32))), {}, "a == 2",
           /*verbosity=*/1);
  BValue assertion = bb.Assert(trace1, bb.ULt(a, bb.Literal(UBits(100, 32))),
                               "a <= 100 must be true");
  bb.Trace(assertion, bb.Eq(a, bb.Literal(UBits(3, 32))), {}, "a == 3",
           /*verbosity=*/1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  EXPECT_THAT(block->nodes(), Contains(m::Trace()));
  absl::StatusOr<bool> graph_changed =
      RunTraceVerbosityPass(block, /*max_trace_verbosity=*/2);
  EXPECT_THAT(graph_changed, IsOkAndHolds(false));

  graph_changed = RunTraceVerbosityPass(block, /*max_trace_verbosity=*/0);
  EXPECT_THAT(graph_changed, IsOkAndHolds(true));
  EXPECT_THAT(block->nodes(), Not(Contains(m::TraceWithVerbosity(Gt(0)))));
  EXPECT_THAT(block->nodes(), Contains(m::TraceWithVerbosity(Eq(0))).Times(1));
}

TEST_F(TraceVerbosityPassTest, FilteredTraceUsageReplaced) {
  auto p = CreatePackage();
  Type* token_type = p->GetTokenType();
  BlockBuilder bb(TestName(), p.get());
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  bb.OutputPort("b", a);
  BValue tok = bb.AfterAll({});
  BValue trace_result =
      bb.Trace(tok, bb.Eq(a, bb.Literal(UBits(0, 32))), {}, "a == 0",
               /*verbosity=*/1);
  FunctionBuilder b("takes_token_arg", p.get());
  b.Param("arg", token_type);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func,
                           b.BuildWithReturnValue(b.Literal(UBits(0, 32))));
  bb.Invoke({trace_result}, func);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());
  EXPECT_THAT(block->nodes(), Contains(m::Trace()));
  EXPECT_THAT(block->nodes(), Contains(m::Invoke(m::Trace())).Times(1));
  absl::StatusOr<bool> graph_changed =
      RunTraceVerbosityPass(block, /*max_trace_verbosity=*/2);
  EXPECT_THAT(graph_changed, IsOkAndHolds(false));

  graph_changed = RunTraceVerbosityPass(block, /*max_trace_verbosity=*/0);
  EXPECT_THAT(graph_changed, IsOkAndHolds(true));
  EXPECT_THAT(block->nodes(), Not(Contains(m::Trace())));
  EXPECT_THAT(block->nodes(), Contains(m::Invoke(m::AfterAll())).Times(1));
}

}  // namespace
}  // namespace xls
