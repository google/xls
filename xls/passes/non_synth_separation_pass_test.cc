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

#include "xls/passes/non_synth_separation_pass.h"

#include <vector>
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_test_helpers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

namespace m = xls::op_matchers;

class NonSynthSeparationPassTest : public IrTestBase {
 protected:
  NonSynthSeparationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    return NonSynthSeparationPass().Run(p, OptimizationPassOptions(), &results,
                                        context);
  }
  absl::StatusOr<bool> RunWithDCE(Package* p) {
    PassResults results;
    OptimizationContext context;
    OptimizationCompoundPass pass("TestPass", TestName());
    bool run_result = false;
    pass.Add<RecordIfPassChanged<NonSynthSeparationPass>>(&run_result);
    pass.Add<DeadCodeEliminationPass>();
    XLS_ASSIGN_OR_RETURN(
        bool compound_result,
        pass.Run(p, OptimizationPassOptions(), &results, context));
    return compound_result && run_result;
  }
};

TEST_F(NonSynthSeparationPassTest, FunctionsAreCloned) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
}

TEST_F(NonSynthSeparationPassTest, FunctionsAreClonedWithNoDuplicateNames) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(fb.Build());
  FunctionBuilder non_synth_fb("non_synth_f", p.get());
  non_synth_fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK(non_synth_fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("f"));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK(p->GetFunction("_non_synth_f"));
  XLS_ASSERT_OK(p->GetFunction("non_synth_non_synth_f"));
}

TEST_F(NonSynthSeparationPassTest, NonSynthFunctionReturnValuesAreVoided) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(1, 1)));
  EXPECT_THAT(non_synth_f->return_value(),
              m::Tuple(std::vector<testing::Matcher<const Node*>>()));
}

TEST_F(NonSynthSeparationPassTest, InvokeToNonSynthFunctionIsInserted) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * invoke_node, f->GetNode("invoke.4"));
  EXPECT_THAT(invoke_node,
              m::Invoke(std::vector<testing::Matcher<const Node*>>()));
}

TEST_F(NonSynthSeparationPassTest,
       InvokeToNonSynthFunctionIsInsertedWithParams) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  fb.Param("p0", p->GetBitsType(1));
  fb.Param("p1", p->GetTupleType({}));
  fb.Param("p2", p->GetArrayType(1, p->GetBitsType(1)));
  fb.Literal(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK(p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * invoke_node, f->GetNode("invoke.10"));
  EXPECT_THAT(invoke_node,
              m::Invoke(m::Param("p0"), m::Param("p1"), m::Param("p2")));
}

// This test fails if the nodes are not topologically sorted.
TEST_F(NonSynthSeparationPassTest, NonSynthNodesAreRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue assert_token = fb.AfterAll({});
  BValue assert_condition = fb.Literal(UBits(0, 1));
  BValue assert = fb.Assert(assert_token, assert_condition, "");
  BValue trace_token = fb.AfterAll({});
  BValue trace_condition = fb.Literal(UBits(0, 1));
  BValue trace = fb.Trace(trace_token, trace_condition, {}, "");
  BValue cover_condition = fb.Literal(UBits(0, 1));
  BValue cover = fb.Cover(cover_condition, "cover_label");
  fb.Tuple({assert, trace, cover});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  std::vector<testing::Matcher<const Node*>> empty_operands;
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::AfterAll(empty_operands), m::AfterAll(empty_operands),
                       m::Tuple(empty_operands)));
  EXPECT_THAT(f->node_count(), 8);
  EXPECT_THAT(non_synth_f->node_count(), 10);
}

TEST_F(NonSynthSeparationPassTest, NonSynthNodesAreRemovedWithDCE) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue assert_token = fb.AfterAll({});
  BValue assert_condition = fb.Literal(UBits(0, 1));
  BValue assert = fb.Assert(assert_token, assert_condition, "");
  BValue trace_token = fb.AfterAll({});
  BValue trace_condition = fb.Literal(UBits(0, 1));
  BValue trace = fb.Trace(trace_token, trace_condition, {}, "");
  BValue cover_condition = fb.Literal(UBits(0, 1));
  BValue cover = fb.Cover(cover_condition, "cover_label");
  fb.Tuple({assert, trace, cover});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(RunWithDCE(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  std::vector<testing::Matcher<const Node*>> empty_operands;
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::AfterAll(empty_operands), m::AfterAll(empty_operands),
                       m::Tuple(empty_operands)));
  EXPECT_THAT(f->node_count(), 5);
  EXPECT_THAT(non_synth_f->node_count(), 9);
}

// This test fails if the nodes are not topologically sorted.
TEST_F(NonSynthSeparationPassTest, GateNodesAreReplacedWithSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  BValue gate_condition = fb.Literal(UBits(0, 1));
  BValue gate_data = fb.Literal(UBits(0, 2));
  BValue gate = fb.Gate(gate_condition, gate_data);
  fb.Identity(gate);
  fb.Literal(UBits(0, 3));
  XLS_ASSERT_OK(fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * non_synth_f,
                           p->GetFunction("non_synth_f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * select_node, non_synth_f->GetNode("sel.13"));
  EXPECT_THAT(select_node,
              m::Select(m::Literal(UBits(0, 1)), {m::Literal(UBits(0, 2))},
                        m::Literal(UBits(0, 2))));
}

void IrFuzzNonSynthSeparation(FuzzPackageWithArgs fuzz_package_with_args) {
  NonSynthSeparationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNonSynthSeparation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
