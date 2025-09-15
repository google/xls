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

#include "xls/scheduling/scheduling_options.h"

#include <cstdint>
#include <string_view>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls {
namespace {

using SchedulingOptionsTest = IrTestBase;

int64_t NumberOfOp(FunctionBase* f, Op op) {
  int64_t result = 0;
  for (Node* node : f->nodes()) {
    if (node->op() == op) {
      ++result;
    }
  }
  return result;
}

TEST_F(SchedulingOptionsTest, MergeOnMutualExclusion) {
  SchedulingOptionsFlagsProto proto;
  proto.set_merge_on_mutual_exclusion(true);
  XLS_ASSERT_OK_AND_ASSIGN(SchedulingOptions options_true,
                           SetUpSchedulingOptions(proto, /*p=*/nullptr));
  EXPECT_TRUE(options_true.merge_on_mutual_exclusion());

  proto.set_merge_on_mutual_exclusion(false);
  XLS_ASSERT_OK_AND_ASSIGN(SchedulingOptions options_false,
                           SetUpSchedulingOptions(proto, /*p=*/nullptr));
  EXPECT_FALSE(options_false.merge_on_mutual_exclusion());
}

class MergeOnMutualExclusionAffectsPipelineTest
    : public IrTestBase,
      public ::testing::WithParamInterface<bool> {};

TEST_P(MergeOnMutualExclusionAffectsPipelineTest,
       MergeOnMutualExclusionAffectsPipeline) {
  const bool merge_on_mutual_exclusion = GetParam();

  constexpr std::string_view kTwoSendsIr = R"(
package test_module
chan test_channel(bits[32], id=0, kind=streaming, ops=send_only, flow_control=ready_valid)
top proc main(__state: bits[1], init={0}) {
  not_st: bits[1] = not(__state)
  lit50: bits[32] = literal(value=50)
  lit60: bits[32] = literal(value=60)
  tok: token = literal(value=token)
  send0: token = send(tok, lit50, predicate=__state, channel=test_channel)
  send1: token = send(tok, lit60, predicate=not_st, channel=test_channel)
  after_all_0: token = after_all(send0, send1)
  next (not_st)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(kTwoSendsIr));
  SchedulingOptionsFlagsProto proto;
  proto.set_delay_model("unit");
  proto.set_merge_on_mutual_exclusion(merge_on_mutual_exclusion);
  proto.set_pipeline_stages(1);
  XLS_ASSERT_OK_AND_ASSIGN(SchedulingOptions options,
                           SetUpSchedulingOptions(proto, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           SetUpDelayEstimator(proto));
  OptimizationContext opt_ctx;
  SchedulingPassOptions pass_options{.scheduling_options = options,
                                     .delay_estimator = estimator};
  SchedulingContext sched_ctx =
      SchedulingContext::CreateForWholePackage(p.get());
  PassResults results;
  XLS_ASSERT_OK(CreateSchedulingPassPipeline(opt_ctx, options)
                    ->Run(p.get(), pass_options, &results, sched_ctx));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_EQ(NumberOfOp(proc, Op::kSend), merge_on_mutual_exclusion ? 1 : 2);
}

INSTANTIATE_TEST_SUITE_P(
    MergeOnMutualExclusionAffectsPipelineTestInstance,
    MergeOnMutualExclusionAffectsPipelineTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<
        MergeOnMutualExclusionAffectsPipelineTest::ParamType>& info) {
      return info.param ? "Merge" : "NoMerge";
    });

}  // namespace
}  // namespace xls
