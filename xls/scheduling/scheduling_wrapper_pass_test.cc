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

#include "xls/scheduling/scheduling_wrapper_pass.h"

#include <memory>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/literal_uncommoning_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_scheduling_pass.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Key;
using ::testing::UnorderedElementsAre;

using SchedulingWrapperPassTest = IrTestBase;

absl::StatusOr<bool> RunSchedulingPass(
    Package* package, std::unique_ptr<OptimizationPass> wrapped_pass,
    bool reschedule_new_nodes, SchedulingContext& context) {
  PassResults scheduling_results;
  TestDelayEstimator delay_estimator;
  SchedulingPassOptions options{
      .scheduling_options = SchedulingOptions().pipeline_stages(10),
      .delay_estimator = &delay_estimator};
  OptimizationContext optimization_context;
  return SchedulingWrapperPass(std::move(wrapped_pass), optimization_context,
                               /*opt_level=*/kMaxOptLevel,
                               /*eliminate_noop_next=*/false,
                               /*reschedule_new_nodes=*/reschedule_new_nodes)
      .Run(package, options, &scheduling_results, context);
}

absl::StatusOr<bool> RunSchedulingPass(Package* package,
                                       const SchedulingPass& pass,
                                       SchedulingContext& context) {
  PassResults scheduling_results;
  TestDelayEstimator delay_estimator;
  SchedulingPassOptions options{
      .scheduling_options = SchedulingOptions().pipeline_stages(10),
      .delay_estimator = &delay_estimator};
  return pass.Run(package, options, &scheduling_results, context);
}

absl::StatusOr<Function*> AddFunction(Package* p, std::string_view name) {
  FunctionBuilder fb(name, p);
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Add(x, y);
  return fb.Build();
}

absl::StatusOr<Function*> AddFunctionWithDeadNode(Package* p,
                                                  std::string_view name) {
  FunctionBuilder fb(name, p);
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Not(y, SourceInfo(), "dead_node");
  fb.Add(x, y);
  return fb.Build();
}

// Makes a function that shares a literal between two ops. The literal
// uncommoning pass will add new nodes if run on this function.
absl::StatusOr<Function*> CommonLiteralFunction(Package* p,
                                                std::string_view name) {
  FunctionBuilder fb(name, p);
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue literal = fb.Literal(UBits(0, 32));
  fb.Add(fb.Add(x, literal), fb.Add(y, literal));
  return fb.Build();
}

// Makes a function that shares a literal between two dead ops. The literal
// uncommoning pass will add new nodes if run on this function. DCE will
// remove the common literals if run on this function.
absl::StatusOr<Function*> DeadCommonLiteralFunction(Package* p,
                                                    std::string_view name) {
  FunctionBuilder fb(name, p);
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue literal = fb.Literal(UBits(0, 32), SourceInfo(), "dead_node");
  fb.Identity(literal);
  fb.Identity(literal);
  fb.Add(x, y);
  return fb.Build();
}

// Makes a function that invokes an add() function.
absl::StatusOr<Function*> AddFunctionViaInvoke(Package* p,
                                               std::string_view name) {
  XLS_ASSIGN_OR_RETURN(Function * add_func,
                       AddFunction(p, absl::StrCat(name, "_sub")));
  FunctionBuilder fb(name, p);
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Invoke({x, y}, add_func);
  return fb.Build();
}

TEST_F(SchedulingWrapperPassTest, DCEDoesntChangeWhenRunOnSingleProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunction(p.get(), "add_func"));
  auto context = SchedulingContext::CreateForSingleFunction(add_func);
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(add_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(add_func, std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<DeadCodeEliminationPass>(),
                        /*reschedule_new_nodes=*/true, context),
      IsOkAndHolds(false));
}

TEST_F(SchedulingWrapperPassTest, DCEWorksOnUnscheduledFunction) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunction(p.get(), "add_func"));
  XLS_ASSERT_OK(
      AddFunctionWithDeadNode(p.get(), "add_func_with_dead_node").status());

  auto context = SchedulingContext::CreateForSingleFunction(add_func);
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(add_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(add_func, std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<DeadCodeEliminationPass>(),
                        /*reschedule_new_nodes=*/true, context),
      IsOkAndHolds(true));

  EXPECT_THAT(context.package_schedule().GetSchedules(),
              UnorderedElementsAre(Key(add_func)));
}

TEST_F(SchedulingWrapperPassTest, DCEWorksWhenDCEdProcIsUnscheduled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunction(p.get(), "add_func"));
  XLS_ASSERT_OK(
      AddFunctionWithDeadNode(p.get(), "add_func_with_dead_node").status());

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(add_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(add_func, std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<DeadCodeEliminationPass>(),
                        /*reschedule_new_nodes=*/true, context),
      IsOkAndHolds(true));

  EXPECT_THAT(context.package_schedule().GetSchedules(),
              UnorderedElementsAre(Key(add_func)));
}

TEST_F(SchedulingWrapperPassTest, DCEFixesScheduleOfChangedProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunction(p.get(), "add_func"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * add_func_with_dead_node,
      AddFunctionWithDeadNode(p.get(), "add_func_with_dead_node"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * dead_node,
                           add_func_with_dead_node->GetNode("dead_node"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(add_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(add_func, std::move(schedule)));

  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, PipelineSchedule::SingleStage(add_func_with_dead_node));
  XLS_ASSERT_OK(context.package_schedule().AddSchedule(add_func_with_dead_node,
                                                       std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<DeadCodeEliminationPass>(),
                        /*reschedule_new_nodes=*/true, context),
      IsOkAndHolds(true));

  ASSERT_THAT(
      context.package_schedule().GetSchedules(),
      UnorderedElementsAre(Key(add_func), Key(add_func_with_dead_node)));

  EXPECT_FALSE(
      context.package_schedule().GetSchedule(add_func).IsScheduled(dead_node));
}

TEST_F(SchedulingWrapperPassTest,
       LiteralUncommoningReturnsErrorWhenReschedulingDisabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * lit_func,
                           CommonLiteralFunction(p.get(), "lit_func"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(lit_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(lit_func, std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<LiteralUncommoningPass>(),
                        /*reschedule_new_nodes=*/false, context),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("can't create new nodes")));
}

TEST_F(SchedulingWrapperPassTest,
       LiteralUncommoningClearsScheduleWhenReschedulingEnabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * lit_func,
                           CommonLiteralFunction(p.get(), "add_func"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::SingleStage(lit_func));
  XLS_ASSERT_OK(
      context.package_schedule().AddSchedule(lit_func, std::move(schedule)));

  EXPECT_THAT(
      RunSchedulingPass(p.get(), std::make_unique<LiteralUncommoningPass>(),
                        /*reschedule_new_nodes=*/true, context),
      IsOkAndHolds(true));

  EXPECT_THAT(context.package_schedule().GetSchedules(), IsEmpty());
}

TEST_F(SchedulingWrapperPassTest, DCEMakesLiteralUncommoningANoop) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * dead_common_literal_func,
      DeadCommonLiteralFunction(p.get(), "dead_common_literal_func"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * dead_node,
                           dead_common_literal_func->GetNode("dead_node"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::SingleStage(dead_common_literal_func));
  XLS_ASSERT_OK(context.package_schedule().AddSchedule(dead_common_literal_func,
                                                       std::move(schedule)));
  SchedulingCompoundPass pass_pipeline("scheduling", "DCE + literal commoning");
  OptimizationContext opt_context;
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<LiteralUncommoningPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  EXPECT_THAT(RunSchedulingPass(p.get(), pass_pipeline, context),
              IsOkAndHolds(true));

  ASSERT_THAT(context.package_schedule().GetSchedules(),
              UnorderedElementsAre(Key(dead_common_literal_func)));
  EXPECT_FALSE(context.package_schedule()
                   .GetSchedule(dead_common_literal_func)
                   .IsScheduled(dead_node));
}

TEST_F(SchedulingWrapperPassTest,
       LiteralUncommoningBeforeDCEClearsTheScheduleWhenReschedulingEnabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * dead_common_literal_func,
      DeadCommonLiteralFunction(p.get(), "dead_common_literal_func"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::SingleStage(dead_common_literal_func));
  XLS_ASSERT_OK(context.package_schedule().AddSchedule(dead_common_literal_func,
                                                       std::move(schedule)));
  SchedulingCompoundPass pass_pipeline("scheduling", "DCE + literal commoning");
  OptimizationContext opt_context;
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<LiteralUncommoningPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false,
      /*reschedule_new_nodes=*/true);
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  EXPECT_THAT(RunSchedulingPass(p.get(), pass_pipeline, context),
              IsOkAndHolds(true));
  ASSERT_THAT(context.package_schedule().GetSchedules(), IsEmpty());
}

TEST_F(SchedulingWrapperPassTest,
       LiteralUncommoningBeforeDCEResultsInErrorWhenReschedulingDisabled) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * dead_common_literal_func,
      DeadCommonLiteralFunction(p.get(), "dead_common_literal_func"));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::SingleStage(dead_common_literal_func));
  XLS_ASSERT_OK(context.package_schedule().AddSchedule(dead_common_literal_func,
                                                       std::move(schedule)));
  SchedulingCompoundPass pass_pipeline("scheduling", "DCE + literal commoning");
  OptimizationContext opt_context;
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<LiteralUncommoningPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false,
      /*reschedule_new_nodes=*/false);
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>(), opt_context, kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  EXPECT_THAT(RunSchedulingPass(p.get(), pass_pipeline, context),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("can't create new nodes")));
}

TEST_F(SchedulingWrapperPassTest, FunctionInliningScheduleDFEWorks) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunctionViaInvoke(p.get(), "add"));
  // Need to set top for DFE to remove invoked function.
  XLS_ASSERT_OK(p->SetTop(add_func));

  auto context = SchedulingContext::CreateForWholePackage(p.get());
  SchedulingCompoundPass pass_pipeline("scheduling", "inline + schedule + dfe");
  OptimizationContext opt_context;
  // Inlining removes `invoke` ops but leaves the invoked function in.
  pass_pipeline.Add<SchedulingWrapperPass>(std::make_unique<InliningPass>(),
                                           opt_context, kMaxOptLevel,
                                           /*eliminate_noop_next=*/false,
                                           /*reschedule_new_nodes=*/true);
  // This should schedule both functions.
  pass_pipeline.Add<PipelineSchedulingPass>();
  // This should remove the invoked function.
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<DeadFunctionEliminationPass>(), opt_context,
      kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  EXPECT_THAT(RunSchedulingPass(p.get(), pass_pipeline, context),
              IsOkAndHolds(true));
  ASSERT_THAT(context.package_schedule().GetSchedules(),
              UnorderedElementsAre(Key(add_func)));
}

TEST_F(SchedulingWrapperPassTest,
       FunctionInliningScheduleDFEThrowsErrorWhenContextRemoved) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * add_func,
                           AddFunctionViaInvoke(p.get(), "add"));
  // Need to set top for DFE to remove invoked function.
  XLS_ASSERT_OK(p->SetTop(add_func));

  // Schedule the function that will be removed by DFE.
  auto context = SchedulingContext::CreateForSingleFunction(
      p->GetFunction("add_sub").value());
  SchedulingCompoundPass pass_pipeline("scheduling", "inline + schedule + dfe");
  OptimizationContext opt_context;
  // Inlining removes `invoke` ops but leaves the invoked function in.
  pass_pipeline.Add<SchedulingWrapperPass>(std::make_unique<InliningPass>(),
                                           opt_context, kMaxOptLevel,
                                           /*eliminate_noop_next=*/false,
                                           /*reschedule_new_nodes=*/true);
  // This should schedule both functions.
  pass_pipeline.Add<PipelineSchedulingPass>();
  // This should remove the invoked function.
  pass_pipeline.Add<SchedulingWrapperPass>(
      std::make_unique<DeadFunctionEliminationPass>(), opt_context,
      kMaxOptLevel,
      /*eliminate_noop_next=*/false);
  EXPECT_THAT(RunSchedulingPass(p.get(), pass_pipeline, context),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("not found")));
}

}  // namespace
}  // namespace xls
