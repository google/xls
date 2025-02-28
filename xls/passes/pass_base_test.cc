// Copyright 2023 The XLS Authors
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

#include "xls/passes/pass_base.h"

#include <memory>
#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;

class PassBaseTest : public IrTestBase {};

// A sneaky pass that tries to avoid returning unlucky numbers just like
// architects. Any number >=13 is increased by 1.
class ArchitectNumber : public OptimizationFunctionBasePass {
 public:
  ArchitectNumber()
      : OptimizationFunctionBasePass("architect_number", "Architect Number") {}
  ~ArchitectNumber() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    Node* original = f->AsFunctionOrDie()->return_value();
    if (!original->GetType()->IsBits()) {
      return false;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * lit_one,
        f->MakeNode<Literal>(original->loc(),
                             Value(UBits(1, original->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * lit_thirteen,
        f->MakeNode<Literal>(original->loc(),
                             Value(UBits(13, original->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * add_one,
        f->MakeNode<BinOp>(original->loc(), lit_one, original, Op::kAdd));
    XLS_ASSIGN_OR_RETURN(Node * is_unlucky,
                         f->MakeNode<CompareOp>(original->loc(), original,
                                                lit_thirteen, Op::kUGe));
    XLS_ASSIGN_OR_RETURN(
        Node * maybe_add,
        f->MakeNode<Select>(original->loc(), is_unlucky,
                            absl::Span<Node* const>{original, add_one},
                            std::nullopt));
    XLS_RETURN_IF_ERROR(f->AsFunctionOrDie()->set_return_value(maybe_add));
    // Oops, we changed things and should return true here!
    return false;
  }
};

// Bigger numbers are always better!
// Replace all constant numbers with their successor as long as doing so does
// not cause the number to wrap around.
class LevelUpPass : public OptimizationFunctionBasePass {
 public:
  LevelUpPass() : OptimizationFunctionBasePass("level_up", "Level up Pass") {}
  ~LevelUpPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    bool changed = false;
    for (Node* n : context.TopoSort(f)) {
      if (n->Is<Literal>() && n->GetType()->IsBits() &&
          !n->As<Literal>()->value().bits().IsAllOnes()) {
        changed = true;
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<Literal>(
                 Value(bits_ops::Increment(n->As<Literal>()->value().bits())))
                .status());
      }
    }
    return changed;
  }
};

auto DceInvoke() { return Field(&PassInvocation::pass_name, Eq("dce")); }
auto LevelUpInvoke() {
  return Field(&PassInvocation::pass_name, Eq("level_up"));
}

TEST_F(PassBaseTest, DetectEasyIncorrectReturn) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(13, 64));
  ASSERT_THAT(fb.Build(), absl_testing::IsOk());
  ArchitectNumber pass;
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(pass.Run(p.get(), OptimizationPassOptions(), &results, context),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::ContainsRegex(
                      "Pass architect_number indicated IR unchanged, but IR is "
                      "changed: \\[Before\\] 1 nodes != \\[after\\] 6 nodes")));
}

TEST_F(PassBaseTest, DetectEasyIncorrectReturnInCompound) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(13, 64));
  ASSERT_THAT(fb.Build(), absl_testing::IsOk());
  ArchitectNumber pass;
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(pass.Run(p.get(), OptimizationPassOptions(), &results, context),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::ContainsRegex(
                      "Pass architect_number indicated IR unchanged, but IR is "
                      "changed: \\[Before\\] 1 nodes != \\[after\\] 6 nodes")));
}

TEST_F(PassBaseTest, BisectLimitMid) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  PassResults results;
  OptimizationContext context;
  for (int i = 0; i < 4; ++i) {
    opt.Add<LevelUpPass>();
    opt.Add<DeadCodeEliminationPass>();
  }
  // Should run Level DCE level DCE
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 4}),
              &results, context),
      IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(2, 64)));
  EXPECT_EQ(f->node_count(), 1);
}

TEST_F(PassBaseTest, BisectLimitAfterEnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  PassResults results;
  OptimizationContext context;
  for (int i = 0; i < 4; ++i) {
    opt.Add<LevelUpPass>();
    opt.Add<DeadCodeEliminationPass>();
  }
  // Should run Level DCE Level DCE Level DCE Level DCE
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 8}),
              &results, context),
      IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(4, 64)));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_EQ(results.invocations.size(), 8);
  auto is_dce = Field(&PassInvocation::pass_name, Eq("dce"));
  auto is_level_up = Field(&PassInvocation::pass_name, Eq("level_up"));
  EXPECT_THAT(
      results.invocations,
      ElementsAre(LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke(),
                  LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke()));
}

TEST_F(PassBaseTest, BisectLimitInFixedPoint) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 16));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  {
    auto fp =
        std::make_unique<OptimizationFixedPointCompoundPass>("fixed", "fixed");
    fp->Add<LevelUpPass>();
    fp->Add<DeadCodeEliminationPass>();
    opt.AddOwned(std::move(fp));
  }
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 16}),
              &results, context),
      IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(8, 16)));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_EQ(results.invocations.size(), 16);
  EXPECT_THAT(
      results.invocations,
      ElementsAre(LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke(),
                  LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke(),
                  LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke(),
                  LevelUpInvoke(), DceInvoke(), LevelUpInvoke(), DceInvoke()));
}
TEST_F(PassBaseTest, BisectLimitInMiddleOfFixedPoint) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 16));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  {
    auto fp =
        std::make_unique<OptimizationFixedPointCompoundPass>("fixed", "fixed");
    fp->Add<DeadCodeEliminationPass>();
    fp->Add<LevelUpPass>();
    fp->Add<DeadCodeEliminationPass>();
    fp->Add<DeadCodeEliminationPass>();
    opt.AddOwned(std::move(fp));
  }
  PassResults results;
  OptimizationContext context;
  // Run 3 times all the way through then the first 3 passes of one last
  // go-around.
  EXPECT_THAT(opt.Run(p.get(),
                      OptimizationPassOptions(
                          PassOptionsBase{.bisect_limit = (3 * 4 + 3)}),
                      &results, context),
              IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(4, 16)));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_EQ(results.invocations.size(), 15);
  EXPECT_THAT(
      results.invocations,
      ElementsAre(DceInvoke(), LevelUpInvoke(), DceInvoke(), DceInvoke(),
                  DceInvoke(), LevelUpInvoke(), DceInvoke(), DceInvoke(),
                  DceInvoke(), LevelUpInvoke(), DceInvoke(), DceInvoke(),
                  DceInvoke(), LevelUpInvoke(), DceInvoke()));
}

TEST_F(PassBaseTest, BisectLimitZero) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  PassResults results;
  OptimizationContext context;
  for (int i = 0; i < 4; ++i) {
    opt.Add<LevelUpPass>();
    opt.Add<DeadCodeEliminationPass>();
  }
  // Should run nothing
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 0}),
              &results, context),
      IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(0, 64)));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_THAT(results.invocations, IsEmpty());
}

}  // namespace
}  // namespace xls
