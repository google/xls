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

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
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
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

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

// Pass which adds literal nodes to the graph.
class NodeAdderPass : public OptimizationFunctionBasePass {
 public:
  explicit NodeAdderPass(int64_t nodes_to_add)
      : OptimizationFunctionBasePass(
            absl::StrFormat("node_adder_%d", nodes_to_add),
            absl::StrFormat("Node adder %d", nodes_to_add)),
        nodes_to_add_(nodes_to_add) {}
  ~NodeAdderPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    for (int64_t i = 0; i < nodes_to_add_; ++i) {
      XLS_RETURN_IF_ERROR(
          f->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 32))).status());
    }
    return nodes_to_add_ > 0;
  }

  int64_t nodes_to_add_;
};

// Pass which adds a single literal nodes if the graph has less than N nodes.
class AddNodesUpToNPass : public OptimizationFunctionBasePass {
 public:
  explicit AddNodesUpToNPass(int64_t n)
      : OptimizationFunctionBasePass(absl::StrFormat("add_up_to_%d", n),
                                     absl::StrFormat("Add up to %d", n)),
        n_(n) {}
  ~AddNodesUpToNPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    if (f->node_count() < n_) {
      XLS_RETURN_IF_ERROR(
          f->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 32))).status());
      return true;
    }
    return false;
  }

  int64_t n_;
};

// Pass which replaces the first node in the function base with a literal zero
// value N times.
class NodeReplacerPass : public OptimizationFunctionBasePass {
 public:
  explicit NodeReplacerPass(int64_t replacement_count)
      : OptimizationFunctionBasePass(
            absl::StrFormat("replacer_%d", replacement_count),
            absl::StrFormat("Replacer %d", replacement_count)),
        replacement_count_(replacement_count) {}
  ~NodeReplacerPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    for (int64_t i = 0; i < replacement_count_; ++i) {
      Node* first_node = *(f->nodes().begin());
      XLS_RETURN_IF_ERROR(
          first_node
              ->ReplaceUsesWithNew<Literal>(ZeroOfType(first_node->GetType()))
              .status());
    }
    return replacement_count_ > 0;
  }

  int64_t replacement_count_;
};

// Invariant checker which counts the number of times it was invoked.
class CounterChecker : public OptimizationInvariantChecker {
 public:
  explicit CounterChecker() = default;

  absl::Status Run(Package* package, const OptimizationPassOptions& options,
                   PassResults* results,
                   OptimizationContext& context) const override {
    counter_++;
    return absl::OkStatus();
  }

  int64_t run_count() const { return counter_; }

 private:
  mutable int64_t counter_ = 0;
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
  EXPECT_EQ(results.invocation.pass_name, "opt");

  EXPECT_EQ(results.invocation.nested_invocations.size(), 8);

  auto is_dce = Field(&PassInvocation::pass_name, Eq("dce"));
  auto is_level_up = Field(&PassInvocation::pass_name, Eq("level_up"));
  EXPECT_THAT(
      results.invocation.nested_invocations,
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

  const std::vector<PassInvocation>& fixed_point_invocations =
      results.invocation.nested_invocations[0].nested_invocations;
  EXPECT_EQ(fixed_point_invocations.size(), 16);
  EXPECT_THAT(
      fixed_point_invocations,
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
  EXPECT_EQ(results.invocation.pass_name, "opt");
  EXPECT_EQ(results.invocation.nested_invocations.size(), 1);
  EXPECT_EQ(results.invocation.nested_invocations[0].pass_name, "fixed");

  const std::vector<PassInvocation>& fixed_point_invocations =
      results.invocation.nested_invocations[0].nested_invocations;
  EXPECT_EQ(fixed_point_invocations.size(), 15);
  EXPECT_THAT(
      fixed_point_invocations,
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
  EXPECT_THAT(results.invocation.nested_invocations, IsEmpty());
}

TEST_F(PassBaseTest, EmptyCompoundPass) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  auto* checker = opt.AddInvariantChecker<CounterChecker>();

  EXPECT_EQ(checker->run_count(), 0);

  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(opt.Run(p.get(), OptimizationPassOptions(), &results, context),
              IsOkAndHolds(false));
  EXPECT_EQ(results.total_invocations, 0);

  // The invariant checker runs at the beginning of each compound pass.
  EXPECT_EQ(checker->run_count(), 1);
}

TEST_F(PassBaseTest, NopCompoundPass) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);

  auto* checker = opt.AddInvariantChecker<CounterChecker>();

  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(opt.Run(p.get(), OptimizationPassOptions(), &results, context),
              IsOkAndHolds(false));
  EXPECT_EQ(results.total_invocations, 5);

  // The invariant checker runs at the beginning of each compound pass but not
  // after the individual passes because the passes don't change the IR.
  EXPECT_EQ(checker->run_count(), 1);
}

TEST_F(PassBaseTest, SimpleCompoundPass) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass opt("my_compound", "My Compound Pass");
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/42);
  opt.Add<DeadCodeEliminationPass>();
  auto* checker = opt.AddInvariantChecker<CounterChecker>();

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(opt.Run(p.get(), options, &results, context), IsOkAndHolds(true));
  EXPECT_EQ(results.total_invocations, 3);

  EXPECT_EQ(results.invocation.pass_name, "my_compound");
  EXPECT_TRUE(results.invocation.ir_changed);
  EXPECT_EQ(results.invocation.metrics.nodes_added, 42);
  EXPECT_EQ(results.invocation.metrics.nodes_removed, 42);

  EXPECT_EQ(results.invocation.nested_invocations.size(), 3);

  EXPECT_EQ(results.invocation.nested_invocations[0].pass_name, "node_adder_0");
  EXPECT_FALSE(results.invocation.nested_invocations[0].ir_changed);
  EXPECT_EQ(results.invocation.nested_invocations[0].metrics.nodes_added, 0);
  EXPECT_EQ(results.invocation.nested_invocations[0].metrics.nodes_removed, 0);

  EXPECT_EQ(results.invocation.nested_invocations[1].pass_name,
            "node_adder_42");
  EXPECT_TRUE(results.invocation.nested_invocations[1].ir_changed);
  EXPECT_EQ(results.invocation.nested_invocations[1].metrics.nodes_added, 42);
  EXPECT_EQ(results.invocation.nested_invocations[1].metrics.nodes_removed, 0);

  EXPECT_EQ(results.invocation.nested_invocations[2].pass_name, "dce");
  EXPECT_TRUE(results.invocation.nested_invocations[2].ir_changed);
  EXPECT_EQ(results.invocation.nested_invocations[2].metrics.nodes_added, 0);
  EXPECT_EQ(results.invocation.nested_invocations[2].metrics.nodes_removed, 42);

  EXPECT_EQ(results.total_invocations, 3);

  // The invariant checker runs at the beginning of the compound pass and after
  // each pass which changes the IR.
  EXPECT_EQ(checker->run_count(), 3);
}

TEST_F(PassBaseTest, NestedCompoundPass) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass top("top_compound", "Top Compound Pass");
  top.Add<NodeAdderPass>(/*nodes_to_add=*/0);

  auto* top_checker = top.AddInvariantChecker<CounterChecker>();

  OptimizationCompoundPass* sub0 =
      top.Add<OptimizationCompoundPass>("subcompound0", "Subcompound Pass 0");
  sub0->Add<NodeAdderPass>(/*nodes_to_add=*/10);

  OptimizationCompoundPass* sub1 =
      top.Add<OptimizationCompoundPass>("subcompound1", "Subcompound Pass 0");
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/0);

  auto* sub1_checker = sub1->AddInvariantChecker<CounterChecker>();

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(top.Run(p.get(), options, &results, context), IsOkAndHolds(true));

  EXPECT_EQ(results.invocation.pass_name, "top_compound");
  EXPECT_TRUE(results.invocation.ir_changed);
  EXPECT_EQ(results.invocation.metrics.nodes_added, 210);
  EXPECT_EQ(results.invocation.metrics.nodes_removed, 0);

  EXPECT_EQ(results.invocation.nested_invocations.size(), 3);
  EXPECT_EQ(results.invocation.nested_invocations[0].pass_name, "node_adder_0");
  EXPECT_EQ(results.invocation.nested_invocations[1].pass_name, "subcompound0");
  EXPECT_EQ(results.invocation.nested_invocations[2].pass_name, "subcompound1");

  const PassInvocation& sub0_invocation =
      results.invocation.nested_invocations[1];
  EXPECT_TRUE(sub0_invocation.ir_changed);
  EXPECT_EQ(sub0_invocation.metrics.nodes_added, 10);
  EXPECT_EQ(sub0_invocation.metrics.nodes_removed, 0);
  EXPECT_EQ(sub0_invocation.nested_invocations.size(), 1);
  EXPECT_EQ(sub0_invocation.nested_invocations[0].pass_name, "node_adder_10");

  const PassInvocation& sub1_invocation =
      results.invocation.nested_invocations[2];
  EXPECT_TRUE(sub1_invocation.ir_changed);
  EXPECT_EQ(sub1_invocation.metrics.nodes_added, 200);
  EXPECT_EQ(sub1_invocation.metrics.nodes_removed, 0);
  EXPECT_EQ(sub1_invocation.nested_invocations.size(), 3);
  EXPECT_EQ(sub1_invocation.nested_invocations[0].pass_name, "node_adder_100");
  EXPECT_EQ(sub1_invocation.nested_invocations[1].pass_name, "node_adder_100");
  EXPECT_EQ(sub1_invocation.nested_invocations[2].pass_name, "node_adder_0");

  EXPECT_EQ(results.total_invocations, 5);

  // The invariant checker runs at the beginning of each compound pass
  // and after each pass which changes the IR.
  EXPECT_EQ(top_checker->run_count(), 6);

  // Invariant checkers run only within (and below) the compound pass they are
  // added.
  EXPECT_EQ(sub1_checker->run_count(), 3);
}

TEST_F(PassBaseTest, DumpIrWithNestedPasses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());

  OptimizationCompoundPass top("top_compound", "Top Compound Pass");
  top.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  OptimizationCompoundPass* sub0 =
      top.Add<OptimizationCompoundPass>("subcompound0", "Subcompound Pass 0");
  sub0->Add<NodeAdderPass>(/*nodes_to_add=*/10);
  OptimizationCompoundPass* sub1 =
      top.Add<OptimizationCompoundPass>("subcompound1", "Subcompound Pass 0");
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  sub1->Add<NodeAdderPass>(/*nodes_to_add=*/0);

  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory dump_dir, TempDirectory::Create());

  OptimizationPassOptions options;
  options.ir_dump_path = dump_dir.path();
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(top.Run(p.get(), options, &results, context), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::filesystem::path> dump_paths,
                           GetDirectoryEntries(options.ir_dump_path));
  std::vector<std::string> dump_filenames;
  dump_filenames.reserve(dump_paths.size());
  for (const std::filesystem::path& path : dump_paths) {
    dump_filenames.push_back(path.filename().string());
  }
  EXPECT_THAT(
      dump_filenames,
      UnorderedElementsAre(
          "DumpIrWithNestedPasses.00004.after_node_adder_0.subcompound1."
          "unchanged.ir",
          "DumpIrWithNestedPasses.00003.after_node_adder_100.subcompound1."
          "changed.ir",
          "DumpIrWithNestedPasses.00002.after_node_adder_100.subcompound1."
          "changed.ir",
          "DumpIrWithNestedPasses.00001.after_node_adder_10.subcompound0."
          "changed.ir",
          "DumpIrWithNestedPasses.00000.after_node_adder_0.top_compound."
          "unchanged.ir",
          "DumpIrWithNestedPasses.00000.start.top_compound.unchanged.ir"));
}

TEST_F(PassBaseTest, MetricsTest) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass opt("opt", "opt");
  opt.Add<NodeAdderPass>(/*n=*/10);
  opt.Add<NodeReplacerPass>(/*n=*/100);

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(opt.Run(p.get(), options, &results, context), IsOkAndHolds(true));
  EXPECT_EQ(results.total_invocations, 2);

  EXPECT_EQ(results.invocation.nested_invocations.size(), 2);
  const TransformMetrics& adder_metrics =
      results.invocation.nested_invocations[0].metrics;
  EXPECT_EQ(adder_metrics.nodes_added, 10);
  EXPECT_EQ(adder_metrics.nodes_removed, 0);
  EXPECT_EQ(adder_metrics.nodes_replaced, 0);

  const TransformMetrics& replacer_metrics =
      results.invocation.nested_invocations[1].metrics;
  EXPECT_EQ(replacer_metrics.nodes_added, 100);
  EXPECT_EQ(replacer_metrics.nodes_removed, 0);
  EXPECT_EQ(replacer_metrics.nodes_replaced, 100);

  const TransformMetrics& top_metrics = results.invocation.metrics;
  EXPECT_EQ(top_metrics.nodes_added, 110);
  EXPECT_EQ(top_metrics.nodes_removed, 0);
  EXPECT_EQ(top_metrics.nodes_replaced, 100);
}

}  // namespace
}  // namespace xls
