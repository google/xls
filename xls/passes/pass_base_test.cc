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

#include "google/protobuf/duration.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/proto_test_utils.h"
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
#include "xls/passes/pass_metrics.pb.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
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
// not cause the number to wrap around or hit the max.
class LevelUpPass : public OptimizationFunctionBasePass {
 public:
  LevelUpPass(std::optional<Bits> max = std::nullopt)
      : OptimizationFunctionBasePass("level_up", "Level up Pass"),
        max_(std::move(max)) {}
  ~LevelUpPass() override = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    bool changed = false;
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes,
                         context.TopoSort(f));
    for (Node* n : topo_sort_nodes) {
      if (n->Is<Literal>() && n->GetType()->IsBits() &&
          !IsTooBig(n->As<Literal>()->value().bits())) {
        changed = true;
        XLS_RETURN_IF_ERROR(
            n->ReplaceUsesWithNew<Literal>(
                 Value(bits_ops::Increment(n->As<Literal>()->value().bits())))
                .status());
      }
    }
    return changed;
  }

 private:
  bool IsTooBig(const Bits& bits) const {
    if (!max_.has_value()) {
      return bits.IsAllOnes();
    }
    int64_t max_bit_count = std::max(max_->bit_count(), bits.bit_count());
    return bits_ops::UGreaterThanOrEqual(
        bits_ops::ZeroExtend(bits, max_bit_count),
        bits_ops::ZeroExtend(*max_, max_bit_count));
  }

  std::optional<Bits> max_;
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

// auto DceInvoke() { return Field(&PassInvocation::pass_name, Eq("dce")); }
// auto LevelUpInvoke() {
//   return Field(&PassInvocation::pass_name, Eq("level_up"));
// }

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

TEST_F(PassBaseTest, BisectLimitDoesNotCauseErrors) {
  for (int64_t i = 1; i < 15; ++i) {
    auto p = CreatePackage();
    FunctionBuilder fb(TestName(), p.get());
    fb.Literal(UBits(0, 16));
    XLS_ASSERT_OK(fb.Build().status());
    OptimizationCompoundPass opt("opt", "opt");
    opt.Add<LevelUpPass>();
    opt.Add<DeadCodeEliminationPass>();
    auto inner =
        std::make_unique<OptimizationFixedPointCompoundPass>("inner", "inner");
    inner->Add<LevelUpPass>();
    inner->Add<DeadCodeEliminationPass>();
    opt.AddOwned(std::move(inner));
    PassResults results;
    OptimizationContext context;
    EXPECT_THAT(
        opt.Run(p.get(),
                OptimizationPassOptions(PassOptionsBase{.bisect_limit = i}),
                &results, context),
        IsOk())
        << "iteration: " << i;
  }
}

TEST_F(PassBaseTest, BisectLimitMid) {
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
    fp->Add<LevelUpPass>();
    fp->Add<DeadCodeEliminationPass>();
    fp->Add<LevelUpPass>();
    fp->Add<DeadCodeEliminationPass>();
    opt.AddOwned(std::move(fp));
  }
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 11}),
              &results, context),
      IsOk());
  // 1 for opt, 2 for fixed runs, 8 for passes, 4 are level up.
  EXPECT_THAT(f->return_value(), m::Literal(UBits(4, 16)));
  EXPECT_EQ(f->node_count(), 1);

  PassPipelineMetricsProto proto = results.ToProto();
  EXPECT_THAT(
      proto, proto_testing::Partially(proto_testing::EqualsProto(R"pb(
        total_passes: 11
        pass_metrics {
          pass_name: "opt"
          changed: 1
          pass_numbers: 0
          nested_results {
            pass_name: "fixed"
            changed: 1
            pass_numbers: 1
            pass_numbers: 8
            fixed_point_iterations: 2
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 2 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 3 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 4 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 5 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 6 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 7 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 9 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 10 }
          }
        }
      )pb")));
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
  // Should run opt Level DCE Level DCE Level DCE Level DCE
  EXPECT_THAT(
      opt.Run(p.get(),
              OptimizationPassOptions(PassOptionsBase{.bisect_limit = 9}),
              &results, context),
      IsOk());
  EXPECT_THAT(f->return_value(), m::Literal(UBits(4, 64)));
  EXPECT_EQ(f->node_count(), 1);
  PassPipelineMetricsProto proto = results.ToProto();
  EXPECT_THAT(
      proto, proto_testing::Partially(proto_testing::EqualsProto(R"pb(
        total_passes: 9
        pass_metrics {
          pass_name: "opt"
          changed: 1
          pass_numbers: 0
          nested_results { pass_name: "level_up" changed: 1 pass_numbers: 1 }
          nested_results { pass_name: "dce" changed: 1 pass_numbers: 2 }
          nested_results { pass_name: "level_up" changed: 1 pass_numbers: 3 }
          nested_results { pass_name: "dce" changed: 1 pass_numbers: 4 }
          nested_results { pass_name: "level_up" changed: 1 pass_numbers: 5 }
          nested_results { pass_name: "dce" changed: 1 pass_numbers: 6 }
          nested_results { pass_name: "level_up" changed: 1 pass_numbers: 7 }
          nested_results { pass_name: "dce" changed: 1 pass_numbers: 8 }
        }
      )pb")));
  EXPECT_THAT(f->return_value(), m::Literal(UBits(4, 64)));
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
  // 1 for opt, 5 for fixed runs, 10 for passes, 5 are level up.
  EXPECT_THAT(f->return_value(), m::Literal(UBits(5, 16)));
  EXPECT_EQ(f->node_count(), 1);

  PassPipelineMetricsProto proto = results.ToProto();
  EXPECT_THAT(
      proto, proto_testing::Partially(proto_testing::EqualsProto(R"pb(
        total_passes: 16
        pass_metrics {
          pass_name: "opt"
          changed: 1
          pass_numbers: 0
          nested_results {
            pass_name: "fixed"
            changed: 1
            pass_numbers: 1
            pass_numbers: 4
            pass_numbers: 7
            pass_numbers: 10
            pass_numbers: 13
            fixed_point_iterations: 5
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 2 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 3 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 5 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 6 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 8 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 9 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 11 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 12 }
            nested_results { pass_name: "level_up" changed: 1 pass_numbers: 14 }
            nested_results { pass_name: "dce" changed: 1 pass_numbers: 15 }
          }
        }
      )pb")));
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
  EXPECT_THAT(results.root_invocation().nested_invocations(), IsEmpty());
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
  EXPECT_EQ(results.total_invocations(), 1);

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
  // opt and the 5 adders
  EXPECT_EQ(results.total_invocations(), 6);

  // The invariant checker runs at the beginning of each compound pass but not
  // after the individual passes because the passes don't change the IR.
  EXPECT_EQ(checker->run_count(), 1);
}

TEST_F(PassBaseTest, SimpleCompoundPasses) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());
  OptimizationCompoundPass opt("my_compound_1", "My Compound Pass");
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/42);
  opt.Add<DeadCodeEliminationPass>();
  auto opt2 = std::make_unique<OptimizationCompoundPass>("my_compound_2",
                                                         "My Compound Pass 2");
  opt2->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  opt2->Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.AddOwned(std::move(opt2));
  opt.Add<DeadCodeEliminationPass>();

  auto* checker = opt.AddInvariantChecker<CounterChecker>();

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(opt.Run(p.get(), options, &results, context), IsOkAndHolds(true));

  auto proto = results.ToProto();
  EXPECT_THAT(proto, proto_testing::Partially(proto_testing::EqualsProto(R"pb(
                total_passes: 8
                pass_metrics {
                  pass_name: "my_compound_1"
                  changed: 1
                  pass_numbers: 0
                  nested_results {
                    pass_name: "node_adder_0"
                    changed: 0
                    pass_numbers: 1
                  }
                  nested_results {
                    pass_name: "node_adder_42"
                    changed: 1
                    pass_numbers: 2
                  }
                  nested_results { pass_name: "dce" changed: 1 pass_numbers: 3 }
                  nested_results {
                    pass_name: "my_compound_2"
                    changed: 1
                    pass_numbers: 4
                    nested_results {
                      pass_name: "node_adder_100"
                      changed: 1
                      pass_numbers: 5
                    }
                    nested_results {
                      pass_name: "node_adder_0"
                      changed: 0
                      pass_numbers: 6
                    }
                  }
                  nested_results { pass_name: "dce" changed: 1 pass_numbers: 7 }
                }
              )pb")));

  // The invariant checker runs at the beginning of the compound pass and after
  // each pass which changes the IR.
  EXPECT_EQ(checker->run_count(), 6);
}

TEST_F(PassBaseTest, InvariantCheckerAddedEarly) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());

  OptimizationCompoundPass opt("my_compound_1", "My Compound Pass");

  auto* checker = opt.AddInvariantChecker<CounterChecker>();

  opt.Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/42);
  opt.Add<DeadCodeEliminationPass>();
  auto opt2 = std::make_unique<OptimizationCompoundPass>("my_compound_2",
                                                         "My Compound Pass 2");
  opt2->Add<NodeAdderPass>(/*nodes_to_add=*/100);
  opt2->Add<NodeAdderPass>(/*nodes_to_add=*/0);
  opt.AddOwned(std::move(opt2));
  opt.Add<DeadCodeEliminationPass>();

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(opt.Run(p.get(), options, &results, context), IsOkAndHolds(true));

  auto proto = results.ToProto();
  EXPECT_THAT(proto, proto_testing::Partially(proto_testing::EqualsProto(R"pb(
                total_passes: 8
                pass_metrics {
                  pass_name: "my_compound_1"
                  changed: 1
                  pass_numbers: 0
                  nested_results {
                    pass_name: "node_adder_0"
                    changed: 0
                    pass_numbers: 1
                  }
                  nested_results {
                    pass_name: "node_adder_42"
                    changed: 1
                    pass_numbers: 2
                  }
                  nested_results { pass_name: "dce" changed: 1 pass_numbers: 3 }
                  nested_results {
                    pass_name: "my_compound_2"
                    changed: 1
                    pass_numbers: 4
                    nested_results {
                      pass_name: "node_adder_100"
                      changed: 1
                      pass_numbers: 5
                    }
                    nested_results {
                      pass_name: "node_adder_0"
                      changed: 0
                      pass_numbers: 6
                    }
                  }
                  nested_results { pass_name: "dce" changed: 1 pass_numbers: 7 }
                }
              )pb")));

  // The invariant checker runs at the beginning of the compound pass and after
  // each pass which changes the IR.
  EXPECT_EQ(checker->run_count(), 6);
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

  EXPECT_THAT(results.ToProto(),
              proto_testing::Partially(proto_testing::EqualsProto(R"pb(
                total_passes: 8
                pass_metrics {
                  pass_name: "top_compound"
                  changed: 1
                  pass_numbers: 0
                  nested_results {
                    pass_name: "node_adder_0"
                    changed: 0
                    pass_numbers: 1
                  }
                  nested_results {
                    pass_name: "subcompound0"
                    changed: 1
                    pass_numbers: 2
                    nested_results {
                      pass_name: "node_adder_10"
                      changed: 1
                      pass_numbers: 3
                    }
                  }
                  nested_results {
                    pass_name: "subcompound1"
                    changed: 1
                    pass_numbers: 4
                    nested_results {
                      pass_name: "node_adder_100"
                      changed: 1
                      pass_numbers: 5
                    }
                    nested_results {
                      pass_name: "node_adder_100"
                      changed: 1
                      pass_numbers: 6
                    }
                    nested_results {
                      pass_name: "node_adder_0"
                      changed: 0
                      pass_numbers: 7
                    }
                  }
                }
              )pb")));

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
          "DumpIrWithNestedPasses.00007.after_top_compound.root.changed.ir",
          "DumpIrWithNestedPasses.00006.after_subcompound1.top_compound."
          "changed.ir",
          "DumpIrWithNestedPasses.00005.after_node_adder_0.subcompound1."
          "unchanged.ir",
          "DumpIrWithNestedPasses.00004.after_node_adder_100.subcompound1."
          "changed.ir",
          "DumpIrWithNestedPasses.00003.after_node_adder_100.subcompound1."
          "changed.ir",
          "DumpIrWithNestedPasses.00002.after_subcompound0.top_compound."
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
  EXPECT_THAT(results.ToProto(),
              proto_testing::Partially(proto_testing::EqualsProto(R"pb(
                total_passes: 3
                pass_metrics {
                  pass_name: "opt"
                  changed: 1
                  pass_numbers: 0
                  transformation_metrics {
                    nodes_added: 110
                    nodes_removed: 0
                    nodes_replaced: 100
                    operands_replaced: 0
                    operands_removed: 0
                  }
                  nested_results {
                    pass_name: "node_adder_10"
                    changed: 1
                    pass_numbers: 1
                    transformation_metrics {
                      nodes_added: 10
                      nodes_removed: 0
                      nodes_replaced: 0
                      operands_replaced: 0
                      operands_removed: 0
                    }
                  }
                  nested_results {
                    pass_name: "replacer_100"
                    changed: 1
                    pass_numbers: 2
                    transformation_metrics {
                      nodes_added: 100
                      nodes_removed: 0
                      nodes_replaced: 100
                      operands_replaced: 0
                      operands_removed: 0
                    }
                  }
                }
              )pb")));
}

class MutatingIdempotentPass : public OptimizationFunctionBasePass {
 public:
  explicit MutatingIdempotentPass(int64_t nodes_to_add)
      : OptimizationFunctionBasePass("mutating_idempotent",
                                     "Mutating Idempotent Pass"),
        nodes_to_add_(nodes_to_add) {}
  ~MutatingIdempotentPass() override = default;

  bool IsIdempotent() const override { return true; }

  RedundancyGuard GetRedundancyGuard(
      const OptimizationPassOptions& options,
      OptimizationContext& context) const override {
    return RedundancyGuard::CanSkip(
        absl::StrCat("nodes_to_add=", nodes_to_add_));
  }

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    bool changed = false;
    for (int64_t i = 0; i < nodes_to_add_; ++i) {
      std::string node_name = absl::StrCat("added_node_", i);
      if (f->HasNode(node_name)) {
        continue;
      }
      XLS_RETURN_IF_ERROR(f->MakeNodeWithName<Literal>(
                               SourceInfo(), Value(UBits(0, 32)), node_name)
                              .status());
      changed = true;
    }
    return changed;
  }

  int64_t nodes_to_add_;
};

TEST_F(PassBaseTest, IdempotentPassTracking) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Literal(UBits(0, 64));
  XLS_ASSERT_OK(fb.Build());

  OptimizationCompoundPass opt("opt", "opt");
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/1);
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/1);
  opt.Add<NodeAdderPass>(/*nodes_to_add=*/1);
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/1);
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/1);
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/0);
  opt.Add<MutatingIdempotentPass>(/*nodes_to_add=*/2);

  PassResults results;
  OptimizationContext context;
  OptimizationPassOptions options;
  EXPECT_THAT(opt.Run(p.get(), options, &results, context), IsOkAndHolds(true));

  PassPipelineMetricsProto metrics_proto = results.ToProto();
  EXPECT_THAT(metrics_proto,
              proto_testing::Partially(proto_testing::EqualsProto(R"pb(
                total_passes: 8
                pass_metrics {
                  pass_name: "opt"
                  changed: 1
                  pass_numbers: 0
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 1
                    pass_numbers: 1
                  }
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 0
                    pass_numbers: 2
                    skip_reason: SKIP_REASON_KNOWN_REDUNDANT
                  }
                  nested_results {
                    pass_name: "node_adder_1"
                    changed: 1
                    pass_numbers: 3
                  }
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 0
                    pass_numbers: 4
                    skip_reason: SKIP_REASON_NOT_SKIPPED
                  }
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 0
                    pass_numbers: 5
                    skip_reason: SKIP_REASON_KNOWN_REDUNDANT
                  }
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 0  # This version hasn't run, so we don't skip it -
                                # but it won't mutate the IR.
                    pass_numbers: 6
                    skip_reason: SKIP_REASON_NOT_SKIPPED
                  }
                  nested_results {
                    pass_name: "mutating_idempotent"
                    changed: 1  # This version will actually mutate the IR.
                    pass_numbers: 7
                  }
                }
              )pb")));
}

}  // namespace
}  // namespace xls
