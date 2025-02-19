// Copyright 2020 The XLS Authors
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

#include "xls/passes/optimization_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class DummyPass : public OptimizationPass {
 public:
  // `change` indicates whether the pass will return that the IR was changed.
  DummyPass(std::string short_name, std::string long_name, bool change = false)
      : OptimizationPass(short_name, long_name), change_(change) {}

  absl::StatusOr<bool> RunInternal(
      Package* p, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext* context) const override {
    return change_;
  }

 private:
  bool change_;
};

// BuildShift0 builds an IR that initially looks like this:
//   sub
//     shrl
//       param(x)
//       sub
//         sub
//           lit(3)
//           lit(2)
//         lit(1)
//     lit(0)
//
// It also starts out with some dead code.
// Simplifications should reduce the whole tree to:
//    param(x)
//
std::pair<std::unique_ptr<Package>, Function*> BuildShift0() {
  auto m = std::make_unique<Package>("m");
  FunctionBuilder b("simple_arith", m.get());
  Type* bits_32 = m->GetBitsType(32);
  auto x = b.Param("x", bits_32);
  auto y = b.Param("y", bits_32);
  auto imm_0 = b.Literal(UBits(0, /*bit_count=*/32));
  auto imm_1 = b.Literal(UBits(1, /*bit_count=*/32));
  auto imm_2 = b.Literal(UBits(2, /*bit_count=*/32));
  auto imm_3 = b.Literal(UBits(3, /*bit_count=*/32));
  auto deadcode = (y - imm_0);
  (void)deadcode;
  auto imm = imm_3 - imm_2 - imm_1;
  auto result = ((x >> imm) - imm_0);
  absl::StatusOr<Function*> f_or_status = b.BuildWithReturnValue(result);
  CHECK_OK(f_or_status.status());
  return {std::move(m), *f_or_status};
}

TEST(PassesTest, AddPasses) {
  std::unique_ptr<Package> p = BuildShift0().first;

  OptimizationCompoundPass pass_mgr("TOP", "Top level pass manager");

  pass_mgr.Add<DummyPass>("d1", "Dummy Pass 1");
  pass_mgr.Add<DummyPass>("d2", "Dummy Pass 2");
  pass_mgr.Add<DummyPass>("d3", "Dummy Pass 3");

  auto comp_pass =
      pass_mgr.Add<OptimizationCompoundPass>("C1", "Some Compound Pass");
  comp_pass->Add<DummyPass>("d4", "Dummy Pass 4");
  comp_pass->Add<DummyPass>("d5", "Dummy Pass 5");

  pass_mgr.Add<DummyPass>("d6", "Dummy Pass 6");

  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(
      pass_mgr.Run(p.get(), OptimizationPassOptions(), &results, &context),
      IsOkAndHolds(false));
  std::vector<std::string> invocation_names;
  invocation_names.reserve(results.invocations.size());
  for (const PassInvocation& invocation : results.invocations) {
    invocation_names.push_back(invocation.pass_name);
  }
  EXPECT_THAT(invocation_names,
              ElementsAre("d1", "d2", "d3", "d4", "d5", "d6"));
}

// Invariant checker which returns an error if the package has function with a
// particular name.
class PackageNameChecker : public OptimizationInvariantChecker {
 public:
  explicit PackageNameChecker(std::string_view str) : str_(str) {}

  absl::Status Run(Package* package, const OptimizationPassOptions& options,
                   PassResults* results) const override {
    for (auto& function : package->functions()) {
      if (function->name() == str_) {
        return absl::InternalError(
            absl::StrFormat("Function has name '%s'", str_));
      }
    }
    return absl::OkStatus();
  }

 private:
  std::string str_;
};

// A trivial package used in the invariant tests.
const char kInvariantTesterPackage[] = R"(
package invariant_tester

fn foo(x:bits[8]) -> bits[8] {
  ret neg.2: bits[8] = neg(x)
}
)";

// Makes and returns a nesting of compound passes.
std::unique_ptr<OptimizationCompoundPass> MakeNestedPasses() {
  auto top = std::make_unique<OptimizationCompoundPass>(
      "top", "Top level pass manager");
  auto comp_pass =
      top->Add<OptimizationCompoundPass>("comp_pass", "compound pass");
  auto nested_comp_pass = comp_pass->Add<OptimizationCompoundPass>(
      "nested_comp_pass", "nested pass");
  nested_comp_pass->Add<DummyPass>("dummy", "blah");
  return top;
}

TEST(PassesTest, InvariantChecker) {
  // Verify the test invariant checker works as expected.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  PassResults results;
  EXPECT_THAT(PackageNameChecker("foo")
                  .Run(p.get(), OptimizationPassOptions(), &results)
                  .message(),
              HasSubstr("Function has name 'foo'"));
  XLS_EXPECT_OK(PackageNameChecker("bar").Run(
      p.get(), OptimizationPassOptions(), &results));
}

TEST(PassesTest, RunWithNoInvariantChecker) {
  // Verify no error when running with no invariant checkers added.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<OptimizationCompoundPass> top = MakeNestedPasses();
  PassResults results;
  OptimizationContext context;
  XLS_EXPECT_OK(top->Run(p.get(), OptimizationPassOptions(), &results, &context)
                    .status());
}

TEST(PassesTest, RunWithPassingInvariantChecker) {
  // With an invariant checker that always passes, running should return ok.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<OptimizationCompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("bar");
  PassResults results;
  OptimizationContext context;
  XLS_EXPECT_OK(top->Run(p.get(), OptimizationPassOptions(), &results, &context)
                    .status());
}

TEST(PassesTest, RunWithFailingInvariantChecker) {
  // With an invariant checker that fails, should return an error.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<OptimizationCompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("foo");
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(top->Run(p.get(), OptimizationPassOptions(), &results, &context),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'foo'; [start of compound "
                                 "pass 'Top level pass manager']")));
}

TEST(PassesTest, RunWithFailingNestedInvariantChecker) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  auto top = std::make_unique<OptimizationCompoundPass>(
      "top", "Top level pass manager");
  top->AddInvariantChecker<PackageNameChecker>("bar");
  auto nested_pass =
      top->Add<OptimizationCompoundPass>("comp_pass", "nested pass");
  nested_pass->AddInvariantChecker<PackageNameChecker>("foo");
  PassResults results;
  OptimizationContext context;
  EXPECT_THAT(top->Run(p.get(), OptimizationPassOptions(), &results, &context),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'foo'; [start of compound "
                                 "pass 'nested pass']")));
}

// Pass which adds a function of a particular name
class FunctionAdderPass : public OptimizationPass {
 public:
  explicit FunctionAdderPass(std::string_view name)
      : OptimizationPass("function_adder",
                         absl::StrCat("Adds function named ", name)),
        name_(name) {}

  // Adds a function named 'str_' to the package.
  absl::StatusOr<bool> RunInternal(
      Package* package, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext* context) const override {
    const char format_string[] =
        R"(
fn %s() -> bits[32] {
 ret forty_two: bits[32] = literal(value=42)
}
)";
    XLS_RETURN_IF_ERROR(
        Parser::ParseFunction(absl::StrFormat(format_string, name_), package)
            .status());
    return true;
  }

 private:
  std::string name_;
};

TEST(PassesTest, InvariantCheckerFailsAfterPass) {
  // Verify the error message when the invariant checker fails only after
  // running a particular pass.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<OptimizationCompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("bar");
  PassResults results;
  OptimizationContext context;
  XLS_EXPECT_OK(top->Run(p.get(), OptimizationPassOptions(), &results, &context)
                    .status());
  down_cast<OptimizationCompoundPass*>(top->passes()[0])
      ->Add<FunctionAdderPass>("bar");
  auto result =
      top->Run(p.get(), OptimizationPassOptions(), &results, &context);
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'bar'; [after 'Adds "
                                 "function named bar' pass")))
      << result.status();
}

// Invariant checker which counts the number of times it was invoked.
class CounterChecker : public OptimizationInvariantChecker {
 public:
  explicit CounterChecker() = default;

  absl::Status Run(Package* package, const OptimizationPassOptions& options,
                   PassResults* results) const override {
    counter_++;
    return absl::OkStatus();
  }

  int64_t run_count() const { return counter_; }

 private:
  mutable int64_t counter_ = 0;
};

TEST(PassesTest, InvariantCheckerOnlyRunsAfterChangedPasses) {
  // Verify the error message when the invariant checker fails only after
  // running a particular pass.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  OptimizationCompoundPass pass_mgr("TOP", "Top level pass manager");
  pass_mgr.Add<DummyPass>("d1", "Dummy pass 1", /*change=*/false);
  pass_mgr.Add<DummyPass>("d2", "Dummy pass 2", /*change=*/false);
  pass_mgr.Add<DummyPass>("d3", "Dummy pass 3", /*change=*/true);
  pass_mgr.Add<DummyPass>("d4", "Dummy pass 4", /*change=*/false);
  pass_mgr.Add<DummyPass>("d5", "Dummy pass 5", /*change=*/true);
  pass_mgr.Add<DummyPass>("d6", "Dummy pass 6", /*change=*/false);

  CounterChecker* checker = pass_mgr.AddInvariantChecker<CounterChecker>();
  EXPECT_EQ(checker->run_count(), 0);

  PassResults results;
  OptimizationContext context;
  XLS_EXPECT_OK(
      pass_mgr.Run(p.get(), OptimizationPassOptions(), &results, &context)
          .status());

  // Checkers should run once at the beginning then only after a passes that
  // indicate the IR has been changed.
  EXPECT_EQ(checker->run_count(), 3);
}

// Pass which just records that it was run via a shared vector of strings.
class RecordingPass : public OptimizationPass {
 public:
  RecordingPass(std::string name, std::vector<std::string>* record)
      : OptimizationPass(name, name), record_(record) {}

  absl::StatusOr<bool> RunInternal(
      Package* p, const OptimizationPassOptions& options, PassResults* results,
      OptimizationContext* context) const override {
    record_->push_back(short_name());
    return false;
  }

 private:
  std::vector<std::string>* record_;
};

TEST(PassesTest, SkipPassesOption) {
  std::unique_ptr<Package> p = BuildShift0().first;

  std::vector<std::string> record;
  OptimizationCompoundPass compound_0("compound_0", "Compound pass 0");
  compound_0.Add<RecordingPass>("foo", &record);
  compound_0.Add<RecordingPass>("bar", &record);
  auto compound_1 =
      compound_0.Add<OptimizationCompoundPass>("compound_1", "Compound pass 1");
  compound_1->Add<RecordingPass>("qux", &record);
  compound_1->Add<RecordingPass>("foo", &record);

  {
    OptimizationPassOptions options;
    options.skip_passes = {"blah"};
    PassResults results;
    OptimizationContext context;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results, &context),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("foo", "bar", "qux", "foo"));
  }

  {
    OptimizationPassOptions options;
    options.skip_passes = {"foo"};
    PassResults results;
    OptimizationContext context;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results, &context),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("bar", "qux"));
  }

  {
    OptimizationPassOptions options;
    options.skip_passes = {"foo", "qux"};
    PassResults results;
    OptimizationContext context;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results, &context),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("bar"));
  }
}

class RecordPass : public OptimizationFunctionBasePass {
 public:
  explicit RecordPass(std::vector<int64_t>* record)
      : OptimizationFunctionBasePass("record", "record opt level"),
        record_(record) {}
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext* context) const override {
    record_->push_back(options.opt_level);
    return false;
  }

 private:
  std::vector<int64_t>* record_;
};
TEST(PassesTest, CapOptLevel) {
  // Verify the test invariant checker works as expected.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::vector<int64_t> record;
  OptimizationCompoundPass passes("test", "test");
  passes.Add<RecordPass>(&record);
  passes.Add<CapOptLevel<kMaxOptLevel + 1, RecordPass>>(&record);
  PassResults res;
  OptimizationContext ctx;
  {
    ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions(), &res, &ctx),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre(kMaxOptLevel, kMaxOptLevel));
  }
  record.clear();
  {
    ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions().WithOptLevel(2),
                           &res, &ctx),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre(2, 2));
  }
  record.clear();
  {
    ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions().WithOptLevel(12),
                           &res, &ctx),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre(12, kMaxOptLevel + 1));
  }
}

TEST(PassesTest, OptLevelIsAtLeast) {
  // Verify the test invariant checker works as expected.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::vector<int64_t> record;
  OptimizationCompoundPass passes("test", "test");
  passes.Add<RecordPass>(&record);
  passes.Add<IfOptLevelAtLeast<1, RecordPass>>(&record);
  PassResults res;
  OptimizationContext ctx;
  {
    ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions(), &res, &ctx),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre(kMaxOptLevel, kMaxOptLevel));
  }
  record.clear();
  {
    ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions().WithOptLevel(0),
                           &res, &ctx),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre(0));
  }
}

TEST(PassesTest, WithOptLevel) {
  // Verify the test invariant checker works as expected.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::vector<int64_t> record;
  OptimizationCompoundPass passes("test", "test");
  passes.Add<RecordPass>(&record);
  passes.Add<WithOptLevel<1, RecordPass>>(&record);
  PassResults res;
  OptimizationContext ctx;
  ASSERT_THAT(passes.Run(p.get(), OptimizationPassOptions(), &res, &ctx),
              IsOkAndHolds(false));
  EXPECT_THAT(record, ElementsAre(kMaxOptLevel, 1));
}

class NaiveDcePass : public OptimizationFunctionBasePass {
 public:
  NaiveDcePass() : OptimizationFunctionBasePass("naive_dce", "naive dce") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext* context) const override {
    return TransformNodesToFixedPoint(f, [](Node* n) -> absl::StatusOr<bool> {
      if (!n->Is<Param>() && n->IsDead()) {
        XLS_RETURN_IF_ERROR(n->function_base()->RemoveNode(n));
        return true;
      }
      return false;
    });
  }
};

TEST(PassesTest, TestTransformNodesToFixedPointWhileRemovingNodes) {
  auto m = std::make_unique<Package>("m");
  FunctionBuilder fb("test", m.get());
  BValue x = fb.Param("x", m->GetBitsType(32));
  BValue a = fb.Not(fb.Add(x, x));
  BValue b = fb.Concat({x, x, x, x, x, x});
  fb.Tuple({a, b, fb.Concat({a, x})});
  // Make parameter x the return value which means everything is dead but x.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(x));
  PassResults results;
  OptimizationContext context;
  EXPECT_EQ(f->node_count(), 6);
  ASSERT_THAT(NaiveDcePass().RunOnFunctionBase(f, OptimizationPassOptions(),
                                               &results, &context),
              IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  ASSERT_THAT(NaiveDcePass().RunOnFunctionBase(f, OptimizationPassOptions(),
                                               &results, &context),
              IsOkAndHolds(false));
}

TEST(RamDatastructuresTest, AddrWidthCorrect) {
  RamConfig config{.kind = RamKind::kAbstract, .depth = 2};
  EXPECT_EQ(config.addr_width(), 1);
  config.depth = 3;
  EXPECT_EQ(config.addr_width(), 2);
  config.depth = 4;
  EXPECT_EQ(config.addr_width(), 2);
  config.depth = 1023;
  EXPECT_EQ(config.addr_width(), 10);
  config.depth = 1024;
  EXPECT_EQ(config.addr_width(), 10);
  config.depth = 1025;
  EXPECT_EQ(config.addr_width(), 11);
}

TEST(RamDatastructuresTest, MaskWidthCorrect) {
  int64_t data_width = 32;
  RamConfig config{.kind = RamKind::kAbstract,
                   .depth = 2,
                   .word_partition_size = std::nullopt};
  EXPECT_EQ(config.mask_width(data_width), std::nullopt);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(data_width), 32);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(data_width), 16);
  config.word_partition_size = 32;
  EXPECT_EQ(config.mask_width(data_width), 1);

  data_width = 7;
  config.word_partition_size = std::nullopt;
  EXPECT_EQ(config.mask_width(data_width), std::nullopt);
  config.word_partition_size = 1;
  EXPECT_EQ(config.mask_width(data_width), 7);
  config.word_partition_size = 2;
  EXPECT_EQ(config.mask_width(data_width), 4);
  config.word_partition_size = 3;
  EXPECT_EQ(config.mask_width(data_width), 3);
  config.word_partition_size = 4;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 5;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 6;
  EXPECT_EQ(config.mask_width(data_width), 2);
  config.word_partition_size = 7;
  EXPECT_EQ(config.mask_width(data_width), 1);
}

TEST(RamDatastructuresTest, RamKindProtoTest) {
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_ABSTRACT),
              IsOkAndHolds(RamKind::kAbstract));
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_1RW),
              IsOkAndHolds(RamKind::k1RW));
  EXPECT_THAT(RamKindFromProto(RamKindProto::RAM_INVALID),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RamDatastructuresTest, RamConfigProtoTest) {
  RamConfigProto proto;
  proto.set_kind(RamKindProto::RAM_ABSTRACT);
  proto.set_depth(1024);
  XLS_EXPECT_OK(RamConfig::FromProto(proto));
  EXPECT_EQ(RamConfig::FromProto(proto)->kind, RamKind::kAbstract);
  EXPECT_EQ(RamConfig::FromProto(proto)->depth, 1024);
  EXPECT_EQ(RamConfig::FromProto(proto)->word_partition_size, std::nullopt);
  EXPECT_EQ(RamConfig::FromProto(proto)->initial_value, std::nullopt);

  proto.set_word_partition_size(1);
  XLS_EXPECT_OK(RamConfig::FromProto(proto));
  EXPECT_EQ(RamConfig::FromProto(proto)->kind, RamKind::kAbstract);
  EXPECT_EQ(RamConfig::FromProto(proto)->depth, 1024);
  EXPECT_EQ(RamConfig::FromProto(proto)->word_partition_size, 1);
  EXPECT_EQ(RamConfig::FromProto(proto)->initial_value, std::nullopt);
}

TEST(RamDatastructuresTest, RamRewriteProtoTest) {
  RamRewriteProto proto;
  proto.mutable_from_config()->set_kind(RamKindProto::RAM_ABSTRACT);
  proto.mutable_from_config()->set_depth(1024);
  proto.mutable_to_config()->set_kind(RamKindProto::RAM_1RW);
  proto.mutable_to_config()->set_depth(1024);
  proto.mutable_from_channels_logical_to_physical()->insert(
      {"read_req", "ram_read_req"});
  proto.set_to_name_prefix("ram");

  XLS_EXPECT_OK(RamRewrite::FromProto(proto));
  EXPECT_EQ(RamRewrite::FromProto(proto)->from_config.kind, RamKind::kAbstract);
  EXPECT_EQ(RamRewrite::FromProto(proto)->from_config.depth, 1024);
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_config.kind, RamKind::k1RW);
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_config.depth, 1024);
  EXPECT_EQ(
      RamRewrite::FromProto(proto)->from_channels_logical_to_physical.size(),
      1);
  EXPECT_THAT(RamRewrite::FromProto(proto)->from_channels_logical_to_physical,
              Contains(std::make_pair("read_req", "ram_read_req")));
  EXPECT_EQ(RamRewrite::FromProto(proto)->to_name_prefix, "ram");
}

TEST(RamDatastructuresTest, RamRewritesProtoTest) {
  RamRewritesProto proto;
  RamRewriteProto rewrite_proto;
  rewrite_proto.mutable_from_config()->set_kind(RamKindProto::RAM_ABSTRACT);
  rewrite_proto.mutable_from_config()->set_depth(1024);
  rewrite_proto.mutable_to_config()->set_kind(RamKindProto::RAM_1RW);
  rewrite_proto.mutable_to_config()->set_depth(1024);
  rewrite_proto.mutable_from_channels_logical_to_physical()->insert(
      {"read_req", "ram_read_req"});
  rewrite_proto.set_to_name_prefix("ram");
  proto.mutable_rewrites()->Add(std::move(rewrite_proto));

  XLS_EXPECT_OK(RamRewritesFromProto(proto));
  EXPECT_EQ(RamRewritesFromProto(proto)->size(), 1);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).from_config.kind,
            RamKind::kAbstract);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).from_config.depth, 1024);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_config.kind, RamKind::k1RW);
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_config.depth, 1024);
  EXPECT_EQ(RamRewritesFromProto(proto)
                ->at(0)
                .from_channels_logical_to_physical.size(),
            1);
  EXPECT_THAT(
      RamRewritesFromProto(proto)->at(0).from_channels_logical_to_physical,
      Contains(std::make_pair("read_req", "ram_read_req")));
  EXPECT_EQ(RamRewritesFromProto(proto)->at(0).to_name_prefix, "ram");
}

}  // namespace
}  // namespace xls
