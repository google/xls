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

#include "xls/passes/passes.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/examples/sample_packages.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class DummyPass : public Pass {
 public:
  // `change` indicates whether the pass will return that the IR was changed.
  DummyPass(std::string short_name, std::string long_name, bool change = false)
      : Pass(short_name, long_name), change_(change) {}

  absl::StatusOr<bool> RunInternal(Package* p, const PassOptions& options,
                                   PassResults* results) const override {
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
  XLS_CHECK_OK(f_or_status.status());
  return {absl::move(m), *f_or_status};
}

TEST(PassesTest, AddPasses) {
  std::unique_ptr<Package> p = BuildShift0().first;

  CompoundPass pass_mgr("TOP", "Top level pass manager");

  pass_mgr.Add<DummyPass>("d1", "Dummy Pass 1");
  pass_mgr.Add<DummyPass>("d2", "Dummy Pass 2");
  pass_mgr.Add<DummyPass>("d3", "Dummy Pass 3");

  auto comp_pass = pass_mgr.Add<CompoundPass>("C1", "Some Compound Pass");
  comp_pass->Add<DummyPass>("d4", "Dummy Pass 4");
  comp_pass->Add<DummyPass>("d5", "Dummy Pass 5");

  pass_mgr.Add<DummyPass>("d6", "Dummy Pass 6");

  PassResults results;
  EXPECT_THAT(pass_mgr.Run(p.get(), PassOptions(), &results),
              IsOkAndHolds(false));
  std::vector<std::string> invocation_names;
  for (const PassInvocation& invocation : results.invocations) {
    invocation_names.push_back(invocation.pass_name);
  }
  EXPECT_THAT(invocation_names,
              ElementsAre("d1", "d2", "d3", "d4", "d5", "d6"));
}

// Invariant checker which returns an error if the package has function with a
// particular name.
class PackageNameChecker : public InvariantChecker {
 public:
  explicit PackageNameChecker(std::string_view str) : str_(str) {}

  absl::Status Run(Package* package, const PassOptions& options,
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
std::unique_ptr<CompoundPass> MakeNestedPasses() {
  auto top = std::make_unique<CompoundPass>("top", "Top level pass manager");
  auto comp_pass = top->Add<CompoundPass>("comp_pass", "compound pass");
  auto nested_comp_pass =
      comp_pass->Add<CompoundPass>("nested_comp_pass", "nested pass");
  nested_comp_pass->Add<DummyPass>("dummy", "blah");
  return top;
}

TEST(PassesTest, InvariantChecker) {
  // Verify the test invariant checker works as expected.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  PassResults results;
  EXPECT_THAT(
      PackageNameChecker("foo").Run(p.get(), PassOptions(), &results).message(),
      HasSubstr("Function has name 'foo'"));
  XLS_EXPECT_OK(
      PackageNameChecker("bar").Run(p.get(), PassOptions(), &results));
}

TEST(PassesTest, RunWithNoInvariantChecker) {
  // Verify no error when running with no invariant checkers added.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<CompoundPass> top = MakeNestedPasses();
  PassResults results;
  XLS_EXPECT_OK(top->Run(p.get(), PassOptions(), &results).status());
}

TEST(PassesTest, RunWithPassingInvariantChecker) {
  // With an invariant checker that always passes, running should return ok.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<CompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("bar");
  PassResults results;
  XLS_EXPECT_OK(top->Run(p.get(), PassOptions(), &results).status());
}

TEST(PassesTest, RunWithFailingInvariantChecker) {
  // With an invariant checker that fails, should return an error.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  std::unique_ptr<CompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("foo");
  PassResults results;
  EXPECT_THAT(top->Run(p.get(), PassOptions(), &results),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'foo'; [start of compound "
                                 "pass 'Top level pass manager']")));
}

TEST(PassesTest, RunWithFailingNestedInvariantChecker) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(kInvariantTesterPackage));
  auto top = std::make_unique<CompoundPass>("top", "Top level pass manager");
  top->AddInvariantChecker<PackageNameChecker>("bar");
  auto nested_pass = top->Add<CompoundPass>("comp_pass", "nested pass");
  nested_pass->AddInvariantChecker<PackageNameChecker>("foo");
  PassResults results;
  EXPECT_THAT(top->Run(p.get(), PassOptions(), &results),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'foo'; [start of compound "
                                 "pass 'nested pass']")));
}

// Pass which adds a function of a particular name
class FunctionAdderPass : public Pass {
 public:
  explicit FunctionAdderPass(std::string_view name)
      : Pass("function_adder", absl::StrCat("Adds function named ", name)),
        name_(name) {}

  // Adds a function named 'str_' to the package.
  absl::StatusOr<bool> RunInternal(Package* package, const PassOptions& options,
                                   PassResults* results) const override {
    const char format_string[] =
        R"(
fn %s() -> bits[32] {
 ret literal.3: bits[32] = literal(value=42)
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
  std::unique_ptr<CompoundPass> top = MakeNestedPasses();
  top->AddInvariantChecker<PackageNameChecker>("bar");
  PassResults results;
  XLS_EXPECT_OK(top->Run(p.get(), PassOptions(), &results).status());
  down_cast<CompoundPass*>(top->passes()[0])->Add<FunctionAdderPass>("bar");
  auto result = top->Run(p.get(), PassOptions(), &results);
  EXPECT_THAT(result,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Function has name 'bar'; [after 'Adds "
                                 "function named bar' pass")))
      << result.status();
}

// Invariant checker which counts the number of times it was invoked.
class CounterChecker : public InvariantChecker {
 public:
  explicit CounterChecker() {}

  absl::Status Run(Package* package, const PassOptions& options,
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
  CompoundPass pass_mgr("TOP", "Top level pass manager");
  pass_mgr.Add<DummyPass>("d1", "Dummy pass 1", /*change=*/false);
  pass_mgr.Add<DummyPass>("d2", "Dummy pass 2", /*change=*/false);
  pass_mgr.Add<DummyPass>("d3", "Dummy pass 3", /*change=*/true);
  pass_mgr.Add<DummyPass>("d4", "Dummy pass 4", /*change=*/false);
  pass_mgr.Add<DummyPass>("d5", "Dummy pass 5", /*change=*/true);
  pass_mgr.Add<DummyPass>("d6", "Dummy pass 6", /*change=*/false);

  CounterChecker* checker = pass_mgr.AddInvariantChecker<CounterChecker>();
  EXPECT_EQ(checker->run_count(), 0);

  PassResults results;
  XLS_EXPECT_OK(pass_mgr.Run(p.get(), PassOptions(), &results).status());

  // Checkers should run once at the beginning then only after a passes that
  // indicate the IR has been changed.
  EXPECT_EQ(checker->run_count(), 3);
}

// Pass which just records that it was run via a shared vector of strings.
class RecordingPass : public Pass {
 public:
  RecordingPass(std::string name, std::vector<std::string>* record)
      : Pass(name, name), record_(record) {}

  absl::StatusOr<bool> RunInternal(Package* p, const PassOptions& options,
                                   PassResults* results) const override {
    record_->push_back(short_name());
    return false;
  }

 private:
  std::vector<std::string>* record_;
};

TEST(PassesTest, RunOnlyPassesOption) {
  std::unique_ptr<Package> p = BuildShift0().first;

  std::vector<std::string> record;
  CompoundPass compound_0("compound_0", "Compound pass 0");
  compound_0.Add<RecordingPass>("foo", &record);
  compound_0.Add<RecordingPass>("bar", &record);
  auto compound_1 =
      compound_0.Add<CompoundPass>("compound_1", "Compound pass 1");
  compound_1->Add<RecordingPass>("qux", &record);
  compound_1->Add<RecordingPass>("foo", &record);

  {
    PassOptions options;
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("foo", "bar", "qux", "foo"));
  }

  {
    PassOptions options;
    options.run_only_passes = {"foo"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("foo", "foo"));
  }

  {
    PassOptions options;
    options.run_only_passes = {"bar", "qux"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("bar", "qux"));
  }

  {
    PassOptions options;
    options.run_only_passes = std::vector<std::string>();
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre());
  }

  {
    PassOptions options;
    options.run_only_passes = {"blah"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre());
  }
}

TEST(PassesTest, SkipPassesOption) {
  std::unique_ptr<Package> p = BuildShift0().first;

  std::vector<std::string> record;
  CompoundPass compound_0("compound_0", "Compound pass 0");
  compound_0.Add<RecordingPass>("foo", &record);
  compound_0.Add<RecordingPass>("bar", &record);
  auto compound_1 =
      compound_0.Add<CompoundPass>("compound_1", "Compound pass 1");
  compound_1->Add<RecordingPass>("qux", &record);
  compound_1->Add<RecordingPass>("foo", &record);

  {
    PassOptions options;
    options.skip_passes = {"blah"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("foo", "bar", "qux", "foo"));
  }

  {
    PassOptions options;
    options.skip_passes = {"foo"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("bar", "qux"));
  }

  {
    PassOptions options;
    options.skip_passes = {"foo", "qux"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("bar"));
  }
}

TEST(PassesTest, RunOnlyAndSkipPassesOption) {
  std::unique_ptr<Package> p = BuildShift0().first;

  std::vector<std::string> record;
  CompoundPass compound_0("compound_0", "Compound pass 0");
  compound_0.Add<RecordingPass>("foo", &record);
  compound_0.Add<RecordingPass>("bar", &record);
  auto compound_1 =
      compound_0.Add<CompoundPass>("compound_1", "Compound pass 1");
  compound_1->Add<RecordingPass>("qux", &record);
  compound_1->Add<RecordingPass>("foo", &record);

  {
    PassOptions options;
    options.run_only_passes = {"foo", "qux"};
    options.skip_passes = {"foo", "bar"};
    PassResults results;
    record.clear();
    EXPECT_THAT(compound_0.Run(p.get(), options, &results),
                IsOkAndHolds(false));
    EXPECT_THAT(record, ElementsAre("qux"));
  }
}

class NaiveDcePass : public FunctionBasePass {
 public:
  NaiveDcePass() : FunctionBasePass("naive_dce", "naive dce") {}

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* f, const PassOptions& options,
      PassResults* results) const override {
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
  // Make paramter x the return value which means everything is dead but x.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(x));
  PassResults results;
  EXPECT_EQ(f->node_count(), 6);
  ASSERT_THAT(NaiveDcePass().RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  ASSERT_THAT(NaiveDcePass().RunOnFunctionBase(f, PassOptions(), &results),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
