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

#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/solvers/ir_equivalence.h"
#include "xls/solvers/prover_matchers.h"
#include "xls/solvers/solver.h"

namespace xls::solvers::bitwuzla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::IsProvenFalse;
using ::xls::solvers::IsProvenTrue;
using ::xls::solvers::SolverKind;
using ::xls::solvers::TryProveEquivalence;

class BitwuzlaEquivalenceTest : public IrTestBase {};

TEST_F(BitwuzlaEquivalenceTest, IdenticalFunctionsEquivalent) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  fb1.Add(fb1.Param("x", p1->GetBitsType(32)),
          fb1.Param("y", p1->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  fb2.Add(fb2.Param("x", p2->GetBitsType(32)),
          fb2.Param("y", p2->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  EXPECT_THAT(TryProveEquivalence(f1, f2, SolverKind::kBitwuzla),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaEquivalenceTest, CommutativeAddEquivalent) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  BValue x1 = fb1.Param("x", p1->GetBitsType(32));
  BValue y1 = fb1.Param("y", p1->GetBitsType(32));
  fb1.Add(x1, y1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  BValue x2 = fb2.Param("x", p2->GetBitsType(32));
  BValue y2 = fb2.Param("y", p2->GetBitsType(32));
  fb2.Add(y2, x2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  EXPECT_THAT(TryProveEquivalence(f1, f2, SolverKind::kBitwuzla),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaEquivalenceTest, ModifiedFunctionNotEquivalent) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  BValue x1 = fb1.Param("x", p1->GetBitsType(32));
  BValue y1 = fb1.Param("y", p1->GetBitsType(32));
  fb1.Add(x1, y1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  BValue x2 = fb2.Param("x", p2->GetBitsType(32));
  BValue y2 = fb2.Param("y", p2->GetBitsType(32));
  fb2.Add(fb2.Add(x2, fb2.Literal(UBits(1, 32))), y2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  EXPECT_THAT(TryProveEquivalence(f1, f2, SolverKind::kBitwuzla),
              IsOkAndHolds(IsProvenFalse()));
}

TEST_F(BitwuzlaEquivalenceTest, OptimizeMergedIrEquivalentAndNotEquivalent) {
  auto run_opt = [](Package* merged) -> absl::Status {
    return RunOptimizationPassPipeline(merged).status();
  };

  {
    std::unique_ptr<Package> p1 = CreatePackage();
    FunctionBuilder fb1(TestName(), p1.get());
    BValue x1 = fb1.Param("x", p1->GetBitsType(16));
    BValue y1 = fb1.Param("y", p1->GetBitsType(16));
    fb1.Add(fb1.Xor(x1, x1), y1);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

    std::unique_ptr<Package> p2 = CreatePackage();
    FunctionBuilder fb2(TestName(), p2.get());
    fb2.Param("x", p2->GetBitsType(16));
    BValue y2 = fb2.Param("y", p2->GetBitsType(16));
    fb2.Identity(y2);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

    EXPECT_THAT(TryProveEquivalence(f1, f2, /*ignore_asserts=*/false,
                                    SolverKind::kBitwuzla, {}, run_opt),
                IsOkAndHolds(IsProvenTrue()));
  }

  {
    std::unique_ptr<Package> p1 = CreatePackage();
    FunctionBuilder fb1(TestName(), p1.get());
    BValue x1 = fb1.Param("x", p1->GetBitsType(8));
    fb1.Assert(fb1.Literal(Value::Token()),
               fb1.Eq(x1, fb1.Literal(UBits(0, 8))), "msg", "lbl");
    fb1.Identity(x1);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

    std::unique_ptr<Package> p2 = CreatePackage();
    FunctionBuilder fb2(TestName(), p2.get());
    BValue x2 = fb2.Param("x", p2->GetBitsType(8));
    fb2.Assert(fb2.Literal(Value::Token()),
               fb2.Ne(x2, fb2.Literal(UBits(0, 8))), "msg", "lbl");
    fb2.Identity(x2);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

    XLS_ASSERT_OK_AND_ASSIGN(
        ProverResult r,
        TryProveEquivalence(f1, f2, /*ignore_asserts=*/false,
                            SolverKind::kBitwuzla, {}, run_opt));
    ASSERT_THAT(r, IsProvenFalse());
    ProvenFalse f = std::get<ProvenFalse>(r);
    ASSERT_THAT(f.counterexample, ::absl_testing::IsOk());
    EXPECT_TRUE(f.counterexample->contains(f1->param(0)));
    EXPECT_TRUE(f.counterexample->contains(f2->param(0)));
  }
}

TEST_F(BitwuzlaEquivalenceTest, TransformMergedSynthesizingAssertIsStripped) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  BValue x1 = fb1.Param("x", p1->GetBitsType(8));
  BValue y1 = fb1.Param("y", p1->GetBitsType(8));
  fb1.Add(x1, y1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  BValue x2 = fb2.Param("x", p2->GetBitsType(8));
  BValue y2 = fb2.Param("y", p2->GetBitsType(8));
  fb2.Add(y2, x2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  auto synthesize_assert = [](Package* merged) -> absl::Status {
    Function* top = (*merged->GetTop())->AsFunctionOrDie();
    XLS_ASSIGN_OR_RETURN(Node * tkn,
                         top->MakeNode<Literal>(SourceInfo(), Value::Token()));
    XLS_ASSIGN_OR_RETURN(Node * cond,
                         top->MakeNode<CompareOp>(SourceInfo(), top->param(0),
                                                  top->param(1), Op::kEq));
    XLS_RETURN_IF_ERROR(
        top->MakeNode<Assert>(SourceInfo(), tkn, cond, "synthesized invariant",
                              "__xls_invariant_test_selector_one_hot_A",
                              /*original_label=*/std::nullopt)
            .status());
    return absl::OkStatus();
  };

  EXPECT_THAT(TryProveEquivalence(f1, f2, /*ignore_asserts=*/false,
                                  SolverKind::kBitwuzla, {}, synthesize_assert),
              IsOkAndHolds(IsProvenTrue()));
}

}  // namespace
}  // namespace xls::solvers::bitwuzla
