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

#include "xls/solvers/symex/symex_engine.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "xls/solvers/symex/test_util.h"
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

// Returns true if `antecedent` implies `consequent` across the entire input
// domain, i.e. `antecedent && !consequent` is unsatisfiable.
bool Implies(Z3_context ctx, Z3_ast antecedent, Z3_ast consequent) {
  Z3_solver solver = Z3_mk_solver(ctx);
  Z3_solver_inc_ref(ctx, solver);
  Z3_ast conjuncts[] = {antecedent, Z3_mk_not(ctx, consequent)};
  Z3_solver_assert(ctx, solver, Z3_mk_and(ctx, 2, conjuncts));
  const bool implies = Z3_solver_check(ctx, solver) == Z3_L_FALSE;
  Z3_solver_dec_ref(ctx, solver);
  return implies;
}

class SymExEngineTest : public IrTestBase {
 protected:
  void SetUp() override {
    IrTestBase::SetUp();
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
  }

  void TearDown() override {
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
    IrTestBase::TearDown();
  }

  bool AreMutuallyExclusive(Z3_ast cond1, Z3_ast cond2) {
    Z3_solver solver = Z3_mk_solver(ctx_);
    Z3_solver_inc_ref(ctx_, solver);
    Z3_solver_assert(ctx_, solver, cond1);
    Z3_solver_assert(ctx_, solver, cond2);
    Z3_lbool result = Z3_solver_check(ctx_, solver);
    Z3_solver_dec_ref(ctx_, solver);
    return result == Z3_L_FALSE;
  }

  // Builds `sel(s1, cases=[sel(s0, cases=[a, b]), literal])`. The inner
  // multiplexer reaches the result only through arm 0 of the outer one, so
  // `s0` is an observability don't care whenever `s1` selects arm 1.
  absl::StatusOr<Function*> BuildNestedMuxFunction(Package* p) {
    FunctionBuilder fb("nested_mux", p);
    BValue s0 = fb.Param("s0", p->GetBitsType(1));
    BValue s1 = fb.Param("s1", p->GetBitsType(1));
    BValue a = fb.Param("a", p->GetBitsType(8));
    BValue b = fb.Param("b", p->GetBitsType(8));
    BValue inner = fb.Select(s0, {a, b});
    fb.Select(s1, {inner, fb.Literal(UBits(0, 8))});
    return fb.Build();
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(SymExEngineTest, InitializesWithDefaultOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  EXPECT_EQ(engine.options().max_paths, 1000);
}

TEST_F(SymExEngineTest, RespectsMaxPathsLimit) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(3));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue c = fb.Param("c", p.GetBitsType(8));
  BValue def = fb.Param("def", p.GetBitsType(8));
  fb.PrioritySelect(sel, {a, b, c}, def);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  SymExOptions options;
  options.max_paths = 2;
  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_THAT(paths, SizeIs(2));
}

TEST_F(SymExEngineTest, NoSelectsSinglePath) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue a = fb.Param("a", p.GetBitsType(32));
  BValue b = fb.Param("b", p.GetBitsType(32));
  fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(1));
  EXPECT_THAT(paths[0].branch_decisions, IsEmpty());
}

TEST_F(SymExEngineTest, SingleSelectTwoPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  fb.Select(sel, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(2));
  EXPECT_THAT(paths,
              ElementsAre(SymbolicPathIs(0, false), SymbolicPathIs(1, false)));
  EXPECT_TRUE(
      AreMutuallyExclusive(paths[0].path_condition, paths[1].path_condition));
}

TEST_F(SymExEngineTest, PrioritySelectFourPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(3));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue c = fb.Param("c", p.GetBitsType(8));
  BValue def = fb.Param("def", p.GetBitsType(8));
  fb.PrioritySelect(sel, {a, b, c}, def);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(4));
  EXPECT_THAT(paths[0].branch_decisions,
              ElementsAre(BranchDecisionIs(0, false)));
  EXPECT_THAT(paths[1].branch_decisions,
              ElementsAre(BranchDecisionIs(1, false)));
  EXPECT_THAT(paths[2].branch_decisions,
              ElementsAre(BranchDecisionIs(2, false)));
  EXPECT_THAT(paths[3].branch_decisions,
              ElementsAre(BranchDecisionIs(3, true)));
}

TEST_F(SymExEngineTest, CascadedMuxesFourPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s0 = fb.Param("s0", p.GetBitsType(1));
  BValue s1 = fb.Param("s1", p.GetBitsType(1));
  BValue v0 = fb.Literal(UBits(0, 8));
  BValue v1 = fb.Literal(UBits(1, 8));
  BValue v2 = fb.Literal(UBits(2, 8));
  BValue v3 = fb.Literal(UBits(3, 8));
  BValue m0 = fb.Select(s0, {v0, v1});
  BValue m1 = fb.Select(s0, {v2, v3});
  fb.Select(s1, {m0, m1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(4));
  for (size_t i = 0; i < paths.size(); ++i) {
    for (size_t j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }
}

TEST_F(SymExEngineTest, PrunesInfeasibleBranchDecisions) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  // Create an unsatisfiable branch condition: constrained_sel is forced to 0,
  // so branch 1 (constrained_sel == 1) is unsatisfiable.
  BValue zero = fb.Literal(UBits(0, 1));
  BValue constrained_sel = fb.And(sel, zero);
  fb.Select(constrained_sel, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Branch 1 is pruned by incremental solver push/pop, leaving 1 path.
  ASSERT_THAT(paths, SizeIs(1));
  EXPECT_THAT(paths[0].branch_decisions,
              ElementsAre(BranchDecisionIs(0, false)));
}

TEST_F(SymExEngineTest, ConcolicExecutionPrunesIncompatibleBranches) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(32));
  fb.Param("b", p->GetBitsType(32));
  BValue c0 = fb.Literal(UBits(10, 32));
  BValue c1 = fb.Literal(UBits(20, 32));
  BValue c2 = fb.Literal(UBits(30, 32));
  fb.Select(sel, {c0, c1, c2}, a);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  SymExOptions options;
  options.concrete_inputs.BindParam("sel", Value(UBits(1, 2)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Only the path corresponding to sel == 1 should be feasible.
  EXPECT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions[0].arm_index, 1);
}

TEST_F(SymExEngineTest, ConcolicExecutionRejectsMismatchedBitwidth) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue c0 = fb.Literal(UBits(10, 32));
  fb.Select(sel, std::vector<BValue>{c0}, a);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  SymExOptions options;
  // sel is bits[2], but binding bits[4] should return InvalidArgument.
  options.concrete_inputs.BindParam("sel", Value(UBits(1, 4)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  EXPECT_THAT(engine.ExplorePaths(fn).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("bitwidth mismatch")));
}

TEST_F(SymExEngineTest,
       ConcolicExecutionPrunesViaIntermediateSelectorComputation) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue two = fb.Literal(UBits(2, 32));
  BValue zero = fb.Literal(UBits(0, 32));
  BValue five = fb.Literal(UBits(5, 32));
  BValue eight = fb.Literal(UBits(8, 32));
  BValue mod_x = fb.UMod(x, two);
  BValue cond_x = fb.Eq(mod_x, zero);
  BValue real_x = fb.Select(cond_x, {five, x});
  BValue cond_y = fb.Eq(y, real_x);
  BValue real_y = fb.Select(cond_y, {y, eight});
  fb.Add(real_x, real_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  // Without concolic constraint, there are 4 paths (2 branches at each of 2
  // muxes).
  {
    XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                             SymExEngine::Create(ctx_, SymExOptions()));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> all_paths,
                             engine.ExplorePaths(fn));
    EXPECT_THAT(all_paths, SizeIs(4));
  }

  // Restricting x = 3 forces mod_x = 1, pruning the cond_x == 1 branch.
  // Exactly 2 feasible paths remain for y.
  SymExOptions options;
  options.concrete_inputs.BindParam("x", Value(UBits(3, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_THAT(paths, SizeIs(2));
  for (const SymbolicPath& path : paths) {
    ASSERT_THAT(path.branch_decisions, SizeIs(2));
    // The first decision (mux for real_x) must be arm 0 (five).
    EXPECT_EQ(path.branch_decisions[0].arm_index, 0);
  }
}

TEST_F(SymExEngineTest, ConcolicExecutionWithTupleInputPrunesBranches) {
  std::unique_ptr<VerifiedPackage> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* tuple_type = p->GetTupleType({p->GetBitsType(2), p->GetBitsType(32)});
  BValue t = fb.Param("t", tuple_type);
  BValue sel = fb.TupleIndex(t, 0);
  BValue val = fb.TupleIndex(t, 1);
  BValue c0 = fb.Literal(UBits(10, 32));
  BValue c1 = fb.Literal(UBits(20, 32));
  BValue c2 = fb.Literal(UBits(30, 32));
  fb.Select(sel, {c0, c1, c2}, val);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  SymExOptions options;
  options.concrete_inputs.BindParam(
      "t", Value::Tuple({Value(UBits(1, 2)), Value(UBits(42, 32))}));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions[0].arm_index, 1);
}

TEST_F(SymExEngineTest, UnobservableMuxIsLeftUnexplored) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // s1 == 0 exposes the inner multiplexer and yields one path per inner arm.
  // s1 == 1 discards it, so both of its arms collapse into a single path.
  EXPECT_THAT(paths, SizeIs(3));
}

TEST_F(SymExEngineTest, PruningDisabledExploresEveryArmCombination) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  SymExOptions options;
  options.prune_unobservable = false;

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_THAT(paths, SizeIs(4));
  for (const SymbolicPath& path : paths) {
    EXPECT_THAT(path.branch_decisions, SizeIs(2));
    EXPECT_THAT(path.unobservable_muxes, IsEmpty());
  }
}

TEST_F(SymExEngineTest, UnobservableMuxesAreReportedOnThePath) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  int64_t collapsed_paths = 0;
  for (const SymbolicPath& path : paths) {
    if (path.unobservable_muxes.empty()) {
      EXPECT_THAT(path.branch_decisions, SizeIs(2));
      continue;
    }
    ++collapsed_paths;
    // Only the inner multiplexer is skipped, so only the outer one is decided.
    ASSERT_THAT(path.unobservable_muxes, SizeIs(1));
    EXPECT_EQ(path.unobservable_muxes[0]->operand(0)->GetName(), "s0");
    EXPECT_THAT(path.branch_decisions, ElementsAre(BranchDecisionIs(1, false)));
  }
  EXPECT_EQ(collapsed_paths, 1);
}

TEST_F(SymExEngineTest,
       UnobservableMuxDoesNotLeakIntoSiblingObservableBranches) {
  // `outer` selects arm 0 (an unrelated literal) or arm 1 (`inner` mux).
  // Arm 0 is explored first, where `inner` is unobservable.
  // Arm 1 is explored second, where `inner` is observable.
  // Paths under arm 1 must have unobservable_muxes empty, ensuring
  // unobservable mux state doesn't leak into sibling subtrees.
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s0 = fb.Param("s0", p.GetBitsType(1));
  BValue s1 = fb.Param("s1", p.GetBitsType(1));
  BValue inner =
      fb.Select(s0, {fb.Literal(UBits(0, 8)), fb.Literal(UBits(1, 8))});
  BValue unrelated = fb.Literal(UBits(2, 8));
  fb.Select(s1, {unrelated, inner});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(3));
  EXPECT_THAT(paths[0].unobservable_muxes, SizeIs(1));
  EXPECT_THAT(paths[1].unobservable_muxes, IsEmpty());
  EXPECT_THAT(paths[2].unobservable_muxes, IsEmpty());
}

TEST_F(SymExEngineTest, PrunedPathsStillPartitionTheInputDomain) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Skipping a multiplexer coarsens the partition without making it overlap or
  // leave gaps, and the symbolic return values stay free of the multiplexer
  // variables they were built from.
  XLS_EXPECT_OK(CheckFormalProperties(ctx_, fn, paths,
                                      FormalCheckConfig{.expected_paths = 3}));
}

TEST_F(SymExEngineTest, ConcolicInputsComposeWithObservabilityPruning) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  // Concolic bindings hold at every feasibility check, so pinning s1 = 1 leaves
  // only the path that discards the inner multiplexer.
  SymExOptions options;
  options.concrete_inputs.BindParam("s1", Value(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(1));
  EXPECT_THAT(paths[0].branch_decisions,
              ElementsAre(BranchDecisionIs(1, false)));
  EXPECT_THAT(paths[0].unobservable_muxes, SizeIs(1));
  EXPECT_EQ(paths[0].GetParamValue("s1"), Value(UBits(1, 1)));

  // The witness must still execute to the value this path stands for.
  // Exhaustiveness and ITE equivalence do not apply here: the concolic binding
  // restricts the input domain to a subset of the function's.
  XLS_EXPECT_OK(CheckFormalProperties(
      ctx_, fn, paths,
      FormalCheckConfig{.check_exhaustiveness = false,
                        .check_smt_ite_equivalence = false,
                        .expected_paths = 1}));
  XLS_ASSERT_OK_AND_ASSIGN(Value result, InterpretPath(fn, paths[0]));
  EXPECT_EQ(result, Value(UBits(0, 8)));
}

TEST_F(SymExEngineTest, PruningCoarsensTheExhaustivePartition) {
  Package p(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, BuildNestedMuxFunction(&p));

  SymExOptions exhaustive_options;
  exhaustive_options.prune_unobservable = false;
  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine exhaustive_engine,
                           SymExEngine::Create(ctx_, exhaustive_options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> exhaustive_paths,
                           exhaustive_engine.ExplorePaths(fn));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine pruned_engine,
                           SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> pruned_paths,
                           pruned_engine.ExplorePaths(fn));

  // Each mode must partition the input domain on its own terms.
  XLS_EXPECT_OK(CheckFormalProperties(ctx_, fn, exhaustive_paths));
  XLS_EXPECT_OK(CheckFormalProperties(ctx_, fn, pruned_paths));

  // This function is chosen so that pruning really does collapse arms; without
  // that the containment check below would pass vacuously.
  EXPECT_LT(pruned_paths.size(), exhaustive_paths.size());

  // Every exhaustive path must lie entirely inside exactly one pruned path.
  // Equal coverage would be too weak a property: two partitions can cover the
  // same domain without either being a coarsening of the other, and only a
  // coarsening makes a pruned path a valid stand-in for the arm combinations it
  // replaces.
  for (const SymbolicPath& exhaustive : exhaustive_paths) {
    int64_t containing_paths = 0;
    for (const SymbolicPath& pruned : pruned_paths) {
      if (!Implies(ctx_, exhaustive.path_condition, pruned.path_condition)) {
        continue;
      }
      ++containing_paths;
      // Where they overlap, both modes must compute the same result.
      EXPECT_TRUE(Implies(
          ctx_, exhaustive.path_condition,
          Z3_mk_eq(ctx_, exhaustive.return_value, pruned.return_value)));
    }
    EXPECT_EQ(containing_paths, 1);
  }
}

}  // namespace
}  // namespace xls::solvers::symex
