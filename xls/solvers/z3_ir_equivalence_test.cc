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

#include "xls/solvers/z3_ir_equivalence.h"

#include <cstdint>
#include <memory>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest-spi.h"
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
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace m = xls::op_matchers;
namespace xls::solvers::z3 {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;

using ::testing::AnyOf;
using ::testing::Not;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

static constexpr bool kHasMsan =
#if defined(ABSL_HAVE_MEMORY_SANITIZER)
    true;
#else
    false;
#endif

class EquivalenceTest : public IrTestBase {};

TEST_F(EquivalenceTest, NoOpIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.Add(x, y), fb.Subtract(x, y), fb.Subtract(y, x), fb.UMul(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(
      TryProveEquivalence(f, [](auto p, auto f) { return absl::OkStatus(); }),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(EquivalenceTest, ScopedNoOpIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.Add(x, y), fb.Subtract(x, y), fb.Subtract(y, x), fb.UMul(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence check_equivalence(f);
}

TEST_F(EquivalenceTest, EquivalentTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.Add(x, y), fb.UMul(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(
      TryProveEquivalence(f,
                          [](Package* p, Function* f) -> absl::Status {
                            // Reverse all the binary operations
                            for (Node* n : TopoSort(f)) {
                              if (n->Is<BinOp>()) {
                                XLS_RETURN_IF_ERROR(
                                    n->ReplaceUsesWithNew<BinOp>(
                                         n->operand(1), n->operand(0), n->op())
                                        .status());
                              }
                            }
                            return absl::OkStatus();
                          }),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(EquivalenceTest, ScopedEquivalentTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.Add(x, y), fb.UMul(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence check_equivalence(f);
  for (Node* n : TopoSort(f)) {
    if (n->Is<BinOp>()) {
      XLS_ASSERT_OK(
          n->ReplaceUsesWithNew<BinOp>(n->operand(1), n->operand(0), n->op())
              .status());
    }
  }
}

TEST_F(EquivalenceTest, EquivalentArrayTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Array({fb.Add(x, y), fb.UMul(x, y)}, p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(
      TryProveEquivalence(f,
                          [](Package* p, Function* f) -> absl::Status {
                            // Reverse all the binary operations
                            for (Node* n : TopoSort(f)) {
                              if (n->Is<BinOp>()) {
                                XLS_RETURN_IF_ERROR(
                                    n->ReplaceUsesWithNew<BinOp>(
                                         n->operand(1), n->operand(0), n->op())
                                        .status());
                              }
                            }
                            return absl::OkStatus();
                          }),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(EquivalenceTest, ScopedEquivalentArrayTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Array({fb.Add(x, y), fb.UMul(x, y)}, p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence check_equivalence(f);
  // Reverse all the binary operations
  for (Node* n : TopoSort(f)) {
    if (n->Is<BinOp>()) {
      XLS_ASSERT_OK(
          n->ReplaceUsesWithNew<BinOp>(n->operand(1), n->operand(0), n->op())
              .status());
    }
  }
}

TEST_F(EquivalenceTest, EquivalentBitsTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(
      TryProveEquivalence(f,
                          [](Package* p, Function* f) -> absl::Status {
                            // Reverse all the binary operations
                            for (Node* n : TopoSort(f)) {
                              if (n->Is<BinOp>()) {
                                XLS_RETURN_IF_ERROR(
                                    n->ReplaceUsesWithNew<BinOp>(
                                         n->operand(1), n->operand(0), n->op())
                                        .status());
                              }
                            }
                            return absl::OkStatus();
                          }),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(EquivalenceTest, ScopedEquivalentBitsTransformIsEquivalent) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScopedVerifyEquivalence check_equivalence(f);
  // Reverse all the binary operations
  for (Node* n : TopoSort(f)) {
    if (n->Is<BinOp>()) {
      XLS_ASSERT_OK(
          n->ReplaceUsesWithNew<BinOp>(n->operand(1), n->operand(0), n->op())
              .status());
    }
  }
}

TEST_F(EquivalenceTest, DetectsNonEquivalentTransform) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(
      TryProveEquivalence(f,
                          [](Package* p, Function* f) -> absl::Status {
                            // Reverse all the binary operations
                            for (Node* n : TopoSort(f)) {
                              if (n->Is<BinOp>()) {
                                XLS_RETURN_IF_ERROR(
                                    n->ReplaceUsesWithNew<BinOp>(
                                         n->operand(1), n->operand(0), n->op())
                                        .status());
                              }
                            }
                            return absl::OkStatus();
                          }),
      IsOkAndHolds(IsProvenFalse()));
}

constexpr int64_t kScopedNonEquivalentTransformCheckLine = __LINE__ + 2;
void ScopedNonEquivalentTransform(Function* f) {
  ScopedVerifyEquivalence check_equivalence(f);
  // Reverse all the binary operations
  for (Node* n : TopoSort(f)) {
    if (n->Is<BinOp>()) {
      XLS_EXPECT_OK(
          n->ReplaceUsesWithNew<BinOp>(n->operand(1), n->operand(0), n->op())
              .status())
          << "Transform failed";
    }
  }
}

TEST_F(EquivalenceTest, ScopedDetectsNonEquivalentTransform) {
  if (kHasMsan) {
    GTEST_SKIP() << "Skipping ScopedVerifyEquivalence test; MSAN is enabled.";
  }
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_test_function"), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_NONFATAL_FAILURE(
      ScopedNonEquivalentTransform(f),
      absl::StrCat(__FILE__, ":", kScopedNonEquivalentTransformCheckLine,
                   ": ScopedVerifyEquivalence failed to prove equivalence of "
                   "function ",
                   f->name(), " before & after changes"));
}

TEST_F(EquivalenceTest, DetectsReturnTypeChange) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(TryProveEquivalence(f,
                                  [](Package* p, Function* f) -> absl::Status {
                                    return f->set_return_value(f->param(0));
                                  }),
              Not(IsOk()));
}

void ScopedReturnTypeChange(Function* f) {
  ScopedVerifyEquivalence check_equivalence(f);
  XLS_ASSERT_OK(f->set_return_value(f->param(0)));
}

TEST_F(EquivalenceTest, ScopedDetectsReturnTypeChange) {
  if (kHasMsan) {
    GTEST_SKIP() << "Skipping ScopedVerifyEquivalence test; MSAN is enabled.";
  }
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_NONFATAL_FAILURE(ScopedReturnTypeChange(f), "differing return types");
}

TEST_F(EquivalenceTest, CounterexampleIsValidPointers) {
  std::unique_ptr<Package> p = CreatePackage();
  Function* f1;
  Function* f2;
  BValue f1_x1;
  BValue f2_x1;
  {
    FunctionBuilder fb(absl::StrCat(TestName(), "_1"), p.get());
    f1_x1 = fb.Param("x1", p->GetBitsType(1));
    BValue x2 = fb.Param("x2", p->GetBitsType(1));
    fb.Concat({f1_x1, x2});
    XLS_ASSERT_OK_AND_ASSIGN(f1, fb.Build());
  }
  {
    FunctionBuilder fb(absl::StrCat(TestName(), "_2"), p.get());
    f2_x1 = fb.Param("x1", p->GetBitsType(1));
    BValue x2 = fb.Param("x2", p->GetBitsType(1));
    fb.Concat({fb.Literal(UBits(0, 1)), x2});

    XLS_ASSERT_OK_AND_ASSIGN(f2, fb.Build());
  }
  XLS_ASSERT_OK_AND_ASSIGN(ProverResult r, TryProveEquivalence(f1, f2));
  ASSERT_THAT(r, IsProvenFalse());
  ProvenFalse f = std::get<ProvenFalse>(r);
  EXPECT_THAT(
      f.counterexample,
      IsOkAndHolds(UnorderedElementsAre(
          Pair(m::Param("x1"), Value(UBits(1, 1))),
          Pair(testing::_, AnyOf(Value(UBits(1, 1)), Value(UBits(0, 1)))))));
}

TEST_F(EquivalenceTest, DetectsParamShift) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x1 = fb.Param("x1", p->GetBitsType(16));
  BValue x2 = fb.Param("x2", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue x = fb.Concat({x1, x2});
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(TryProveEquivalence(f,
                                  [](Package* p, Function* f) -> absl::Status {
                                    return f->MoveParamToIndex(f->param(2), 0);
                                  }),
              Not(IsOk()));
}

void ScopedParamShift(Function* f) {
  ScopedVerifyEquivalence check_equivalence(f);
  XLS_ASSERT_OK(f->MoveParamToIndex(f->param(2), 0));
}

TEST_F(EquivalenceTest, ScopedDetectsParamShift) {
  if (kHasMsan) {
    GTEST_SKIP() << "Skipping ScopedVerifyEquivalence test; MSAN is enabled.";
  }

  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x1 = fb.Param("x1", p->GetBitsType(16));
  BValue x2 = fb.Param("x2", p->GetBitsType(16));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue x = fb.Concat({x1, x2});
  fb.Tuple({fb.UDiv(x, y), fb.Subtract(x, y), fb.UMod(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_NONFATAL_FAILURE(ScopedParamShift(f), "differing parameter 0 types");
}

TEST_F(EquivalenceTest, MultiFunctionDetectsSame) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  fb1.Add(fb1.Param("x", p1->GetBitsType(32)),
          fb1.Param("y", p1->GetBitsType(32)));

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  fb2.Add(fb2.Param("x", p2->GetBitsType(32)),
          fb2.Param("y", p2->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  EXPECT_THAT(TryProveEquivalence(f1, f2), IsOkAndHolds(IsProvenTrue()));
}

TEST_F(EquivalenceTest, MultiFunctionDetectsDifference) {
  std::unique_ptr<Package> p1 = CreatePackage();
  FunctionBuilder fb1(TestName(), p1.get());
  fb1.Add(fb1.Param("x", p1->GetBitsType(32)),
          fb1.Param("y", p1->GetBitsType(32)));

  std::unique_ptr<Package> p2 = CreatePackage();
  FunctionBuilder fb2(TestName(), p2.get());
  fb2.Add(
      fb2.Add(fb2.Literal(UBits(1, 32)), fb2.Param("x", p2->GetBitsType(32))),
      fb2.Param("y", p2->GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f1, fb1.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f2, fb2.Build());

  EXPECT_THAT(TryProveEquivalence(f1, f2), IsOkAndHolds(IsProvenFalse()));
}

}  // namespace
}  // namespace xls::solvers::z3
