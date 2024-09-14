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

#include "xls/passes/array_simplification_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ArraySimplificationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    bool changed = false;
    bool changed_this_iteration = true;
    while (changed_this_iteration) {
      XLS_ASSIGN_OR_RETURN(changed_this_iteration,
                           ArraySimplificationPass().RunOnFunctionBase(
                               f, OptimizationPassOptions(), &results));
      // Run dce and constant folding to clean things up.
      XLS_RETURN_IF_ERROR(
          ConstantFoldingPass()
              .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
              .status());
      XLS_RETURN_IF_ERROR(
          DeadCodeEliminationPass()
              .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
              .status());
      changed = changed || changed_this_iteration;
    }

    return changed;
  }
};

TEST_F(ArraySimplificationPassTest, ArrayWithOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(3, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(UBits(123, 32)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Index should be clipped at the max legal index.
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(2)}));
}

TEST_F(ArraySimplificationPassTest, ArrayWithWideOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(42, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(Bits::AllOnes(1234)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Index should be clipped at the max legal index.
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(41)}));
}

TEST_F(ArraySimplificationPassTest, ArrayWithWideInBoundsLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(3, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(UBits(1, 1000)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // No transformation, but shouldn't crash.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest, LiteralArrayWithNonLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Param("idx", p->GetBitsType(32));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayIndex());
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u16 = p->GetBitsType(16);
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Param("i", u16);
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("i"), {m::Param("x"), m::Param("y")}, m::Param("z")));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperationExactFit) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u2 = p->GetBitsType(2);
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array({fb.Param("x", u32), fb.Param("y", u32),
                       fb.Param("z", u32), fb.Param("w", u32)},
                      u32);
  BValue index = fb.Param("i", u2);
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("i"), {m::Param("x"), m::Param("y"),
                                        m::Param("z"), m::Param("w")}));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperationUndersizedIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u2 = p->GetBitsType(2);
  Type* u32 = p->GetBitsType(32);
  BValue a =
      fb.Array({fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32),
                fb.Param("w", u32), fb.Param("q", u32)},
               u32);
  BValue index = fb.Param("i", u2);
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("i"), {m::Param("x"), m::Param("y"),
                                        m::Param("z"), m::Param("w")}));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperationWithLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Literal(Value(UBits(2, 16)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperationWithOobLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Literal(Value(UBits(5, 16)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayUpdateOperationSameIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue update_index = fb.Literal(Value(UBits(2, 16)));
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue index = fb.Literal(Value(UBits(2, 32)));
  fb.ArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("q"));
}

TEST_F(ArraySimplificationPassTest, OobArrayIndexOfArrayUpdateSameIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue update_index = fb.Literal(Value(UBits(2, 16)));
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue oob_index = fb.Literal(Value(UBits(1000, 32)));
  fb.ArrayIndex(array_update, {oob_index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // The ArrayIndex should be replaced by the update value of the ArrayUpdate
  // operation because the OOB ArrayIndex index is clamped to 2, the same index
  // which is updated by the ArrayUpdate.
  EXPECT_THAT(f->return_value(), m::Param("q"));
}

TEST_F(ArraySimplificationPassTest, OobArrayIndexOfArrayUpdateDifferentIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue update_index = fb.Literal(Value(UBits(1, 16)));
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue oob_index = fb.Literal(Value(UBits(1000, 32)));
  fb.ArrayIndex(array_update, {oob_index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(ArraySimplificationPassTest,
       IndexingArrayUpdateOperationDifferentIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue update_index = fb.Literal(Value(UBits(1, 16)));
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue index = fb.Literal(Value(UBits(2, 32)));
  fb.ArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayUpdateOperationUnknownIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue update_index = fb.Param("idx", u32);
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.ArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), {m::Literal(1)}));
}

TEST_F(ArraySimplificationPassTest,
       IndexingArrayUpdateOperationUnknownButSameIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Param("idx", u32);
  BValue array_update = fb.ArrayUpdate(a, fb.Param("q", u32), {index});
  fb.ArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // Though this is an array_index of an array_update and they have the same
  // index, we can't optimize this because of different behaviors of OOB access
  // for array_index and array_update operations.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::ArrayUpdate(),
                                               /*indices=*/{m::Param()}));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayParameter) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", p->GetArrayType(42, u32));
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param(), /*indices=*/{m::Literal()}));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param(), /*indices=*/{m::Literal()}));
}

TEST_F(ArraySimplificationPassTest, SimpleUnboxingArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[2]) -> bits[2] {
  a: bits[2][1] = array(x)
  zero: bits[1] = literal(value=0)
  ret array_index.4: bits[2] = array_index(a, indices=[zero])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(ArraySimplificationPassTest, UnboxingTwoElementArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[2], y: bits[2]) -> bits[2] {
  a: bits[2][2] = array(x, y)
  zero: bits[1] = literal(value=0)
  one: bits[1] = literal(value=1)
  element_0: bits[2] = array_index(a, indices=[zero])
  element_1: bits[2] = array_index(a, indices=[one])
  ret sum: bits[2] = add(element_0, element_1)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Param("x"), m::Param("y")));
}

TEST_F(ArraySimplificationPassTest, SequentialArrayUpdatesToSameLocation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7], idx: bits[32], x: bits[32], y: bits[32]) -> bits[32][7] {
  update0: bits[32][7] = array_update(a, x, indices=[idx])
  ret update1: bits[32][7] = array_update(update0, y, indices=[idx])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ArrayUpdate(m::Param("a"), m::Param("y"),
                                                /*indices=*/{m::Param("idx")}));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToSameLocationWithMultipleUses) {
  // Cannot squash the first update of sequential updates to the same location
  // if one of the updates in the chain has multiple uses.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7], idx0: bits[32], idx1: bits[32], x: bits[32], y: bits[32]) -> (bits[32][7], bits[32][7], bits[32][7]) {
  update0: bits[32][7] = array_update(a, x, indices=[idx0])
  update1: bits[32][7] = array_update(update0, y, indices=[idx1])
  update2: bits[32][7] = array_update(update1, y, indices=[idx0])
  ret result: (bits[32][7], bits[32][7], bits[32][7]) = tuple(update0, update1, update2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToSameLocationMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7][8][9], i: bits[32], j: bits[32], k: bits[32], x: bits[32], y: bits[32]) -> bits[32][7][8][9] {
  update0: bits[32][7][8][9] = array_update(a, x, indices=[i, j, k])
  ret update1: bits[32][7][8][9] = array_update(update0, y, indices=[i, j, k])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"), m::Param("y"),
                  /*indices=*/{m::Param("i"), m::Param("j"), m::Param("k")}));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToDifferentLocationMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7][8][9], i: bits[32], j: bits[32], k: bits[32], x: bits[32], y: bits[32]) -> bits[32][7][8][9] {
  update0: bits[32][7][8][9] = array_update(a, x, indices=[j, i, k])
  ret update1: bits[32][7][8][9] = array_update(update0, y, indices=[i, j, k])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       NonSequentialArrayUpdatesToSameLocationMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7][8][9], i: bits[32], j: bits[32], x: bits[32], y: bits[32]) -> bits[32][7][8][9] {
  one: bits[32] = literal(value=1)
  two: bits[32] = literal(value=2)
  update0: bits[32][7][8][9] = array_update(a, x, indices=[one, i, j])
  update1: bits[32][7][8][9] = array_update(update0, x, indices=[two, i, j])
  ret update2: bits[32][7][8][9] = array_update(update1, y, indices=[one, i, j])
 }
  )",
                                                       p.get()));
  // The first update (update0) can be elided because update2 overwrites the
  // same location.
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(
          m::ArrayUpdate(
              m::Param("a"), m::Param("x"),
              /*indices=*/{m::Literal(2), m::Param("i"), m::Param("j")}),
          m::Param("y"),
          /*indices=*/{m::Literal(1), m::Param("i"), m::Param("j")}));
}

TEST_F(ArraySimplificationPassTest,
       NonSequentialArrayUpdatesToSameLocationMultidimensionalDifferentSizes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7][8][9], i: bits[32], j: bits[32], x: bits[32], y: bits[32][7]) -> bits[32][7][8][9] {
  one: bits[32] = literal(value=1)
  two: bits[32] = literal(value=2)
  update0: bits[32][7][8][9] = array_update(a, x, indices=[one, i, j])
  update1: bits[32][7][8][9] = array_update(update0, x, indices=[two, i, j])
  ret update2: bits[32][7][8][9] = array_update(update1, y, indices=[one, i])
 }
  )",
                                                       p.get()));
  // The first update (update0) can be elided because update2 overwrites the
  // same location.
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(
          m::ArrayUpdate(
              m::Param("a"), m::Param("x"),
              /*indices=*/{m::Literal(2), m::Param("i"), m::Param("j")}),
          m::Param("y"), /*indices=*/{m::Literal(1), m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToSameLiteralLocation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7], x: bits[32], y: bits[32]) -> bits[32][7] {
  one: bits[4] = literal(value=1)
  update0: bits[32][7] = array_update(a, x, indices=[one])
  big_one: bits[1234] = literal(value=1)
  ret update1: bits[32][7] = array_update(update0, y, indices=[big_one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::ArrayUpdate(m::Param("a"), m::Param("y"),
                                                /*indices=*/{m::Literal(1)}));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToNonliteralLocations) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], idx0: bits[2], x: bits[32], idx1: bits[8], y: bits[32]) -> bits[32][4] {
  update0: bits[32][4] = array_update(a, x, indices=[idx0])
  ret update1: bits[32][4] = array_update(update0, y, indices=[idx1])
 }
  )",
                                                       p.get()));
  solvers::z3::ScopedVerifyEquivalence stays_equivalent(f);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Array(m::PrioritySelect(
                   m::Concat(m::Eq(m::ZeroExt(m::Param("idx0")), m::Literal(0)),
                             m::Eq(m::Param("idx1"), m::Literal(0))),
                   {m::Param("y"), m::Param("x")},
                   m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(0)})),
               m::PrioritySelect(
                   m::Concat(m::Eq(m::ZeroExt(m::Param("idx0")), m::Literal(1)),
                             m::Eq(m::Param("idx1"), m::Literal(1))),
                   {m::Param("y"), m::Param("x")},
                   m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(1)})),
               m::PrioritySelect(
                   m::Concat(m::Eq(m::ZeroExt(m::Param("idx0")), m::Literal(2)),
                             m::Eq(m::Param("idx1"), m::Literal(2))),
                   {m::Param("y"), m::Param("x")},
                   m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(2)})),
               m::PrioritySelect(
                   m::Concat(m::Eq(m::ZeroExt(m::Param("idx0")), m::Literal(3)),
                             m::Eq(m::Param("idx1"), m::Literal(3))),
                   {m::Param("y"), m::Param("x")},
                   m::ArrayIndex(m::Param("a"), /*indices=*/{m::Literal(3)}))));
}

TEST_F(ArraySimplificationPassTest, ArrayConstructedBySequenceOfArrayUpdates) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  a: bits[32][4] = literal(value=[999, 888, 777, 666])
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, x, indices=[one])
  update2: bits[32][4] = array_update(update1, y, indices=[two])
  ret update3: bits[32][4] = array_update(update2, z, indices=[three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesFromParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, x, indices=[one])
  update2: bits[32][4] = array_update(update1, y, indices=[two])
  ret update3: bits[32][4] = array_update(update2, z, indices=[three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesDifferentOrder) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  a: bits[32][4] = literal(value=[999, 888, 777, 666])
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, y, indices=[two])
  update2: bits[32][4] = array_update(update1, z, indices=[three])
  ret update3: bits[32][4] = array_update(update2, x, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesDifferentOrderFromParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, y, indices=[two])
  update2: bits[32][4] = array_update(update1, z, indices=[three])
  ret update3: bits[32][4] = array_update(update2, x, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       MultidimensionalArrayConstructedBySequenceOfArrayUpdates) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4][5], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4][5] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4][5] = array_update(a, w, indices=[two, zero])
  update1: bits[32][4][5] = array_update(update0, x, indices=[two, one])
  update2: bits[32][4][5] = array_update(update1, y, indices=[two, two])
  ret update3: bits[32][4][5] = array_update(update2, z, indices=[two, three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"),
                             m::Array(m::Param("w"), m::Param("x"),
                                      m::Param("y"), m::Param("z")),
                             /*indices=*/{m::Literal(2)}));
}

TEST_F(
    ArraySimplificationPassTest,
    MultidimensionalArrayConstructedBySequenceOfArrayUpdatesDifferentPrefix) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5][5], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][5][5] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][5][5] = array_update(a, w, indices=[two, zero])
  update1: bits[32][5][5] = array_update(update0, x, indices=[one, one])
  update2: bits[32][5][5] = array_update(update1, y, indices=[two, two])
  ret update3: bits[32][5][5] = array_update(update2, z, indices=[one, three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesNotDense) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  a: bits[32][4] = literal(value=[999, 888, 777, 666])
  zero: bits[4] = literal(value=0)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, y, indices=[two])
  ret update2: bits[32][4] = array_update(update1, z, indices=[three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Literal(888),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesDuplicateIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  a: bits[32][4] = literal(value=[999, 888, 777, 666])
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, w, indices=[zero])
  update1: bits[32][4] = array_update(update0, y, indices=[two])
  update2: bits[32][4] = array_update(update1, z, indices=[three])
  update3: bits[32][4] = array_update(update2, x, indices=[one])
  ret update4: bits[32][4] = array_update(update3, w, indices=[two])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("w"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest, SimplifyDecomposedArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  element_0: bits[32] = array_index(a, indices=[zero])
  element_1: bits[32] = array_index(a, indices=[one])
  element_2: bits[32] = array_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("a"));
}

TEST_F(ArraySimplificationPassTest, SimplifyDecomposedNestedArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][123][55]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  forty: bits[10] = literal(value=40)
  other_forty: bits[55] = literal(value=40)
  fifty: bits[10] = literal(value=50)
  other_fifty: bits[1000] = literal(value=50)
  element_0: bits[32] = array_index(a, indices=[forty, fifty, zero])
  element_1: bits[32] = array_index(a, indices=[other_forty, fifty, one])
  element_2: bits[32] = array_index(a, indices=[forty, other_fifty, two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param("a"),
                            /*indices=*/{m::Literal(40), m::Literal(50)}));
}

TEST_F(ArraySimplificationPassTest, SimplifyDecomposedArraySwizzledElements) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  element_0: bits[32] = array_index(a, indices=[one])
  element_1: bits[32] = array_index(a, indices=[zero])
  element_2: bits[32] = array_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Array());
}

TEST_F(ArraySimplificationPassTest, SimplifyDecomposedArrayMismatchingType) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  element_0: bits[32] = array_index(a, indices=[zero])
  element_1: bits[32] = array_index(a, indices=[one])
  element_2: bits[32] = array_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Array());
}

TEST_F(ArraySimplificationPassTest, ChainedArrayUpdate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][16][64][256], x: bits[8], y: bits[6], z: bits[4]) -> bits[32][16][64][256] {
  subarray: bits[32][16] = array_index(a, indices=[x,y])
  value: bits[32] = literal(value=42)
  update: bits[32][16] = array_update(subarray, value, indices=[z])
  ret update2: bits[32][16][64][256] = array_update(a, update, indices=[x,y])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"), m::Literal(42),
                  /*indices=*/{m::Param("x"), m::Param("y"), m::Param("z")}));
}

TEST_F(ArraySimplificationPassTest, IndexOfSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][4], b: bits[32][3][4], i: bits[10], j: bits[10], p: bits[1]) -> bits[32] {
  sel: bits[32][3][4] = sel(p, cases=[a, b])
  ret result: bits[32] = array_index(sel, indices=[i, j])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(
          m::Param("p"), /*cases=*/{
              m::ArrayIndex(m::Param("a"), {m::Param("i"), m::Param("j")}),
              m::ArrayIndex(m::Param("b"), {m::Param("i"), m::Param("j")})}));
}

TEST_F(ArraySimplificationPassTest, IndexOfLargeSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][4], b: bits[32][3][4], c: bits[32][3][4], d: bits[32][3][4], i: bits[10], j: bits[10], p: bits[3]) -> bits[32] {
  sel: bits[32][3][4] = sel(p, cases=[a, b, c], default=d)
  ret result: bits[32] = array_index(sel, indices=[i, j])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("p"), /*cases=*/
                {m::ArrayIndex(m::Param("a"), {m::Param("i"), m::Param("j")}),
                 m::ArrayIndex(m::Param("b"), {m::Param("i"), m::Param("j")}),
                 m::ArrayIndex(m::Param("c"), {m::Param("i"), m::Param("j")})},
                /*default_value=*/
                m::ArrayIndex(m::Param("d"), {m::Param("i"), m::Param("j")})));
}

TEST_F(ArraySimplificationPassTest, IndexOfPrioritySelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][4], b: bits[32][3][4], c: bits[32][3][4], d: bits[32][3][4], i: bits[10], j: bits[10], p: bits[3]) -> bits[32] {
  sel: bits[32][3][4] = priority_sel(p, cases=[a, b, c], default=d)
  ret result: bits[32] = array_index(sel, indices=[i, j])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::PrioritySelect(
          m::Param("p"), /*cases=*/
          {m::ArrayIndex(m::Param("a"), {m::Param("i"), m::Param("j")}),
           m::ArrayIndex(m::Param("b"), {m::Param("i"), m::Param("j")}),
           m::ArrayIndex(m::Param("c"), {m::Param("i"), m::Param("j")})},
          /*default_value=*/
          m::ArrayIndex(m::Param("d"), {m::Param("i"), m::Param("j")})));
}

TEST_F(ArraySimplificationPassTest, SelectAmongArrayUpdates) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5][47], x: bits[10], y: bits[24],
         pred: bits[1], value0: bits[32], value1: bits[32]) -> bits[32][5][47] {
  update0: bits[32][5][47] = array_update(a, value0, indices=[x, y])
  update1: bits[32][5][47] = array_update(a, value1, indices=[x, y])
  ret result: bits[32][5][47] = sel(pred, cases=[update0, update1])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"),
                  m::Select(m::Param("pred"),
                            /*cases=*/{m::Param("value0"), m::Param("value1")}),
                  /*indices=*/{m::Param("x"), m::Param("y")}));
}

TEST_F(ArraySimplificationPassTest, PrioritySelectAmongArrayUpdates) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5][47], x: bits[10], y: bits[24],
         pred: bits[1], value0: bits[32], value1: bits[32]) -> bits[32][5][47] {
  update0: bits[32][5][47] = array_update(a, value0, indices=[x, y])
  update1: bits[32][5][47] = array_update(a, value1, indices=[x, y])
  ret result: bits[32][5][47] = priority_sel(pred, cases=[update1], default=update0)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(m::Param("a"),
                     m::PrioritySelect(m::Param("pred"),
                                       /*cases=*/{m::Param("value1")},
                                       /*default_value=*/m::Param("value0")),
                     /*indices=*/{m::Param("x"), m::Param("y")}));
}

TEST_F(ArraySimplificationPassTest, SelectAmongArrayUpdatesOfDifferentArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5][47], b: bits[32][5][47], x: bits[10], y: bits[24],
         pred: bits[1], value0: bits[32], value1: bits[32]) -> bits[32][5][47] {
  update0: bits[32][5][47] = array_update(a, value0, indices=[x, y])
  update1: bits[32][5][47] = array_update(b, value1, indices=[x, y])
  ret result: bits[32][5][47] = sel(pred, cases=[update0, update1])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       PrioritySelectAmongArrayUpdatesOfDifferentArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][5][47], b: bits[32][5][47], x: bits[10], y: bits[24],
         pred: bits[1], value0: bits[32], value1: bits[32]) -> bits[32][5][47] {
  update0: bits[32][5][47] = array_update(a, value0, indices=[x, y])
  update1: bits[32][5][47] = array_update(b, value1, indices=[x, y])
  ret result: bits[32][5][47] = priority_sel(pred, cases=[update1], default=update0)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest, NilIndexUpdate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], v: bits[32][4]) -> bits[32][4] {
  ret result: bits[32][4] = array_update(a, v, indices=[])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("v"));
}

TEST_F(ArraySimplificationPassTest, UpdateOfArrayOp) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[32], y: bits[32], z: bits[32], v: bits[32]) -> bits[32][3] {
  a: bits[32][3] = array(x, y, z)
  one: bits[16] = literal(value=1)
  ret results: bits[32][3] = array_update(a, v, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::Param("x"), m::Param("v"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest, UpdateOfArrayOpMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[32][6], y: bits[32][6], z: bits[32][6], v: bits[32], i: bits[17]) -> bits[32][6][3] {
  a: bits[32][6][3] = array(x, y, z)
  one: bits[16] = literal(value=1)
  ret results: bits[32][6][3] = array_update(a, v, indices=[one, i])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::Param("x"),
                       m::ArrayUpdate(m::Param("y"), m::Param("v"),
                                      /*indices=*/{m::Param("i")}),
                       m::Param("z")));
}

TEST_F(ArraySimplificationPassTest, NestedArrayUpdate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][6][7], v: bits[32]) -> bits[32][6][7] {
  one: bits[16] = literal(value=1)
  two: bits[16] = literal(value=2)
  orig_element: bits[32][6] = array_index(a, indices=[one])
  update0: bits[32][6] = array_update(orig_element, v, indices=[two])
  ret result: bits[32][6][7] = array_update(a, update0, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("v"),
                             /*indices=*/{m::Literal(1), m::Literal(2)}));
}

TEST_F(ArraySimplificationPassTest, NestedArrayUpdateDifferentIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][6][7], v: bits[32]) -> bits[32][6][7] {
  one: bits[16] = literal(value=1)
  two: bits[16] = literal(value=2)
  three: bits[16] = literal(value=3)
  orig_element: bits[32][6] = array_index(a, indices=[one])
  update0: bits[32][6] = array_update(orig_element, v, indices=[two])
  ret result: bits[32][6][7] = array_update(a, update0, indices=[three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest, NestedArrayUpdateUnknownIndex) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][6][7], v: bits[32], i: bits[16]) -> bits[32][6][7] {
  one: bits[16] = literal(value=1)
  orig_element: bits[32][6] = array_index(a, indices=[i])
  update0: bits[32][6] = array_update(orig_element, v, indices=[one])
  ret result: bits[32][6][7] = array_update(a, update0, indices=[i])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest, NestedArrayUpdateMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][6][7][8][9], v: bits[32][6], i: bits[32]) -> bits[32][6][7][8][9] {
  one: bits[16] = literal(value=1)
  two: bits[16] = literal(value=2)
  orig_element: bits[32][6][7] = array_index(a, indices=[one, two])
  update0: bits[32][6][7] = array_update(orig_element, v, indices=[i])
  ret result: bits[32][6][7][8][9] = array_update(a, update0, indices=[one, two])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"), m::Param("v"),
                  /*indices=*/{m::Literal(1), m::Literal(2), m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest, UpdateOfLiteralArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(v: bits[32]) -> bits[32][3] {
  one: bits[14] = literal(value=1)
  a: bits[32][3] = literal(value=[11,22,33])
  ret update: bits[32][3] = array_update(a, v, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::Literal(11), m::Param("v"), m::Literal(33)));
}

TEST_F(ArraySimplificationPassTest, UpdateOfLiteralArrayNested) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(v: bits[32]) -> bits[32][3][2] {
  zero: bits[14] = literal(value=0)
  one: bits[14] = literal(value=1)
  a: bits[32][3][2] = literal(value=[[11,22,33], [44,55,66]])
  ret update: bits[32][3][2] = array_update(a, v, indices=[zero, one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Array(m::Array(m::Literal(11), m::Param("v"), m::Literal(33)),
               m::Literal(Value::UBitsArray({44, 55, 66}, 32).value())));
}

TEST_F(ArraySimplificationPassTest,
       ConditionalAssignmentOfArrayElementUpdatedOnTrue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32]) -> bits[32][4] {
  one: bits[14] = literal(value=1)
  updated_a: bits[32][4] = array_update(a, v, indices=[one])
  ret result: bits[32][4] = sel(p, cases=[a, updated_a])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(
          m::Param("a"),
          m::Select(m::Param("p"),
                    /*cases=*/{m::ArrayIndex(m::Param("a"),
                                             /*indices=*/{m::Literal(1)}),
                               m::Param("v")}),
          /*indices=*/{m::Literal(1)}));
}

TEST_F(ArraySimplificationPassTest,
       ConditionalAssignmentOfArrayElementWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32]) -> (bits[32][4], bits[32][4]) {
  one: bits[14] = literal(value=1)
  updated_a: bits[32][4] = array_update(a, v, indices=[one])
  sel: bits[32][4] = sel(p, cases=[a, updated_a])
  ret result: (bits[32][4], bits[32][4]) = tuple(updated_a, sel)
 }
  )",
                                                       p.get()));
  // Multiple uses of the updated array prevents the optimization.
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       ConditionalAssignmentOfArrayElementUpdatedOnFalse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32], i: bits[16]) -> bits[32][4] {
  updated_a: bits[32][4] = array_update(a, v, indices=[i])
  ret result: bits[32][4] = sel(p, cases=[updated_a, a])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(
          m::Param("a"),
          m::Select(m::Param("p"),
                    /*cases=*/{m::Param("v"),
                               m::ArrayIndex(m::Param("a"),
                                             /*indices=*/{m::Param("i")})}),
          /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       ConditionalAssignmentOfArrayElementWithMultipleCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[3], a: bits[32][4], v1: bits[32], v3: bits[32], i: bits[16]) -> bits[32][4] {
  a_update1: bits[32][4] = array_update(a, v1, indices=[i])
  a_update3: bits[32][4] = array_update(a, v3, indices=[i])
  ret result: bits[32][4] = sel(p, cases=[a, a_update1, a, a_update3], default=a)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(
          m::Param("a"),
          m::Select(m::Param("p"),
                    /*cases=*/
                    {m::ArrayIndex(m::Param("a"),
                                   /*indices=*/{m::Param("i")}),
                     m::Param("v1"),
                     m::ArrayIndex(m::Param("a"),
                                   /*indices=*/{m::Param("i")}),
                     m::Param("v3")},
                    /*default_value=*/
                    m::ArrayIndex(m::Param("a"), /*indices=*/{m::Param("i")})),
          /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       ConditionalAssignmentOfArrayElementWithMultipleCasesAllChanged) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v1: bits[32], v2: bits[32], i: bits[16]) -> bits[32][4] {
  a_update1: bits[32][4] = array_update(a, v1, indices=[i])
  a_update2: bits[32][4] = array_update(a, v2, indices=[i])
  ret result: bits[32][4] = sel(p, cases=[a_update1, a_update2])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"),
                             m::Select(m::Param("p"),
                                       /*cases=*/
                                       {m::Param("v1"), m::Param("v2")}),
                             /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       PriorityConditionalAssignmentOfArrayElementUpdatedOnTrue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32]) -> bits[32][4] {
  one: bits[14] = literal(value=1)
  updated_a: bits[32][4] = array_update(a, v, indices=[one])
  ret result: bits[32][4] = priority_sel(p, cases=[updated_a], default=a)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"),
                  m::PrioritySelect(m::Param("p"),
                                    /*cases=*/{m::Param("v")},
                                    /*default_value=*/
                                    m::ArrayIndex(m::Param("a"),
                                                  /*indices=*/{m::Literal(1)})),
                  /*indices=*/{m::Literal(1)}));
}

TEST_F(ArraySimplificationPassTest,
       PriorityConditionalAssignmentOfArrayElementWithMultipleUses) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32]) -> (bits[32][4], bits[32][4]) {
  one: bits[14] = literal(value=1)
  updated_a: bits[32][4] = array_update(a, v, indices=[one])
  sel: bits[32][4] = priority_sel(p, cases=[updated_a], default=a)
  ret result: (bits[32][4], bits[32][4]) = tuple(updated_a, sel)
 }
  )",
                                                       p.get()));
  // Multiple uses of the updated array prevents the optimization.
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(ArraySimplificationPassTest,
       PriorityConditionalAssignmentOfArrayElementUpdatedOnFalse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v: bits[32], i: bits[16]) -> bits[32][4] {
  updated_a: bits[32][4] = array_update(a, v, indices=[i])
  ret result: bits[32][4] = priority_sel(p, cases=[a], default=updated_a)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"),
                             m::PrioritySelect(
                                 m::Param("p"),
                                 /*cases=*/
                                 {m::ArrayIndex(m::Param("a"),
                                                /*indices=*/{m::Param("i")})},
                                 /*default_value=*/m::Param("v")),
                             /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       PriorityConditionalAssignmentOfArrayElementWithMultipleCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[4], a: bits[32][4], v1: bits[32], v3: bits[32], i: bits[16]) -> bits[32][4] {
  a_update1: bits[32][4] = array_update(a, v1, indices=[i])
  a_update3: bits[32][4] = array_update(a, v3, indices=[i])
  ret result: bits[32][4] = priority_sel(p, cases=[a, a_update1, a, a_update3], default=a)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(
                  m::Param("a"),
                  m::PrioritySelect(m::Param("p"),
                                    /*cases=*/
                                    {m::ArrayIndex(m::Param("a"),
                                                   /*indices=*/{m::Param("i")}),
                                     m::Param("v1"),
                                     m::ArrayIndex(m::Param("a"),
                                                   /*indices=*/{m::Param("i")}),
                                     m::Param("v3")},
                                    /*default_value=*/
                                    m::ArrayIndex(m::Param("a"),
                                                  /*indices=*/{m::Param("i")})),
                  /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest,
       PriorityConditionalAssignmentOfArrayElementWithMultipleCasesAllChanged) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], a: bits[32][4], v1: bits[32], v2: bits[32], i: bits[16]) -> bits[32][4] {
  a_update1: bits[32][4] = array_update(a, v1, indices=[i])
  a_update2: bits[32][4] = array_update(a, v2, indices=[i])
  ret result: bits[32][4] = priority_sel(p, cases=[a_update1], default=a_update2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ArrayUpdate(m::Param("a"),
                     m::PrioritySelect(m::Param("p"),
                                       /*cases=*/{m::Param("v1")},
                                       /*default_value=*/m::Param("v2")),
                     /*indices=*/{m::Param("i")}));
}

TEST_F(ArraySimplificationPassTest, SimplifySelectOfArrays) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][2] {
  a: bits[32][2] = array(w, x)
  b: bits[32][2] = array(y, z)
  ret result: bits[32][2] = sel(p, cases=[a, b])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::Select(m::Param("p"),
                                 /*cases=*/{m::Param("w"), m::Param("y")}),
                       m::Select(m::Param("p"),
                                 /*cases=*/{m::Param("x"), m::Param("z")})));
}

TEST_F(ArraySimplificationPassTest, SimplifyPrioritySelectOfArrays) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(p: bits[1], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][2] {
  a: bits[32][2] = array(w, x)
  d: bits[32][2] = array(y, z)
  ret result: bits[32][2] = priority_sel(p, cases=[a], default=d)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::PrioritySelect(m::Param("p"),
                                         /*cases=*/{m::Param("w")},
                                         /*default_value=*/m::Param("y")),
                       m::PrioritySelect(m::Param("p"),
                                         /*cases=*/{m::Param("x")},
                                         /*default_value=*/m::Param("z"))));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayConcat) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* subarray_type = p->GetArrayType(42, p->GetBitsType(10));
  BValue a = fb.Param("A", p->GetArrayType(10, subarray_type));
  BValue b = fb.Param("B", p->GetArrayType(20, subarray_type));
  BValue c = fb.Param("C", p->GetArrayType(30, subarray_type));
  BValue concat = fb.ArrayConcat({a, b, c});
  fb.ArrayIndex(concat,
                /*indices=*/{fb.Literal(Value(UBits(15, 32))),
                             fb.Param("x", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::Param("B"), {m::Literal(5), m::Param("x")}));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayConcatNonConstant) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* subarray_type = p->GetArrayType(42, p->GetBitsType(10));
  BValue a = fb.Param("A", p->GetArrayType(10, subarray_type));
  BValue b = fb.Param("B", p->GetArrayType(20, subarray_type));
  BValue c = fb.Param("C", p->GetArrayType(30, subarray_type));
  BValue concat = fb.ArrayConcat({a, b, c});
  fb.ArrayIndex(concat,
                /*indices=*/{fb.Param("x", p->GetBitsType(32)),
                             fb.Param("y", p->GetBitsType(32))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayConcat(), {m::Param(), m::Param()}));
}

}  // namespace
}  // namespace xls
