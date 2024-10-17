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

#include "xls/passes/dataflow_simplification_pass.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class DataflowSimplificationPassTest : public IrTestBase {
 protected:
  DataflowSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         DataflowSimplificationPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
            .status());
    return changed;
  }
};

TEST_F(DataflowSimplificationPassTest, EmptyTypesNotReplaced) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetTupleType({}));
  BValue b = fb.Param("b", p->GetTupleType({}));
  BValue c = fb.Param("c", p->GetTupleType({p->GetTupleType({})}));
  BValue d = fb.Tuple({});
  BValue e = fb.Tuple({fb.Tuple({})});
  fb.Tuple({a, b, c, d, e});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  // None of the empty types should be replaced by each other though they are
  // equivalent.
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
}

TEST_F(DataflowSimplificationPassTest, SingleSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x:bits[2], y:bits[42]) -> bits[42] {
        tuple.1: (bits[2], bits[42]) = tuple(x, y)
        ret tuple_index.2: bits[42] = tuple_index(tuple.1, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 4);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_F(DataflowSimplificationPassTest, NoSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: (bits[2], bits[42])) -> bits[42] {
        ret tuple_index.2: bits[42] = tuple_index(x, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 2);
}

TEST_F(DataflowSimplificationPassTest, NestedSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[3], z: bits[73]) -> bits[73] {
        tuple.1: (bits[42], bits[73]) = tuple(x, z)
        tuple.2: ((bits[42], bits[73]), bits[3]) = tuple(tuple.1, y)
        tuple.3: ((bits[42], bits[73]), ((bits[42], bits[73]), bits[3])) = tuple(tuple.1, tuple.2)
        tuple_index.4: ((bits[42], bits[73]), bits[3]) = tuple_index(tuple.3, index=1)
        tuple_index.5: (bits[42], bits[73]) = tuple_index(tuple_index.4, index=0)
        ret tuple_index.6: bits[73] = tuple_index(tuple_index.5, index=1)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(DataflowSimplificationPassTest, ChainOfTuplesSimplification) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[3]) -> bits[42] {
        tuple.1: (bits[42], bits[3]) = tuple(x, y)
        tuple_index.2: bits[42] = tuple_index(tuple.1, index=0)
        tuple.3: (bits[42], bits[3]) = tuple(tuple_index.2, y)
        tuple_index.4: bits[42] = tuple_index(tuple.3, index=0)
        tuple.5: (bits[42], bits[3]) = tuple(tuple_index.4, y)
        ret tuple_index.6: bits[42] = tuple_index(tuple.5, index=0)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 8);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 2);
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(DataflowSimplificationPassTest, TupleReductionEmptyTuple) {
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  fb.Tuple({});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 1);
}

TEST_F(DataflowSimplificationPassTest, TupleReductionDifferentSize) {
  const int64_t kTupleIndex = 0;
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  Type* u32 = p.GetBitsType(32);
  BValue x = fb.Param("x", p.GetTupleType({u32, u32}));
  fb.Tuple({fb.TupleIndex(x, kTupleIndex)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::TupleIndex(m::Param("x"), kTupleIndex)));
}

TEST_F(DataflowSimplificationPassTest, TupleReductionDifferentIndex) {
  const int64_t kTupleIndex0 = 0;
  const int64_t kTupleIndex1 = 1;
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  Type* u32 = p.GetBitsType(32);
  BValue x = fb.Param("x", p.GetTupleType({u32, u32}));
  fb.Tuple({fb.TupleIndex(x, kTupleIndex1), fb.TupleIndex(x, kTupleIndex0)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::TupleIndex(m::Param("x"), kTupleIndex1),
                       m::TupleIndex(m::Param("x"), kTupleIndex0)));
}

TEST_F(DataflowSimplificationPassTest, TupleReductionDifferentSubject) {
  const int64_t kTupleIndex0 = 0;
  const int64_t kTupleIndex1 = 1;
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  Type* u32 = p.GetBitsType(32);
  BValue x = fb.Param("x", p.GetTupleType({u32, u32}));
  BValue y = fb.Param("y", p.GetTupleType({u32, u32}));
  fb.Tuple({fb.TupleIndex(x, kTupleIndex0), fb.TupleIndex(y, kTupleIndex1)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::Tuple(m::TupleIndex(m::Param("x"), kTupleIndex0),
                       m::TupleIndex(m::Param("y"), kTupleIndex1)));
}

TEST_F(DataflowSimplificationPassTest, TupleReductionTupleIndex) {
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  Type* u32 = p.GetBitsType(32);
  BValue x = fb.Param("x", p.GetTupleType({u32, u32, u32}));
  fb.Tuple({fb.TupleIndex(x, 0), fb.TupleIndex(x, 1), fb.TupleIndex(x, 2)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 1);
  EXPECT_THAT(f->return_value(), m::Param("x"));
}

TEST_F(DataflowSimplificationPassTest, IndexingArrayOperation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Literal(Value(UBits(2, 16)));
  fb.ArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("z"));
}

TEST_F(DataflowSimplificationPassTest, IndexingArrayUpdateOperationSameIndex) {
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

TEST_F(DataflowSimplificationPassTest, OobArrayIndexOfArrayUpdateSameIndex) {
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

TEST_F(DataflowSimplificationPassTest,
       OobArrayIndexOfArrayUpdateDifferentIndex) {
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

TEST_F(DataflowSimplificationPassTest,
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

TEST_F(DataflowSimplificationPassTest,
       IndexingArrayUpdateOperationUnknownIndex) {
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

TEST_F(DataflowSimplificationPassTest,
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

TEST_F(DataflowSimplificationPassTest, IndexingArrayParameter) {
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

TEST_F(DataflowSimplificationPassTest, SimpleUnboxingArray) {
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

TEST_F(DataflowSimplificationPassTest, UnboxingTwoElementArray) {
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

TEST_F(DataflowSimplificationPassTest, SimplifyDecomposedArray) {
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

TEST_F(DataflowSimplificationPassTest,
       SimplifyDecomposedArraySwizzledElements) {
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

TEST_F(DataflowSimplificationPassTest, SimplifyDecomposedArrayMismatchingType) {
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

TEST_F(DataflowSimplificationPassTest, NilIndexUpdate) {
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

TEST_F(DataflowSimplificationPassTest, ArrayConstructedFromConcats) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue array_ab = fb.Array({a, b}, u32);
  BValue array_c = fb.Array({c}, u32);
  fb.ArrayConcat({array_ab, array_c});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Array(m::Param("a"), m::Param("b"), m::Param("c")));
}

TEST_F(DataflowSimplificationPassTest,
       ArrayConstructedFromConcatsSimplificationNotPossible) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue array_ab = fb.Param("ab", p->GetArrayType(2, u32));
  BValue c = fb.Param("c", u32);
  BValue array_c = fb.Array({c}, u32);
  fb.ArrayConcat({array_ab, array_c});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayConcat());
}

TEST_F(DataflowSimplificationPassTest, ArrayConstructedWithArrayUpdates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  BValue b = fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue zero = fb.Literal(UBits(0, 32));
  BValue x = fb.Array({zero, zero, zero}, u32);
  x = fb.ArrayUpdate(x, a, {fb.Literal(UBits(0, 8))});
  x = fb.ArrayUpdate(x, b, {fb.Literal(UBits(1, 8))});
  x = fb.ArrayUpdate(x, c, {fb.Literal(UBits(2, 8))});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Array(m::Param("a"), m::Param("b"), m::Param("c")));
}

TEST_F(DataflowSimplificationPassTest,
       ArrayConstructedWithPartialArrayUpdates) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", u32);
  fb.Param("b", u32);
  BValue c = fb.Param("c", u32);
  BValue zero = fb.Literal(UBits(0, 32));
  BValue x = fb.Array({zero, zero, zero}, u32);
  x = fb.ArrayUpdate(x, a, {fb.Literal(UBits(0, 8))});
  x = fb.ArrayUpdate(x, c, {fb.Literal(UBits(2, 8))});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_THAT(f->return_value(),
              m::Array(m::Param("a"), m::Literal(0), m::Param("c")));
}

}  // namespace
}  // namespace xls
