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
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"

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
                               f, PassOptions(), &results));
      // Run dce to clean things up.
      XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                              .RunOnFunctionBase(f, PassOptions(), &results)
                              .status());
      changed |= changed_this_iteration;
    }

    return changed;
  }
};

TEST_F(ArraySimplificationPassTest, ConvertArrayIndexToMultiArrayIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(42, p->GetBitsType(100)));
  BValue idx = fb.Param("idx", p->GetBitsType(32));
  fb.ArrayIndex(a, idx);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Param("idx")));
}

TEST_F(ArraySimplificationPassTest, ArrayWithOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(3, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(UBits(123, 32)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Index should be clipped at the max legal index.
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Literal(2)));
}

TEST_F(ArraySimplificationPassTest, ArrayWithWideOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(42, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(Bits::AllOnes(1234)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  // Index should be clipped at the max legal index.
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Literal(41)));
}

TEST_F(ArraySimplificationPassTest, ArrayWithWideInBoundsLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetArrayType(3, p->GetBitsType(32)));
  BValue index = fb.Literal(Value(UBits(1, 1000)));
  fb.MultiArrayIndex(a, {index});
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
  fb.MultiArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::MultiArrayIndex());
}

TEST_F(ArraySimplificationPassTest, IndexingArrayOperation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Literal(Value(UBits(2, 16)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
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
  BValue array_update = fb.ArrayUpdate(a, update_index, fb.Param("q", u32));
  BValue index = fb.Literal(Value(UBits(2, 32)));
  fb.ArrayIndex(array_update, index);
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
  BValue array_update = fb.ArrayUpdate(a, update_index, fb.Param("q", u32));
  BValue oob_index = fb.Literal(Value(UBits(1000, 32)));
  fb.ArrayIndex(array_update, oob_index);
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
  BValue array_update = fb.ArrayUpdate(a, update_index, fb.Param("q", u32));
  BValue oob_index = fb.Literal(Value(UBits(1000, 32)));
  fb.ArrayIndex(array_update, oob_index);
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
  BValue array_update = fb.ArrayUpdate(a, update_index, fb.Param("q", u32));
  BValue index = fb.Literal(Value(UBits(2, 32)));
  fb.ArrayIndex(array_update, index);
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
  BValue array_update =
      fb.MultiArrayUpdate(a, fb.Param("q", u32), {update_index});
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.MultiArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::MultiArrayUpdate(), m::Literal(1)));
}

TEST_F(ArraySimplificationPassTest,
       IndexingArrayUpdateOperationUnknownButSameIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Array(
      {fb.Param("x", u32), fb.Param("y", u32), fb.Param("z", u32)}, u32);
  BValue index = fb.Param("idx", u32);
  BValue array_update = fb.MultiArrayUpdate(a, fb.Param("q", u32), {index});
  fb.MultiArrayIndex(array_update, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // Though this is an array_index of an array_update and they have the same
  // index, we can't optimize this because of different behaviors of OOB access
  // for array_index and array_update operations.
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::MultiArrayUpdate(), m::Param()));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayParameter) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", p->GetArrayType(42, u32));
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.MultiArrayIndex(a, {index});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), m::MultiArrayIndex(m::Param(), m::Literal()));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::MultiArrayIndex(m::Param(), m::Literal()));
}

TEST_F(ArraySimplificationPassTest, SimpleUnboxingArray) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[2]) -> bits[2] {
  a: bits[2][1] = array(x)
  zero: bits[1] = literal(value=0)
  ret multiarray_index.4: bits[2] = multiarray_index(a, indices=[zero])
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
  element_0: bits[2] = multiarray_index(a, indices=[zero])
  element_1: bits[2] = multiarray_index(a, indices=[one])
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
  update0: bits[32][7] = array_update(a, idx, x)
  ret update1: bits[32][7] = array_update(update0, idx, y)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Param("idx"), m::Param("y")));
}

TEST_F(ArraySimplificationPassTest,
       SequentialArrayUpdatesToSameLiteralLocation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][7], x: bits[32], y: bits[32]) -> bits[32][7] {
  one: bits[4] = literal(value=1)
  update0: bits[32][7] = array_update(a, one, x)
  big_one: bits[1234] = literal(value=1)
  ret update1: bits[32][7] = array_update(update0, big_one, y)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ArrayUpdate(m::Param("a"), m::Literal(1), m::Param("y")));
}

TEST_F(ArraySimplificationPassTest, ArrayConstructedBySequenceOfArrayUpdates) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, zero, w)
  update1: bits[32][4] = array_update(update0, one, x)
  update2: bits[32][4] = array_update(update1, two, y)
  ret update3: bits[32][4] = array_update(update2, three, z)
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
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, zero, w)
  update1: bits[32][4] = array_update(update0, two, y)
  update2: bits[32][4] = array_update(update1, three, z)
  ret update3: bits[32][4] = array_update(update2, one, x)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Array(m::Param("w"), m::Param("x"),
                                          m::Param("y"), m::Param("z")));
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesNotDense) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, zero, w)
  update1: bits[32][4] = array_update(update0, two, y)
  ret update2: bits[32][4] = array_update(update1, three, z)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayUpdate());
}

TEST_F(ArraySimplificationPassTest,
       ArrayConstructedBySequenceOfArrayUpdatesDuplicate) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][4], w: bits[32], x: bits[32], y: bits[32], z: bits[32]) -> bits[32][4] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  three: bits[4] = literal(value=3)
  update0: bits[32][4] = array_update(a, zero, w)
  update1: bits[32][4] = array_update(update0, two, y)
  update2: bits[32][4] = array_update(update1, three, z)
  update3: bits[32][4] = array_update(update2, one, x)
  ret update4: bits[32][4] = array_update(update3, two, w)
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
  element_0: bits[32] = multiarray_index(a, indices=[zero])
  element_1: bits[32] = multiarray_index(a, indices=[one])
  element_2: bits[32] = multiarray_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("a"));
}

TEST_F(ArraySimplificationPassTest, DecomposedArrayDifferentSources) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3], b: bits[32][3]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  element_0: bits[32] = multiarray_index(a, indices=[zero])
  element_1: bits[32] = multiarray_index(b, indices=[one])
  element_2: bits[32] = multiarray_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Array());
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
  element_0: bits[32] = multiarray_index(a, indices=[forty, fifty, zero])
  element_1: bits[32] = multiarray_index(a, indices=[other_forty, fifty, one])
  element_2: bits[32] = multiarray_index(a, indices=[forty, other_fifty, two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::MultiArrayIndex(m::Param("a"), m::Literal(40), m::Literal(50)));
}

TEST_F(ArraySimplificationPassTest,
       DecomposedNestedArrayWithNonmatchingIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][123][55]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  forty: bits[10] = literal(value=40)
  fifty: bits[10] = literal(value=50)
  element_0: bits[32] = multiarray_index(a, indices=[forty, fifty, zero])
  element_1: bits[32] = multiarray_index(a, indices=[forty, forty, one])
  element_2: bits[32] = multiarray_index(a, indices=[forty, fifty, two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Array());
}

TEST_F(ArraySimplificationPassTest, SimplifyDecomposedArraySwizzledElements) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3]) -> bits[32][3] {
  zero: bits[4] = literal(value=0)
  one: bits[4] = literal(value=1)
  two: bits[4] = literal(value=2)
  element_0: bits[32] = multiarray_index(a, indices=[one])
  element_1: bits[32] = multiarray_index(a, indices=[zero])
  element_2: bits[32] = multiarray_index(a, indices=[two])
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
  element_0: bits[32] = multiarray_index(a, indices=[zero])
  element_1: bits[32] = multiarray_index(a, indices=[one])
  element_2: bits[32] = multiarray_index(a, indices=[two])
  ret array: bits[32][3] = array(element_0, element_1, element_2)
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::Array());
}

TEST_F(ArraySimplificationPassTest, ClampingArrayIndexIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][42][99]) -> bits[32] {
  zero: bits[44] = literal(value=0)
  hundred: bits[32] = literal(value=100)
  ret result: bits[32] = multiarray_index(a, indices=[hundred, hundred, zero])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Literal(98), m::Literal(41),
                                 m::Literal(0)));
}

TEST_F(ArraySimplificationPassTest, ArrayIndexWithEmptyIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][42][99]) -> bits[32][10][42][99] {
  ret result: bits[32][10][42][99] = multiarray_index(a, indices=[])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("a"));
}

TEST_F(ArraySimplificationPassTest, ArrayIndexWithEmptyIndicesBitsType) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32]) -> bits[32] {
  ret result: bits[32] = multiarray_index(a, indices=[])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("a"));
}

TEST_F(ArraySimplificationPassTest, IndexOfArrayOp) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  a: bits[32][3] = array(x, y, z)
  one: bits[32] = literal(value=1)
  ret result: bits[32] = multiarray_index(a, indices=[one])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("y"));
}

TEST_F(ArraySimplificationPassTest, IndexOfArrayOpMultidimensional) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(x: bits[32][3][4], y: bits[32][3][4], z: bits[32][3][4], i: bits[10], j: bits[10]) -> bits[32] {
  a: bits[32][3][4][3] = array(x, y, z)
  one: bits[32] = literal(value=1)
  ret result: bits[32] = multiarray_index(a, indices=[one, i, j])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("y"), m::Param("i"), m::Param("j")));
}

TEST_F(ArraySimplificationPassTest, ConsecutiveArrayIndices) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][4][5][6], i: bits[10], j: bits[10], k: bits[32], l: bits[100]) -> bits[32] {
  foo: bits[32][3][4] = multiarray_index(a, indices=[i, j])
  ret bar: bits[32] = multiarray_index(foo, indices=[k, l])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Param("i"), m::Param("j"),
                                 m::Param("k"), m::Param("l")));
}

TEST_F(ArraySimplificationPassTest, IndexOfSelect) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][3][4], b: bits[32][3][4], i: bits[10], j: bits[10], p: bits[1]) -> bits[32] {
  sel: bits[32][3][4] = sel(p, cases=[a, b])
  ret result: bits[32] = multiarray_index(sel, indices=[i, j])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Select(m::Param("p"), /*cases=*/{
                            m::MultiArrayIndex(m::Param("a"), m::Param("i"),
                                               m::Param("j")),
                            m::MultiArrayIndex(m::Param("b"), m::Param("i"),
                                               m::Param("j"))}));
}

TEST_F(ArraySimplificationPassTest, IndexOfUpdateIndicesMatch) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][11][12], v: bits[32][10]) -> bits[32] {
  one: bits[15] = literal(value=1)
  two: bits[15] = literal(value=2)
  three: bits[15] = literal(value=3)
  update: bits[32][10][11][12] = multiarray_update(a, v, indices=[one, two])
  ret result: bits[32] = multiarray_index(update, indices=[one, two, three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("v"), m::Literal(3)));
}

TEST_F(ArraySimplificationPassTest, IndexOfUpdateIndicesDoNotMatch) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][11][12], v: bits[32][10]) -> bits[32] {
  one: bits[15] = literal(value=1)
  two: bits[15] = literal(value=2)
  three: bits[15] = literal(value=3)
  update: bits[32][10][11][12] = multiarray_update(a, v, indices=[two, one])
  ret result: bits[32] = multiarray_index(update, indices=[one, two, three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::Param("a"), m::Literal(1), m::Literal(2),
                                 m::Literal(3)));
}

TEST_F(ArraySimplificationPassTest,
       IndexOfUpdateIndicesMatchButMightBeOutOfBounds) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][11][12], v: bits[32][10], i: bits[33]) -> bits[32] {
  two: bits[15] = literal(value=2)
  three: bits[15] = literal(value=3)
  update: bits[32][10][11][12] = multiarray_update(a, v, indices=[i, two])
  ret result: bits[32] = multiarray_index(update, indices=[i, two, three])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::MultiArrayIndex(m::MultiArrayUpdate(), m::Param("i"),
                                 m::Literal(2), m::Literal(3)));
}

TEST_F(ArraySimplificationPassTest, IndexOfUpdateIndicesIdentical) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn func(a: bits[32][10][11][12], v: bits[32][10], i: bits[3]) -> bits[32][10] {
  two: bits[15] = literal(value=2)
  update: bits[32][10][11][12] = multiarray_update(a, v, indices=[i, two])
  ret result: bits[32][10] = multiarray_index(update, indices=[i, two])
 }
  )",
                                                       p.get()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Param("v"));
}

}  // namespace
}  // namespace xls
