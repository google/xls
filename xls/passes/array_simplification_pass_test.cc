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

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ArraySimplificationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed, ArraySimplificationPass().RunOnFunction(
                                           f, PassOptions(), &results));
    return changed;
  }
};

TEST_F(ArraySimplificationPassTest, LiteralArrayWithLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Literal(Value(UBits(1, 32)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(4));
}

TEST_F(ArraySimplificationPassTest, LiteralArrayWithOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Literal(Value(UBits(123, 32)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(6));
}

TEST_F(ArraySimplificationPassTest, LiteralArrayWithWideOOBLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Literal(Value(Bits::AllOnes(1234)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(6));
}

TEST_F(ArraySimplificationPassTest, LiteralArrayWithWideInBoundsLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Literal(Value(UBits(1, 1000)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Literal(4));
}

TEST_F(ArraySimplificationPassTest, LiteralArrayWithNonLiteralIndex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(
      Parser::ParseTypedValue("[bits[32]:2, bits[32]:4, bits[32]:6]").value());
  BValue index = fb.Param("idx", p->GetBitsType(32));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayIndex());
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
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(2)));
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

  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(1000)));
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

  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(1000)));
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
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(2)));
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
  BValue array_update = fb.ArrayUpdate(a, update_index, fb.Param("q", u32));
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.ArrayIndex(array_update, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(1)));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(),
              m::ArrayIndex(m::ArrayUpdate(), m::Literal(1)));
}

TEST_F(ArraySimplificationPassTest, IndexingArrayParameter) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue a = fb.Param("a", p->GetArrayType(42, u32));
  BValue index = fb.Literal(Value(UBits(1, 16)));
  fb.ArrayIndex(a, index);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Param(), m::Literal()));
  ASSERT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_THAT(f->return_value(), m::ArrayIndex(m::Param(), m::Literal()));
}

}  // namespace
}  // namespace xls
