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

#include "xls/passes/tuple_simplification_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class TupleSimplificationPassTest : public IrTestBase {
 protected:
  TupleSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TupleSimplificationPass().RunOnFunctionBase(
                             f, PassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .RunOnFunctionBase(f, PassOptions(), &results)
                            .status());
    // Return whether tuple simplification changed anything.
    return changed;
  }
};

TEST_F(TupleSimplificationPassTest, SingleSimplification) {
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

TEST_F(TupleSimplificationPassTest, NoSimplification) {
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

TEST_F(TupleSimplificationPassTest, NestedSimplification) {
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

TEST_F(TupleSimplificationPassTest, ChainOfTuplesSimplification) {
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

TEST_F(TupleSimplificationPassTest, TupleReductionEmptyTuple) {
  Package p("p");
  FunctionBuilder fb(TestName(), &p);
  fb.Tuple({});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 1);
}

TEST_F(TupleSimplificationPassTest, TupleReductionDifferentSize) {
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

TEST_F(TupleSimplificationPassTest, TupleReductionDifferentIndex) {
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

TEST_F(TupleSimplificationPassTest, TupleReductionDifferentSubject) {
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

TEST_F(TupleSimplificationPassTest, TupleReductionTupleIndex) {
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

}  // namespace
}  // namespace xls
