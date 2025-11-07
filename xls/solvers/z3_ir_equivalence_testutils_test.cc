// Copyright 2025 The XLS Authors
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

#include "xls/solvers/z3_ir_equivalence_testutils.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls::solvers::z3 {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::ContainsRegex;
class Z3IrEquivalenceTestutilsTest : public IrTestBase {};

TEST_F(Z3IrEquivalenceTestutilsTest, DumpWithNodeValues) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Tuple({fb.Add(x, y), fb.UMul(x, y)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  absl::flat_hash_map<const Param*, Value> counterexample{
      {x.node()->As<Param>(), Value(UBits(1, 32))}};
  EXPECT_THAT(
      DumpWithNodeValues(
          f, ProvenFalse{.counterexample = std::move(counterexample)}),
      IsOkAndHolds(AllOf(ContainsRegex("x: bits\\[32\\] id=[0-9]+ \\(1\\)"),
                         ContainsRegex("y: bits\\[32\\] id=[0-9]+ \\(0\\)"))));
}

TEST_F(Z3IrEquivalenceTestutilsTest, EquivWithAssert) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  fb.Assert(fb.Literal(Value::Token()), fb.Eq(x, y), "foo");
  BValue add = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f, /*ignore_asserts=*/true);
  XLS_ASSERT_OK(
      add.node()
          ->ReplaceUsesWithNew<BinOp>(y.node(), x.node(), add.node()->op())
          .status());
  XLS_ASSERT_OK(f->RemoveNode(add.node()));
}

}  // namespace
}  // namespace xls::solvers::z3
