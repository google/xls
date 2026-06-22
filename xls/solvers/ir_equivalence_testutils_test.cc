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

#include "xls/solvers/ir_equivalence_testutils.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

namespace xls::solvers {
namespace {

class Z3IrEquivalenceTestutilsTest : public IrTestBase {};

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

class FunctionInterpreter : public IrInterpreter {
 public:
  using IrInterpreter::IrInterpreter;
  absl::Status HandleParam(Param* param) override {
    XLS_RET_CHECK(HasResult(param)) << param;
    return absl::OkStatus();
  }
};

TEST_F(Z3IrEquivalenceTestutilsTest, EquivArraySlice) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetArrayType(2, p->GetBitsType(4)));
  BValue y = fb.Param("y", p->GetBitsType(4));
  BValue eq_0 = fb.Eq(y, fb.Literal(UBits(0, 4)));
  fb.ArrayIndex(x, {fb.Literal(UBits(0, 4))});
  BValue element_1 = fb.ArrayIndex(x, {fb.Literal(UBits(1, 4))});
  BValue end = fb.Array({element_1, element_1}, element_1.GetType());
  BValue transformed = fb.Select(eq_0, /*on_true=*/x, /*on_false=*/end);
  BValue res = fb.ArraySlice(x, y, 2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  // Exhaustively check all inputs for equivalence.
  for (uint64_t x0 = 0; x0 < 16; ++x0) {
    for (uint64_t x1 = 0; x1 < 16; ++x1) {
      for (uint64_t y0 = 0; y0 < 16; ++y0) {
        InterpreterEvents events;
        XLS_ASSERT_OK_AND_ASSIGN(Value x_val, Value::UBitsArray({x0, x1}, 4));
        absl::flat_hash_map<Node*, Value> node_values{
            {x.node(), x_val}, {y.node(), Value(UBits(y0, 4))}};
        FunctionInterpreter interp(&node_values, &events);
        XLS_ASSERT_OK(f->Accept(&interp));
        EXPECT_EQ(node_values.at(res.node()),
                  node_values.at(transformed.node()))
            << " @ [" << x0 << ", " << x1 << "], " << y;
      }
    }
  }
  XLS_ASSERT_OK(
      res.node()->ReplaceImplicitUsesWith(transformed.node()).status());
}

}  // namespace
}  // namespace xls::solvers
