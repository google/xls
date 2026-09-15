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

#include "xls/solvers/symex/symbolic_path.h"

#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::ElementsAre;
using ::testing::Optional;

class SymbolicPathTest : public IrTestBase {
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

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(SymbolicPathTest, InitializesDefaultFields) {
  SymbolicPath path;
  EXPECT_EQ(path.path_condition, nullptr);
  EXPECT_EQ(path.return_value, nullptr);
  EXPECT_TRUE(path.branch_decisions.empty());
  EXPECT_TRUE(path.generated_test.empty());
  EXPECT_TRUE(path.input_values().empty());
}

TEST_F(SymbolicPathTest, BranchDecisionEvaluatesIsDefault) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn mux_fn(sel: bits[2]) -> bits[32] {
      c0: bits[32] = literal(value=10)
      c1: bits[32] = literal(value=20)
      dflt: bits[32] = literal(value=99)
      ret mux: bits[32] = sel(sel, cases=[c0, c1], default=dflt)
    }
  )",
                                                        p.get()));

  const Node* mux_node = FindNode("mux", fn);

  BranchDecision case0{.mux_node = mux_node, .arm_index = 0};
  EXPECT_FALSE(case0.is_default());

  BranchDecision case1{.mux_node = mux_node, .arm_index = 1};
  EXPECT_FALSE(case1.is_default());

  BranchDecision default_case{.mux_node = mux_node, .arm_index = 2};
  EXPECT_TRUE(default_case.is_default());

  BranchDecision null_mux{.mux_node = nullptr, .arm_index = 0};
  EXPECT_FALSE(null_mux.is_default());
}

TEST_F(SymbolicPathTest, StoresPathConditionAndDecisions) {
  Z3_ast true_ast = Z3_mk_true(ctx_);
  Z3_sort bv32 = Z3_mk_bv_sort(ctx_, 32);
  Z3_symbol ret_sym = Z3_mk_string_symbol(ctx_, "ret");
  Z3_ast ret_ast = Z3_mk_const(ctx_, ret_sym, bv32);

  SymbolicPath path;
  path.path_condition = true_ast;
  path.return_value = ret_ast;
  path.branch_decisions.push_back(BranchDecision{.arm_index = 1});

  EXPECT_EQ(path.path_condition, true_ast);
  EXPECT_EQ(path.return_value, ret_ast);
  ASSERT_EQ(path.branch_decisions.size(), 1);
  EXPECT_EQ(path.branch_decisions[0].arm_index, 1);
}

TEST_F(SymbolicPathTest, StoresGeneratedTestAndAccessors) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn param_fn(x: bits[32]) -> bits[32] {
      ret x: bits[32] = param(name=x)
    }
  )",
                                                        p.get()));

  SymbolicPath path;
  path.generated_test.push_back(
      ParamAssignment{.param = fn->param(0), .value = Value(UBits(42, 32))});

  ASSERT_EQ(path.generated_test.size(), 1);
  EXPECT_EQ(path.generated_test[0].param, fn->param(0));
  EXPECT_EQ(path.generated_test[0].value, Value(UBits(42, 32)));

  EXPECT_THAT(path.GetParamValue("x"), Optional(Value(UBits(42, 32))));
  EXPECT_THAT(path.GetParamValue(fn->param(0)), Optional(Value(UBits(42, 32))));
  EXPECT_EQ(path.GetParamValue("nonexistent"), std::nullopt);

  EXPECT_THAT(path.input_values(), ElementsAre(Value(UBits(42, 32))));
}

}  // namespace
}  // namespace xls::solvers::symex
