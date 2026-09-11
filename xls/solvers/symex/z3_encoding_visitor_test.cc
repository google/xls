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

#include "xls/solvers/symex/z3_encoding_visitor.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

class Z3EncodingVisitorTest : public IrTestBase {
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

  int64_t GetNodeBvSize(const Z3EncodingVisitor& encoder, const Node* node) {
    return Z3_get_bv_sort_size(ctx_,
                               Z3_get_sort(ctx_, encoder.GetNodeAst(node)));
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(Z3EncodingVisitorTest, TranslatesBitsLiteralsAndTypes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn test_fn(x: bits[32]) -> bits[32] {
      ret lit: bits[32] = literal(value=42)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  Z3_ast lit_ast = encoder.GetNodeAst(FindNode("lit", fn));
  EXPECT_NE(lit_ast, nullptr);
  EXPECT_EQ(Z3_get_sort_kind(ctx_, Z3_get_sort(ctx_, lit_ast)), Z3_BV_SORT);
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, lit_ast)), 32);
}

TEST_F(Z3EncodingVisitorTest, TranslatesArithmeticAndExtension) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn arithmetic_fn(a: bits[8], b: bits[8]) -> bits[9] {
      a_ext: bits[9] = zero_ext(a, new_bit_count=9)
      b_ext: bits[9] = zero_ext(b, new_bit_count=9)
      ret add: bits[9] = add(a_ext, b_ext)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("a_ext", fn)), 9);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("b_ext", fn)), 9);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("add", fn)), 9);
}

TEST_F(Z3EncodingVisitorTest, TranslatesBitwiseAndComparisons) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn bitwise_fn(a: bits[16], b: bits[16]) -> bits[1] {
      and_res: bits[16] = and(a, b)
      eq_res: bits[1] = eq(a, b)
      ret ugt_res: bits[1] = ugt(a, b)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  EXPECT_NE(encoder.GetNodeAst(FindNode("and_res", fn)), nullptr);
  EXPECT_NE(encoder.GetNodeAst(FindNode("eq_res", fn)), nullptr);
  EXPECT_NE(encoder.GetNodeAst(FindNode("ugt_res", fn)), nullptr);
}

TEST_F(Z3EncodingVisitorTest, TranslatesBitSlice) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn slice_fn(a: bits[32]) -> bits[16] {
      ret slice: bits[16] = bit_slice(a, start=8, width=16)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("slice", fn)), 16);
}

TEST_F(Z3EncodingVisitorTest, TranslatesTuplesAndIndexing) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn tuple_fn(a: bits[8], b: bits[16]) -> bits[16] {
      t: (bits[8], bits[16]) = tuple(a, b)
      ret idx: bits[16] = tuple_index(t, index=1)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  Z3_ast idx_ast = encoder.GetNodeAst(FindNode("idx", fn));
  Z3_ast b_ast = encoder.GetNodeAst(FindNode("b", fn));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, idx_ast)), 16);

  // Verify semantic correctness with solver: check(!(idx == b)) is UNSAT.
  Z3_solver solver = Z3_mk_solver(ctx_);
  Z3_solver_inc_ref(ctx_, solver);
  Z3_solver_assert(ctx_, solver,
                   Z3_mk_not(ctx_, Z3_mk_eq(ctx_, idx_ast, b_ast)));
  EXPECT_EQ(Z3_solver_check(ctx_, solver), Z3_L_FALSE);
  Z3_solver_dec_ref(ctx_, solver);
}

TEST_F(Z3EncodingVisitorTest, TranslatesTupleValuesAndNestedTuples) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn tuple_val_fn() -> (bits[8], (bits[16], bits[32])) {
      ret lit: (bits[8], (bits[16], bits[32])) = literal(value=(42, (100, 200)))
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  Z3_ast ast = encoder.GetNodeAst(FindNode("lit", fn));
  EXPECT_NE(ast, nullptr);
  EXPECT_EQ(Z3_get_sort_kind(ctx_, Z3_get_sort(ctx_, ast)), Z3_DATATYPE_SORT);
}

TEST_F(Z3EncodingVisitorTest, TranslatesNaryOps) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn nary_fn(a: bits[8], b: bits[8], c: bits[8]) -> bits[8] {
      add_2: bits[8] = add(a, b)
      and_1: bits[8] = and(a)
      and_3: bits[8] = and(a, b, c)
      or_3: bits[8] = or(a, b, c)
      ret xor_3: bits[8] = xor(a, b, c)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("add_2", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("and_1", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("and_3", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("or_3", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("xor_3", fn)), 8);
}

TEST_F(Z3EncodingVisitorTest, EncodesSelectMuxBranchConditions) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn select_fn(sel: bits[2]) -> bits[32] {
      c0: bits[32] = literal(value=10)
      c1: bits[32] = literal(value=20)
      c2: bits[32] = literal(value=30)
      dflt: bits[32] = literal(value=99)
      ret mux: bits[32] = sel(sel, cases=[c0, c1, c2], default=dflt)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  Node* mux = FindNode("mux", fn);
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm0,
                           encoder.EncodeMuxBranchCondition(mux, 0));
  EXPECT_NE(cond_arm0, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm1,
                           encoder.EncodeMuxBranchCondition(mux, 1));
  EXPECT_NE(cond_arm1, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_default,
                           encoder.EncodeMuxBranchCondition(mux, 3));
  EXPECT_NE(cond_default, nullptr);
}

TEST_F(Z3EncodingVisitorTest, EncodesPrioritySelectMuxBranchConditions) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn psel_fn(sel: bits[2]) -> bits[32] {
      c0: bits[32] = literal(value=10)
      c1: bits[32] = literal(value=20)
      dflt: bits[32] = literal(value=99)
      ret mux: bits[32] = priority_sel(sel, cases=[c0, c1], default=dflt)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  Node* mux = FindNode("mux", fn);
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm0,
                           encoder.EncodeMuxBranchCondition(mux, 0));
  EXPECT_NE(cond_arm0, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm1,
                           encoder.EncodeMuxBranchCondition(mux, 1));
  EXPECT_NE(cond_arm1, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_default,
                           encoder.EncodeMuxBranchCondition(mux, 2));
  EXPECT_NE(cond_default, nullptr);
}

TEST_F(Z3EncodingVisitorTest, TranslatesNotNegConcat) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn ops_fn(a: bits[8], b: bits[8]) -> bits[16] {
      not_a: bits[8] = not(a)
      neg_b: bits[8] = neg(b)
      ret cat: bits[16] = concat(not_a, neg_b)
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("not_a", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("neg_b", fn)), 8);
  EXPECT_EQ(GetNodeBvSize(encoder, FindNode("cat", fn)), 16);
}

TEST_F(Z3EncodingVisitorTest, EncodesFunctionAndMuxArmEquality) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn mux_fn(sel: bits[1], a: bits[32], b: bits[32]) -> bits[32] {
      ret res: bits[32] = sel(sel, cases=[a, b])
    }
  )",
                                                        p.get()));

  Z3EncodingVisitor encoder(ctx_, fn);
  XLS_ASSERT_OK(fn->Accept(&encoder));

  Node* res_node = FindNode("res", fn);
  Node* a_node = FindNode("a", fn);
  Node* b_node = FindNode("b", fn);

  EXPECT_NE(encoder.GetNodeAst(res_node), nullptr);
  EXPECT_NE(encoder.GetNodeAst(a_node), nullptr);
  EXPECT_NE(encoder.GetNodeAst(b_node), nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast eq_arm0,
                           encoder.EncodeMuxArmEquality(res_node, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast eq_arm1,
                           encoder.EncodeMuxArmEquality(res_node, 1));
  EXPECT_NE(eq_arm0, nullptr);
  EXPECT_NE(eq_arm1, nullptr);
}

}  // namespace
}  // namespace xls::solvers::symex
