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

#include "xls/solvers/symex/z3_semantics_encoder.h"

#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "z3/src/api/z3.h"  // IWYU pragma: keep
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

class Z3SemanticsEncoderTest : public IrTestBase {
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

TEST_F(Z3SemanticsEncoderTest, TranslatesBitsLiteralsAndTypes) {
  Z3SemanticsEncoder encoder(ctx_);
  auto p = CreatePackage();
  Type* u32_type = p->GetBitsType(32);
  Z3_sort sort = encoder.GetTypeSort(*u32_type);
  EXPECT_EQ(Z3_get_sort_kind(ctx_, sort), Z3_BV_SORT);
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, sort), 32);

  Value val = Value(UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast ast, encoder.TranslateValue(u32_type, val));
  EXPECT_NE(ast, nullptr);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesArithmeticAndExtension) {
  Z3SemanticsEncoder encoder(ctx_);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn arithmetic_fn(a: bits[8], b: bits[8]) -> bits[9] {
      a_ext: bits[9] = zero_ext(a, new_bit_count=9)
      b_ext: bits[9] = zero_ext(b, new_bit_count=9)
      ret add: bits[9] = add(a_ext, b_ext)
    }
  )",
                                                        p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast a_ext_ast, encoder.TranslateNode(FindNode("a_ext", fn), {a_ast}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, a_ext_ast)), 9);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast b_ext_ast, encoder.TranslateNode(FindNode("b_ext", fn), {b_ast}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, b_ext_ast)), 9);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast add_ast,
      encoder.TranslateNode(FindNode("add", fn), {a_ext_ast, b_ext_ast}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, add_ast)), 9);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesBitwiseAndComparisons) {
  Z3SemanticsEncoder encoder(ctx_);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn bitwise_fn(a: bits[16], b: bits[16]) -> bits[1] {
      and_res: bits[16] = and(a, b)
      eq_res: bits[1] = eq(a, b)
      ret ugt_res: bits[1] = ugt(a, b)
    }
  )",
                                                        p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));
  std::vector<Z3_ast> operands = {a_ast, b_ast};

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast and_ast, encoder.TranslateNode(FindNode("and_res", fn), operands));
  EXPECT_NE(and_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast eq_ast, encoder.TranslateNode(FindNode("eq_res", fn), operands));
  EXPECT_NE(eq_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast ugt_ast, encoder.TranslateNode(FindNode("ugt_res", fn), operands));
  EXPECT_NE(ugt_ast, nullptr);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesBitSlice) {
  Z3SemanticsEncoder encoder(ctx_);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn slice_fn(a: bits[32]) -> bits[16] {
      ret slice: bits[16] = bit_slice(a, start=8, width=16)
    }
  )",
                                                        p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast slice_ast, encoder.TranslateNode(FindNode("slice", fn), {a_ast}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, slice_ast)), 16);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesTuplesAndIndexing) {
  Z3SemanticsEncoder encoder(ctx_);
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn tuple_fn(a: bits[8], b: bits[16]) -> bits[16] {
      t: (bits[8], bits[16]) = tuple(a, b)
      ret idx: bits[16] = tuple_index(t, index=1)
    }
  )",
                                                        p.get()));

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast tuple_ast,
      encoder.TranslateNode(FindNode("t", fn), {a_ast, b_ast}));
  EXPECT_NE(tuple_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast idx_ast, encoder.TranslateNode(FindNode("idx", fn), {tuple_ast}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, idx_ast)), 16);
}

TEST_F(Z3SemanticsEncoderTest, EncodesSelectMuxBranchConditions) {
  Z3SemanticsEncoder encoder(ctx_);
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

}  // namespace
}  // namespace xls::solvers::symex
