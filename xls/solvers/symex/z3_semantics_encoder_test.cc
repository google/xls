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

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_utils.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {
namespace {

class Z3SemanticsEncoderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
    package_ = std::make_unique<xls::Package>("test_pkg");
  }

  void TearDown() override {
    package_.reset();
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
  std::unique_ptr<xls::Package> package_;
};

TEST_F(Z3SemanticsEncoderTest, TranslatesBitsLiteralsAndTypes) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::Type* u32_type = package_->GetBitsType(32);
  Z3_sort sort = encoder.GetTypeSort(*u32_type);
  EXPECT_EQ(Z3_get_sort_kind(ctx_, sort), Z3_BV_SORT);
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, sort), 32);

  xls::Value val = xls::Value(xls::UBits(42, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast ast, encoder.TranslateValue(u32_type, val));
  EXPECT_NE(ast, nullptr);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesArithmeticAndExtension) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("arithmetic_fn", package_.get());
  auto a = fb.Param("a", package_->GetBitsType(8));
  auto b = fb.Param("b", package_->GetBitsType(8));
  auto a_ext = fb.ZeroExtend(a, 9);
  auto b_ext = fb.ZeroExtend(b, 9);
  auto add_node = fb.Add(a_ext, b_ext);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast a_ext_ast,
      encoder.TranslateNode(a_ext.node(), absl::Span<const Z3_ast>{&a_ast, 1}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, a_ext_ast)), 9);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast b_ext_ast,
      encoder.TranslateNode(b_ext.node(), absl::Span<const Z3_ast>{&b_ast, 1}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, b_ext_ast)), 9);

  std::vector<Z3_ast> operands = {a_ext_ast, b_ext_ast};
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast add_ast,
                           encoder.TranslateNode(add_node.node(), operands));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, add_ast)), 9);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesBitwiseAndComparisons) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("bitwise_fn", package_.get());
  auto a = fb.Param("a", package_->GetBitsType(16));
  auto b = fb.Param("b", package_->GetBitsType(16));
  auto and_node = fb.And(a, b);
  auto eq_node = fb.Eq(a, b);
  auto ugt_node = fb.UGt(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));
  std::vector<Z3_ast> operands = {a_ast, b_ast};

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast and_ast,
                           encoder.TranslateNode(and_node.node(), operands));
  EXPECT_NE(and_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast eq_ast,
                           encoder.TranslateNode(eq_node.node(), operands));
  EXPECT_NE(eq_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast ugt_ast,
                           encoder.TranslateNode(ugt_node.node(), operands));
  EXPECT_NE(ugt_ast, nullptr);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesBitSlice) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("slice_fn", package_.get());
  auto a = fb.Param("a", package_->GetBitsType(32));
  auto slice = fb.BitSlice(a, /*start=*/8, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast slice_ast,
      encoder.TranslateNode(slice.node(), absl::Span<const Z3_ast>{&a_ast, 1}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, slice_ast)), 16);
}

TEST_F(Z3SemanticsEncoderTest, TranslatesTuplesAndIndexing) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("tuple_fn", package_.get());
  auto a = fb.Param("a", package_->GetBitsType(8));
  auto b = fb.Param("b", package_->GetBitsType(16));
  auto tuple = fb.Tuple({a, b});
  auto tuple_idx = fb.TupleIndex(tuple, 1);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast a_ast, encoder.TranslateParam(fn->param(0)));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast b_ast, encoder.TranslateParam(fn->param(1)));
  std::vector<Z3_ast> tuple_ops = {a_ast, b_ast};

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast tuple_ast,
                           encoder.TranslateNode(tuple.node(), tuple_ops));
  EXPECT_NE(tuple_ast, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(
      Z3_ast idx_ast,
      encoder.TranslateNode(tuple_idx.node(),
                            absl::Span<const Z3_ast>{&tuple_ast, 1}));
  EXPECT_EQ(Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, idx_ast)), 16);
}

TEST_F(Z3SemanticsEncoderTest, EncodesSelectMuxBranchConditions) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("select_fn", package_.get());
  auto sel = fb.Param("sel", package_->GetBitsType(2));
  auto c0 = fb.Literal(xls::UBits(10, 32));
  auto c1 = fb.Literal(xls::UBits(20, 32));
  auto c2 = fb.Literal(xls::UBits(30, 32));
  auto dflt = fb.Literal(xls::UBits(99, 32));
  auto mux = fb.Select(sel, {c0, c1, c2}, dflt);
  XLS_ASSERT_OK(fb.Build().status());

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm0,
                           encoder.EncodeMuxBranchCondition(mux.node(), 0));
  EXPECT_NE(cond_arm0, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_arm1,
                           encoder.EncodeMuxBranchCondition(mux.node(), 1));
  EXPECT_NE(cond_arm1, nullptr);

  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast cond_default,
                           encoder.EncodeMuxBranchCondition(mux.node(), 3));
  EXPECT_NE(cond_default, nullptr);
}

TEST_F(Z3SemanticsEncoderTest, EncodesParamBindingEquality) {
  Z3SemanticsEncoder encoder(ctx_);
  xls::FunctionBuilder fb("binding_fn", package_.get());
  fb.Param("p", package_->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Function * fn, fb.Build());

  xls::Value concrete_val(xls::UBits(0xdeadbeef, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Z3_ast binding_ast, encoder.EncodeParamBinding(
                                                   fn->param(0), concrete_val));
  EXPECT_NE(binding_ast, nullptr);
}

}  // namespace
}  // namespace xls::solvers::symex
