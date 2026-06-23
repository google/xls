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

#include "xls/solvers/bitwuzla_ir_translator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/solvers/prover_matchers.h"
#include "xls/solvers/solver.h"

namespace xls::solvers::bitwuzla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::xls::solvers::IsProvenFalse;
using ::xls::solvers::IsProvenTrue;

class BitwuzlaIrTranslatorTest : public IrTestBase {
 protected:
  std::unique_ptr<BitwuzlaSolver> solver_ = std::make_unique<BitwuzlaSolver>();
};

TEST_F(BitwuzlaIrTranslatorTest, ZeroIsZero) {
  auto package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Literal(UBits(0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  EXPECT_THAT(
      solver_->TryProve(f, x.node(), Predicate::EqualToZero(), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, AddCounterexample) {
  auto package = CreatePackage();
  FunctionBuilder b("f", package.get());
  auto x = b.Param("x", package->GetBitsType(32));
  auto y = b.Literal(UBits(1, 32));
  auto add = b.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, b.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult res,
      solver_->TryProve(f, add.node(), Predicate::EqualToZero(),
                        SolverLimit()));
  EXPECT_THAT(res, IsProvenFalse());
  auto false_res = std::get<ProvenFalse>(res);
  ASSERT_TRUE(false_res.counterexample.ok());
  EXPECT_EQ(false_res.counterexample.value().at(x.node()).bits(), UBits(0, 32));
}

TEST_F(BitwuzlaIrTranslatorTest, ComplexAggregateParity) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u32 = package->GetBitsType(32);
  Type* arr_t = package->GetArrayType(4, u32);
  Type* tup_t = package->GetTupleType({u32, arr_t});
  auto a = fb.Param("a", tup_t);
  auto b = fb.Param("b", tup_t);
  auto prop = fb.And(fb.Eq(a, b), fb.Ne(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ShiftParity) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(16));
  auto shamt = fb.Literal(UBits(20, 16));  // out of bounds shift amount
  auto shifted = fb.Shll(x, shamt);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shifted));
  EXPECT_THAT(solver_->TryProve(f, shifted.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, TimeoutHandling) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(32));
  auto y = fb.Param("y", package->GetBitsType(32));
  auto prop = fb.Eq(fb.UMul(x, x), y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  SolverLimit limit;
  limit.deterministic_limit = 1;
  auto res = solver_->TryProve(f, prop.node(), Predicate::EqualToZero(), limit);
  EXPECT_FALSE(res.ok());
  EXPECT_THAT(res.status(), StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST_F(BitwuzlaIrTranslatorTest, UMulDifferentWidths) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(8));
  auto y = fb.Param("y", package->GetBitsType(4));
  auto mul = fb.UMul(x, y, /*result_width=*/12);
  auto x_ext = fb.ZeroExtend(x, 12);
  auto y_ext = fb.ZeroExtend(y, 12);
  auto mul_expected = fb.UMul(x_ext, y_ext);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({mul, mul_expected})));
  EXPECT_THAT(solver_->TryProve(f, mul.node(),
                                Predicate::IsEqualTo(mul_expected.node()),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, UMulDifferentWidthsTruncating) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(8));
  auto y = fb.Param("y", package->GetBitsType(4));
  auto mul = fb.UMul(x, y, /*result_width=*/5);
  auto x_ext = fb.ZeroExtend(x, 12);
  auto y_ext = fb.ZeroExtend(y, 12);
  auto mul_expected = fb.BitSlice(fb.UMul(x_ext, y_ext), 0, 5);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({mul, mul_expected})));
  EXPECT_THAT(solver_->TryProve(f, mul.node(),
                                Predicate::IsEqualTo(mul_expected.node()),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulDifferentWidths) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(8));
  auto y = fb.Param("y", package->GetBitsType(4));
  auto mul = fb.SMul(x, y, /*result_width=*/12);
  auto x_ext = fb.SignExtend(x, 12);
  auto y_ext = fb.SignExtend(y, 12);
  auto mul_expected = fb.SMul(x_ext, y_ext);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({mul, mul_expected})));
  EXPECT_THAT(solver_->TryProve(f, mul.node(),
                                Predicate::IsEqualTo(mul_expected.node()),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulDifferentWidthsTruncating) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(8));
  auto y = fb.Param("y", package->GetBitsType(4));
  auto mul = fb.SMul(x, y, /*result_width=*/5);
  auto x_ext = fb.SignExtend(x, 12);
  auto y_ext = fb.SignExtend(y, 12);
  auto mul_expected = fb.BitSlice(fb.SMul(x_ext, y_ext), 0, 5);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({mul, mul_expected})));
  EXPECT_THAT(solver_->TryProve(f, mul.node(),
                                Predicate::IsEqualTo(mul_expected.node()),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulZeroWidth) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(1, 1));
  auto y = fb.Literal(UBits(1, 1));
  auto mul = fb.SMul(x, y, /*result_width=*/0);
  auto expected = fb.Literal(UBits(0, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({mul, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, mul.node(), Predicate::IsEqualTo(expected.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulZeroWidthExtend) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(1, 1));
  auto y = fb.Literal(UBits(1, 1));
  auto mul = fb.SMul(x, y, /*result_width=*/0);
  auto ext = fb.ZeroExtend(mul, 4);
  auto expected = fb.Literal(UBits(0, 4));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({ext, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, ext.node(), Predicate::IsEqualTo(expected.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulZeroWidthNested) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(1, 1));
  auto y = fb.Literal(UBits(1, 1));
  auto mul = fb.SMul(x, y, /*result_width=*/0);
  auto z = fb.Param("z", package->GetBitsType(8));
  auto mul2 = fb.SMul(mul, z, /*result_width=*/12);
  auto expected = fb.Literal(UBits(0, 12));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({mul2, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, mul2.node(), Predicate::IsEqualTo(expected.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, SMulpZeroWidth) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(1, 1));
  auto y = fb.Literal(UBits(1, 1));
  auto zero_w = fb.SMul(x, y, /*result_width=*/0);
  auto z = fb.Param("z", package->GetBitsType(8));
  auto mulp = fb.SMulp(zero_w, z, /*result_width=*/12);
  auto offset = fb.TupleIndex(mulp, 0);
  auto diff = fb.TupleIndex(mulp, 1);
  auto sum = fb.Add(offset, diff);
  auto expected = fb.Literal(UBits(0, 12));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({sum, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, sum.node(), Predicate::IsEqualTo(expected.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ShllZeroWidthShiftAmount) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(255, 8));  // 0b11111111
  auto z = fb.Literal(UBits(1, 1));
  auto zero_w = fb.SMul(z, z, /*result_width=*/0);
  auto shifted = fb.Shll(x, zero_w);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({shifted, x})));
  EXPECT_THAT(solver_->TryProve(f, shifted.node(),
                                Predicate::IsEqualTo(x.node()), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ShllZeroWidthShiftValue) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Literal(UBits(0, 0));  // 0-width shift value
  auto shamt = fb.Param("shamt", package->GetBitsType(8));
  auto shifted = fb.Shll(x, shamt);
  auto ext = fb.ZeroExtend(shifted, 8);
  auto expected = fb.Literal(UBits(0, 8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({ext, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, ext.node(), Predicate::IsEqualTo(expected.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ZeroBitTupleAndConcat) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto x = fb.Param("x", package->GetBitsType(8));
  auto y = fb.Literal(UBits(0, 0));  // 0-bit literal
  auto z = fb.Param("z", package->GetBitsType(4));
  auto conc = fb.Concat({x, y, z});
  auto tup = fb.Tuple({x, y, z});
  auto idx0 = fb.TupleIndex(tup, 0);
  auto idx2 = fb.TupleIndex(tup, 2);
  auto prop = fb.And(fb.Eq(conc, fb.Concat({x, z})),
                     fb.And(fb.Eq(idx0, x), fb.Eq(idx2, z)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, MultipleZeroWidthElements) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u0 = package->GetBitsType(0);
  Type* u8 = package->GetBitsType(8);
  Type* u16 = package->GetBitsType(16);

  auto lit0 = fb.Literal(UBits(0, 0));
  auto p0 = fb.Param("p0", u0);
  auto p8 = fb.Param("p8", u8);
  auto p16 = fb.Param("p16", u16);
  auto lit0_2 = fb.Literal(UBits(0, 0));

  // Concat with multiple 0-bit literals and params interspersed
  auto conc = fb.Concat({lit0, p16, p0, p8, lit0_2});
  auto expected_conc = fb.Concat({p16, p8});

  // Tuple with multiple 0-bit literals and params interspersed
  auto tup = fb.Tuple({lit0, p16, p0, p8, lit0_2});
  auto elem0 = fb.TupleIndex(tup, 0);
  auto elem1 = fb.TupleIndex(tup, 1);
  auto elem2 = fb.TupleIndex(tup, 2);
  auto elem3 = fb.TupleIndex(tup, 3);
  auto elem4 = fb.TupleIndex(tup, 4);

  // All 0-bit concat and tuple
  auto conc_zero = fb.Concat({lit0, p0, lit0_2});
  auto tup_zero = fb.Tuple({p0, lit0});
  auto tup_zero_elem0 = fb.TupleIndex(tup_zero, 0);
  auto tup_zero_elem1 = fb.TupleIndex(tup_zero, 1);

  std::vector<BValue> checks = {
      fb.Eq(conc, expected_conc),
      fb.Eq(elem0, lit0),
      fb.Eq(elem1, p16),
      fb.Eq(elem2, p0),
      fb.Eq(elem3, p8),
      fb.Eq(elem4, lit0_2),
      fb.Eq(conc_zero, lit0),
      fb.Eq(tup_zero_elem0, p0),
      fb.Eq(tup_zero_elem1, lit0),
  };
  auto prop = fb.And(checks);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, conc_zero.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, tup_zero_elem0.node(),
                                Predicate::EqualToZero(), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, DeeplyNestedTuples) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u0 = package->GetBitsType(0);
  Type* u4 = package->GetBitsType(4);
  Type* u16 = package->GetBitsType(16);
  Type* u32 = package->GetBitsType(32);

  auto p0 = fb.Param("p0", u0);
  auto p32 = fb.Param("p32", u32);
  auto p16 = fb.Param("p16", u16);
  auto p4 = fb.Param("p4", u4);

  // t1 = (bits[0], bits[32])
  auto t1 = fb.Tuple({p0, p32});
  // t2 = ()
  auto t2 = fb.Tuple({});
  // t3_inner = (bits[0], bits[4])
  auto lit0 = fb.Literal(UBits(0, 0));
  auto t3_inner = fb.Tuple({lit0, p4});
  // t3 = (bits[16], (bits[0], bits[4]))
  auto t3 = fb.Tuple({p16, t3_inner});

  // nested = ((bits[0], bits[32]), (), (bits[16], (bits[0], bits[4])))
  auto nested = fb.Tuple({t1, t2, t3});

  // Rebuild identical structure from scratch
  auto nested_copy = fb.Tuple({
      fb.Tuple({p0, p32}),
      fb.Tuple({}),
      fb.Tuple({p16, fb.Tuple({lit0, p4})}),
  });

  std::vector<BValue> checks = {
      fb.Eq(nested, nested_copy),
      fb.Eq(fb.TupleIndex(nested, 0), t1),
      fb.Eq(fb.TupleIndex(nested, 1), t2),
      fb.Eq(fb.TupleIndex(nested, 2), t3),
  };
  auto prop = fb.And(checks);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, t2.node(), Predicate::EqualToZero(), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, MultiLevelTupleIndexing) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u0 = package->GetBitsType(0);
  Type* u4 = package->GetBitsType(4);
  Type* u16 = package->GetBitsType(16);
  Type* u32 = package->GetBitsType(32);

  auto p0 = fb.Param("p0", u0);
  auto p32 = fb.Param("p32", u32);
  auto p16 = fb.Param("p16", u16);
  auto p4 = fb.Param("p4", u4);

  // Structure: ((bits[0], bits[32]), (), (bits[16], (bits[0], bits[4])))
  auto nested = fb.Tuple({
      fb.Tuple({p0, p32}),
      fb.Tuple({}),
      fb.Tuple({p16, fb.Tuple({fb.Literal(UBits(0, 0)), p4})}),
  });

  // Extract at level 1
  auto extr_t1 = fb.TupleIndex(nested, 0);
  auto extr_t2 = fb.TupleIndex(nested, 1);
  auto extr_t3 = fb.TupleIndex(nested, 2);

  // Extract at level 2
  auto leaf0_from_t1 = fb.TupleIndex(extr_t1, 0);
  auto leaf32_from_t1 = fb.TupleIndex(extr_t1, 1);
  auto leaf16_from_t3 = fb.TupleIndex(extr_t3, 0);
  auto extr_t3_inner = fb.TupleIndex(extr_t3, 1);

  // Extract at level 3
  auto leaf0_from_inner = fb.TupleIndex(extr_t3_inner, 0);
  auto leaf4_from_inner = fb.TupleIndex(extr_t3_inner, 1);

  auto rebuilt_nested = fb.Tuple({
      fb.Tuple({leaf0_from_t1, leaf32_from_t1}),
      extr_t2,
      fb.Tuple(
          {leaf16_from_t3, fb.Tuple({leaf0_from_inner, leaf4_from_inner})}),
  });

  std::vector<BValue> checks = {
      fb.Eq(nested, rebuilt_nested), fb.Eq(leaf32_from_t1, p32),
      fb.Eq(leaf16_from_t3, p16),    fb.Eq(leaf4_from_inner, p4),
      fb.Eq(leaf0_from_t1, p0),
  };
  auto prop = fb.And(checks);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));

  // Verify non-zero leaf members and aggregate structure
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));

  // Verify 0-bit leaf members prove EqualToZero()
  EXPECT_THAT(solver_->TryProve(f, leaf0_from_t1.node(),
                                Predicate::EqualToZero(), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, leaf0_from_inner.node(),
                                Predicate::EqualToZero(), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, extr_t2.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ArrayOfAggregates) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u0 = package->GetBitsType(0);
  Type* u24 = package->GetBitsType(24);
  Type* empty_tup_t = package->GetTupleType({});
  Type* elem_tup_t = package->GetTupleType({u0, u24, empty_tup_t});

  auto e0 = fb.Param("e0", elem_tup_t);
  auto e1 = fb.Param("e1", elem_tup_t);
  auto e2 = fb.Param("e2", elem_tup_t);

  auto arr = fb.Array({e0, e1, e2}, elem_tup_t);

  auto idx0 = fb.Literal(UBits(0, 32));
  auto idx1 = fb.Literal(UBits(1, 32));
  auto idx2 = fb.Literal(UBits(2, 32));

  auto elem0 = fb.ArrayIndex(arr, {idx0});
  auto elem1 = fb.ArrayIndex(arr, {idx1});
  auto elem2 = fb.ArrayIndex(arr, {idx2});

  auto field0 = fb.TupleIndex(elem1, 0);
  auto field24 = fb.TupleIndex(elem1, 1);
  auto field_empty = fb.TupleIndex(elem1, 2);

  auto rebuilt_elem1 = fb.Tuple({field0, field24, field_empty});
  auto rebuilt_arr = fb.Array({elem0, rebuilt_elem1, elem2}, elem_tup_t);
  auto updated_arr = fb.ArrayUpdate(arr, rebuilt_elem1, {idx1});

  std::vector<BValue> checks = {
      fb.Eq(elem1, e1),
      fb.Eq(rebuilt_elem1, e1),
      fb.Eq(arr, rebuilt_arr),
      fb.Eq(arr, updated_arr),
  };
  auto prop = fb.And(checks);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, field0.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, field_empty.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ZeroWidthAggregateCounterexample) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u0 = package->GetBitsType(0);
  Type* u16 = package->GetBitsType(16);
  Type* tup_t = package->GetTupleType({u0, u16, u0});

  auto x = fb.Param("x", tup_t);
  auto y = fb.Param("y", tup_t);
  auto neq = fb.Ne(x, y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(neq));
  XLS_ASSERT_OK_AND_ASSIGN(
      ProverResult res,
      solver_->TryProve(f, neq.node(), Predicate::EqualToZero(),
                        SolverLimit()));
  EXPECT_THAT(res, IsProvenFalse());
  auto false_res = std::get<ProvenFalse>(res);
  ASSERT_TRUE(false_res.counterexample.ok());
  EXPECT_EQ(false_res.counterexample.value().at(x.node()).GetFlatBitCount(),
            16);
  EXPECT_EQ(false_res.counterexample.value().at(x.node()).elements().size(), 3);
  EXPECT_EQ(false_res.counterexample.value()
                .at(x.node())
                .elements()[0]
                .GetFlatBitCount(),
            0);
}

// Microbenchmarks comparing Bitwuzla against Z3 on representative IR operations

static void BM_OneHotLsb(benchmark::State& state, SolverKind kind) {
  int64_t w = state.range(0);
  Package package("p");
  FunctionBuilder fb("f", &package);
  Type* bits_w = package.GetBitsType(w);
  BValue x = fb.Param("x", bits_w);
  BValue oh = fb.OneHot(x, LsbOrMsb::kLsb);
  BValue oh_minus_1 = fb.Subtract(oh, fb.Literal(UBits(1, w + 1)));
  BValue oh_and = fb.And(oh, oh_minus_1);
  fb.Eq(oh_and, fb.Literal(UBits(0, w + 1)));
  auto f_or = fb.Build();
  CHECK(f_or.ok());
  Function* f = f_or.value();
  auto solver = CreateSolver(kind);
  for (auto _ : state) {
    auto proven = (*solver)->TryProve(
        f, f->return_value(), Predicate::NotEqualToZero(), SolverLimit());
    CHECK(proven.ok());
    benchmark::DoNotOptimize(proven.value());
  }
}

static void BM_OneHotMsb(benchmark::State& state, SolverKind kind) {
  int64_t w = state.range(0);
  Package package("p");
  FunctionBuilder fb("f", &package);
  Type* bits_w = package.GetBitsType(w);
  BValue x = fb.Param("x", bits_w);
  BValue oh = fb.OneHot(x, LsbOrMsb::kMsb);
  BValue oh_minus_1 = fb.Subtract(oh, fb.Literal(UBits(1, w + 1)));
  BValue oh_and = fb.And(oh, oh_minus_1);
  fb.Eq(oh_and, fb.Literal(UBits(0, w + 1)));
  auto f_or = fb.Build();
  CHECK(f_or.ok());
  Function* f = f_or.value();
  auto solver = CreateSolver(kind);
  for (auto _ : state) {
    auto proven = (*solver)->TryProve(
        f, f->return_value(), Predicate::NotEqualToZero(), SolverLimit());
    CHECK(proven.ok());
    benchmark::DoNotOptimize(proven.value());
  }
}

static void BM_OneHotSelect(benchmark::State& state, SolverKind kind) {
  int64_t n = state.range(0);
  Package package("p");
  FunctionBuilder fb("f", &package);
  Type* bits_8 = package.GetBitsType(8);
  BValue sel = fb.Param("sel", package.GetBitsType(n));
  std::vector<BValue> cases;
  for (int64_t i = 0; i < n; ++i) {
    cases.push_back(fb.Param(absl::StrCat("case_", i), bits_8));
  }
  BValue oh_sel = fb.OneHotSelect(sel, cases);
  BValue one_or_zero_hot =
      fb.Eq(fb.And(sel, fb.Subtract(sel, fb.Literal(UBits(1, n)))),
            fb.Literal(UBits(0, n)));
  std::vector<BValue> or_terms = {fb.Not(one_or_zero_hot),
                                  fb.Eq(oh_sel, fb.Literal(UBits(0, 8)))};
  for (const auto& c : cases) {
    or_terms.push_back(fb.Eq(oh_sel, c));
  }
  fb.Or(or_terms);
  auto f_or = fb.Build();
  CHECK(f_or.ok());
  Function* f = f_or.value();
  auto solver = CreateSolver(kind);
  for (auto _ : state) {
    auto proven = (*solver)->TryProve(
        f, f->return_value(), Predicate::NotEqualToZero(), SolverLimit());
    CHECK(proven.ok());
    benchmark::DoNotOptimize(proven.value());
  }
}

static void BM_PrioritySelect(benchmark::State& state, SolverKind kind) {
  int64_t n = state.range(0);
  Package package("p");
  FunctionBuilder fb("f", &package);
  Type* bits_8 = package.GetBitsType(8);
  BValue sel = fb.Param("sel", package.GetBitsType(n));
  BValue def_v = fb.Param("def_v", bits_8);
  std::vector<BValue> cases;
  for (int64_t i = 0; i < n; ++i) {
    cases.push_back(fb.Param(absl::StrCat("case_", i), bits_8));
  }
  BValue pri_sel = fb.PrioritySelect(sel, cases, def_v);
  std::vector<BValue> or_terms = {fb.Eq(pri_sel, def_v)};
  for (const auto& c : cases) {
    or_terms.push_back(fb.Eq(pri_sel, c));
  }
  fb.Or(or_terms);
  auto f_or = fb.Build();
  CHECK(f_or.ok());
  Function* f = f_or.value();
  auto solver = CreateSolver(kind);
  for (auto _ : state) {
    auto proven = (*solver)->TryProve(
        f, f->return_value(), Predicate::NotEqualToZero(), SolverLimit());
    CHECK(proven.ok());
    benchmark::DoNotOptimize(proven.value());
  }
}

static bool RegisterAllBenchmarks() {
  for (SolverKind kind : {SolverKind::kZ3, SolverKind::kBitwuzla}) {
    for (int64_t w : {64, 128}) {
      benchmark::RegisterBenchmark(
          absl::StrFormat("BM_OneHotLsb/solver=%v", kind),
          [kind](benchmark::State& state) { BM_OneHotLsb(state, kind); })
          ->Arg(w);
    }
    for (int64_t w : {64, 128}) {
      benchmark::RegisterBenchmark(
          absl::StrFormat("BM_OneHotMsb/solver=%v", kind),
          [kind](benchmark::State& state) { BM_OneHotMsb(state, kind); })
          ->Arg(w);
    }
    for (int64_t n : {16, 32}) {
      benchmark::RegisterBenchmark(
          absl::StrFormat("BM_OneHotSelect/solver=%v", kind),
          [kind](benchmark::State& state) { BM_OneHotSelect(state, kind); })
          ->Arg(n);
    }
    for (int64_t n : {16, 32}) {
      benchmark::RegisterBenchmark(
          absl::StrFormat("BM_PrioritySelect/solver=%v", kind),
          [kind](benchmark::State& state) { BM_PrioritySelect(state, kind); })
          ->Arg(n);
    }
  }
  return true;
}

static bool dummy = RegisterAllBenchmarks();

}  // namespace
}  // namespace xls::solvers::bitwuzla
