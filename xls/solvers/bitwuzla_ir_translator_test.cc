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

#include <cstddef>
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
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
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

TEST_F(BitwuzlaIrTranslatorTest, NaryNandAndNor) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  auto a = fb.Param("a", package->GetBitsType(8));
  auto b = fb.Param("b", package->GetBitsType(8));
  auto c = fb.Param("c", package->GetBitsType(8));
  auto nand_res = fb.Nand({a, b, c});
  auto nor_res = fb.Nor({a, b, c});
  auto expected_nand = fb.Not(fb.And({a, b, c}));
  auto expected_nor = fb.Not(fb.Or({a, b, c}));
  auto prop =
      fb.And(fb.Eq(nand_res, expected_nand), fb.Eq(nor_res, expected_nor));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
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

TEST_F(BitwuzlaIrTranslatorTest, TupleWithArrayLiteral) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u32 = package->GetBitsType(32);
  Type* arr_t = package->GetArrayType(3, u32);
  Type* tup_t = package->GetTupleType({arr_t, package->GetBitsType(1)});

  XLS_ASSERT_OK_AND_ASSIGN(Value val_arr, Value::UBitsArray({1, 2, 3}, 32));
  Value val_tup = Value::Tuple({val_arr, Value(UBits(0, 1))});

  auto lit = fb.Literal(val_tup);
  EXPECT_EQ(lit.node()->GetType(), tup_t);
  auto prop = fb.Eq(lit, lit);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, WideLiteralTranslation) {
  // Test that literals wider than 64 bits with multiple non-zero words
  // and non-power-of-two widths preserve exact bit order and endianness.

  // 1. 128-bit literal: two 64-bit words
  {
    auto package = CreatePackage();
    FunctionBuilder fb("wide_128", package.get());
    Bits hi_64 = UBits(0x1122334455667788ULL, 64);
    Bits lo_64 = UBits(0x99aabbccddeeff01ULL, 64);
    Bits val_128 = bits_ops::Concat({hi_64, lo_64});
    BValue lit_128 = fb.Literal(val_128);

    BValue slice_lo = fb.BitSlice(lit_128, /*start=*/0, /*width=*/64);
    BValue slice_hi = fb.BitSlice(lit_128, /*start=*/64, /*width=*/64);

    BValue eq_lo = fb.Eq(slice_lo, fb.Literal(lo_64));
    BValue eq_hi = fb.Eq(slice_hi, fb.Literal(hi_64));
    BValue swapped = fb.Eq(slice_lo, fb.Literal(hi_64));
    BValue eq_concat =
        fb.Eq(lit_128, fb.Concat({fb.Literal(hi_64), fb.Literal(lo_64)}));

    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eq_lo));
    EXPECT_THAT(solver_->TryProve(f, eq_lo.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_hi.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, swapped.node(),
                                  Predicate::NotEqualToZero(), SolverLimit()),
                IsOkAndHolds(IsProvenFalse()));
    EXPECT_THAT(solver_->TryProve(f, eq_concat.node(),
                                  Predicate::NotEqualToZero(), SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
  }

  // 2. 100-bit literal: 36-bit top word (MSB) + 64-bit bottom word (LSB)
  {
    auto package = CreatePackage();
    FunctionBuilder fb("wide_100", package.get());
    Bits hi_36 = UBits(0x123456789ULL, 36);
    Bits lo_64 = UBits(0xfedcba9876543210ULL, 64);
    Bits val_100 = bits_ops::Concat({hi_36, lo_64});
    BValue lit_100 = fb.Literal(val_100);

    BValue slice_lo = fb.BitSlice(lit_100, /*start=*/0, /*width=*/64);
    BValue slice_hi = fb.BitSlice(lit_100, /*start=*/64, /*width=*/36);
    // Cross-boundary slice: bits 60..67 (8 bits spanning the 64-bit word
    // boundary) Low 4 bits come from lo_64 bits 60..63 (top nibble of
    // 0xfedcba9876543210: 0xf) High 4 bits come from hi_36 bits 0..3 (bottom
    // nibble of 0x123456789: 0x9) Combined byte: (0x9 << 4) | 0xf = 0x9f.
    BValue slice_cross = fb.BitSlice(lit_100, /*start=*/60, /*width=*/8);
    BValue expected_cross = fb.Literal(UBits(0x9f, 8));

    BValue eq_lo = fb.Eq(slice_lo, fb.Literal(lo_64));
    BValue eq_hi = fb.Eq(slice_hi, fb.Literal(hi_36));
    BValue eq_cross = fb.Eq(slice_cross, expected_cross);
    BValue eq_concat =
        fb.Eq(lit_100, fb.Concat({fb.Literal(hi_36), fb.Literal(lo_64)}));

    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eq_lo));
    EXPECT_THAT(solver_->TryProve(f, eq_lo.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_hi.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_cross.node(),
                                  Predicate::NotEqualToZero(), SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_concat.node(),
                                  Predicate::NotEqualToZero(), SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
  }

  // 3. 256-bit literal: 4 distinct 64-bit words
  {
    auto package = CreatePackage();
    FunctionBuilder fb("wide_256", package.get());
    Bits w3 = UBits(0x0123456789abcdefULL, 64);
    Bits w2 = UBits(0xfedcba9876543210ULL, 64);
    Bits w1 = UBits(0xa5a5a5a55a5a5a5aULL, 64);
    Bits w0 = UBits(0x5a5a5a5aa5a5a5a5ULL, 64);
    Bits val_256 = bits_ops::Concat({w3, w2, w1, w0});
    BValue lit_256 = fb.Literal(val_256);

    BValue s0 = fb.BitSlice(lit_256, /*start=*/0, /*width=*/64);
    BValue s1 = fb.BitSlice(lit_256, /*start=*/64, /*width=*/64);
    BValue s2 = fb.BitSlice(lit_256, /*start=*/128, /*width=*/64);
    BValue s3 = fb.BitSlice(lit_256, /*start=*/192, /*width=*/64);

    BValue eq_s0 = fb.Eq(s0, fb.Literal(w0));
    BValue eq_s1 = fb.Eq(s1, fb.Literal(w1));
    BValue eq_s2 = fb.Eq(s2, fb.Literal(w2));
    BValue eq_s3 = fb.Eq(s3, fb.Literal(w3));

    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eq_s0));
    EXPECT_THAT(solver_->TryProve(f, eq_s0.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_s1.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_s2.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
    EXPECT_THAT(solver_->TryProve(f, eq_s3.node(), Predicate::NotEqualToZero(),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()));
  }
}

TEST_F(BitwuzlaIrTranslatorTest, HandlesSDiv) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* s8 = package->GetBitsType(8);
  BValue lhs = fb.Param("lhs", s8);
  BValue rhs = fb.Literal(SBits(0, 8));
  BValue sdiv_zero = fb.SDiv(lhs, rhs);
  BValue is_neg = fb.SLt(lhs, fb.Literal(SBits(0, 8)));
  BValue expected =
      fb.Select(is_neg, fb.Literal(SBits(-128, 8)), fb.Literal(SBits(127, 8)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({sdiv_zero, expected})));
  EXPECT_THAT(
      solver_->TryProve(f, sdiv_zero.node(),
                        Predicate::IsEqualTo(expected.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));

  for (int64_t a : {-128, -127, -4, -1, 0, 1, 4, 126, 127}) {
    for (int64_t b : {-128, -4, -1, 0, 1, 4, 127}) {
      auto pkg = CreatePackage();
      FunctionBuilder cfb("c", pkg.get());
      Bits lhs_bits = SBits(a, 8);
      Bits rhs_bits = SBits(b, 8);
      BValue res = cfb.SDiv(cfb.Literal(lhs_bits), cfb.Literal(rhs_bits));
      BValue exp = cfb.Literal(bits_ops::SDiv(lhs_bits, rhs_bits));
      XLS_ASSERT_OK_AND_ASSIGN(Function * cf,
                               cfb.BuildWithReturnValue(cfb.Tuple({res, exp})));
      EXPECT_THAT(
          solver_->TryProve(cf, res.node(), Predicate::IsEqualTo(exp.node()),
                            SolverLimit()),
          IsOkAndHolds(IsProvenTrue()))
          << a << " sdiv " << b;
    }
  }
}

TEST_F(BitwuzlaIrTranslatorTest, HandlesSDiv1BitWide) {
  for (int64_t a : {0, -1}) {
    for (int64_t b : {0, -1}) {
      auto pkg = CreatePackage();
      FunctionBuilder cfb("c", pkg.get());
      Bits lhs_bits = SBits(a, 1);
      Bits rhs_bits = SBits(b, 1);
      BValue res = cfb.SDiv(cfb.Literal(lhs_bits), cfb.Literal(rhs_bits));
      BValue exp = cfb.Literal(bits_ops::SDiv(lhs_bits, rhs_bits));
      XLS_ASSERT_OK_AND_ASSIGN(Function * cf,
                               cfb.BuildWithReturnValue(cfb.Tuple({res, exp})));
      EXPECT_THAT(
          solver_->TryProve(cf, res.node(), Predicate::IsEqualTo(exp.node()),
                            SolverLimit()),
          IsOkAndHolds(IsProvenTrue()))
          << a << " sdiv " << b;
    }
  }
}

TEST_F(BitwuzlaIrTranslatorTest, HandlesOneHotSelArrayMultiHot) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  Type* arr_t = package->GetArrayType(2, u8);
  BValue sel = fb.Param("sel", package->GetBitsType(2));
  BValue case0 = fb.Param("case0", arr_t);
  BValue case1 = fb.Param("case1", arr_t);
  BValue ohs = fb.OneHotSelect(sel, {case0, case1});

  BValue s0 = fb.BitSlice(sel, 0, 1);
  BValue s1 = fb.BitSlice(sel, 1, 1);
  BValue zero = fb.Literal(UBits(0, 8));
  BValue idx0 = fb.Literal(UBits(0, 1));
  BValue idx1 = fb.Literal(UBits(1, 1));
  BValue exp0 = fb.Or(fb.Select(s0, fb.ArrayIndex(case0, {idx0}), zero),
                      fb.Select(s1, fb.ArrayIndex(case1, {idx0}), zero));
  BValue exp1 = fb.Or(fb.Select(s0, fb.ArrayIndex(case0, {idx1}), zero),
                      fb.Select(s1, fb.ArrayIndex(case1, {idx1}), zero));
  BValue actual0 = fb.ArrayIndex(ohs, {idx0});
  BValue actual1 = fb.ArrayIndex(ohs, {idx1});

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      fb.BuildWithReturnValue(fb.Tuple({actual0, actual1, exp0, exp1})));
  EXPECT_THAT(
      solver_->TryProve(f, actual0.node(), Predicate::IsEqualTo(exp0.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, actual1.node(), Predicate::IsEqualTo(exp1.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, HandlesOneHotSelNestedArrayMultiHot) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u4 = package->GetBitsType(4);
  Type* inner_t = package->GetArrayType(2, u4);
  Type* outer_t = package->GetArrayType(2, inner_t);
  BValue sel = fb.Param("sel", package->GetBitsType(2));
  BValue case0 = fb.Param("case0", outer_t);
  BValue case1 = fb.Param("case1", outer_t);
  BValue ohs = fb.OneHotSelect(sel, {case0, case1});

  BValue s0 = fb.BitSlice(sel, 0, 1);
  BValue s1 = fb.BitSlice(sel, 1, 1);
  BValue zero = fb.Literal(UBits(0, 4));
  std::vector<BValue> actuals;
  std::vector<BValue> expecteds;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      BValue bi = fb.Literal(UBits(i, 1));
      BValue bj = fb.Literal(UBits(j, 1));
      actuals.push_back(fb.ArrayIndex(ohs, {bi, bj}));
      expecteds.push_back(
          fb.Or(fb.Select(s0, fb.ArrayIndex(case0, {bi, bj}), zero),
                fb.Select(s1, fb.ArrayIndex(case1, {bi, bj}), zero)));
    }
  }
  std::vector<BValue> ret_elems = actuals;
  ret_elems.insert(ret_elems.end(), expecteds.begin(), expecteds.end());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple(ret_elems)));
  for (size_t k = 0; k < actuals.size(); ++k) {
    EXPECT_THAT(solver_->TryProve(f, actuals[k].node(),
                                  Predicate::IsEqualTo(expecteds[k].node()),
                                  SolverLimit()),
                IsOkAndHolds(IsProvenTrue()))
        << "element " << k;
  }
}

TEST_F(BitwuzlaIrTranslatorTest, ZeroBitParamSDivAndUDiv) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  BValue x0 = fb.Param("x0", package->GetBitsType(0));
  BValue zero0 = fb.Literal(UBits(0, 0));
  BValue sdiv0 = fb.SDiv(x0, zero0);
  BValue udiv0 = fb.UDiv(x0, zero0);
  BValue x_param = fb.Param("xp0", package->GetBitsType(0));
  BValue sdiv_p = fb.SDiv(x_param, zero0);

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({sdiv0, udiv0, sdiv_p})));
  EXPECT_THAT(solver_->TryProve(f, sdiv0.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, udiv0.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, sdiv_p.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, x_param.node(), Predicate::EqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, NonPow2ArrayEqualityAndMultidimUpdate) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  Type* arr3_t = package->GetArrayType(3, u8);
  BValue a = fb.Param("a", arr3_t);
  BValue rebuilt_a = fb.Array({fb.ArrayIndex(a, {fb.Literal(UBits(0, 2))}),
                               fb.ArrayIndex(a, {fb.Literal(UBits(1, 2))}),
                               fb.ArrayIndex(a, {fb.Literal(UBits(2, 2))})},
                              u8);

  Type* u32 = package->GetBitsType(32);
  Type* unit_arr_t = package->GetArrayType(
      1, package->GetArrayType(2, package->GetArrayType(1, u32)));
  BValue unit_arr = fb.Param("unit_arr", unit_arr_t);
  BValue idx = fb.Param("idx", u32);
  BValue val = fb.Param("val", u32);
  BValue updated = fb.ArrayUpdate(unit_arr, val, {idx, idx, idx});
  BValue direct = fb.PrioritySelect(
      fb.Eq(idx, fb.Literal(UBits(0, 32))),
      {fb.Array(
          {fb.ArrayUpdate(fb.ArrayIndex(unit_arr, {fb.Literal(UBits(0, 1))}),
                          val, {idx, idx})},
          unit_arr_t->AsArrayOrDie()->element_type())},
      unit_arr);

  BValue prop = fb.And({fb.Eq(a, rebuilt_a), fb.Eq(updated, direct)});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ArraySliceOobAndPrioritySelectWithArrayNode) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u4 = package->GetBitsType(4);
  BValue e0 = fb.Param("e0", u4);
  BValue e1 = fb.Param("e1", u4);
  BValue arr2 = fb.Array({e0, e1}, u4);
  // Start=14 >= arr2.size()=2 -> every element in slice must clamp to e1.
  BValue oob_slice = fb.ArraySlice(arr2, fb.Literal(UBits(14, 4)), 2);

  BValue outer = fb.Param("outer", package->GetArrayType(1, arr2.GetType()));
  BValue cond = fb.Param("cond", package->GetBitsType(1));
  BValue psel =
      fb.PrioritySelect(cond, {fb.Array({arr2}, arr2.GetType())}, outer);
  BValue psel_idx =
      fb.ArrayIndex(psel, {fb.Literal(UBits(0, 1)), fb.Literal(UBits(1, 2))});

  BValue check_slice = fb.Eq(oob_slice, fb.Array({e1, e1}, u4));
  BValue check_psel =
      fb.Eq(psel_idx,
            fb.Select(cond, {fb.ArrayIndex(outer, {fb.Literal(UBits(0, 1)),
                                                   fb.Literal(UBits(1, 2))}),
                             e1}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(check_slice));
  EXPECT_THAT(solver_->TryProve(f, check_slice.node(),
                                Predicate::NotEqualToZero(), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, check_psel.node(),
                                Predicate::NotEqualToZero(), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, LargeSmtArrayOperations) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u4 = package->GetBitsType(4);
  Type* arr_t = package->GetArrayType(65, u4);
  BValue a = fb.Param("a", arr_t);
  BValue b = fb.Param("b", arr_t);
  BValue sel = fb.Param("sel", package->GetBitsType(2));
  BValue idx = fb.Param("idx", package->GetBitsType(8));
  BValue val = fb.Param("val", u4);

  BValue ohs = fb.OneHotSelect(sel, {a, b});
  BValue ohs_idx = fb.ArrayIndex(ohs, {idx});
  BValue expected_idx =
      fb.OneHotSelect(sel, {fb.ArrayIndex(a, {idx}), fb.ArrayIndex(b, {idx})});

  BValue gated = fb.Gate(fb.Literal(UBits(0, 1)), a);
  BValue gated_idx = fb.ArrayIndex(gated, {idx});

  BValue oob_update = fb.ArrayUpdate(a, val, {fb.Literal(UBits(65, 8))});
  BValue oob_idx = fb.ArrayIndex(oob_update, {fb.Literal(UBits(65, 8))});

  BValue prop = fb.And({
      fb.Eq(ohs_idx, expected_idx),
      fb.Eq(gated_idx, fb.Literal(UBits(0, 4))),
      fb.Eq(oob_idx, fb.ArrayIndex(a, {fb.Literal(UBits(64, 8))})),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, PredicateOnNestedArraySortNode) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  Type* inner_t = package->GetArrayType(3, u8);
  Type* nested_t = package->GetArrayType(2, inner_t);
  BValue x = fb.Param("x", u8);
  BValue y = fb.Param("y", u8);
  BValue masked =
      fb.Or(fb.And(x, fb.Literal(UBits(0x0F, 8))), fb.Literal(UBits(0xA0, 8)));
  BValue row0 = fb.Array({masked, fb.Literal(UBits(0x55, 8)), y}, u8);
  BValue row1 =
      fb.Array({y, fb.Literal(UBits(0xFF, 8)), fb.Literal(UBits(0x01, 8))}, u8);
  BValue arr = fb.Array({row0, row1}, inner_t);
  BValue same_update =
      fb.ArrayUpdate(arr, fb.Literal(UBits(0x55, 8)),
                     {fb.Literal(UBits(0, 2)), fb.Literal(UBits(1, 2))});
  BValue diff_update = fb.ArrayUpdate(
      arr, fb.Not(y), {fb.Literal(UBits(1, 2)), fb.Literal(UBits(0, 2))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(arr));

  EXPECT_THAT(
      solver_->TryProve(f, arr.node(), Predicate::IsEqualTo(same_update.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, arr.node(), Predicate::IsEqualTo(diff_update.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenFalse()));
  EXPECT_THAT(
      solver_->TryProve(f, arr.node(), Predicate::EqualToZero(), SolverLimit()),
      StatusIs(absl::StatusCode::kInvalidArgument));

  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector prefix_a,
                           StringToTernaryVector("0b1010_XXXX"));
  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector any_u8,
                           StringToTernaryVector("0bXXXX_XXXX"));
  XLS_ASSERT_OK_AND_ASSIGN(TernaryVector val_55,
                           StringToTernaryVector("0b0101_0101"));
  std::vector<TernaryVector> leaves = {prefix_a, val_55, any_u8,
                                       any_u8,   any_u8, any_u8};
  EXPECT_THAT(solver_->TryProve(
                  f, arr.node(),
                  Predicate::IsCompatibleWith(
                      LeafTypeTree<TernaryVector>(nested_t, leaves).AsShared()),
                  SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  leaves[0] = val_55;
  EXPECT_THAT(solver_->TryProve(
                  f, arr.node(),
                  Predicate::IsCompatibleWith(
                      LeafTypeTree<TernaryVector>(nested_t, leaves).AsShared()),
                  SolverLimit()),
              IsOkAndHolds(IsProvenFalse()));
}

TEST_F(BitwuzlaIrTranslatorTest, SingletonOuterArrayUpdateSemantics) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  Type* u2 = package->GetBitsType(2);
  Type* inner_t = package->GetArrayType(2, u8);
  Type* singleton_t = package->GetArrayType(1, inner_t);
  BValue arr = fb.Param("arr", singleton_t);
  BValue val = fb.Param("val", u8);
  BValue idx = fb.Param("idx", u2);

  BValue oob_upd = fb.ArrayUpdate(
      arr, val, {fb.Literal(UBits(1, 2)), fb.Literal(UBits(0, 1))});
  BValue ib_upd = fb.ArrayUpdate(
      arr, val, {fb.Literal(UBits(0, 2)), fb.Literal(UBits(1, 1))});
  BValue sym_upd = fb.ArrayUpdate(arr, val, {idx, fb.Literal(UBits(1, 1))});

  BValue idx0 = fb.Literal(UBits(0, 1));
  BValue idx1 = fb.Literal(UBits(1, 1));
  BValue expected_elem1 = fb.Select(fb.Eq(idx, fb.Literal(UBits(0, 2))),
                                    {fb.ArrayIndex(arr, {idx0, idx1}), val});
  BValue prop = fb.And({
      fb.Eq(fb.ArrayIndex(ib_upd, {idx0, idx1}), val),
      fb.Eq(fb.ArrayIndex(ib_upd, {idx0, idx0}),
            fb.ArrayIndex(arr, {idx0, idx0})),
      fb.Eq(fb.ArrayIndex(sym_upd, {idx0, idx1}), expected_elem1),
      fb.Eq(fb.ArrayIndex(sym_upd, {idx0, idx0}),
            fb.ArrayIndex(arr, {idx0, idx0})),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(
      solver_->TryProve(f, oob_upd.node(), Predicate::IsEqualTo(arr.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, WideIndexAndSliceOob) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  BValue a3 = fb.Param("a3", package->GetArrayType(3, u8));
  BValue v = fb.Param("v", u8);
  BValue idx65 = fb.Concat({fb.Literal(UBits(1, 1)), fb.Literal(UBits(0, 64))});
  BValue upd65 = fb.ArrayUpdate(a3, v, {idx65});
  BValue u8_1 = fb.Literal(UBits(1, 8));
  BValue u8_2 = fb.Literal(UBits(2, 8));
  BValue a3_1 = fb.ArrayIndex(a3, {u8_1});
  BValue a3_2 = fb.ArrayIndex(a3, {u8_2});

  BValue max_slice = fb.ArraySlice(a3, fb.Literal(Bits::AllOnes(64)), 2);
  BValue max_exp = fb.Array({a3_2, a3_2}, u8);
  BValue straddle = fb.ArraySlice(a3, u8_1, 3);
  BValue straddle_exp = fb.Array({a3_1, a3_2, a3_2}, u8);
  BValue prop = fb.Eq(fb.ArrayIndex(a3, {idx65}), a3_2);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, upd65.node(),
                                Predicate::IsEqualTo(a3.node()), SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, max_slice.node(),
                        Predicate::IsEqualTo(max_exp.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(solver_->TryProve(f, straddle.node(),
                                Predicate::IsEqualTo(straddle_exp.node()),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, MultidimArrayOobAndZeroIndexUpdate) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  Type* row_t = package->GetArrayType(3, u8);
  Type* mat_t = package->GetArrayType(2, row_t);
  BValue arr = fb.Param("arr", mat_t);
  BValue other = fb.Param("other", mat_t);
  BValue row = fb.Param("row", row_t);
  BValue val = fb.Param("val", u8);

  BValue i0 = fb.Literal(UBits(0, 2));
  BValue i1 = fb.Literal(UBits(1, 2));
  BValue i2 = fb.Literal(UBits(2, 2));
  BValue oob = fb.Literal(UBits(3, 2));
  BValue upd_inner_oob = fb.ArrayUpdate(arr, val, {i0, oob});
  BValue upd_partial_oob = fb.ArrayUpdate(arr, row, {oob});
  BValue upd_zero_idx = fb.ArrayUpdate(arr, other, {});
  BValue prop = fb.And({
      fb.Eq(fb.ArrayIndex(arr, {oob, i1}), fb.ArrayIndex(arr, {i1, i1})),
      fb.Eq(fb.ArrayIndex(arr, {i0, oob}), fb.ArrayIndex(arr, {i0, i2})),
      fb.Eq(fb.ArrayIndex(arr, {oob, oob}), fb.ArrayIndex(arr, {i1, i2})),
  });

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, upd_inner_oob.node(),
                        Predicate::IsEqualTo(arr.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, upd_partial_oob.node(),
                        Predicate::IsEqualTo(arr.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, upd_zero_idx.node(),
                        Predicate::IsEqualTo(other.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
}

TEST_F(BitwuzlaIrTranslatorTest, ZeroBitComputedIndexAndOps) {
  auto package = CreatePackage();
  FunctionBuilder fb("f", package.get());
  Type* u8 = package->GetBitsType(8);
  BValue p0 = fb.Param("p0", package->GetBitsType(0));
  BValue inv0 = fb.Not(p0);
  BValue a2 = fb.Param("a2", package->GetArrayType(2, u8));
  BValue a1 = fb.Param("a1", package->GetArrayType(1, u8));
  BValue val = fb.Param("val", u8);

  BValue idx0 = fb.Literal(UBits(0, 1));
  BValue one1 = fb.Literal(UBits(1, 1));
  BValue a2_0 = fb.ArrayIndex(a2, {idx0});
  BValue upd_a1 = fb.ArrayUpdate(a1, val, {inv0});
  BValue exp_a1 = fb.Array({val}, u8);
  BValue slice_a2 = fb.ArraySlice(a2, inv0, 1);
  BValue exp_slice = fb.Array({a2_0}, u8);
  BValue prop = fb.And({
      fb.Eq(fb.ArrayIndex(a2, {inv0}), a2_0),
      fb.Eq(fb.AndReduce(inv0), one1),
      fb.Eq(fb.OneHot(inv0, LsbOrMsb::kLsb), one1),
      fb.Eq(fb.OneHot(inv0, LsbOrMsb::kMsb), one1),
  });

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(prop));
  EXPECT_THAT(solver_->TryProve(f, prop.node(), Predicate::NotEqualToZero(),
                                SolverLimit()),
              IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, upd_a1.node(), Predicate::IsEqualTo(exp_a1.node()),
                        SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
  EXPECT_THAT(
      solver_->TryProve(f, slice_a2.node(),
                        Predicate::IsEqualTo(exp_slice.node()), SolverLimit()),
      IsOkAndHolds(IsProvenTrue()));
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
