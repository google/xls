// Copyright 2024 The XLS Authors
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

#include "xls/codegen/fold_vast_constants.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {
namespace {

using ::xls::status_testing::IsOkAndHolds;

class FoldVastConstantsTest : public ::testing::Test {
 public:
  FoldVastConstantsTest() : file_(FileType::kSystemVerilog) {
    module_ = file_.AddModule("test_module", SourceInfo());
  }

  Literal* BareLiteral(int32_t value) {
    absl::StatusOr<Bits> bits = ParseNumber(std::to_string(value));
    CHECK_OK(bits);
    return file_.Make<Literal>(SourceInfo(),
                               value < 0 ? bits_ops::SignExtend(*bits, 32)
                                         : bits_ops::ZeroExtend(*bits, 32),
                               FormatPreference::kDefault,
                               /*declared_bit_count=*/32,
                               /*emit_bit_count=*/true,
                               /*declared_as_signed=*/true);
  }

  std::string FoldConstantsToString(Expression* expr) {
    absl::StatusOr<Expression*> folded = FoldVastConstants(expr);
    XLS_EXPECT_OK(folded);
    return (*folded)->Emit(nullptr);
  }

  absl::StatusOr<int64_t> FoldConstantsAndGetBitCount(DataType* data_type) {
    XLS_ASSIGN_OR_RETURN(DataType * folded, FoldVastConstants(data_type));
    return folded->FlatBitCountAsInt64();
  }

  VerilogFile file_;
  Module* module_;
};

TEST_F(FoldVastConstantsTest, FoldLiteral) {
  SourceInfo loc;
  EXPECT_EQ(FoldConstantsToString(BareLiteral(2)), "2");
}

TEST_F(FoldVastConstantsTest, FoldLiteralArithmetic) {
  SourceInfo loc;
  EXPECT_EQ(
      FoldConstantsToString(file_.Add(BareLiteral(2), BareLiteral(3), loc)),
      "5");
  EXPECT_EQ(
      FoldConstantsToString(file_.Sub(BareLiteral(5), BareLiteral(3), loc)),
      "2");
  EXPECT_EQ(
      FoldConstantsToString(file_.Sub(BareLiteral(2), BareLiteral(3), loc)),
      "32'shffff_ffff");
  EXPECT_EQ(
      FoldConstantsToString(file_.Mul(
          BareLiteral(4), file_.Add(BareLiteral(2), BareLiteral(3), loc), loc)),
      "20");
  EXPECT_EQ(FoldConstantsToString(
                file_.Div(BareLiteral(24),
                          file_.Mul(BareLiteral(4), BareLiteral(3), loc), loc)),
            "2");
  EXPECT_EQ(FoldConstantsToString(
                file_.Mod(BareLiteral(24),
                          file_.Mul(BareLiteral(5), BareLiteral(2), loc), loc)),
            "4");
}

TEST_F(FoldVastConstantsTest, FoldLiteralComparison) {
  SourceInfo loc;
  EXPECT_EQ(
      FoldConstantsToString(file_.Equals(BareLiteral(1), BareLiteral(1), loc)),
      "1");
  EXPECT_EQ(
      FoldConstantsToString(file_.Equals(BareLiteral(1), BareLiteral(2), loc)),
      "0");
  EXPECT_EQ(FoldConstantsToString(
                file_.NotEquals(BareLiteral(1), BareLiteral(1), loc)),
            "0");
  EXPECT_EQ(FoldConstantsToString(
                file_.NotEquals(BareLiteral(1), BareLiteral(2), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.GreaterThan(BareLiteral(5), BareLiteral(3), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.GreaterThan(BareLiteral(3), BareLiteral(5), loc)),
            "0");
  EXPECT_EQ(FoldConstantsToString(
                file_.LessThan(BareLiteral(5), BareLiteral(3), loc)),
            "0");
  EXPECT_EQ(FoldConstantsToString(
                file_.LessThan(BareLiteral(3), BareLiteral(5), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(file_.GreaterThan(
                BareLiteral(std::numeric_limits<int32_t>::max()),
                BareLiteral(std::numeric_limits<int32_t>::min()), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(file_.GreaterThan(
                BareLiteral(std::numeric_limits<int32_t>::max()),
                BareLiteral(0), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(file_.LessThan(
                BareLiteral(std::numeric_limits<int32_t>::min()),
                BareLiteral(std::numeric_limits<int32_t>::min() + 1), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.LessThanEquals(BareLiteral(1), BareLiteral(1), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.LessThanEquals(BareLiteral(1), BareLiteral(0), loc)),
            "0");
  EXPECT_EQ(FoldConstantsToString(
                file_.LessThanEquals(BareLiteral(0), BareLiteral(1), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.GreaterThanEquals(BareLiteral(1), BareLiteral(1), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.GreaterThanEquals(BareLiteral(1), BareLiteral(0), loc)),
            "1");
  EXPECT_EQ(FoldConstantsToString(
                file_.GreaterThanEquals(BareLiteral(0), BareLiteral(1), loc)),
            "0");
  // Weird-sized values not created with the convenience constructor.
  Bits three_ones = Bits::AllOnes(3);
  Bits two_of_three_ones(3);
  two_of_three_ones.SetRange(0, 2, true);
  // Try as unsigned.
  EXPECT_EQ(
      FoldConstantsToString(file_.GreaterThan(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          loc)),
      "1");
  EXPECT_EQ(
      FoldConstantsToString(file_.GreaterThanEquals(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          loc)),
      "1");
  EXPECT_EQ(
      FoldConstantsToString(file_.LessThan(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          loc)),
      "0");
  EXPECT_EQ(
      FoldConstantsToString(file_.LessThanEquals(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/false),
          loc)),
      "0");
  // Try as signed.
  EXPECT_EQ(
      FoldConstantsToString(file_.GreaterThan(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          loc)),
      "0");
  EXPECT_EQ(
      FoldConstantsToString(file_.GreaterThanEquals(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          loc)),
      "0");
  EXPECT_EQ(
      FoldConstantsToString(file_.LessThan(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          loc)),
      "1");
  EXPECT_EQ(
      FoldConstantsToString(file_.LessThanEquals(
          file_.Make<Literal>(loc, three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          file_.Make<Literal>(loc, two_of_three_ones, FormatPreference::kHex, 3,
                              /*emit_bit_count=*/true,
                              /*declared_as_signed=*/true),
          loc)),
      "1");
}

TEST_F(FoldVastConstantsTest, FoldLiteralAndParameter) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(3), loc);
  EXPECT_EQ(FoldConstantsToString(file_.Mul(
                BareLiteral(4), file_.Add(BareLiteral(2), foo, loc), loc)),
            "20");
}

TEST_F(FoldVastConstantsTest, FoldLiteralAndEnumValue) {
  SourceInfo loc;
  Enum* enum_def =
      file_.Make<Enum>(loc, DataKind::kLogic, file_.BitVectorType(16, loc));
  auto* foo = enum_def->AddMember("foo", BareLiteral(3), loc);
  EXPECT_EQ(FoldConstantsToString(file_.Mul(
                BareLiteral(4), file_.Add(BareLiteral(2), foo, loc), loc)),
            "20");
}

TEST_F(FoldVastConstantsTest, FoldTernary) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(5), loc);
  EXPECT_EQ(FoldConstantsToString(
                file_.Ternary(file_.GreaterThan(foo, BareLiteral(7), loc),
                              file_.Add(foo, BareLiteral(1), loc),
                              file_.Mul(foo, BareLiteral(2), loc), loc)),
            "10");
}

TEST_F(FoldVastConstantsTest, FoldComplexBitVectorSpec) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(3), loc);
  EXPECT_THAT(
      FoldConstantsAndGetBitCount(file_.Make<BitVectorType>(
          loc,
          file_.Mul(BareLiteral(4), file_.Add(BareLiteral(2), foo, loc), loc),
          /*is_signed=*/false, /*size_expr_is_max=*/false)),
      IsOkAndHolds(20));
}

TEST_F(FoldVastConstantsTest, FoldEnumBaseType) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(3), loc);
  EXPECT_THAT(FoldConstantsAndGetBitCount(file_.Make<Enum>(
                  loc, DataKind::kLogic,
                  file_.Make<BitVectorType>(
                      loc,
                      file_.Mul(BareLiteral(4),
                                file_.Add(BareLiteral(2), foo, loc), loc),
                      /*is_signed=*/false, /*size_expr_is_max=*/false))),
              IsOkAndHolds(20));
}

TEST_F(FoldVastConstantsTest, FoldComplexPackedArraySpec) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(3), loc);
  EXPECT_THAT(FoldConstantsAndGetBitCount(file_.Make<PackedArrayType>(
                  loc,
                  file_.Make<BitVectorType>(
                      loc,
                      file_.Mul(BareLiteral(4),
                                file_.Add(BareLiteral(2), foo, loc), loc),
                      /*is_signed=*/false, /*size_expr_is_max=*/false),
                  std::vector<Expression*>{file_.Sub(foo, BareLiteral(1), loc)},
                  /*dims_are_max=*/true)),
              IsOkAndHolds(40));
}

TEST_F(FoldVastConstantsTest, FoldStructDef) {
  SourceInfo loc;
  auto* foo = module_->AddParameter("foo", BareLiteral(3), loc);
  auto* bit_vector_type = file_.Make<BitVectorType>(
      loc, file_.Mul(BareLiteral(4), file_.Add(BareLiteral(2), foo, loc), loc),
      /*is_signed=*/false, /*size_expr_is_max=*/false);
  EXPECT_THAT(
      FoldConstantsAndGetBitCount(file_.Make<Struct>(
          loc,
          std::vector<Def*>{file_.Make<Def>(loc, "member1", DataKind::kLogic,
                                            bit_vector_type),
                            file_.Make<Def>(loc, "member2", DataKind::kLogic,
                                            bit_vector_type)})),
      IsOkAndHolds(40));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
