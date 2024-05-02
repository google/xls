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

#include "xls/codegen/vast.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class VastTest : public testing::TestWithParam<bool> {
 protected:
  bool UseSystemVerilog() const { return GetParam(); }
  FileType GetFileType() const {
    return GetParam() ? FileType::kSystemVerilog : FileType::kVerilog;
  }
};

TEST_P(VastTest, SanitizeIdentifier) {
  EXPECT_EQ("foo", SanitizeIdentifier("foo"));
  EXPECT_EQ("foo_bar", SanitizeIdentifier("foo_bar"));
  EXPECT_EQ("__bar77__", SanitizeIdentifier("__bar77__"));

  EXPECT_EQ("__foo_", SanitizeIdentifier("  foo "));
  EXPECT_EQ("_42", SanitizeIdentifier("42"));
  EXPECT_EQ("_42_", SanitizeIdentifier("42 "));
  EXPECT_EQ("name_of_a_thing", SanitizeIdentifier("name of a thing"));
  EXPECT_EQ("_qux", SanitizeIdentifier(".qux"));
  EXPECT_EQ("add_1234", SanitizeIdentifier("add.1234"));
}

TEST_P(VastTest, DataTypes) {
  VerilogFile f(GetFileType());

  LineInfo line_info;
  DataType* scalar = f.ScalarType(SourceInfo());
  EXPECT_EQ(scalar->EmitWithIdentifier(&line_info, "foo"), " foo");
  EXPECT_THAT(scalar->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(scalar->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(scalar->width(), std::nullopt);
  EXPECT_FALSE(scalar->is_signed());
  EXPECT_EQ(line_info.LookupNode(scalar),
            std::make_optional(std::vector<LineSpan>{LineSpan(0, 0)}));

  // A width 1 data type returned from BitVectorType should be a scalar.
  DataType* u1 = f.BitVectorType(1, SourceInfo());
  EXPECT_EQ(u1->EmitWithIdentifier(nullptr, "foo"), " foo");
  EXPECT_THAT(u1->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(u1->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(u1->width(), std::nullopt);
  EXPECT_FALSE(u1->is_signed());

  DataType* s1 = f.BitVectorType(1, SourceInfo(), /*is_signed=*/true);
  EXPECT_EQ(s1->EmitWithIdentifier(nullptr, "foo"), " signed [0:0] foo");
  EXPECT_THAT(s1->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(s1->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_TRUE(s1->is_signed());

  DataType* u2 = f.BitVectorType(2, SourceInfo());
  EXPECT_EQ(u2->EmitWithIdentifier(nullptr, "foo"), " [1:0] foo");
  EXPECT_THAT(u2->WidthAsInt64(), IsOkAndHolds(2));
  EXPECT_THAT(u2->FlatBitCountAsInt64(), IsOkAndHolds(2));
  EXPECT_FALSE(u2->is_signed());

  DataType* u32 = f.BitVectorType(32, SourceInfo());
  EXPECT_EQ(u32->EmitWithIdentifier(nullptr, "foo"), " [31:0] foo");
  EXPECT_THAT(u32->WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(u32->FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_FALSE(u32->is_signed());

  DataType* s32 = f.BitVectorType(32, SourceInfo(), /*is_signed=*/true);
  EXPECT_EQ(s32->EmitWithIdentifier(nullptr, "foo"), " signed [31:0] foo");
  EXPECT_THAT(s32->WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(s32->FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_TRUE(s32->is_signed());

  DataType* packed_array = f.PackedArrayType(10, {3, 2}, SourceInfo());
  EXPECT_EQ(packed_array->EmitWithIdentifier(nullptr, "foo"),
            " [9:0][2:0][1:0] foo");
  EXPECT_THAT(packed_array->WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(packed_array->FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_FALSE(packed_array->is_signed());

  DataType* spacked_array =
      f.PackedArrayType(10, {3, 2}, SourceInfo(), /*is_signed=*/true);
  EXPECT_EQ(spacked_array->EmitWithIdentifier(nullptr, "foo"),
            " signed [9:0][2:0][1:0] foo");
  EXPECT_THAT(spacked_array->WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(spacked_array->FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_TRUE(spacked_array->is_signed());

  DataType* unpacked_array = f.UnpackedArrayType(10, {3, 2}, SourceInfo());
  if (f.use_system_verilog()) {
    EXPECT_EQ(unpacked_array->EmitWithIdentifier(nullptr, "foo"),
              " [9:0] foo[3][2]");
  } else {
    EXPECT_EQ(unpacked_array->EmitWithIdentifier(nullptr, "foo"),
              " [9:0] foo[0:2][0:1]");
  }
  EXPECT_THAT(unpacked_array->WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(unpacked_array->FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_FALSE(unpacked_array->is_signed());

  // Bit vector type with non-literal range.
  DataType* bv = f.Make<BitVectorType>(
      SourceInfo(),
      /*width=*/
      f.Mul(f.PlainLiteral(10, SourceInfo()), f.PlainLiteral(5, SourceInfo()),
            SourceInfo()),
      /*is_signed=*/false);
  EXPECT_EQ(bv->EmitWithIdentifier(nullptr, "foo"), " [10 * 5 - 1:0] foo");
  EXPECT_THAT(bv->WidthAsInt64(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Width is not a literal")));
  EXPECT_THAT(bv->FlatBitCountAsInt64(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Width is not a literal")));
  EXPECT_FALSE(bv->is_signed());
}

TEST_P(VastTest, ModuleWithManyVariableDefinitions) {
  VerilogFile f(GetFileType());
  Module* module = f.Make<Module>(SourceInfo(), "my_module");
  LogicRef* a_ref =
      module->AddInput("a", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* b_ref =
      module->AddOutput("b", f.BitVectorType(4, SourceInfo()), SourceInfo());
  LogicRef* array = module->AddInput(
      "array", f.PackedArrayType(8, {42, 3}, SourceInfo()), SourceInfo());

  // Define a bunch of random regs.
  LogicRef* r1 =
      module->AddReg("r1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* r2 =
      module->AddReg("r2", f.BitVectorType(2, SourceInfo()), SourceInfo());
  LogicRef* r1_init = module->AddReg(
      "r1_init",
      f.Make<BitVectorType>(SourceInfo(), f.PlainLiteral(1, SourceInfo()),
                            /*is_signed=*/false),
      SourceInfo(), f.PlainLiteral(1, SourceInfo()));
  LogicRef* s =
      module->AddReg("s", f.BitVectorType(42, SourceInfo()), SourceInfo());
  LogicRef* s_init =
      module->AddReg("s_init", f.BitVectorType(42, SourceInfo()), SourceInfo(),
                     f.Literal(123, 42, SourceInfo()));
  LogicRef* t = module->AddReg(
      "t",
      f.Make<PackedArrayType>(
          SourceInfo(),
          /*width=*/f.PlainLiteral(42, SourceInfo()), /*packed_dims=*/
          std::vector<Expression*>(
              {f.PlainLiteral(8, SourceInfo()),
               f.Add(f.PlainLiteral(8, SourceInfo()),
                     f.PlainLiteral(42, SourceInfo()), SourceInfo())}),
          /*is_signed=*/false),
      SourceInfo());
  LogicRef* signed_foo = module->AddReg(
      "signed_foo", f.BitVectorType(8, SourceInfo(), /*is_signed=*/true),
      SourceInfo());

  // Define a bunch of random wires.
  LogicRef* x =
      module->AddWire("x", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* y =
      module->AddWire("y", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* z = module->AddWire(
      "z",
      f.Make<PackedArrayType>(
          SourceInfo(),
          f.Mul(f.PlainLiteral(3, SourceInfo()),
                f.PlainLiteral(3, SourceInfo()), SourceInfo()),
          /*packed_dims=*/
          std::vector<Expression*>(
              {f.PlainLiteral(8, SourceInfo()),
               f.Add(f.PlainLiteral(8, SourceInfo()),
                     f.PlainLiteral(42, SourceInfo()), SourceInfo())}),
          /*is_signed=*/true),
      SourceInfo());
  LogicRef* i = module->AddInteger("i", SourceInfo());

  VastNode* assign = module->Add<ContinuousAssignment>(
      SourceInfo(), b_ref,
      f.Concat({a_ref, a_ref, a_ref, a_ref}, SourceInfo()));
  LineInfo line_info;
  EXPECT_EQ(module->Emit(&line_info),
            R"(module my_module(
  input wire a,
  output wire [3:0] b,
  input wire [7:0][41:0][2:0] array
);
  reg r1;
  reg [1:0] r2;
  reg [0:0] r1_init = 1;
  reg [41:0] s;
  reg [41:0] s_init = 42'h000_0000_007b;
  reg [41:0][7:0][8 + 42 - 1:0] t;
  reg signed [7:0] signed_foo;
  wire x;
  wire y;
  wire signed [3 * 3 - 1:0][7:0][8 + 42 - 1:0] z;
  integer i;
  assign b = {a, a, a, a};
endmodule)");

  EXPECT_EQ(line_info.LookupNode(module).value(),
            std::vector<LineSpan>{LineSpan(0, 17)});
  EXPECT_EQ(line_info.LookupNode(a_ref->def()).value(),
            std::vector<LineSpan>{LineSpan(1, 1)});
  EXPECT_EQ(line_info.LookupNode(b_ref->def()).value(),
            std::vector<LineSpan>{LineSpan(2, 2)});
  EXPECT_EQ(line_info.LookupNode(array->def()).value(),
            std::vector<LineSpan>{LineSpan(3, 3)});
  EXPECT_EQ(line_info.LookupNode(r1->def()).value(),
            std::vector<LineSpan>{LineSpan(5, 5)});
  EXPECT_EQ(line_info.LookupNode(r2->def()).value(),
            std::vector<LineSpan>{LineSpan(6, 6)});
  EXPECT_EQ(line_info.LookupNode(r1_init->def()).value(),
            std::vector<LineSpan>{LineSpan(7, 7)});
  EXPECT_EQ(line_info.LookupNode(s->def()).value(),
            std::vector<LineSpan>{LineSpan(8, 8)});
  EXPECT_EQ(line_info.LookupNode(s_init->def()).value(),
            std::vector<LineSpan>{LineSpan(9, 9)});
  EXPECT_EQ(line_info.LookupNode(t->def()).value(),
            std::vector<LineSpan>{LineSpan(10, 10)});
  EXPECT_EQ(line_info.LookupNode(signed_foo->def()).value(),
            std::vector<LineSpan>{LineSpan(11, 11)});
  EXPECT_EQ(line_info.LookupNode(x->def()).value(),
            std::vector<LineSpan>{LineSpan(12, 12)});
  EXPECT_EQ(line_info.LookupNode(y->def()).value(),
            std::vector<LineSpan>{LineSpan(13, 13)});
  EXPECT_EQ(line_info.LookupNode(z->def()).value(),
            std::vector<LineSpan>{LineSpan(14, 14)});
  EXPECT_EQ(line_info.LookupNode(i->def()).value(),
            std::vector<LineSpan>{LineSpan(15, 15)});
  EXPECT_EQ(line_info.LookupNode(assign).value(),
            std::vector<LineSpan>{LineSpan(16, 16)});
}

TEST_P(VastTest, ModuleWithUnpackedArrayRegWithSize) {
  VerilogFile f(GetFileType());
  Module* module = f.Make<Module>(SourceInfo(), "my_module");
  LogicRef* out_ref =
      module->AddOutput("out", f.BitVectorType(64, SourceInfo()), SourceInfo());
  LogicRef* arr_ref = module->AddReg(
      "arr", f.UnpackedArrayType(4, {8, 64}, SourceInfo()), SourceInfo());
  module->Add<ContinuousAssignment>(
      SourceInfo(), out_ref,
      f.Index(f.Index(arr_ref, 2, SourceInfo()), 1, SourceInfo()));
  if (UseSystemVerilog()) {
    EXPECT_EQ(module->Emit(nullptr),
              R"(module my_module(
  output wire [63:0] out
);
  reg [3:0] arr[8][64];
  assign out = arr[2][1];
endmodule)");
  } else {
    EXPECT_EQ(module->Emit(nullptr),
              R"(module my_module(
  output wire [63:0] out
);
  reg [3:0] arr[0:7][0:63];
  assign out = arr[2][1];
endmodule)");
  }
}

TEST_P(VastTest, ModuleWithUnpackedArrayRegWithPackedDims) {
  VerilogFile f(GetFileType());
  Module* module = f.Make<Module>(SourceInfo(), "my_module");
  LogicRef* out_ref =
      module->AddOutput("out", f.BitVectorType(64, SourceInfo()), SourceInfo());
  DataType* element_type = f.Make<PackedArrayType>(
      SourceInfo(),
      /*width=*/f.PlainLiteral(4, SourceInfo()),
      /*packed_dims=*/
      std::vector<Expression*>(
          {f.PlainLiteral(42, SourceInfo()), f.PlainLiteral(7, SourceInfo())}),
      /*is_signed=*/false);
  DataType* array_type = f.Make<UnpackedArrayType>(
      SourceInfo(), element_type,
      /*unpacked_dims=*/
      std::vector<Expression*>(
          {f.PlainLiteral(8, SourceInfo()), f.PlainLiteral(64, SourceInfo())}));
  LogicRef* arr_ref = module->AddReg("arr", array_type, SourceInfo());
  module->Add<ContinuousAssignment>(
      SourceInfo(), out_ref,
      f.Index(f.Index(arr_ref, 2, SourceInfo()), 1, SourceInfo()));
  if (UseSystemVerilog()) {
    EXPECT_EQ(module->Emit(nullptr),
              R"(module my_module(
  output wire [63:0] out
);
  reg [3:0][41:0][6:0] arr[8][64];
  assign out = arr[2][1];
endmodule)");
  } else {
    EXPECT_EQ(module->Emit(nullptr),
              R"(module my_module(
  output wire [63:0] out
);
  reg [3:0][41:0][6:0] arr[0:7][0:63];
  assign out = arr[2][1];
endmodule)");
  }
}

TEST_P(VastTest, Literals) {
  VerilogFile f(GetFileType());
  EXPECT_EQ("32'd44", f.Literal(UBits(44, 32), SourceInfo(),
                                FormatPreference::kUnsignedDecimal)
                          ->Emit(nullptr));
  EXPECT_EQ("1'b1",
            f.Literal(UBits(1, 1), SourceInfo(), FormatPreference::kBinary)
                ->Emit(nullptr));
  EXPECT_EQ("4'b1010",
            f.Literal(UBits(10, 4), SourceInfo(), FormatPreference::kBinary)
                ->Emit(nullptr));
  EXPECT_EQ("42'h000_0000_3039",
            f.Literal(UBits(12345, 42), SourceInfo(), FormatPreference::kHex)
                ->Emit(nullptr));

  EXPECT_EQ("13579", f.PlainLiteral(13579, SourceInfo())->Emit(nullptr));

  Bits b0 = UBits(0, 1);
  Bits b2 = UBits(2, 3);
  Bits b55 = UBits(55, 32);
  Bits b1234 = UBits(1234, 55);
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber("0xabcd_000_0000_1234_4321_0000_aaaa_bbbb_cccc_dddd_eeee"));

  EXPECT_EQ(f.Literal(b0, SourceInfo(), FormatPreference::kUnsignedDecimal)
                ->Emit(nullptr),
            "1'd0");

  EXPECT_EQ(f.Literal(b2, SourceInfo(), FormatPreference::kHex)->Emit(nullptr),
            "3'h2");
  EXPECT_EQ(
      f.Literal(b2, SourceInfo(), FormatPreference::kBinary)->Emit(nullptr),
      "3'b010");

  EXPECT_EQ(
      f.Literal(b55, SourceInfo(), FormatPreference::kDefault)->Emit(nullptr),
      "55");
  EXPECT_EQ(
      f.Literal(b55, SourceInfo(), FormatPreference::kBinary)->Emit(nullptr),
      "32'b0000_0000_0000_0000_0000_0000_0011_0111");
  EXPECT_TRUE(f.Literal(b55, SourceInfo())->IsLiteralWithValue(55));

  EXPECT_EQ(
      f.Literal(b1234, SourceInfo(), FormatPreference::kHex)->Emit(nullptr),
      "55'h00_0000_0000_04d2");
  EXPECT_EQ(f.Literal(b1234, SourceInfo(), FormatPreference::kUnsignedDecimal)
                ->Emit(nullptr),
            "55'd1234");
  EXPECT_EQ(
      f.Literal(b1234, SourceInfo(), FormatPreference::kBinary)->Emit(nullptr),
      "55'b000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0100_"
      "1101_0010");
  EXPECT_TRUE(f.Literal(b1234, SourceInfo())->IsLiteralWithValue(1234));

  EXPECT_TRUE(f.Literal(huge, SourceInfo())->IsLiteral());
  EXPECT_FALSE(f.Literal(huge, SourceInfo())->IsLiteralWithValue(42));

  Literal* zero = f.Literal(UBits(0, 32), SourceInfo());
  EXPECT_TRUE(zero->IsLiteral());
  EXPECT_TRUE(zero->IsLiteralWithValue(0));
  EXPECT_FALSE(zero->IsLiteralWithValue(1));

  Literal* plain_zero = f.PlainLiteral(0, SourceInfo());
  EXPECT_TRUE(plain_zero->IsLiteral());
  EXPECT_TRUE(plain_zero->IsLiteralWithValue(0));
  EXPECT_FALSE(plain_zero->IsLiteralWithValue(1));

  Literal* forty_two = f.Literal(UBits(42, 32), SourceInfo());
  EXPECT_TRUE(forty_two->IsLiteral());
  EXPECT_FALSE(forty_two->IsLiteralWithValue(0));
  EXPECT_TRUE(forty_two->IsLiteralWithValue(42));

  Literal* wide_forty_two = f.Literal(UBits(42, 123456), SourceInfo());
  EXPECT_TRUE(wide_forty_two->IsLiteral());
  EXPECT_FALSE(wide_forty_two->IsLiteralWithValue(0));
  EXPECT_TRUE(wide_forty_two->IsLiteralWithValue(42));

  Literal* all_ones = f.Literal(UBits(15, 4), SourceInfo());
  EXPECT_TRUE(all_ones->IsLiteral());
  EXPECT_FALSE(all_ones->IsLiteralWithValue(0));
  EXPECT_TRUE(all_ones->IsLiteralWithValue(15));
  EXPECT_FALSE(all_ones->IsLiteralWithValue(-1));
}

TEST_P(VastTest, Precedence) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("precedence", SourceInfo());
  auto a = m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  auto b = m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  auto c = m->AddReg("c", f.BitVectorType(8, SourceInfo()), SourceInfo());
  EXPECT_EQ("-a", f.Negate(a, SourceInfo())->Emit(nullptr));
  // Though technically not required by precedence the Verilog consumers are
  // happier if nested unary are wrapped in parens.
  EXPECT_EQ(
      "-(~a)",
      f.Negate(f.BitwiseNot(a, SourceInfo()), SourceInfo())->Emit(nullptr));
  EXPECT_EQ(
      "-(a | b)",
      f.Negate(f.BitwiseOr(a, b, SourceInfo()), SourceInfo())->Emit(nullptr));

  EXPECT_EQ("a + b", f.Add(a, b, SourceInfo())->Emit(nullptr));
  EXPECT_EQ("a + (b + c)",
            f.Add(a, f.Add(b, c, SourceInfo()), SourceInfo())->Emit(nullptr));
  EXPECT_EQ("a + b + c",
            f.Add(f.Add(a, b, SourceInfo()), c, SourceInfo())->Emit(nullptr));

  EXPECT_EQ("a + b * c",
            f.Add(a, f.Mul(b, c, SourceInfo()), SourceInfo())->Emit(nullptr));
  EXPECT_EQ("a * (b + c)",
            f.Mul(a, f.Add(b, c, SourceInfo()), SourceInfo())->Emit(nullptr));

  EXPECT_EQ(
      "a | (a + b || b)",
      f.BitwiseOr(a, f.LogicalOr(f.Add(a, b, SourceInfo()), b, SourceInfo()),
                  SourceInfo())
          ->Emit(nullptr));
  EXPECT_EQ("a * b << (b & c)",
            f.Shll(f.Mul(a, b, SourceInfo()), f.BitwiseAnd(b, c, SourceInfo()),
                   SourceInfo())
                ->Emit(nullptr));
}

TEST_P(VastTest, UnaryReductionOperations) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("precedence", SourceInfo());
  auto a = m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  auto b = m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  auto c = m->AddReg("c", f.BitVectorType(8, SourceInfo()), SourceInfo());
  // Verify reduction operations are parentheses-wrapped when inside of binary
  // infix expressions.
  EXPECT_EQ("|a", f.OrReduce(a, SourceInfo())->Emit(nullptr));
  EXPECT_EQ("&a", f.AndReduce(a, SourceInfo())->Emit(nullptr));
  EXPECT_EQ("b && (^c)",
            f.LogicalAnd(b, f.XorReduce(c, SourceInfo()), SourceInfo())
                ->Emit(nullptr));
  EXPECT_EQ("(^a) + b + (|c)",
            f.Add(f.Add(f.XorReduce(a, SourceInfo()), b, SourceInfo()),
                  f.OrReduce(c, SourceInfo()), SourceInfo())
                ->Emit(nullptr));
}

TEST_P(VastTest, NestedUnaryOps) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("NestedUnaryOps", SourceInfo());
  auto a = m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  auto b = m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  EXPECT_EQ("-a", f.Negate(a, SourceInfo())->Emit(nullptr));
  EXPECT_EQ(
      "-(~a)",
      f.Negate(f.BitwiseNot(a, SourceInfo()), SourceInfo())->Emit(nullptr));
  EXPECT_EQ(
      "-(-(-a))",
      f.Negate(f.Negate(f.Negate(a, SourceInfo()), SourceInfo()), SourceInfo())
          ->Emit(nullptr));
  EXPECT_EQ(
      "-(~(a + b)) - b",
      f.Sub(f.Negate(f.BitwiseNot(f.Add(a, b, SourceInfo()), SourceInfo()),
                     SourceInfo()),
            b, SourceInfo())
          ->Emit(nullptr));
}

TEST_P(VastTest, Case) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* thing_next =
      m->AddWire("thing_next", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* thing =
      m->AddWire("thing", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* my_state =
      m->AddWire("my_state", f.BitVectorType(2, SourceInfo()), SourceInfo());
  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Case* case_statement = ac->statements()->Add<Case>(SourceInfo(), my_state);
  StatementBlock* one_block =
      case_statement->AddCaseArm(f.Literal(1, 2, SourceInfo()));
  one_block->Add<BlockingAssignment>(SourceInfo(), thing_next, thing);
  StatementBlock* default_block = case_statement->AddCaseArm(DefaultSentinel());
  default_block->Add<BlockingAssignment>(SourceInfo(), thing_next,
                                         f.Make<XSentinel>(SourceInfo(), 2));

  EXPECT_EQ(case_statement->Emit(nullptr),
            R"(case (my_state)
  2'h1: begin
    thing_next = thing;
  end
  default: begin
    thing_next = 2'dx;
  end
endcase)");
}

TEST_P(VastTest, Casez) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* thing_next =
      m->AddWire("thing_next", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* thing =
      m->AddWire("thing", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* my_state =
      m->AddWire("my_state", f.BitVectorType(2, SourceInfo()), SourceInfo());
  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Case* case_statement = ac->statements()->Add<Case>(
      SourceInfo(), my_state, CaseType(CaseKeyword::kCasez));
  StatementBlock* one_block =
      case_statement->AddCaseArm(f.Literal(1, 2, SourceInfo()));
  one_block->Add<BlockingAssignment>(SourceInfo(), thing_next, thing);
  StatementBlock* default_block = case_statement->AddCaseArm(DefaultSentinel());
  default_block->Add<BlockingAssignment>(SourceInfo(), thing_next,
                                         f.Make<XSentinel>(SourceInfo(), 2));

  EXPECT_EQ(case_statement->Emit(nullptr),
            R"(casez (my_state)
  2'h1: begin
    thing_next = thing;
  end
  default: begin
    thing_next = 2'dx;
  end
endcase)");
}


TEST_P(VastTest, CaseWithHighZ) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* thing_next =
      m->AddWire("thing_next", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* thing =
      m->AddWire("thing", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* my_state =
      m->AddWire("my_state", f.BitVectorType(2, SourceInfo()), SourceInfo());
  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Case* case_statement = ac->statements()->Add<Case>(SourceInfo(), my_state);
  FourValueBinaryLiteral* msb_set = f.Make<FourValueBinaryLiteral>(
      SourceInfo(), std::vector({FourValueBit::kOne, FourValueBit::kHighZ}));
  StatementBlock* one_block =
      case_statement->AddCaseArm(msb_set);
  one_block->Add<BlockingAssignment>(SourceInfo(), thing_next, thing);
  FourValueBinaryLiteral* lsb_unset = f.Make<FourValueBinaryLiteral>(
      SourceInfo(), std::vector({FourValueBit::kUnknown, FourValueBit::kZero}));
  StatementBlock* zero_block =
      case_statement->AddCaseArm(lsb_unset);
  zero_block->Add<BlockingAssignment>(SourceInfo(), thing_next, thing);
  StatementBlock* default_block = case_statement->AddCaseArm(DefaultSentinel());
  default_block->Add<BlockingAssignment>(SourceInfo(), thing_next,
                                         f.Make<XSentinel>(SourceInfo(), 2));

  EXPECT_EQ(case_statement->Emit(nullptr),
            R"(case (my_state)
  2'b1?: begin
    thing_next = thing;
  end
  2'bX0: begin
    thing_next = thing;
  end
  default: begin
    thing_next = 2'dx;
  end
endcase)");
}

TEST_P(VastTest, AlwaysFlopTestNoReset) {
  std::vector<std::string> fsm_signals = {
      "rx_byte_done",
      "state",
      "tx_byte",
      "tx_byte_valid",
  };

  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, SourceInfo()), SourceInfo());
  AlwaysFlop* af = f.Make<AlwaysFlop>(SourceInfo(), clk);
  for (const std::string& signal_name : fsm_signals) {
    LogicRef* signal =
        m->AddReg(signal_name, f.BitVectorType(1, SourceInfo()), SourceInfo());
    LogicRef* signal_next =
        m->AddReg(absl::StrCat(signal_name, "_next"),
                  f.BitVectorType(1, SourceInfo()), SourceInfo());
    af->AddRegister(signal, signal_next, SourceInfo());
  }
  m->AddModuleMember(af);

  EXPECT_EQ(af->Emit(nullptr),
            R"(always @ (posedge my_clk) begin
  rx_byte_done <= rx_byte_done_next;
  state <= state_next;
  tx_byte <= tx_byte_next;
  tx_byte_valid <= tx_byte_valid_next;
end)");
}

TEST_P(VastTest, AlwaysFlopTestSyncReset) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* rst =
      m->AddInput("my_rst", f.BitVectorType(1, SourceInfo()), SourceInfo());

  LogicRef* a = m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* a_next =
      m->AddReg("a_next", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* b = m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* b_next =
      m->AddReg("b_next", f.BitVectorType(8, SourceInfo()), SourceInfo());

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      SourceInfo(), clk, Reset{rst, /*async*/ false, /*active_low*/ false});
  af->AddRegister(a, a_next, SourceInfo(),
                  /*reset_value=*/f.Literal(42, 8, SourceInfo()));
  af->AddRegister(b, b_next, SourceInfo());

  EXPECT_EQ(af->Emit(nullptr),
            R"(always @ (posedge my_clk) begin
  if (my_rst) begin
    a <= 8'h2a;
  end else begin
    a <= a_next;
    b <= b_next;
  end
end)");
}

TEST_P(VastTest, AlwaysFlopTestAsyncResetActiveLow) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* rst =
      m->AddInput("my_rst_n", f.BitVectorType(1, SourceInfo()), SourceInfo());

  LogicRef* a = m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* a_next =
      m->AddReg("a_next", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* b = m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* b_next =
      m->AddReg("b_next", f.BitVectorType(8, SourceInfo()), SourceInfo());

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      SourceInfo(), clk, Reset{rst, /*async*/ true, /*active_low*/ true});
  af->AddRegister(a, a_next, SourceInfo(),
                  /*reset_value=*/f.Literal(42, 8, SourceInfo()));
  af->AddRegister(b, b_next, SourceInfo());

  EXPECT_EQ(af->Emit(nullptr),
            R"(always @ (posedge my_clk or negedge my_rst_n) begin
  if (!my_rst_n) begin
    a <= 8'h2a;
  end else begin
    a <= a_next;
    b <= b_next;
  end
end)");
}

TEST_P(VastTest, AlwaysFf) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* rst_n =
      m->AddInput("rst_n", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* foo =
      m->AddInput("foo", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* bar =
      m->AddInput("bar", f.BitVectorType(1, SourceInfo()), SourceInfo());

  LogicRef* foo_reg =
      m->AddReg("foo_reg", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* bar_reg =
      m->AddReg("bar_reg", f.BitVectorType(1, SourceInfo()), SourceInfo());

  AlwaysFf* always_ff =
      f.Make<AlwaysFf>(SourceInfo(), std::vector<SensitivityListElement>{
                                         f.Make<PosEdge>(SourceInfo(), clk),
                                         f.Make<NegEdge>(SourceInfo(), rst_n)});
  always_ff->statements()->Add<NonblockingAssignment>(SourceInfo(), foo_reg,
                                                      foo);
  always_ff->statements()->Add<NonblockingAssignment>(SourceInfo(), bar_reg,
                                                      bar);
  m->AddModuleMember(always_ff);

  EXPECT_EQ(always_ff->Emit(nullptr),
            R"(always_ff @ (posedge my_clk or negedge rst_n) begin
  foo_reg <= foo;
  bar_reg <= bar;
end)");
}

TEST_P(VastTest, Always) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* foo =
      m->AddReg("foo", f.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* bar =
      m->AddReg("bar", f.BitVectorType(32, SourceInfo()), SourceInfo());

  Always* always = f.Make<Always>(
      SourceInfo(),
      std::vector<SensitivityListElement>{ImplicitEventExpression()});
  always->statements()->Add<BlockingAssignment>(
      SourceInfo(), foo, f.PlainLiteral(7, SourceInfo()));
  always->statements()->Add<BlockingAssignment>(
      SourceInfo(), bar, f.PlainLiteral(42, SourceInfo()));
  m->AddModuleMember(always);

  EXPECT_EQ(always->Emit(nullptr),
            R"(always @ (*) begin
  foo = 7;
  bar = 42;
end)");
}

TEST_P(VastTest, AlwaysSignalInSensitivityList) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* foo =
      m->AddReg("foo", f.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* bar =
      m->AddReg("bar", f.BitVectorType(32, SourceInfo()), SourceInfo());

  Always* always = f.Make<Always>(
      SourceInfo(), std::vector<SensitivityListElement>{foo, bar});
  always->statements()->Add<BlockingAssignment>(
      SourceInfo(), foo, f.PlainLiteral(7, SourceInfo()));
  always->statements()->Add<BlockingAssignment>(
      SourceInfo(), bar, f.PlainLiteral(42, SourceInfo()));
  m->AddModuleMember(always);

  EXPECT_EQ(always->Emit(nullptr),
            R"(always @ (foo or bar) begin
  foo = 7;
  bar = 42;
end)");
}

TEST_P(VastTest, AlwaysCombTest) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* rx_byte_done = m->AddWire(
      "rx_byte_done", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* rx_byte_done_next = m->AddWire(
      "rx_byte_done_next", f.BitVectorType(1, SourceInfo()), SourceInfo());

  LogicRef* tx_byte_next = m->AddWire(
      "tx_byte_next", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* tx_byte =
      m->AddWire("tx_byte", f.BitVectorType(8, SourceInfo()), SourceInfo());

  LogicRef* tx_byte_valid_next = m->AddWire(
      "tx_byte_valid_next", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* tx_byte_valid = m->AddWire(
      "tx_byte_valid", f.BitVectorType(1, SourceInfo()), SourceInfo());

  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  for (auto p : std::vector<std::pair<Expression*, Expression*>>{
           {rx_byte_done_next, rx_byte_done},
           {tx_byte_next, tx_byte},
           {tx_byte_valid_next, tx_byte_valid},
       }) {
    ac->statements()->Add<BlockingAssignment>(SourceInfo(), p.first, p.second);
  }

  EXPECT_EQ(ac->Emit(nullptr),
            R"(always_comb begin
  rx_byte_done_next = rx_byte_done;
  tx_byte_next = tx_byte;
  tx_byte_valid_next = tx_byte_valid;
end)");
}

TEST_P(VastTest, InstantiationTest) {
  VerilogFile f(GetFileType());
  auto* default_clocks_per_baud =
      f.Make<MacroRef>(SourceInfo(), "DEFAULT_CLOCKS_PER_BAUD");
  auto* clk_def =
      f.Make<WireDef>(SourceInfo(), "my_clk", f.BitVectorType(1, SourceInfo()));
  auto* clk_ref = f.Make<LogicRef>(SourceInfo(), clk_def);

  auto* tx_byte_def = f.Make<WireDef>(SourceInfo(), "my_tx_byte",
                                      f.BitVectorType(8, SourceInfo()));
  auto* tx_byte_ref = f.Make<LogicRef>(SourceInfo(), tx_byte_def);
  auto* instantiation =
      f.Make<Instantiation>(SourceInfo(),
                            /*module_name=*/"uart_transmitter",
                            /*instance_name=*/"tx",
                            /*parameters=*/
                            std::vector<Connection>{
                                {"ClocksPerBaud", default_clocks_per_baud},
                            },
                            /*connections=*/
                            std::vector<Connection>{
                                {"clk", clk_ref},
                                {"tx_byte", tx_byte_ref},
                            });

  EXPECT_EQ(instantiation->Emit(nullptr),
            R"(uart_transmitter #(
  .ClocksPerBaud(`DEFAULT_CLOCKS_PER_BAUD)
) tx (
  .clk(my_clk),
  .tx_byte(my_tx_byte)
);)");
}

TEST_P(VastTest, TemplateInstantiationTest) {
  VerilogFile f(GetFileType());
  auto* a_def =
      f.Make<WireDef>(SourceInfo(), "i_a", f.BitVectorType(32, SourceInfo()));
  auto* a_ref = f.Make<LogicRef>(SourceInfo(), a_def);

  auto* ret_def =
      f.Make<WireDef>(SourceInfo(), "o_ret", f.BitVectorType(32, SourceInfo()));
  auto* ret_ref = f.Make<LogicRef>(SourceInfo(), ret_def);

  const std::string_view code_template = "foo {fn} (.x({a}), .out({return}))";

  auto* instantiation =
      f.Make<TemplateInstantiation>(SourceInfo(),
                                    /*instance_name=*/"template_inst_42",
                                    /*code_template=*/code_template,
                                    /*connections=*/
                                    std::vector<Connection>{
                                        {"a", a_ref},
                                        {"return", ret_ref},
                                    });

  EXPECT_EQ(instantiation->Emit(nullptr),
            "foo template_inst_42 (.x(i_a), .out(o_ret));");
}

TEST_P(VastTest, BlockingAndNonblockingAssignments) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* a = m->AddReg("a", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* b = m->AddReg("b", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* c = m->AddReg("c", f.BitVectorType(1, SourceInfo()), SourceInfo());
  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  ac->statements()->Add<BlockingAssignment>(SourceInfo(), a, b);
  ac->statements()->Add<NonblockingAssignment>(SourceInfo(), b, c);

  EXPECT_EQ(ac->Emit(nullptr),
            R"(always_comb begin
  a = b;
  b <= c;
end)");
}

TEST_P(VastTest, ParameterAndLocalParam) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  m->AddParameter("ClocksPerBaud",
                  f.Make<MacroRef>(SourceInfo(), "DEFAULT_CLOCKS_PER_BAUD"),
                  SourceInfo());
  LocalParam* p = m->Add<LocalParam>(SourceInfo());
  LocalParamItemRef* idle =
      p->AddItem("StateIdle", f.Literal(0, 2, SourceInfo()), SourceInfo());
  p->AddItem("StateGotByte", f.Literal(1, 2, SourceInfo()), SourceInfo());
  p->AddItem("StateError", f.Literal(2, 2, SourceInfo()), SourceInfo());

  LocalParamItemRef* state_bits =
      m->Add<LocalParam>(SourceInfo())
          ->AddItem("StateBits", f.Literal(2, 2, SourceInfo()), SourceInfo());
  m->AddReg(
      "state",
      f.Make<BitVectorType>(SourceInfo(), state_bits, /*is_signed=*/false),
      SourceInfo(), idle);

  EXPECT_EQ(m->Emit(nullptr),
            R"(module top;
  parameter ClocksPerBaud = `DEFAULT_CLOCKS_PER_BAUD;
  localparam
    StateIdle = 2'h0,
    StateGotByte = 2'h1,
    StateError = 2'h2;
  localparam StateBits = 2'h2;
  reg [StateBits - 1:0] state = StateIdle;
endmodule)");
}

TEST_P(VastTest, SimpleConditional) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* input =
      m->AddInput("input", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* output =
      m->AddReg("output", f.BitVectorType(1, SourceInfo()), SourceInfo());
  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Conditional* if_statement =
      ac->statements()->Add<Conditional>(SourceInfo(), input);
  if_statement->consequent()->Add<BlockingAssignment>(
      SourceInfo(), output, f.Literal(1, 1, SourceInfo()));
  EXPECT_EQ(if_statement->Emit(nullptr),
            R"(if (input) begin
  output = 1'h1;
end)");
}

TEST_P(VastTest, SignedOperation) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* a =
      m->AddInput("a", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* b =
      m->AddInput("b", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* out =
      m->AddOutput("out", f.BitVectorType(8, SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), out,
      f.Div(f.Make<SignedCast>(SourceInfo(), a),
            f.Make<SignedCast>(SourceInfo(), b), SourceInfo()));
  EXPECT_EQ(m->Emit(nullptr),
            R"(module top(
  input wire [7:0] a,
  input wire [7:0] b,
  output wire [7:0] out
);
  assign out = $signed(a) / $signed(b);
endmodule)");
}

TEST_P(VastTest, ComplexConditional) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* output1 =
      m->AddReg("output1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* output2 =
      m->AddReg("output2", f.BitVectorType(1, SourceInfo()), SourceInfo());

  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Conditional* conditional =
      ac->statements()->Add<Conditional>(SourceInfo(), input1);
  conditional->consequent()->Add<BlockingAssignment>(
      SourceInfo(), output1, f.Literal(1, 1, SourceInfo()));

  StatementBlock* alternate1 =
      conditional->AddAlternate(f.BitwiseAnd(input1, input2, SourceInfo()));
  alternate1->Add<BlockingAssignment>(SourceInfo(), output1,
                                      f.Literal(0, 1, SourceInfo()));

  StatementBlock* alternate2 = conditional->AddAlternate(input2);
  alternate2->Add<BlockingAssignment>(SourceInfo(), output1,
                                      f.Literal(1, 1, SourceInfo()));

  StatementBlock* alternate3 = conditional->AddAlternate();
  alternate3->Add<BlockingAssignment>(SourceInfo(), output1,
                                      f.Literal(0, 1, SourceInfo()));
  alternate3->Add<BlockingAssignment>(SourceInfo(), output2,
                                      f.Literal(1, 1, SourceInfo()));

  EXPECT_EQ(conditional->Emit(nullptr),
            R"(if (input1) begin
  output1 = 1'h1;
end else if (input1 & input2) begin
  output1 = 1'h0;
end else if (input2) begin
  output1 = 1'h1;
end else begin
  output1 = 1'h0;
  output2 = 1'h1;
end)");
}

TEST_P(VastTest, NestedConditional) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* output1 =
      m->AddReg("output1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* output2 =
      m->AddReg("output2", f.BitVectorType(1, SourceInfo()), SourceInfo());

  AlwaysComb* ac = m->Add<AlwaysComb>(SourceInfo());
  Conditional* conditional =
      ac->statements()->Add<Conditional>(SourceInfo(), input1);
  conditional->consequent()->Add<BlockingAssignment>(
      SourceInfo(), output1, f.Literal(1, 1, SourceInfo()));

  StatementBlock* alternate = conditional->AddAlternate();

  Conditional* nested_conditional =
      alternate->Add<Conditional>(SourceInfo(), input2);
  nested_conditional->consequent()->Add<BlockingAssignment>(
      SourceInfo(), output2, f.Literal(1, 1, SourceInfo()));
  StatementBlock* nested_alternate = nested_conditional->AddAlternate();
  nested_alternate->Add<BlockingAssignment>(SourceInfo(), output1,
                                            f.Literal(0, 1, SourceInfo()));
  nested_alternate->Add<BlockingAssignment>(SourceInfo(), output2,
                                            f.Literal(1, 1, SourceInfo()));

  EXPECT_EQ(m->Emit(nullptr),
            R"(module top(
  input wire input1,
  input wire input2
);
  reg output1;
  reg output2;
  always_comb begin
    if (input1) begin
      output1 = 1'h1;
    end else begin
      if (input2) begin
        output2 = 1'h1;
      end else begin
        output1 = 1'h0;
        output2 = 1'h1;
      end
    end
  end
endmodule)");
}

TEST_P(VastTest, TestbenchClock) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("testbench", SourceInfo());
  LogicRef* clk =
      m->AddReg("clk", f.BitVectorType(1, SourceInfo()), SourceInfo());

  Initial* initial = m->Add<Initial>(SourceInfo());
  Statement* clk_equals_zero = f.Make<BlockingAssignment>(
      SourceInfo(), clk, f.PlainLiteral(0, SourceInfo()));
  initial->statements()->Add<DelayStatement>(
      SourceInfo(), f.PlainLiteral(1, SourceInfo()), clk_equals_zero);

  initial->statements()->Add<Forever>(
      SourceInfo(),
      f.Make<DelayStatement>(
          SourceInfo(), f.PlainLiteral(1, SourceInfo()),
          f.Make<BlockingAssignment>(SourceInfo(), clk,
                                     f.LogicalNot(clk, SourceInfo()))));

  EXPECT_EQ(f.Emit(nullptr),
            R"(module testbench;
  reg clk;
  initial begin
    #1 clk = 0;
    forever #1 clk = !clk;
  end
endmodule
)");
}

TEST_P(VastTest, TestbenchDisplayAndMonitor) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("testbench", SourceInfo());
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, SourceInfo()), SourceInfo());
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, SourceInfo()), SourceInfo());

  Initial* initial = m->Add<Initial>(SourceInfo());
  std::vector<Expression*> display_args = {
      f.Make<QuotedString>(SourceInfo(), R"(foo\n)")};
  initial->statements()->Add<Display>(SourceInfo(), display_args);
  initial->statements()->Add<DelayStatement>(SourceInfo(),
                                             f.PlainLiteral(42, SourceInfo()));
  std::vector<Expression*> monitor_args = {
      f.Make<QuotedString>(SourceInfo(), R"(%t %d %d)"),
      f.Make<SystemFunctionCall>(SourceInfo(), "time"), input1, input2};
  initial->statements()->Add<Monitor>(SourceInfo(), monitor_args);
  initial->statements()->Add<DelayStatement>(
      SourceInfo(), f.Add(f.PlainLiteral(123, SourceInfo()),
                          f.PlainLiteral(456, SourceInfo()), SourceInfo()));
  initial->statements()->Add<WaitStatement>(SourceInfo(), input1);
  initial->statements()->Add<Finish>(SourceInfo());

  EXPECT_EQ(f.Emit(nullptr),
            R"(module testbench(
  input wire input1,
  input wire input2
);
  initial begin
    $display("foo\n");
    #42;
    $monitor("%t %d %d", $time, input1, input2);
    #(123 + 456);
    wait(input1);
    $finish;
  end
endmodule
)");
}

TEST_P(VastTest, Concat) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("Concat", SourceInfo());
  EXPECT_EQ(
      "{32'h0000_002a}",
      f.Concat({f.Literal(42, 32, SourceInfo())}, SourceInfo())->Emit(nullptr));
  EXPECT_EQ(
      "{a, 8'h7b, b}",
      f.Concat({m->AddReg("a", f.BitVectorType(1, SourceInfo()), SourceInfo()),
                f.Literal(123, 8, SourceInfo()),
                m->AddReg("b", f.BitVectorType(1, SourceInfo()), SourceInfo())},
               SourceInfo())
          ->Emit(nullptr));
  EXPECT_EQ(
      "{42{a, 8'h7b, b}}",
      f.Concat(/*replication=*/42,
               {m->AddReg("a", f.BitVectorType(1, SourceInfo()), SourceInfo()),
                f.Literal(123, 8, SourceInfo()),
                m->AddReg("b", f.BitVectorType(1, SourceInfo()), SourceInfo())},
               SourceInfo())
          ->Emit(nullptr));
}

TEST_P(VastTest, PartSelect) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("PartSelect", SourceInfo());
  EXPECT_EQ("a[4'h3 +: 16'h0006]",
            f.PartSelect(
                 m->AddReg("a", f.BitVectorType(8, SourceInfo()), SourceInfo()),
                 f.Literal(3, 4, SourceInfo()), f.Literal(6, 16, SourceInfo()),
                 SourceInfo())
                ->Emit(nullptr));
  EXPECT_EQ("b[c +: 16'h0012]",
            f.PartSelect(
                 m->AddReg("b", f.BitVectorType(8, SourceInfo()), SourceInfo()),
                 m->AddReg("c", f.BitVectorType(1, SourceInfo()), SourceInfo()),
                 f.Literal(18, 16, SourceInfo()), SourceInfo())
                ->Emit(nullptr));
}

TEST_P(VastTest, ArrayAssignmentPattern) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("ArrayAssignmentPattern", SourceInfo());
  EXPECT_EQ(
      "'{32'h0000_002a}",
      f.ArrayAssignmentPattern({f.Literal(42, 32, SourceInfo())}, SourceInfo())
          ->Emit(nullptr));
  EXPECT_EQ(
      "'{a, 32'h0000_007b, b}",
      f.ArrayAssignmentPattern(
           {m->AddReg("a", f.BitVectorType(32, SourceInfo()), SourceInfo()),
            f.Literal(123, 32, SourceInfo()),
            m->AddReg("b", f.BitVectorType(32, SourceInfo()), SourceInfo())},
           SourceInfo())
          ->Emit(nullptr));
  EXPECT_EQ("'{'{foo, bar}, '{baz, qux}}",
            f.ArrayAssignmentPattern(
                 {f.ArrayAssignmentPattern(
                      {m->AddReg("foo", f.BitVectorType(32, SourceInfo()),
                                 SourceInfo()),
                       m->AddReg("bar", f.BitVectorType(32, SourceInfo()),
                                 SourceInfo())},
                      SourceInfo()),
                  f.ArrayAssignmentPattern(
                      {m->AddReg("baz", f.BitVectorType(32, SourceInfo()),
                                 SourceInfo()),
                       m->AddReg("qux", f.BitVectorType(32, SourceInfo()),
                                 SourceInfo())},
                      SourceInfo())},
                 SourceInfo())
                ->Emit(nullptr));
}

TEST_P(VastTest, ModuleSections) {
  VerilogFile f(GetFileType());
  Module* module = f.Make<Module>(SourceInfo(), "my_module");
  ModuleSection* s0 = module->Add<ModuleSection>(SourceInfo());
  module->AddReg("foo", f.BitVectorType(1, SourceInfo()), SourceInfo(),
                 f.PlainLiteral(1, SourceInfo()));
  ModuleSection* s1 = module->Add<ModuleSection>(SourceInfo());
  // Create an empty section.
  module->Add<ModuleSection>(SourceInfo());

  // Fill the sections and add something to the module top as well.
  s0->Add<Comment>(SourceInfo(), "section 0");
  s1->Add<Comment>(SourceInfo(), "section 1");
  ModuleSection* nested_section = s1->Add<ModuleSection>(SourceInfo());
  nested_section->Add<Comment>(SourceInfo(), "  nested in section 1");
  module->Add<Comment>(SourceInfo(), "random comment at end");
  s1->Add<Comment>(SourceInfo(), "more stuff in section 1");
  s0->Add<Comment>(SourceInfo(), "more stuff in section 0");
  s0->Add<InlineVerilogStatement>(SourceInfo(), "`SOME_MACRO(42);");
  module->AddReg("section_0_reg", f.BitVectorType(42, SourceInfo()),
                 SourceInfo(), /*init=*/nullptr,
                 /*section=*/s0);
  LineInfo line_info;
  EXPECT_EQ(module->Emit(&line_info),
            R"(module my_module;
  // section 0
  // more stuff in section 0
  `SOME_MACRO(42);
  reg [41:0] section_0_reg;
  reg foo = 1;
  // section 1
  //   nested in section 1
  // more stuff in section 1
  // random comment at end
endmodule)");

  EXPECT_EQ(line_info.LookupNode(module).value(),
            std::vector<LineSpan>{LineSpan(0, 10)});
  EXPECT_EQ(line_info.LookupNode(s0).value(),
            std::vector<LineSpan>{LineSpan(1, 4)});
  EXPECT_EQ(line_info.LookupNode(s1).value(),
            std::vector<LineSpan>{LineSpan(6, 8)});
  EXPECT_EQ(line_info.LookupNode(nested_section).value(),
            std::vector<LineSpan>{LineSpan(7, 7)});
}

TEST_P(VastTest, VerilogFunction) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  VerilogFunction* func = m->Add<VerilogFunction>(
      SourceInfo(), "func", f.BitVectorType(42, SourceInfo()));
  LogicRef* foo =
      func->AddArgument("foo", f.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* bar =
      func->AddArgument("bar", f.BitVectorType(3, SourceInfo()), SourceInfo());
  func->AddStatement<BlockingAssignment>(SourceInfo(), func->return_value_ref(),
                                         f.Shll(foo, bar, SourceInfo()));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), qux,
      f.Make<VerilogFunctionCall>(
          SourceInfo(), func,
          std::vector<Expression*>{f.Literal(UBits(12, 32), SourceInfo()),
                                   f.Literal(UBits(2, 3), SourceInfo())}));
  EXPECT_EQ(m->Emit(nullptr),
            R"(module top;
  function automatic [41:0] func (input reg [31:0] foo, input reg [2:0] bar);
    begin
      func = foo << bar;
    end
  endfunction
  wire [31:0] qux;
  assign qux = func(32'h0000_000c, 3'h2);
endmodule)");
}

TEST_P(VastTest, VerilogFunctionNoArguments) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  VerilogFunction* func = m->Add<VerilogFunction>(
      SourceInfo(), "func", f.BitVectorType(42, SourceInfo()));
  func->AddStatement<BlockingAssignment>(
      SourceInfo(), func->return_value_ref(),
      f.Literal(UBits(0x42, 42), SourceInfo()));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), qux,
      f.Make<VerilogFunctionCall>(SourceInfo(), func,
                                  std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(nullptr),
            R"(module top;
  function automatic [41:0] func ();
    begin
      func = 42'h000_0000_0042;
    end
  endfunction
  wire [31:0] qux;
  assign qux = func();
endmodule)");
}

TEST_P(VastTest, VerilogFunctionWithRegDefs) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  VerilogFunction* func = m->Add<VerilogFunction>(
      SourceInfo(), "func", f.BitVectorType(42, SourceInfo()));
  LogicRef* foo =
      func->AddRegDef(SourceInfo(), "foo", f.BitVectorType(42, SourceInfo()));
  LogicRef* bar =
      func->AddRegDef(SourceInfo(), "bar", f.BitVectorType(42, SourceInfo()));
  func->AddStatement<BlockingAssignment>(
      SourceInfo(), foo, f.Literal(UBits(0x42, 42), SourceInfo()));
  func->AddStatement<BlockingAssignment>(SourceInfo(), bar, foo);
  func->AddStatement<BlockingAssignment>(SourceInfo(), func->return_value_ref(),
                                         bar);

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), qux,
      f.Make<VerilogFunctionCall>(SourceInfo(), func,
                                  std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(nullptr),
            R"(module top;
  function automatic [41:0] func ();
    reg [41:0] foo;
    reg [41:0] bar;
    begin
      foo = 42'h000_0000_0042;
      bar = foo;
      func = bar;
    end
  endfunction
  wire [31:0] qux;
  assign qux = func();
endmodule)");
}

TEST_P(VastTest, VerilogFunctionWithScalarReturn) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  VerilogFunction* func = m->Add<VerilogFunction>(
      SourceInfo(), "func", f.Make<ScalarType>(SourceInfo()));
  func->AddStatement<BlockingAssignment>(SourceInfo(), func->return_value_ref(),
                                         f.PlainLiteral(1, SourceInfo()));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(1, SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), qux,
      f.Make<VerilogFunctionCall>(SourceInfo(), func,
                                  std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(nullptr),
            R"(module top;
  function automatic func ();
    begin
      func = 1;
    end
  endfunction
  wire qux;
  assign qux = func();
endmodule)");
}

TEST_P(VastTest, ConcurrentAssertionTest) {
  if (!UseSystemVerilog()) {
    return;
  }

  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* rst = m->AddInput("rst", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* a =
      m->AddInput("a", f.BitVectorType(8, SourceInfo()), SourceInfo());

  m->Add<ConcurrentAssertion>(
      SourceInfo(), f.Equals(a, f.Literal(0, 8, SourceInfo()), SourceInfo()),
      /*clocking_event=*/f.Make<PosEdge>(SourceInfo(), clk),
      /*disable_iff=*/std::nullopt,
      /*label=*/"",
      /*error_message=*/"");
  m->Add<ConcurrentAssertion>(
      SourceInfo(), f.Equals(a, f.Literal(0x9, 8, SourceInfo()), SourceInfo()),
      /*clocking_event=*/f.Make<PosEdge>(SourceInfo(), clk),
      /*disable_iff=*/rst,
      /*label=*/"my_label",
      /*error_message=*/"");
  m->Add<ConcurrentAssertion>(
      SourceInfo(), f.Equals(a, f.Literal(0x42, 8, SourceInfo()), SourceInfo()),
      /*clocking_event=*/f.Make<PosEdge>(SourceInfo(), clk),
      /*disable_iff=*/rst,
      /*label=*/"",
      /*error_message=*/"a does not equal 0x42");

  LineInfo line_info;
  EXPECT_EQ(m->Emit(&line_info),
            R"(module top(
  input wire clk,
  input wire rst,
  input wire [7:0] a
);
  assert property (@(posedge clk) a == 8'h00) else $fatal(0);
  my_label: assert property (@(posedge clk) disable iff (rst) a == 8'h09) else $fatal(0);
  assert property (@(posedge clk) disable iff (rst) a == 8'h42) else $fatal(0, "a does not equal 0x42");
endmodule)");
}

TEST_P(VastTest, VerilogFunctionWithComplicatedTypes) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  DataType* return_type =
      f.PackedArrayType(6, {3, 33}, SourceInfo(), /*is_signed=*/true);
  DataType* foo_type = f.BitVectorType(1, SourceInfo());
  DataType* bar_type = f.Make<PackedArrayType>(
      SourceInfo(),
      /*width=*/
      f.Add(f.PlainLiteral(6, SourceInfo()), f.PlainLiteral(6, SourceInfo()),
            SourceInfo()),
      /*packed_dims=*/
      std::vector<Expression*>({f.PlainLiteral(111, SourceInfo())}),
      /*is_signed=*/true);
  DataType* baz_type = f.BitVectorType(33, SourceInfo(), /*is_signed=*/true);

  VerilogFunction* func =
      m->Add<VerilogFunction>(SourceInfo(), "func", return_type);
  func->AddArgument("foo", foo_type, SourceInfo());
  func->AddArgument("bar", bar_type, SourceInfo());
  func->AddArgument("baz", baz_type, SourceInfo());
  VastNode* body = func->AddStatement<BlockingAssignment>(
      SourceInfo(), func->return_value_ref(), f.PlainLiteral(0, SourceInfo()));

  LogicRef* a = m->AddReg("a", foo_type, SourceInfo());
  LogicRef* b = m->AddWire("b", bar_type, SourceInfo());
  LogicRef* c = m->AddWire("c", baz_type, SourceInfo());
  LogicRef* qux = m->AddWire("qux", return_type, SourceInfo());
  m->Add<ContinuousAssignment>(
      SourceInfo(), qux,
      f.Make<VerilogFunctionCall>(SourceInfo(), func,
                                  std::vector<Expression*>{a, b, c}));
  LineInfo line_info;
  EXPECT_EQ(m->Emit(&line_info),
            R"(module top;
  function automatic signed [5:0][2:0][32:0] func (input reg foo, input reg signed [6 + 6 - 1:0][110:0] bar, input reg signed [32:0] baz);
    begin
      func = 0;
    end
  endfunction
  reg a;
  wire signed [6 + 6 - 1:0][110:0] b;
  wire signed [32:0] c;
  wire signed [5:0][2:0][32:0] qux;
  assign qux = func(a, b, c);
endmodule)");

  EXPECT_EQ(line_info.LookupNode(m).value(),
            std::vector<LineSpan>{LineSpan(0, 11)});
  EXPECT_EQ(line_info.LookupNode(func).value(),
            std::vector<LineSpan>{LineSpan(1, 5)});
  EXPECT_EQ(line_info.LookupNode(body).value(),
            std::vector<LineSpan>{LineSpan(3, 3)});
  EXPECT_EQ(line_info.LookupNode(a->def()).value(),
            std::vector<LineSpan>{LineSpan(6, 6)});
  EXPECT_EQ(line_info.LookupNode(b->def()).value(),
            std::vector<LineSpan>{LineSpan(7, 7)});
  EXPECT_EQ(line_info.LookupNode(c->def()).value(),
            std::vector<LineSpan>{LineSpan(8, 8)});
  EXPECT_EQ(line_info.LookupNode(qux->def()).value(),
            std::vector<LineSpan>{LineSpan(9, 9)});
}

TEST_P(VastTest, RegAndWireDefWithInit) {
  VerilogFile f(GetFileType());
  Module* m = f.AddModule("top", SourceInfo());
  DataType* return_type =
      f.PackedArrayType(6, {3, 33}, SourceInfo(), /*is_signed=*/true);
  DataType* foo_type = f.BitVectorType(1, SourceInfo());
  DataType* bar_type = f.Make<PackedArrayType>(
      SourceInfo(),
      /*width=*/
      f.Add(f.PlainLiteral(6, SourceInfo()), f.PlainLiteral(6, SourceInfo()),
            SourceInfo()),
      /*packed_dims=*/
      std::vector<Expression*>({f.PlainLiteral(111, SourceInfo())}),
      /*is_signed=*/true);
  DataType* baz_type = f.BitVectorType(33, SourceInfo(), /*is_signed=*/true);

  VerilogFunction* func =
      m->Add<VerilogFunction>(SourceInfo(), "func", return_type);
  func->AddArgument("foo", foo_type, SourceInfo());
  func->AddArgument("bar", bar_type, SourceInfo());
  func->AddArgument("baz", baz_type, SourceInfo());
  VastNode* body = func->AddStatement<BlockingAssignment>(
      SourceInfo(), func->return_value_ref(), f.PlainLiteral(0, SourceInfo()));

  LogicRef* a =
      m->AddReg("a", foo_type, SourceInfo(), f.Literal1(0, SourceInfo()));
  LogicRef* b =
      m->AddWire("b", bar_type, f.PlainLiteral(0, SourceInfo()), SourceInfo());
  LogicRef* c =
      m->AddWire("c", baz_type, f.Literal(0, 33, SourceInfo()), SourceInfo());

  VerilogFunctionCall* func_call = f.Make<VerilogFunctionCall>(
      SourceInfo(), func, std::vector<Expression*>{a, b, c});
  LogicRef* qux = m->AddWire("qux", return_type, func_call, SourceInfo());
  LineInfo line_info;
  EXPECT_EQ(m->Emit(&line_info),
            R"(module top;
  function automatic signed [5:0][2:0][32:0] func (input reg foo, input reg signed [6 + 6 - 1:0][110:0] bar, input reg signed [32:0] baz);
    begin
      func = 0;
    end
  endfunction
  reg a = 1'h0;
  wire signed [6 + 6 - 1:0][110:0] b = 0;
  wire signed [32:0] c = 33'h0_0000_0000;
  wire signed [5:0][2:0][32:0] qux = func(a, b, c);
endmodule)");

  EXPECT_EQ(line_info.LookupNode(m).value(),
            std::vector<LineSpan>{LineSpan(0, 10)});
  EXPECT_EQ(line_info.LookupNode(func).value(),
            std::vector<LineSpan>{LineSpan(1, 5)});
  EXPECT_EQ(line_info.LookupNode(body).value(),
            std::vector<LineSpan>{LineSpan(3, 3)});
  EXPECT_EQ(line_info.LookupNode(a->def()).value(),
            std::vector<LineSpan>{LineSpan(6, 6)});
  EXPECT_EQ(line_info.LookupNode(b->def()).value(),
            std::vector<LineSpan>{LineSpan(7, 7)});
  EXPECT_EQ(line_info.LookupNode(c->def()).value(),
            std::vector<LineSpan>{LineSpan(8, 8)});
  EXPECT_EQ(line_info.LookupNode(qux->def()).value(),
            std::vector<LineSpan>{LineSpan(9, 9)});
}

INSTANTIATE_TEST_SUITE_P(VastTestInstantiation, VastTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "SystemVerilog" : "Verilog";
                         });

}  // namespace
}  // namespace verilog
}  // namespace xls
