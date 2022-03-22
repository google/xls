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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/number_parser.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class VastTest : public testing::TestWithParam<bool> {
 protected:
  bool UseSystemVerilog() const { return GetParam(); }
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
  VerilogFile f(UseSystemVerilog());

  LineInfo line_info;
  DataType* scalar = f.ScalarType(std::nullopt);
  EXPECT_EQ(scalar->EmitWithIdentifier(&line_info, "foo"), " foo");
  EXPECT_THAT(scalar->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(scalar->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(scalar->width(), nullptr);
  EXPECT_FALSE(scalar->is_signed());
  EXPECT_EQ(line_info.LookupNode(scalar),
            std::make_optional(std::vector<LineSpan>{LineSpan(0, 0)}));

  // A width 1 data type returned from BitVectorType should be a scalar.
  DataType* u1 = f.BitVectorType(1, std::nullopt);
  EXPECT_EQ(u1->EmitWithIdentifier(nullptr, "foo"), " foo");
  EXPECT_THAT(u1->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(u1->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(u1->width(), nullptr);
  EXPECT_FALSE(u1->is_signed());

  DataType* s1 = f.BitVectorType(1, std::nullopt, /*is_signed=*/true);
  EXPECT_EQ(s1->EmitWithIdentifier(nullptr, "foo"), " signed foo");
  EXPECT_THAT(s1->WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(s1->FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(s1->width(), nullptr);
  EXPECT_TRUE(s1->is_signed());

  DataType* u2 = f.BitVectorType(2, std::nullopt);
  EXPECT_EQ(u2->EmitWithIdentifier(nullptr, "foo"), " [1:0] foo");
  EXPECT_THAT(u2->WidthAsInt64(), IsOkAndHolds(2));
  EXPECT_THAT(u2->FlatBitCountAsInt64(), IsOkAndHolds(2));
  EXPECT_FALSE(u2->is_signed());

  DataType* u32 = f.BitVectorType(32, std::nullopt);
  EXPECT_EQ(u32->EmitWithIdentifier(nullptr, "foo"), " [31:0] foo");
  EXPECT_THAT(u32->WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(u32->FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_FALSE(u32->is_signed());

  DataType* s32 = f.BitVectorType(32, std::nullopt, /*is_signed=*/true);
  EXPECT_EQ(s32->EmitWithIdentifier(nullptr, "foo"), " signed [31:0] foo");
  EXPECT_THAT(s32->WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(s32->FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_TRUE(s32->is_signed());

  DataType* packed_array = f.PackedArrayType(10, {3, 2}, std::nullopt);
  EXPECT_EQ(packed_array->EmitWithIdentifier(nullptr, "foo"),
            " [9:0][2:0][1:0] foo");
  EXPECT_THAT(packed_array->WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(packed_array->FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_FALSE(packed_array->is_signed());

  DataType* spacked_array =
      f.PackedArrayType(10, {3, 2}, std::nullopt, /*is_signed=*/true);
  EXPECT_EQ(spacked_array->EmitWithIdentifier(nullptr, "foo"),
            " signed [9:0][2:0][1:0] foo");
  EXPECT_THAT(spacked_array->WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(spacked_array->FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_TRUE(spacked_array->is_signed());

  DataType* unpacked_array = f.UnpackedArrayType(10, {3, 2}, std::nullopt);
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
  DataType* bv =
      f.Make<DataType>(std::nullopt,
                       /*width=*/
                       f.Mul(f.PlainLiteral(10, std::nullopt),
                             f.PlainLiteral(5, std::nullopt), std::nullopt),
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
  VerilogFile f(UseSystemVerilog());
  Module* module = f.Make<Module>(std::nullopt, "my_module");
  LogicRef* a_ref =
      module->AddInput("a", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* b_ref =
      module->AddOutput("b", f.BitVectorType(4, std::nullopt), std::nullopt);
  LogicRef* array = module->AddInput(
      "array", f.PackedArrayType(8, {42, 3}, std::nullopt), std::nullopt);

  // Define a bunch of random regs.
  LogicRef* r1 =
      module->AddReg("r1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* r2 =
      module->AddReg("r2", f.BitVectorType(2, std::nullopt), std::nullopt);
  LogicRef* r1_init = module->AddReg(
      "r1_init",
      f.Make<DataType>(std::nullopt, f.PlainLiteral(1, std::nullopt),
                       /*is_signed=*/false),
      std::nullopt, f.PlainLiteral(1, std::nullopt));
  LogicRef* s =
      module->AddReg("s", f.BitVectorType(42, std::nullopt), std::nullopt);
  LogicRef* s_init =
      module->AddReg("s_init", f.BitVectorType(42, std::nullopt), std::nullopt,
                     f.Literal(123, 42, std::nullopt));
  LogicRef* t = module->AddReg(
      "t",
      f.Make<DataType>(
          std::nullopt,
          /*width=*/f.PlainLiteral(42, std::nullopt), /*packed_dims=*/
          std::vector<Expression*>(
              {f.PlainLiteral(8, std::nullopt),
               f.Add(f.PlainLiteral(8, std::nullopt),
                     f.PlainLiteral(42, std::nullopt), std::nullopt)}),
          /*unpacked_dims=*/std::vector<Expression*>(),
          /*is_signed=*/false),
      std::nullopt);
  LogicRef* signed_foo = module->AddReg(
      "signed_foo", f.BitVectorType(8, std::nullopt, /*is_signed=*/true),
      std::nullopt);

  // Define a bunch of random wires.
  LogicRef* x =
      module->AddWire("x", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* y =
      module->AddWire("y", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* z = module->AddWire(
      "z",
      f.Make<DataType>(
          std::nullopt,
          f.Mul(f.PlainLiteral(3, std::nullopt),
                f.PlainLiteral(3, std::nullopt), std::nullopt),
          /*packed_dims=*/
          std::vector<Expression*>(
              {f.PlainLiteral(8, std::nullopt),
               f.Add(f.PlainLiteral(8, std::nullopt),
                     f.PlainLiteral(42, std::nullopt), std::nullopt)}),
          /*unpacked_dims=*/std::vector<Expression*>(),
          /*is_signed=*/true),
      std::nullopt);

  VastNode* assign = module->Add<ContinuousAssignment>(
      std::nullopt, b_ref,
      f.Concat({a_ref, a_ref, a_ref, a_ref}, std::nullopt));
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
  assign b = {a, a, a, a};
endmodule)");

  EXPECT_EQ(line_info.LookupNode(module).value(),
            std::vector<LineSpan>{LineSpan(0, 16)});
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
  EXPECT_EQ(line_info.LookupNode(assign).value(),
            std::vector<LineSpan>{LineSpan(15, 15)});
}

TEST_P(VastTest, ModuleWithUnpackedArrayRegWithSize) {
  VerilogFile f(UseSystemVerilog());
  Module* module = f.Make<Module>(std::nullopt, "my_module");
  LogicRef* out_ref =
      module->AddOutput("out", f.BitVectorType(64, std::nullopt), std::nullopt);
  LogicRef* arr_ref = module->AddReg(
      "arr", f.UnpackedArrayType(4, {8, 64}, std::nullopt), std::nullopt);
  module->Add<ContinuousAssignment>(
      std::nullopt, out_ref,
      f.Index(f.Index(arr_ref, 2, std::nullopt), 1, std::nullopt));
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
  VerilogFile f(UseSystemVerilog());
  Module* module = f.Make<Module>(std::nullopt, "my_module");
  LogicRef* out_ref =
      module->AddOutput("out", f.BitVectorType(64, std::nullopt), std::nullopt);
  DataType* array_type = f.Make<DataType>(
      std::nullopt,
      /*width=*/f.PlainLiteral(4, std::nullopt),
      /*packed_dims=*/
      std::vector<Expression*>(
          {f.PlainLiteral(42, std::nullopt), f.PlainLiteral(7, std::nullopt)}),
      /*unpacked_dims=*/
      std::vector<Expression*>(
          {f.PlainLiteral(8, std::nullopt), f.PlainLiteral(64, std::nullopt)}),
      /*is_signed=*/false);
  LogicRef* arr_ref = module->AddReg("arr", array_type, std::nullopt);
  module->Add<ContinuousAssignment>(
      std::nullopt, out_ref,
      f.Index(f.Index(arr_ref, 2, std::nullopt), 1, std::nullopt));
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
  VerilogFile f(UseSystemVerilog());
  EXPECT_EQ("32'd44",
            f.Literal(UBits(44, 32), std::nullopt, FormatPreference::kDecimal)
                ->Emit(nullptr));
  EXPECT_EQ("1'b1",
            f.Literal(UBits(1, 1), std::nullopt, FormatPreference::kBinary)
                ->Emit(nullptr));
  EXPECT_EQ("4'b1010",
            f.Literal(UBits(10, 4), std::nullopt, FormatPreference::kBinary)
                ->Emit(nullptr));
  EXPECT_EQ("42'h000_0000_3039",
            f.Literal(UBits(12345, 42), std::nullopt, FormatPreference::kHex)
                ->Emit(nullptr));

  EXPECT_EQ("13579", f.PlainLiteral(13579, std::nullopt)->Emit(nullptr));

  Bits b0 = UBits(0, 1);
  Bits b2 = UBits(2, 3);
  Bits b55 = UBits(55, 32);
  Bits b1234 = UBits(1234, 55);
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber("0xabcd_000_0000_1234_4321_0000_aaaa_bbbb_cccc_dddd_eeee"));

  EXPECT_EQ(
      f.Literal(b0, std::nullopt, FormatPreference::kDecimal)->Emit(nullptr),
      "1'd0");

  EXPECT_EQ(f.Literal(b2, std::nullopt, FormatPreference::kHex)->Emit(nullptr),
            "3'h2");
  EXPECT_EQ(
      f.Literal(b2, std::nullopt, FormatPreference::kBinary)->Emit(nullptr),
      "3'b010");

  EXPECT_EQ(
      f.Literal(b55, std::nullopt, FormatPreference::kDefault)->Emit(nullptr),
      "55");
  EXPECT_EQ(
      f.Literal(b55, std::nullopt, FormatPreference::kBinary)->Emit(nullptr),
      "32'b0000_0000_0000_0000_0000_0000_0011_0111");
  EXPECT_TRUE(f.Literal(b55, std::nullopt)->IsLiteralWithValue(55));

  EXPECT_EQ(
      f.Literal(b1234, std::nullopt, FormatPreference::kHex)->Emit(nullptr),
      "55'h00_0000_0000_04d2");
  EXPECT_EQ(
      f.Literal(b1234, std::nullopt, FormatPreference::kDecimal)->Emit(nullptr),
      "55'd1234");
  EXPECT_EQ(
      f.Literal(b1234, std::nullopt, FormatPreference::kBinary)->Emit(nullptr),
      "55'b000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0100_"
      "1101_0010");
  EXPECT_TRUE(f.Literal(b1234, std::nullopt)->IsLiteralWithValue(1234));

  EXPECT_TRUE(f.Literal(huge, std::nullopt)->IsLiteral());
  EXPECT_FALSE(f.Literal(huge, std::nullopt)->IsLiteralWithValue(42));

  Literal* zero = f.Literal(UBits(0, 32), std::nullopt);
  EXPECT_TRUE(zero->IsLiteral());
  EXPECT_TRUE(zero->IsLiteralWithValue(0));
  EXPECT_FALSE(zero->IsLiteralWithValue(1));

  Literal* plain_zero = f.PlainLiteral(0, std::nullopt);
  EXPECT_TRUE(plain_zero->IsLiteral());
  EXPECT_TRUE(plain_zero->IsLiteralWithValue(0));
  EXPECT_FALSE(plain_zero->IsLiteralWithValue(1));

  Literal* forty_two = f.Literal(UBits(42, 32), std::nullopt);
  EXPECT_TRUE(forty_two->IsLiteral());
  EXPECT_FALSE(forty_two->IsLiteralWithValue(0));
  EXPECT_TRUE(forty_two->IsLiteralWithValue(42));

  Literal* wide_forty_two = f.Literal(UBits(42, 123456), std::nullopt);
  EXPECT_TRUE(wide_forty_two->IsLiteral());
  EXPECT_FALSE(wide_forty_two->IsLiteralWithValue(0));
  EXPECT_TRUE(wide_forty_two->IsLiteralWithValue(42));

  Literal* all_ones = f.Literal(UBits(15, 4), std::nullopt);
  EXPECT_TRUE(all_ones->IsLiteral());
  EXPECT_FALSE(all_ones->IsLiteralWithValue(0));
  EXPECT_TRUE(all_ones->IsLiteralWithValue(15));
  EXPECT_FALSE(all_ones->IsLiteralWithValue(-1));
}

TEST_P(VastTest, Precedence) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("precedence", std::nullopt);
  auto a = m->AddReg("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  auto b = m->AddReg("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  auto c = m->AddReg("c", f.BitVectorType(8, std::nullopt), std::nullopt);
  EXPECT_EQ("-a", f.Negate(a, std::nullopt)->Emit(nullptr));
  // Though technically not required by precedence the Verilog consumers are
  // happier if nested unary are wrapped in parens.
  EXPECT_EQ(
      "-(~a)",
      f.Negate(f.BitwiseNot(a, std::nullopt), std::nullopt)->Emit(nullptr));
  EXPECT_EQ(
      "-(a | b)",
      f.Negate(f.BitwiseOr(a, b, std::nullopt), std::nullopt)->Emit(nullptr));

  EXPECT_EQ("a + b", f.Add(a, b, std::nullopt)->Emit(nullptr));
  EXPECT_EQ("a + (b + c)",
            f.Add(a, f.Add(b, c, std::nullopt), std::nullopt)->Emit(nullptr));
  EXPECT_EQ("a + b + c",
            f.Add(f.Add(a, b, std::nullopt), c, std::nullopt)->Emit(nullptr));

  EXPECT_EQ("a + b * c",
            f.Add(a, f.Mul(b, c, std::nullopt), std::nullopt)->Emit(nullptr));
  EXPECT_EQ("a * (b + c)",
            f.Mul(a, f.Add(b, c, std::nullopt), std::nullopt)->Emit(nullptr));

  EXPECT_EQ(
      "a | (a + b || b)",
      f.BitwiseOr(a, f.LogicalOr(f.Add(a, b, std::nullopt), b, std::nullopt),
                  std::nullopt)
          ->Emit(nullptr));
  EXPECT_EQ("a * b << (b & c)",
            f.Shll(f.Mul(a, b, std::nullopt), f.BitwiseAnd(b, c, std::nullopt),
                   std::nullopt)
                ->Emit(nullptr));
}

TEST_P(VastTest, NestedUnaryOps) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("NestedUnaryOps", std::nullopt);
  auto a = m->AddReg("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  auto b = m->AddReg("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  EXPECT_EQ("-a", f.Negate(a, std::nullopt)->Emit(nullptr));
  EXPECT_EQ(
      "-(~a)",
      f.Negate(f.BitwiseNot(a, std::nullopt), std::nullopt)->Emit(nullptr));
  EXPECT_EQ(
      "-(-(-a))",
      f.Negate(f.Negate(f.Negate(a, std::nullopt), std::nullopt), std::nullopt)
          ->Emit(nullptr));
  EXPECT_EQ(
      "-(~(a + b)) - b",
      f.Sub(f.Negate(f.BitwiseNot(f.Add(a, b, std::nullopt), std::nullopt),
                     std::nullopt),
            b, std::nullopt)
          ->Emit(nullptr));
}

TEST_P(VastTest, Case) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* thing_next =
      m->AddWire("thing_next", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* thing =
      m->AddWire("thing", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* my_state =
      m->AddWire("my_state", f.BitVectorType(2, std::nullopt), std::nullopt);
  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  Case* case_statement = ac->statements()->Add<Case>(std::nullopt, my_state);
  StatementBlock* one_block =
      case_statement->AddCaseArm(f.Literal(1, 2, std::nullopt));
  one_block->Add<BlockingAssignment>(std::nullopt, thing_next, thing);
  StatementBlock* default_block = case_statement->AddCaseArm(DefaultSentinel());
  default_block->Add<BlockingAssignment>(std::nullopt, thing_next,
                                         f.Make<XSentinel>(std::nullopt, 2));

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

TEST_P(VastTest, AlwaysFlopTestNoReset) {
  std::vector<std::string> fsm_signals = {
      "rx_byte_done",
      "state",
      "tx_byte",
      "tx_byte_valid",
  };

  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, std::nullopt), std::nullopt);
  AlwaysFlop* af = f.Make<AlwaysFlop>(std::nullopt, clk);
  for (const std::string& signal_name : fsm_signals) {
    LogicRef* signal =
        m->AddReg(signal_name, f.BitVectorType(1, std::nullopt), std::nullopt);
    LogicRef* signal_next =
        m->AddReg(absl::StrCat(signal_name, "_next"),
                  f.BitVectorType(1, std::nullopt), std::nullopt);
    af->AddRegister(signal, signal_next, std::nullopt);
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* rst =
      m->AddInput("my_rst", f.BitVectorType(1, std::nullopt), std::nullopt);

  LogicRef* a = m->AddReg("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* a_next =
      m->AddReg("a_next", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b = m->AddReg("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b_next =
      m->AddReg("b_next", f.BitVectorType(8, std::nullopt), std::nullopt);

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      std::nullopt, clk, Reset{rst, /*async*/ false, /*active_low*/ false});
  af->AddRegister(a, a_next, std::nullopt,
                  /*reset_value=*/f.Literal(42, 8, std::nullopt));
  af->AddRegister(b, b_next, std::nullopt);

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* rst =
      m->AddInput("my_rst_n", f.BitVectorType(1, std::nullopt), std::nullopt);

  LogicRef* a = m->AddReg("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* a_next =
      m->AddReg("a_next", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b = m->AddReg("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b_next =
      m->AddReg("b_next", f.BitVectorType(8, std::nullopt), std::nullopt);

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      std::nullopt, clk, Reset{rst, /*async*/ true, /*active_low*/ true});
  af->AddRegister(a, a_next, std::nullopt,
                  /*reset_value=*/f.Literal(42, 8, std::nullopt));
  af->AddRegister(b, b_next, std::nullopt);

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* clk =
      m->AddInput("my_clk", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* rst_n =
      m->AddInput("rst_n", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* foo =
      m->AddInput("foo", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* bar =
      m->AddInput("bar", f.BitVectorType(1, std::nullopt), std::nullopt);

  LogicRef* foo_reg =
      m->AddReg("foo_reg", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* bar_reg =
      m->AddReg("bar_reg", f.BitVectorType(1, std::nullopt), std::nullopt);

  AlwaysFf* always_ff =
      f.Make<AlwaysFf>(std::nullopt, std::vector<SensitivityListElement>{
                                         f.Make<PosEdge>(std::nullopt, clk),
                                         f.Make<NegEdge>(std::nullopt, rst_n)});
  always_ff->statements()->Add<NonblockingAssignment>(std::nullopt, foo_reg,
                                                      foo);
  always_ff->statements()->Add<NonblockingAssignment>(std::nullopt, bar_reg,
                                                      bar);
  m->AddModuleMember(always_ff);

  EXPECT_EQ(always_ff->Emit(nullptr),
            R"(always_ff @ (posedge my_clk or negedge rst_n) begin
  foo_reg <= foo;
  bar_reg <= bar;
end)");
}

TEST_P(VastTest, Always) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* foo =
      m->AddReg("foo", f.BitVectorType(32, std::nullopt), std::nullopt);
  LogicRef* bar =
      m->AddReg("bar", f.BitVectorType(32, std::nullopt), std::nullopt);

  Always* always = f.Make<Always>(
      std::nullopt,
      std::vector<SensitivityListElement>{ImplicitEventExpression()});
  always->statements()->Add<BlockingAssignment>(
      std::nullopt, foo, f.PlainLiteral(7, std::nullopt));
  always->statements()->Add<BlockingAssignment>(
      std::nullopt, bar, f.PlainLiteral(42, std::nullopt));
  m->AddModuleMember(always);

  EXPECT_EQ(always->Emit(nullptr),
            R"(always @ (*) begin
  foo = 7;
  bar = 42;
end)");
}

TEST_P(VastTest, AlwaysCombTest) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* rx_byte_done = m->AddWire(
      "rx_byte_done", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* rx_byte_done_next = m->AddWire(
      "rx_byte_done_next", f.BitVectorType(1, std::nullopt), std::nullopt);

  LogicRef* tx_byte_next = m->AddWire(
      "tx_byte_next", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* tx_byte =
      m->AddWire("tx_byte", f.BitVectorType(8, std::nullopt), std::nullopt);

  LogicRef* tx_byte_valid_next = m->AddWire(
      "tx_byte_valid_next", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* tx_byte_valid = m->AddWire(
      "tx_byte_valid", f.BitVectorType(1, std::nullopt), std::nullopt);

  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  for (auto p : std::vector<std::pair<Expression*, Expression*>>{
           {rx_byte_done_next, rx_byte_done},
           {tx_byte_next, tx_byte},
           {tx_byte_valid_next, tx_byte_valid},
       }) {
    ac->statements()->Add<BlockingAssignment>(std::nullopt, p.first, p.second);
  }

  EXPECT_EQ(ac->Emit(nullptr),
            R"(always_comb begin
  rx_byte_done_next = rx_byte_done;
  tx_byte_next = tx_byte;
  tx_byte_valid_next = tx_byte_valid;
end)");
}

TEST_P(VastTest, InstantiationTest) {
  VerilogFile f(UseSystemVerilog());
  auto* default_clocks_per_baud =
      f.Make<MacroRef>(std::nullopt, "DEFAULT_CLOCKS_PER_BAUD");
  auto* clk_def =
      f.Make<WireDef>(std::nullopt, "my_clk", f.BitVectorType(1, std::nullopt));
  auto* clk_ref = f.Make<LogicRef>(std::nullopt, clk_def);

  auto* tx_byte_def = f.Make<WireDef>(std::nullopt, "my_tx_byte",
                                      f.BitVectorType(8, std::nullopt));
  auto* tx_byte_ref = f.Make<LogicRef>(std::nullopt, tx_byte_def);
  auto* instantiation =
      f.Make<Instantiation>(std::nullopt,
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

TEST_P(VastTest, BlockingAndNonblockingAssignments) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* a = m->AddReg("a", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* b = m->AddReg("b", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* c = m->AddReg("c", f.BitVectorType(1, std::nullopt), std::nullopt);
  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  ac->statements()->Add<BlockingAssignment>(std::nullopt, a, b);
  ac->statements()->Add<NonblockingAssignment>(std::nullopt, b, c);

  EXPECT_EQ(ac->Emit(nullptr),
            R"(always_comb begin
  a = b;
  b <= c;
end)");
}

TEST_P(VastTest, ParameterAndLocalParam) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  m->AddParameter("ClocksPerBaud",
                  f.Make<MacroRef>(std::nullopt, "DEFAULT_CLOCKS_PER_BAUD"),
                  std::nullopt);
  LocalParam* p = m->Add<LocalParam>(std::nullopt);
  LocalParamItemRef* idle =
      p->AddItem("StateIdle", f.Literal(0, 2, std::nullopt), std::nullopt);
  p->AddItem("StateGotByte", f.Literal(1, 2, std::nullopt), std::nullopt);
  p->AddItem("StateError", f.Literal(2, 2, std::nullopt), std::nullopt);

  LocalParamItemRef* state_bits =
      m->Add<LocalParam>(std::nullopt)
          ->AddItem("StateBits", f.Literal(2, 2, std::nullopt), std::nullopt);
  m->AddReg("state",
            f.Make<DataType>(std::nullopt, state_bits, /*is_signed=*/false),
            std::nullopt, idle);

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* input =
      m->AddInput("input", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* output =
      m->AddReg("output", f.BitVectorType(1, std::nullopt), std::nullopt);
  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  Conditional* if_statement =
      ac->statements()->Add<Conditional>(std::nullopt, input);
  if_statement->consequent()->Add<BlockingAssignment>(
      std::nullopt, output, f.Literal(1, 1, std::nullopt));
  EXPECT_EQ(if_statement->Emit(nullptr),
            R"(if (input) begin
  output = 1'h1;
end)");
}

TEST_P(VastTest, SignedOperation) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* a =
      m->AddInput("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b =
      m->AddInput("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* out =
      m->AddOutput("out", f.BitVectorType(8, std::nullopt), std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, out,
      f.Div(f.Make<SignedCast>(std::nullopt, a),
            f.Make<SignedCast>(std::nullopt, b), std::nullopt));
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* output1 =
      m->AddReg("output1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* output2 =
      m->AddReg("output2", f.BitVectorType(1, std::nullopt), std::nullopt);

  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  Conditional* conditional =
      ac->statements()->Add<Conditional>(std::nullopt, input1);
  conditional->consequent()->Add<BlockingAssignment>(
      std::nullopt, output1, f.Literal(1, 1, std::nullopt));

  StatementBlock* alternate1 =
      conditional->AddAlternate(f.BitwiseAnd(input1, input2, std::nullopt));
  alternate1->Add<BlockingAssignment>(std::nullopt, output1,
                                      f.Literal(0, 1, std::nullopt));

  StatementBlock* alternate2 = conditional->AddAlternate(input2);
  alternate2->Add<BlockingAssignment>(std::nullopt, output1,
                                      f.Literal(1, 1, std::nullopt));

  StatementBlock* alternate3 = conditional->AddAlternate();
  alternate3->Add<BlockingAssignment>(std::nullopt, output1,
                                      f.Literal(0, 1, std::nullopt));
  alternate3->Add<BlockingAssignment>(std::nullopt, output2,
                                      f.Literal(1, 1, std::nullopt));

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* output1 =
      m->AddReg("output1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* output2 =
      m->AddReg("output2", f.BitVectorType(1, std::nullopt), std::nullopt);

  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  Conditional* conditional =
      ac->statements()->Add<Conditional>(std::nullopt, input1);
  conditional->consequent()->Add<BlockingAssignment>(
      std::nullopt, output1, f.Literal(1, 1, std::nullopt));

  StatementBlock* alternate = conditional->AddAlternate();

  Conditional* nested_conditional =
      alternate->Add<Conditional>(std::nullopt, input2);
  nested_conditional->consequent()->Add<BlockingAssignment>(
      std::nullopt, output2, f.Literal(1, 1, std::nullopt));
  StatementBlock* nested_alternate = nested_conditional->AddAlternate();
  nested_alternate->Add<BlockingAssignment>(std::nullopt, output1,
                                            f.Literal(0, 1, std::nullopt));
  nested_alternate->Add<BlockingAssignment>(std::nullopt, output2,
                                            f.Literal(1, 1, std::nullopt));

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("testbench", std::nullopt);
  LogicRef* clk =
      m->AddReg("clk", f.BitVectorType(1, std::nullopt), std::nullopt);

  Initial* initial = m->Add<Initial>(std::nullopt);
  Statement* clk_equals_zero = f.Make<BlockingAssignment>(
      std::nullopt, clk, f.PlainLiteral(0, std::nullopt));
  initial->statements()->Add<DelayStatement>(
      std::nullopt, f.PlainLiteral(1, std::nullopt), clk_equals_zero);

  initial->statements()->Add<Forever>(
      std::nullopt,
      f.Make<DelayStatement>(
          std::nullopt, f.PlainLiteral(1, std::nullopt),
          f.Make<BlockingAssignment>(std::nullopt, clk,
                                     f.LogicalNot(clk, std::nullopt))));

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("testbench", std::nullopt);
  LogicRef* input1 =
      m->AddInput("input1", f.BitVectorType(1, std::nullopt), std::nullopt);
  LogicRef* input2 =
      m->AddInput("input2", f.BitVectorType(1, std::nullopt), std::nullopt);

  Initial* initial = m->Add<Initial>(std::nullopt);
  std::vector<Expression*> display_args = {
      f.Make<QuotedString>(std::nullopt, R"(foo\n)")};
  initial->statements()->Add<Display>(std::nullopt, display_args);
  initial->statements()->Add<DelayStatement>(std::nullopt,
                                             f.PlainLiteral(42, std::nullopt));
  std::vector<Expression*> monitor_args = {
      f.Make<QuotedString>(std::nullopt, R"(%t %d %d)"),
      f.Make<SystemFunctionCall>(std::nullopt, "time"), input1, input2};
  initial->statements()->Add<Monitor>(std::nullopt, monitor_args);
  initial->statements()->Add<DelayStatement>(
      std::nullopt, f.Add(f.PlainLiteral(123, std::nullopt),
                          f.PlainLiteral(456, std::nullopt), std::nullopt));
  initial->statements()->Add<WaitStatement>(std::nullopt, input1);
  initial->statements()->Add<Finish>(std::nullopt);

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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("Concat", std::nullopt);
  EXPECT_EQ(
      "{32'h0000_002a}",
      f.Concat({f.Literal(42, 32, std::nullopt)}, std::nullopt)->Emit(nullptr));
  EXPECT_EQ(
      "{a, 8'h7b, b}",
      f.Concat({m->AddReg("a", f.BitVectorType(1, std::nullopt), std::nullopt),
                f.Literal(123, 8, std::nullopt),
                m->AddReg("b", f.BitVectorType(1, std::nullopt), std::nullopt)},
               std::nullopt)
          ->Emit(nullptr));
  EXPECT_EQ(
      "{42{a, 8'h7b, b}}",
      f.Concat(/*replication=*/42,
               {m->AddReg("a", f.BitVectorType(1, std::nullopt), std::nullopt),
                f.Literal(123, 8, std::nullopt),
                m->AddReg("b", f.BitVectorType(1, std::nullopt), std::nullopt)},
               std::nullopt)
          ->Emit(nullptr));
}

TEST_P(VastTest, PartSelect) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("PartSelect", std::nullopt);
  EXPECT_EQ("a[4'h3 +: 16'h0006]",
            f.PartSelect(
                 m->AddReg("a", f.BitVectorType(8, std::nullopt), std::nullopt),
                 f.Literal(3, 4, std::nullopt), f.Literal(6, 16, std::nullopt),
                 std::nullopt)
                ->Emit(nullptr));
  EXPECT_EQ("b[c +: 16'h0012]",
            f.PartSelect(
                 m->AddReg("b", f.BitVectorType(8, std::nullopt), std::nullopt),
                 m->AddReg("c", f.BitVectorType(1, std::nullopt), std::nullopt),
                 f.Literal(18, 16, std::nullopt), std::nullopt)
                ->Emit(nullptr));
}

TEST_P(VastTest, ArrayAssignmentPattern) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("ArrayAssignmentPattern", std::nullopt);
  EXPECT_EQ(
      "'{32'h0000_002a}",
      f.ArrayAssignmentPattern({f.Literal(42, 32, std::nullopt)}, std::nullopt)
          ->Emit(nullptr));
  EXPECT_EQ(
      "'{a, 32'h0000_007b, b}",
      f.ArrayAssignmentPattern(
           {m->AddReg("a", f.BitVectorType(32, std::nullopt), std::nullopt),
            f.Literal(123, 32, std::nullopt),
            m->AddReg("b", f.BitVectorType(32, std::nullopt), std::nullopt)},
           std::nullopt)
          ->Emit(nullptr));
  EXPECT_EQ("'{'{foo, bar}, '{baz, qux}}",
            f.ArrayAssignmentPattern(
                 {f.ArrayAssignmentPattern(
                      {m->AddReg("foo", f.BitVectorType(32, std::nullopt),
                                 std::nullopt),
                       m->AddReg("bar", f.BitVectorType(32, std::nullopt),
                                 std::nullopt)},
                      std::nullopt),
                  f.ArrayAssignmentPattern(
                      {m->AddReg("baz", f.BitVectorType(32, std::nullopt),
                                 std::nullopt),
                       m->AddReg("qux", f.BitVectorType(32, std::nullopt),
                                 std::nullopt)},
                      std::nullopt)},
                 std::nullopt)
                ->Emit(nullptr));
}

TEST_P(VastTest, ModuleSections) {
  VerilogFile f(UseSystemVerilog());
  Module* module = f.Make<Module>(std::nullopt, "my_module");
  ModuleSection* s0 = module->Add<ModuleSection>(std::nullopt);
  module->AddReg("foo", f.BitVectorType(1, std::nullopt), std::nullopt,
                 f.PlainLiteral(1, std::nullopt));
  ModuleSection* s1 = module->Add<ModuleSection>(std::nullopt);
  // Create an empty section.
  module->Add<ModuleSection>(std::nullopt);

  // Fill the sections and add something to the module top as well.
  s0->Add<Comment>(std::nullopt, "section 0");
  s1->Add<Comment>(std::nullopt, "section 1");
  ModuleSection* nested_section = s1->Add<ModuleSection>(std::nullopt);
  nested_section->Add<Comment>(std::nullopt, "  nested in section 1");
  module->Add<Comment>(std::nullopt, "random comment at end");
  s1->Add<Comment>(std::nullopt, "more stuff in section 1");
  s0->Add<Comment>(std::nullopt, "more stuff in section 0");
  s0->Add<InlineVerilogStatement>(std::nullopt, "`SOME_MACRO(42);");
  module->AddReg("section_0_reg", f.BitVectorType(42, std::nullopt),
                 std::nullopt, /*init=*/nullptr,
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  VerilogFunction* func = m->Add<VerilogFunction>(
      std::nullopt, "func", f.BitVectorType(42, std::nullopt));
  LogicRef* foo =
      func->AddArgument("foo", f.BitVectorType(32, std::nullopt), std::nullopt);
  LogicRef* bar =
      func->AddArgument("bar", f.BitVectorType(3, std::nullopt), std::nullopt);
  func->AddStatement<BlockingAssignment>(std::nullopt, func->return_value_ref(),
                                         f.Shll(foo, bar, std::nullopt));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, std::nullopt), std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, qux,
      f.Make<VerilogFunctionCall>(
          std::nullopt, func,
          std::vector<Expression*>{f.Literal(UBits(12, 32), std::nullopt),
                                   f.Literal(UBits(2, 3), std::nullopt)}));
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  VerilogFunction* func = m->Add<VerilogFunction>(
      std::nullopt, "func", f.BitVectorType(42, std::nullopt));
  func->AddStatement<BlockingAssignment>(
      std::nullopt, func->return_value_ref(),
      f.Literal(UBits(0x42, 42), std::nullopt));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, std::nullopt), std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, qux,
      f.Make<VerilogFunctionCall>(std::nullopt, func,
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  VerilogFunction* func = m->Add<VerilogFunction>(
      std::nullopt, "func", f.BitVectorType(42, std::nullopt));
  LogicRef* foo =
      func->AddRegDef(std::nullopt, "foo", f.BitVectorType(42, std::nullopt));
  LogicRef* bar =
      func->AddRegDef(std::nullopt, "bar", f.BitVectorType(42, std::nullopt));
  func->AddStatement<BlockingAssignment>(
      std::nullopt, foo, f.Literal(UBits(0x42, 42), std::nullopt));
  func->AddStatement<BlockingAssignment>(std::nullopt, bar, foo);
  func->AddStatement<BlockingAssignment>(std::nullopt, func->return_value_ref(),
                                         bar);

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(32, std::nullopt), std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, qux,
      f.Make<VerilogFunctionCall>(std::nullopt, func,
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
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  VerilogFunction* func = m->Add<VerilogFunction>(
      std::nullopt, "func", f.Make<DataType>(std::nullopt));
  func->AddStatement<BlockingAssignment>(std::nullopt, func->return_value_ref(),
                                         f.PlainLiteral(1, std::nullopt));

  LogicRef* qux =
      m->AddWire("qux", f.BitVectorType(1, std::nullopt), std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, qux,
      f.Make<VerilogFunctionCall>(std::nullopt, func,
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

TEST_P(VastTest, AssertTest) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  LogicRef* a_ref =
      m->AddInput("a", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* b_ref =
      m->AddInput("b", f.BitVectorType(8, std::nullopt), std::nullopt);
  LogicRef* c_ref =
      m->AddOutput("c", f.BitVectorType(8, std::nullopt), std::nullopt);

  AlwaysComb* ac = m->Add<AlwaysComb>(std::nullopt);
  VastNode* assert1 = ac->statements()->Add<Assert>(
      std::nullopt,
      f.Equals(a_ref, f.Literal(42, 8, std::nullopt), std::nullopt));
  VastNode* assign = ac->statements()->Add<BlockingAssignment>(
      std::nullopt, c_ref, f.Add(a_ref, b_ref, std::nullopt));
  VastNode* assert2 = ac->statements()->Add<Assert>(
      std::nullopt,
      f.LessThan(c_ref, f.Literal(100, 8, std::nullopt), std::nullopt),
      "Oh noes! c is too big");

  LineInfo line_info;
  EXPECT_EQ(m->Emit(&line_info),
            R"(module top(
  input wire [7:0] a,
  input wire [7:0] b,
  output wire [7:0] c
);
  always_comb begin
    assert #0 (a == 8'h2a) else $fatal(0);
    c = a + b;
    assert #0 (c < 8'h64) else $fatal(0, "Oh noes! c is too big");
  end
endmodule)");

  EXPECT_EQ(line_info.LookupNode(ac).value(),
            std::vector<LineSpan>{LineSpan(5, 9)});
  EXPECT_EQ(line_info.LookupNode(assert1).value(),
            std::vector<LineSpan>{LineSpan(6, 6)});
  EXPECT_EQ(line_info.LookupNode(assign).value(),
            std::vector<LineSpan>{LineSpan(7, 7)});
  EXPECT_EQ(line_info.LookupNode(assert2).value(),
            std::vector<LineSpan>{LineSpan(8, 8)});
}

TEST_P(VastTest, VerilogFunctionWithComplicatedTypes) {
  VerilogFile f(UseSystemVerilog());
  Module* m = f.AddModule("top", std::nullopt);
  DataType* return_type =
      f.PackedArrayType(6, {3, 33}, std::nullopt, /*is_signed=*/true);
  DataType* foo_type = f.BitVectorType(1, std::nullopt);
  DataType* bar_type = f.Make<DataType>(
      std::nullopt,
      /*width=*/
      f.Add(f.PlainLiteral(6, std::nullopt), f.PlainLiteral(6, std::nullopt),
            std::nullopt),
      /*packed_dims=*/
      std::vector<Expression*>({f.PlainLiteral(111, std::nullopt)}),
      /*unpacked_dims=*/std::vector<Expression*>(),
      /*is_signed=*/true);
  DataType* baz_type = f.BitVectorType(33, std::nullopt, /*is_signed=*/true);

  VerilogFunction* func =
      m->Add<VerilogFunction>(std::nullopt, "func", return_type);
  func->AddArgument("foo", foo_type, std::nullopt);
  func->AddArgument("bar", bar_type, std::nullopt);
  func->AddArgument("baz", baz_type, std::nullopt);
  VastNode* body = func->AddStatement<BlockingAssignment>(
      std::nullopt, func->return_value_ref(), f.PlainLiteral(0, std::nullopt));

  LogicRef* a = m->AddReg("a", foo_type, std::nullopt);
  LogicRef* b = m->AddWire("b", bar_type, std::nullopt);
  LogicRef* c = m->AddWire("c", baz_type, std::nullopt);
  LogicRef* qux = m->AddWire("qux", return_type, std::nullopt);
  m->Add<ContinuousAssignment>(
      std::nullopt, qux,
      f.Make<VerilogFunctionCall>(std::nullopt, func,
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

INSTANTIATE_TEST_SUITE_P(VastTestInstantiation, VastTest,
                         testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "SystemVerilog" : "Verilog";
                         });

}  // namespace
}  // namespace verilog
}  // namespace xls
