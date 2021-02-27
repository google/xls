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

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(VastTest, SanitizeIdentifier) {
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

TEST(VastTest, DataTypes) {
  VerilogFile f;

  DataType scalar;
  EXPECT_EQ(scalar.Emit(), "");
  EXPECT_THAT(scalar.WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(scalar.FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(scalar.width(), nullptr);
  EXPECT_FALSE(scalar.is_signed());

  // A width 1 data type returned from DataTypeOfWidth should be a scalar.
  DataType width1 = f.DataTypeOfWidth(1);
  EXPECT_EQ(width1.Emit(), "");
  EXPECT_THAT(width1.WidthAsInt64(), IsOkAndHolds(1));
  EXPECT_THAT(width1.FlatBitCountAsInt64(), IsOkAndHolds(1));
  EXPECT_EQ(width1.width(), nullptr);
  EXPECT_FALSE(width1.is_signed());

  DataType u2 = f.DataTypeOfWidth(2);
  EXPECT_EQ(u2.Emit(), " [1:0]");
  EXPECT_THAT(u2.WidthAsInt64(), IsOkAndHolds(2));
  EXPECT_THAT(u2.FlatBitCountAsInt64(), IsOkAndHolds(2));
  EXPECT_FALSE(u2.is_signed());

  DataType u32 = DataType(f.PlainLiteral(32));
  EXPECT_EQ(u32.Emit(), " [31:0]");
  EXPECT_THAT(u32.WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(u32.FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_FALSE(u32.is_signed());

  DataType s32 = DataType(f.PlainLiteral(32), /*is_signed=*/true);
  EXPECT_EQ(s32.Emit(), " signed [31:0]");
  EXPECT_THAT(s32.WidthAsInt64(), IsOkAndHolds(32));
  EXPECT_THAT(s32.FlatBitCountAsInt64(), IsOkAndHolds(32));
  EXPECT_TRUE(s32.is_signed());

  DataType packed_array =
      DataType(/*width=*/f.PlainLiteral(10),
               /*packed_dims=*/{f.PlainLiteral(3), f.PlainLiteral(2)},
               /*is_signed=*/false);
  EXPECT_EQ(packed_array.Emit(), " [9:0][2:0][1:0]");
  EXPECT_THAT(packed_array.WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(packed_array.FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_FALSE(packed_array.is_signed());

  DataType spacked_array =
      DataType(/*width=*/f.PlainLiteral(10),
               /*packed_dims=*/{f.PlainLiteral(3), f.PlainLiteral(2)},
               /*is_signed=*/true);
  EXPECT_EQ(spacked_array.Emit(), " signed [9:0][2:0][1:0]");
  EXPECT_THAT(spacked_array.WidthAsInt64(), IsOkAndHolds(10));
  EXPECT_THAT(spacked_array.FlatBitCountAsInt64(), IsOkAndHolds(60));
  EXPECT_TRUE(spacked_array.is_signed());

  // Bit vector type with non-literal range.
  DataType bv =
      DataType(/*width=*/(f.Mul(f.PlainLiteral(10), f.PlainLiteral(5))));
  EXPECT_EQ(bv.Emit(), " [10 * 5 - 1:0]");
  EXPECT_THAT(bv.WidthAsInt64(), StatusIs(absl::StatusCode::kFailedPrecondition,
                                          HasSubstr("Width is not a literal")));
  EXPECT_THAT(bv.FlatBitCountAsInt64(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Width is not a literal")));
  EXPECT_FALSE(bv.is_signed());
}

TEST(VastTest, ModuleWithManyVariableDefinitions) {
  VerilogFile f;
  Module* module = f.Make<Module>("my_module", &f);
  LogicRef* a_ref = module->AddInput("a", f.DataTypeOfWidth(1));
  LogicRef* b_ref = module->AddOutput("b", f.DataTypeOfWidth(4));
  module->AddInput("array", DataType(f.PlainLiteral(8),
                                     /*packed_dims=*/{f.PlainLiteral(42),
                                                      f.PlainLiteral(3)}));

  // Define a bunch of random regs.
  module->AddReg("r1", f.DataTypeOfWidth(1));
  module->AddReg("r2", f.DataTypeOfWidth(2));
  module->AddReg("r1_init", DataType(f.PlainLiteral(1)), f.PlainLiteral(1));
  module->AddReg("s", DataType(f.PlainLiteral(42)));
  module->AddReg("s_init", DataType(f.PlainLiteral(42)), f.Literal(123, 42));
  module->AddReg("t", DataType(/*width=*/f.PlainLiteral(42), /*packed_dims=*/
                               {f.PlainLiteral(8),
                                f.Add(f.PlainLiteral(8), f.PlainLiteral(42))}));
  module->AddReg("signed_foo",
                 DataType(/*width=*/f.PlainLiteral(8), /*is_signed=*/true));

  // Define a bunch of random wires.
  module->AddWire("x", f.DataTypeOfWidth(1));
  module->AddWire("y", DataType(f.Literal(1, 16)));
  module->AddWire("z", DataType(f.Mul(f.PlainLiteral(3), f.PlainLiteral(3)),
                                /*packed_dims=*/
                                {f.PlainLiteral(8),
                                 f.Add(f.PlainLiteral(8), f.PlainLiteral(42))},
                                /*is_signed=*/true));

  module->Add<ContinuousAssignment>(b_ref,
                                    f.Concat({a_ref, a_ref, a_ref, a_ref}));
  EXPECT_EQ(module->Emit(),
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
  wire [0:0] y;
  wire signed [3 * 3 - 1:0][7:0][8 + 42 - 1:0] z;
  assign b = {a, a, a, a};
endmodule)");
}

TEST(VastTest, ModuleWithUnpackedArrayRegWithSize) {
  VerilogFile f;
  Module* module = f.Make<Module>("my_module", &f);
  LogicRef* out_ref = module->AddOutput("out", f.DataTypeOfWidth(64));
  LogicRef* arr_ref = module->AddUnpackedArrayReg(
      "arr", f.DataTypeOfWidth(4),
      /*array_bounds=*/{f.PlainLiteral(8), f.PlainLiteral(64)});
  module->Add<ContinuousAssignment>(out_ref, f.Index(f.Index(arr_ref, 2), 1));
  EXPECT_EQ(module->Emit(),
            R"(module my_module(
  output wire [63:0] out
);
  reg [3:0] arr[8][64];
  assign out = arr[2][1];
endmodule)");
}

TEST(VastTest, ModuleWithUnpackedArrayRegWithPackedDims) {
  VerilogFile f;
  Module* module = f.Make<Module>("my_module", &f);
  LogicRef* out_ref = module->AddOutput("out", f.DataTypeOfWidth(64));
  LogicRef* arr_ref = module->AddUnpackedArrayReg(
      "arr",
      DataType(f.PlainLiteral(4),
               /*packed_dims=*/{f.PlainLiteral(42), f.PlainLiteral(7)}),
      /*array_bounds=*/{f.PlainLiteral(8), f.PlainLiteral(64)});
  module->Add<ContinuousAssignment>(out_ref, f.Index(f.Index(arr_ref, 2), 1));
  EXPECT_EQ(module->Emit(),
            R"(module my_module(
  output wire [63:0] out
);
  reg [3:0][41:0][6:0] arr[8][64];
  assign out = arr[2][1];
endmodule)");
}

TEST(VastTest, ModuleWithUnpackedArrayRegWithRanges) {
  VerilogFile f;
  Module* module = f.Make<Module>("my_module", &f);
  LogicRef* out_ref = module->AddOutput("out", DataType(f.PlainLiteral(64)));
  LogicRef* arr_ref = module->AddUnpackedArrayReg(
      "arr", f.DataTypeOfWidth(4),
      /*array_bounds=*/
      {std::make_pair(f.PlainLiteral(0), f.PlainLiteral(7)),
       std::make_pair(f.PlainLiteral(0), f.PlainLiteral(63))});
  module->Add<ContinuousAssignment>(out_ref, f.Index(f.Index(arr_ref, 2), 1));
  EXPECT_EQ(module->Emit(),
            R"(module my_module(
  output wire [63:0] out
);
  reg [3:0] arr[0:7][0:63];
  assign out = arr[2][1];
endmodule)");
}

TEST(VastTest, Literals) {
  EXPECT_EQ("32'd44",
            Literal(UBits(44, 32), FormatPreference::kDecimal).Emit());
  EXPECT_EQ("1'b1",
            Literal(UBits(1, 1), FormatPreference::kBinary).Emit());
  EXPECT_EQ("4'b1010",
            Literal(UBits(10, 4), FormatPreference::kBinary).Emit());
  EXPECT_EQ("42'h000_0000_3039",
            Literal(UBits(12345, 42), FormatPreference::kHex).Emit());

  VerilogFile f;
  EXPECT_EQ("13579", f.PlainLiteral(13579)->Emit());

  Bits b0 = UBits(0, 1);
  Bits b2 = UBits(2, 3);
  Bits b55 = UBits(55, 32);
  Bits b1234 = UBits(1234, 55);

  EXPECT_EQ(Literal(b0, FormatPreference::kDecimal).Emit(), "1'd0");

  EXPECT_EQ(Literal(b2, FormatPreference::kHex).Emit(), "3'h2");
  EXPECT_EQ(Literal(b2, FormatPreference::kBinary).Emit(), "3'b010");

  EXPECT_EQ(Literal(b55, FormatPreference::kDefault).Emit(), "55");
  EXPECT_EQ(Literal(b55, FormatPreference::kBinary).Emit(),
            "32'b0000_0000_0000_0000_0000_0000_0011_0111");

  EXPECT_EQ(Literal(b1234, FormatPreference::kHex).Emit(),
            "55'h00_0000_0000_04d2");
  EXPECT_EQ(Literal(b1234, FormatPreference::kDecimal).Emit(), "55'd1234");
  EXPECT_EQ(Literal(b1234, FormatPreference::kBinary).Emit(),
            "55'b000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0100_"
            "1101_0010");
}

TEST(VastTest, Precedence) {
  VerilogFile f;
  Module* m = f.AddModule("precedence");
  auto a = m->AddReg("a", f.DataTypeOfWidth(8));
  auto b = m->AddReg("b", f.DataTypeOfWidth(8));
  auto c = m->AddReg("c", f.DataTypeOfWidth(8));
  EXPECT_EQ("-a", f.Negate(a)->Emit());
  // Though technically not required by precedence the Verilog consumers are
  // happier if nested unary are wrapped in parens.
  EXPECT_EQ("-(~a)", f.Negate(f.BitwiseNot(a))->Emit());
  EXPECT_EQ("-(a | b)", f.Negate(f.BitwiseOr(a, b))->Emit());

  EXPECT_EQ("a + b", f.Add(a, b)->Emit());
  EXPECT_EQ("a + (b + c)", f.Add(a, f.Add(b, c))->Emit());
  EXPECT_EQ("a + b + c", f.Add(f.Add(a, b), c)->Emit());

  EXPECT_EQ("a + b * c", f.Add(a, f.Mul(b, c))->Emit());
  EXPECT_EQ("a * (b + c)", f.Mul(a, f.Add(b, c))->Emit());

  EXPECT_EQ("a | (a + b || b)",
            f.BitwiseOr(a, f.LogicalOr(f.Add(a, b), b))->Emit());
  EXPECT_EQ("a * b << (b & c)",
            f.Shll(f.Mul(a, b), f.BitwiseAnd(b, c))->Emit());
}

TEST(VastTest, NestedUnaryOps) {
  VerilogFile f;
  Module* m = f.AddModule("NestedUnaryOps");
  auto a = m->AddReg("a", f.DataTypeOfWidth(8));
  auto b = m->AddReg("b", f.DataTypeOfWidth(8));
  EXPECT_EQ("-a", f.Negate(a)->Emit());
  EXPECT_EQ("-(~a)", f.Negate(f.BitwiseNot(a))->Emit());
  EXPECT_EQ("-(-(-a))", f.Negate(f.Negate(f.Negate(a)))->Emit());
  EXPECT_EQ("-(~(a + b)) - b",
            f.Sub(f.Negate(f.BitwiseNot(f.Add(a, b))), b)->Emit());
}

TEST(VastTest, Case) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* thing_next = m->AddWire("thing_next", f.DataTypeOfWidth(1));
  LogicRef* thing = m->AddWire("thing", f.DataTypeOfWidth(1));
  LogicRef* my_state = m->AddWire("my_state", f.DataTypeOfWidth(2));
  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  Case* case_statement = ac->statements()->Add<Case>(&f, my_state);
  StatementBlock* one_block = case_statement->AddCaseArm(f.Literal(1, 2));
  one_block->Add<BlockingAssignment>(thing_next, thing);
  StatementBlock* default_block = case_statement->AddCaseArm(DefaultSentinel());
  default_block->Add<BlockingAssignment>(thing_next, f.Make<XSentinel>(2));

  EXPECT_EQ(case_statement->Emit(),
            R"(case (my_state)
  2'h1: begin
    thing_next = thing;
  end
  default: begin
    thing_next = 2'dx;
  end
endcase)");
}

TEST(VastTest, AlwaysFlopTestNoReset) {
  std::vector<std::string> fsm_signals = {
      "rx_byte_done",
      "state",
      "tx_byte",
      "tx_byte_valid",
  };

  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* clk = m->AddInput("my_clk", f.DataTypeOfWidth(1));
  AlwaysFlop* af = f.Make<AlwaysFlop>(&f, clk);
  for (const std::string& signal_name : fsm_signals) {
    LogicRef* signal = m->AddReg(signal_name, f.DataTypeOfWidth(1));
    LogicRef* signal_next =
        m->AddReg(absl::StrCat(signal_name, "_next"), f.DataTypeOfWidth(1));
    af->AddRegister(signal, signal_next);
  }
  m->AddModuleMember(af);

  EXPECT_EQ(af->Emit(),
            R"(always @ (posedge my_clk) begin
  rx_byte_done <= rx_byte_done_next;
  state <= state_next;
  tx_byte <= tx_byte_next;
  tx_byte_valid <= tx_byte_valid_next;
end)");
}

TEST(VastTest, AlwaysFlopTestSyncReset) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* clk = m->AddInput("my_clk", f.DataTypeOfWidth(1));
  LogicRef* rst = m->AddInput("my_rst", f.DataTypeOfWidth(1));

  LogicRef* a = m->AddReg("a", f.DataTypeOfWidth(8));
  LogicRef* a_next = m->AddReg("a_next", f.DataTypeOfWidth(8));
  LogicRef* b = m->AddReg("b", f.DataTypeOfWidth(8));
  LogicRef* b_next = m->AddReg("b_next", f.DataTypeOfWidth(8));

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      &f, clk, Reset{rst, /*async*/ false, /*active_low*/ false});
  af->AddRegister(a, a_next, /*reset_value=*/f.Literal(42, 8));
  af->AddRegister(b, b_next);

  EXPECT_EQ(af->Emit(),
            R"(always @ (posedge my_clk) begin
  if (my_rst) begin
    a <= 8'h2a;
  end else begin
    a <= a_next;
    b <= b_next;
  end
end)");
}

TEST(VastTest, AlwaysFlopTestAsyncResetActiveLow) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* clk = m->AddInput("my_clk", f.DataTypeOfWidth(1));
  LogicRef* rst = m->AddInput("my_rst_n", f.DataTypeOfWidth(1));

  LogicRef* a = m->AddReg("a", f.DataTypeOfWidth(8));
  LogicRef* a_next = m->AddReg("a_next", f.DataTypeOfWidth(8));
  LogicRef* b = m->AddReg("b", f.DataTypeOfWidth(8));
  LogicRef* b_next = m->AddReg("b_next", f.DataTypeOfWidth(8));

  AlwaysFlop* af = m->Add<AlwaysFlop>(
      &f, clk, Reset{rst, /*async*/ true, /*active_low*/ true});
  af->AddRegister(a, a_next, /*reset_value=*/f.Literal(42, 8));
  af->AddRegister(b, b_next);

  EXPECT_EQ(af->Emit(),
            R"(always @ (posedge my_clk or negedge my_rst_n) begin
  if (!my_rst_n) begin
    a <= 8'h2a;
  end else begin
    a <= a_next;
    b <= b_next;
  end
end)");
}

TEST(VastTest, AlwaysFf) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* clk = m->AddInput("my_clk", f.DataTypeOfWidth(1));
  LogicRef* rst_n = m->AddInput("rst_n", f.DataTypeOfWidth(1));
  LogicRef* foo = m->AddInput("foo", f.DataTypeOfWidth(1));
  LogicRef* bar = m->AddInput("bar", f.DataTypeOfWidth(1));

  LogicRef* foo_reg = m->AddReg("foo_reg", f.DataTypeOfWidth(1));
  LogicRef* bar_reg = m->AddReg("bar_reg", f.DataTypeOfWidth(1));

  AlwaysFf* always_ff =
      f.Make<AlwaysFf>(&f, std::vector<SensitivityListElement>{
                               f.Make<PosEdge>(clk), f.Make<NegEdge>(rst_n)});
  always_ff->statements()->Add<NonblockingAssignment>(foo_reg, foo);
  always_ff->statements()->Add<NonblockingAssignment>(bar_reg, bar);
  m->AddModuleMember(always_ff);

  EXPECT_EQ(always_ff->Emit(),
            R"(always_ff @ (posedge my_clk or negedge rst_n) begin
  foo_reg <= foo;
  bar_reg <= bar;
end)");
}

TEST(VastTest, Always) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* foo = m->AddReg("foo", f.DataTypeOfWidth(32));
  LogicRef* bar = m->AddReg("bar", f.DataTypeOfWidth(32));

  Always* always = f.Make<Always>(
      &f, std::vector<SensitivityListElement>{ImplicitEventExpression()});
  always->statements()->Add<BlockingAssignment>(foo, f.PlainLiteral(7));
  always->statements()->Add<BlockingAssignment>(bar, f.PlainLiteral(42));
  m->AddModuleMember(always);

  EXPECT_EQ(always->Emit(),
            R"(always @ (*) begin
  foo = 7;
  bar = 42;
end)");
}

TEST(VastTest, AlwaysCombTest) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* rx_byte_done = m->AddWire("rx_byte_done", f.DataTypeOfWidth(1));
  LogicRef* rx_byte_done_next =
      m->AddWire("rx_byte_done_next", f.DataTypeOfWidth(1));

  LogicRef* tx_byte_next = m->AddWire("tx_byte_next", f.DataTypeOfWidth(8));
  LogicRef* tx_byte = m->AddWire("tx_byte", f.DataTypeOfWidth(8));

  LogicRef* tx_byte_valid_next =
      m->AddWire("tx_byte_valid_next", f.DataTypeOfWidth(1));
  LogicRef* tx_byte_valid = m->AddWire("tx_byte_valid", f.DataTypeOfWidth(1));

  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  for (auto p : std::vector<std::pair<Expression*, Expression*>>{
           {rx_byte_done_next, rx_byte_done},
           {tx_byte_next, tx_byte},
           {tx_byte_valid_next, tx_byte_valid},
       }) {
    ac->statements()->Add<BlockingAssignment>(p.first, p.second);
  }

  EXPECT_EQ(ac->Emit(),
            R"(always_comb begin
  rx_byte_done_next = rx_byte_done;
  tx_byte_next = tx_byte;
  tx_byte_valid_next = tx_byte_valid;
end)");
}

TEST(VastTest, InstantiationTest) {
  VerilogFile f;
  MacroRef default_clocks_per_baud("DEFAULT_CLOCKS_PER_BAUD");
  WireDef clk_def("my_clk", f.DataTypeOfWidth(1));
  LogicRef clk_ref(&clk_def);

  Literal* value_32d8 = f.PlainLiteral(8);
  WireDef tx_byte_def("my_tx_byte", DataType(value_32d8));
  LogicRef tx_byte_ref(&tx_byte_def);
  Instantiation instantiation(
      /*module_name=*/"uart_transmitter",
      /*instance_name=*/"tx",
      /*parameters=*/
      std::vector<Connection>{
          {"ClocksPerBaud", &default_clocks_per_baud},
      },
      /*connections=*/
      std::vector<Connection>{
          {"clk", &clk_ref},
          {"tx_byte", &tx_byte_ref},
      });

  EXPECT_EQ(instantiation.Emit(),
            R"(uart_transmitter #(
  .ClocksPerBaud(`DEFAULT_CLOCKS_PER_BAUD)
) tx (
  .clk(my_clk),
  .tx_byte(my_tx_byte)
);)");
}

TEST(VastTest, BlockingAndNonblockingAssignments) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* a = m->AddReg("a", f.DataTypeOfWidth(1));
  LogicRef* b = m->AddReg("b", f.DataTypeOfWidth(1));
  LogicRef* c = m->AddReg("c", f.DataTypeOfWidth(1));
  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  ac->statements()->Add<BlockingAssignment>(a, b);
  ac->statements()->Add<NonblockingAssignment>(b, c);

  EXPECT_EQ(ac->Emit(),
            R"(always_comb begin
  a = b;
  b <= c;
end)");
}

TEST(VastTest, ParameterAndLocalParam) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  m->AddParameter("ClocksPerBaud", f.Make<MacroRef>("DEFAULT_CLOCKS_PER_BAUD"));
  LocalParam* p = m->Add<LocalParam>(&f);
  LocalParamItemRef* idle = p->AddItem("StateIdle", f.Literal(0, 2));
  p->AddItem("StateGotByte", f.Literal(1, 2));
  p->AddItem("StateError", f.Literal(2, 2));

  LocalParamItemRef* state_bits =
      m->Add<LocalParam>(&f)->AddItem("StateBits", f.Literal(2, 2));
  m->AddReg("state", DataType(/*width=*/state_bits), idle);

  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, SimpleConditional) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* input = m->AddInput("input", f.DataTypeOfWidth(1));
  LogicRef* output = m->AddReg("output", f.DataTypeOfWidth(1));
  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  Conditional* if_statement = ac->statements()->Add<Conditional>(&f, input);
  if_statement->consequent()->Add<BlockingAssignment>(output, f.Literal(1, 1));
  EXPECT_EQ(if_statement->Emit(),
            R"(if (input) begin
  output = 1'h1;
end)");
}

TEST(VastTest, SignedOperation) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* a = m->AddInput("a", f.DataTypeOfWidth(8));
  LogicRef* b = m->AddInput("b", f.DataTypeOfWidth(8));
  LogicRef* out = m->AddOutput("out", f.DataTypeOfWidth(8));
  m->Add<ContinuousAssignment>(
      out, f.Div(f.Make<SignedCast>(a), f.Make<SignedCast>(b)));
  EXPECT_EQ(m->Emit(),
            R"(module top(
  input wire [7:0] a,
  input wire [7:0] b,
  output wire [7:0] out
);
  assign out = $signed(a) / $signed(b);
endmodule)");
}

TEST(VastTest, ComplexConditional) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* input1 = m->AddInput("input1", f.DataTypeOfWidth(1));
  LogicRef* input2 = m->AddInput("input2", f.DataTypeOfWidth(1));
  LogicRef* output1 = m->AddReg("output1", f.DataTypeOfWidth(1));
  LogicRef* output2 = m->AddReg("output2", f.DataTypeOfWidth(1));

  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  Conditional* conditional = ac->statements()->Add<Conditional>(&f, input1);
  conditional->consequent()->Add<BlockingAssignment>(output1, f.Literal(1, 1));

  StatementBlock* alternate1 =
      conditional->AddAlternate(f.BitwiseAnd(input1, input2));
  alternate1->Add<BlockingAssignment>(output1, f.Literal(0, 1));

  StatementBlock* alternate2 = conditional->AddAlternate(input2);
  alternate2->Add<BlockingAssignment>(output1, f.Literal(1, 1));

  StatementBlock* alternate3 = conditional->AddAlternate();
  alternate3->Add<BlockingAssignment>(output1, f.Literal(0, 1));
  alternate3->Add<BlockingAssignment>(output2, f.Literal(1, 1));

  EXPECT_EQ(conditional->Emit(),
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

TEST(VastTest, NestedConditional) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* input1 = m->AddInput("input1", f.DataTypeOfWidth(1));
  LogicRef* input2 = m->AddInput("input2", f.DataTypeOfWidth(1));
  LogicRef* output1 = m->AddReg("output1", f.DataTypeOfWidth(1));
  LogicRef* output2 = m->AddReg("output2", f.DataTypeOfWidth(1));

  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  Conditional* conditional = ac->statements()->Add<Conditional>(&f, input1);
  conditional->consequent()->Add<BlockingAssignment>(output1, f.Literal(1, 1));

  StatementBlock* alternate = conditional->AddAlternate();

  Conditional* nested_conditional = alternate->Add<Conditional>(&f, input2);
  nested_conditional->consequent()->Add<BlockingAssignment>(output2,
                                                            f.Literal(1, 1));
  StatementBlock* nested_alternate = nested_conditional->AddAlternate();
  nested_alternate->Add<BlockingAssignment>(output1, f.Literal(0, 1));
  nested_alternate->Add<BlockingAssignment>(output2, f.Literal(1, 1));

  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, TestbenchClock) {
  VerilogFile f;
  Module* m = f.AddModule("testbench");
  LogicRef* clk = m->AddReg("clk", f.DataTypeOfWidth(1));

  Initial* initial = m->Add<Initial>(&f);
  Statement* clk_equals_zero =
      f.Make<BlockingAssignment>(clk, f.PlainLiteral(0));
  initial->statements()->Add<DelayStatement>(f.PlainLiteral(1),
                                             clk_equals_zero);

  initial->statements()->Add<Forever>(f.Make<DelayStatement>(
      f.PlainLiteral(1), f.Make<BlockingAssignment>(clk, f.LogicalNot(clk))));

  EXPECT_EQ(f.Emit(),
            R"(module testbench;
  reg clk;
  initial begin
    #1 clk = 0;
    forever #1 clk = !clk;
  end
endmodule
)");
}

TEST(VastTest, TestbenchDisplayAndMonitor) {
  VerilogFile f;
  Module* m = f.AddModule("testbench");
  LogicRef* input1 = m->AddInput("input1", f.DataTypeOfWidth(1));
  LogicRef* input2 = m->AddInput("input2", f.DataTypeOfWidth(1));

  Initial* initial = m->Add<Initial>(&f);
  std::vector<Expression*> display_args = {f.Make<QuotedString>(R"(foo\n)")};
  initial->statements()->Add<Display>(display_args);
  initial->statements()->Add<DelayStatement>(f.PlainLiteral(42));
  std::vector<Expression*> monitor_args = {f.Make<QuotedString>(R"(%t %d %d)"),
                                           f.Make<SystemFunctionCall>("time"),
                                           input1, input2};
  initial->statements()->Add<Monitor>(monitor_args);
  initial->statements()->Add<DelayStatement>(
      f.Add(f.PlainLiteral(123), f.PlainLiteral(456)));
  initial->statements()->Add<WaitStatement>(input1);
  initial->statements()->Add<Finish>();

  EXPECT_EQ(f.Emit(),
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

TEST(VastTest, Concat) {
  VerilogFile f;
  Module* m = f.AddModule("Concat");
  EXPECT_EQ("{32'h0000_002a}", f.Concat({f.Literal(42, 32)})->Emit());
  EXPECT_EQ("{a, 8'h7b, b}",
            f.Concat({m->AddReg("a", f.DataTypeOfWidth(1)), f.Literal(123, 8),
                      m->AddReg("b", f.DataTypeOfWidth(1))})
                ->Emit());
  EXPECT_EQ("{42{a, 8'h7b, b}}",
            f.Concat(/*replication=*/42,
                     {m->AddReg("a", f.DataTypeOfWidth(1)), f.Literal(123, 8),
                      m->AddReg("b", f.DataTypeOfWidth(1))})
                ->Emit());
}

TEST(VastTest, PartSelect) {
  VerilogFile f;
  Module* m = f.AddModule("PartSelect");
  EXPECT_EQ("a[4'h3 +: 16'h0006]",
            f.PartSelect(m->AddReg("a", f.DataTypeOfWidth(8)), f.Literal(3, 4),
                         f.Literal(6, 16))
                ->Emit());
  EXPECT_EQ(
      "b[c +: 16'h0012]",
      f.PartSelect(m->AddReg("b", f.DataTypeOfWidth(8)),
                   m->AddReg("c", f.DataTypeOfWidth(1)), f.Literal(18, 16))
          ->Emit());
}

TEST(VastTest, ArrayAssignmentPattern) {
  VerilogFile f;
  Module* m = f.AddModule("ArrayAssignmentPattern");
  EXPECT_EQ("'{32'h0000_002a}",
            f.ArrayAssignmentPattern({f.Literal(42, 32)})->Emit());
  EXPECT_EQ("'{a, 32'h0000_007b, b}",
            f.ArrayAssignmentPattern({m->AddReg("a", f.DataTypeOfWidth(32)),
                                      f.Literal(123, 32),
                                      m->AddReg("b", f.DataTypeOfWidth(32))})
                ->Emit());
  EXPECT_EQ(
      "'{'{foo, bar}, '{baz, qux}}",
      f.ArrayAssignmentPattern(
           {f.ArrayAssignmentPattern({m->AddReg("foo", f.DataTypeOfWidth(32)),
                                      m->AddReg("bar", f.DataTypeOfWidth(32))}),
            f.ArrayAssignmentPattern(
                {m->AddReg("baz", f.DataTypeOfWidth(32)),
                 m->AddReg("qux", f.DataTypeOfWidth(32))})})
          ->Emit());
}

TEST(VastTest, ModuleSections) {
  VerilogFile f;
  Module* module = f.Make<Module>("my_module", &f);
  ModuleSection* s0 = module->Add<ModuleSection>(&f);
  module->AddReg("foo", f.DataTypeOfWidth(1), f.PlainLiteral(1));
  ModuleSection* s1 = module->Add<ModuleSection>(&f);
  // Create an empty section.
  module->Add<ModuleSection>(&f);

  // Fill the sections and add something to the module top as well.
  s0->Add<Comment>("section 0");
  s1->Add<Comment>("section 1");
  ModuleSection* nested_section = s1->Add<ModuleSection>(&f);
  nested_section->Add<Comment>("  nested in section 1");
  module->Add<Comment>("random comment at end");
  s1->Add<Comment>("more stuff in section 1");
  s0->Add<Comment>("more stuff in section 0");
  s0->Add<RawStatement>("`SOME_MACRO(42);");
  module->AddReg("section_0_reg", f.DataTypeOfWidth(42), /*init=*/nullptr,
                 /*section=*/s0);
  EXPECT_EQ(module->Emit(),
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
}

TEST(VastTest, VerilogFunction) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  VerilogFunction* func =
      m->Add<VerilogFunction>("func", f.DataTypeOfWidth(42), &f);
  LogicRef* foo = func->AddArgument("foo", f.DataTypeOfWidth(32));
  LogicRef* bar = func->AddArgument("bar", f.DataTypeOfWidth(3));
  func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                         f.Shll(foo, bar));

  LogicRef* qux = m->AddWire("qux", f.DataTypeOfWidth(32));
  m->Add<ContinuousAssignment>(
      qux, f.Make<VerilogFunctionCall>(
               func, std::vector<Expression*>{f.Literal(UBits(12, 32)),
                                              f.Literal(UBits(2, 3))}));
  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, VerilogFunctionNoArguments) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  VerilogFunction* func =
      m->Add<VerilogFunction>("func", f.DataTypeOfWidth(42), &f);
  func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                         f.Literal(UBits(0x42, 42)));

  LogicRef* qux = m->AddWire("qux", f.DataTypeOfWidth(32));
  m->Add<ContinuousAssignment>(
      qux, f.Make<VerilogFunctionCall>(func, std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, VerilogFunctionWithRegDefs) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  VerilogFunction* func =
      m->Add<VerilogFunction>("func", f.DataTypeOfWidth(42), &f);
  LogicRef* foo = func->AddRegDef("foo", f.DataTypeOfWidth(42));
  LogicRef* bar = func->AddRegDef("bar", f.DataTypeOfWidth(42));
  func->AddStatement<BlockingAssignment>(foo, f.Literal(UBits(0x42, 42)));
  func->AddStatement<BlockingAssignment>(bar, foo);
  func->AddStatement<BlockingAssignment>(func->return_value_ref(), bar);

  LogicRef* qux = m->AddWire("qux", f.DataTypeOfWidth(32));
  m->Add<ContinuousAssignment>(
      qux, f.Make<VerilogFunctionCall>(func, std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, VerilogFunctionWithScalarReturn) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  VerilogFunction* func = m->Add<VerilogFunction>("func", DataType(), &f);
  func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                         f.PlainLiteral(1));

  LogicRef* qux = m->AddWire("qux", f.DataTypeOfWidth(1));
  m->Add<ContinuousAssignment>(
      qux, f.Make<VerilogFunctionCall>(func, std::vector<Expression*>{}));
  EXPECT_EQ(m->Emit(),
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

TEST(VastTest, AssertTest) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  LogicRef* a_ref = m->AddInput("a", f.DataTypeOfWidth(8));
  LogicRef* b_ref = m->AddInput("b", f.DataTypeOfWidth(8));
  LogicRef* c_ref = m->AddOutput("c", f.DataTypeOfWidth(8));

  AlwaysComb* ac = m->Add<AlwaysComb>(&f);
  ac->statements()->Add<Assert>(f.Equals(a_ref, f.Literal(42, 8)));
  ac->statements()->Add<BlockingAssignment>(c_ref, f.Add(a_ref, b_ref));
  ac->statements()->Add<Assert>(f.LessThan(c_ref, f.Literal(100, 8)),
                                "Oh noes! c is too big");
  EXPECT_EQ(m->Emit(),
            R"(module top(
  input wire [7:0] a,
  input wire [7:0] b,
  output wire [7:0] c
);
  always_comb begin
    assert (a == 8'h2a) else $fatal(0);
    c = a + b;
    assert (c < 8'h64) else $fatal(0, "Oh noes! c is too big");
  end
endmodule)");
}

TEST(VastTest, VerilogFunctionWithComplicatedTypes) {
  VerilogFile f;
  Module* m = f.AddModule("top");
  DataType return_type =
      DataType(/*width=*/f.PlainLiteral(6),
               /*packed_dims=*/{f.PlainLiteral(3), f.PlainLiteral(33)},
               /*is_signed=*/true);
  DataType foo_type = f.DataTypeOfWidth(1);
  DataType bar_type =
      DataType(/*width=*/f.Add(f.PlainLiteral(6), f.PlainLiteral(6)),
               /*packed_dims=*/{f.PlainLiteral(111)},
               /*is_signed=*/true);
  DataType baz_type =
      DataType(/*width=*/f.PlainLiteral(33), /*is_signed=*/true);

  VerilogFunction* func = m->Add<VerilogFunction>("func", return_type, &f);
  func->AddArgument("foo", foo_type);
  func->AddArgument("bar", bar_type);
  func->AddArgument("baz", baz_type);
  func->AddStatement<BlockingAssignment>(func->return_value_ref(),
                                         f.PlainLiteral(0));

  LogicRef* a = m->AddReg("a", foo_type);
  LogicRef* b = m->AddWire("b", bar_type);
  LogicRef* c = m->AddWire("c", baz_type);
  LogicRef* qux = m->AddWire("qux", return_type);
  m->Add<ContinuousAssignment>(
      qux,
      f.Make<VerilogFunctionCall>(func, std::vector<Expression*>{a, b, c}));
  EXPECT_EQ(m->Emit(),
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
}

}  // namespace
}  // namespace verilog
}  // namespace xls
