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
#include "xls/netlist/interpreter.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/status/matchers.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/fake_cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist_parser.h"

namespace xls {
namespace netlist {
namespace {

// Smoke test to make sure anything works.
TEST(InterpreterTest, BasicFunctionality) {
  // Make a very simple A * B module.
  auto module = std::make_unique<rtl::Module>("the_module");
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "A"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "B"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kOutput, "O"));

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* entry,
                           library.GetEntry("AND"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef a_ref, module->ResolveNet("A"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef b_ref, module->ResolveNet("B"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef o_ref, module->ResolveNet("O"));

  absl::flat_hash_map<std::string, rtl::NetRef> params;
  params["A"] = a_ref;
  params["B"] = b_ref;
  params["Z"] = o_ref;

  XLS_ASSERT_OK_AND_ASSIGN(
      rtl::Cell tmp_cell,
      rtl::Cell::Create(entry, "the_cell", params, std::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto cell, module->AddCell(tmp_cell));
  a_ref->NoteConnectedCell(cell);
  b_ref->NoteConnectedCell(cell);
  o_ref->NoteConnectedCell(cell);

  rtl::Netlist netlist;
  netlist.AddModule(std::move(module));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module_ptr,
                           netlist.GetModule("the_module"));
  Interpreter interpreter(&netlist);

  NetRef2Value inputs, outputs;
  inputs[module_ptr->inputs()[0]] = true;
  inputs[module_ptr->inputs()[1]] = false;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module_ptr, inputs));
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module_ptr->outputs()[0]], 0);
}

// Verifies that a simple XOR(AND(), OR()) tree is interpreted correctly.
TEST(InterpreterTest, Tree) {
  // Make a very simple A * B module.
  auto module = std::make_unique<rtl::Module>("the_module");
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i0"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i1"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i2"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kInput, "i3"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kWire, "and_o"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kWire, "or_o"));
  XLS_ASSERT_OK(module->AddNetDecl(rtl::NetDeclKind::kOutput, "xor_o"));

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary library, MakeFakeCellLibrary());
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* and_entry,
                           library.GetEntry("AND"));
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* or_entry,
                           library.GetEntry("OR"));
  XLS_ASSERT_OK_AND_ASSIGN(const CellLibraryEntry* xor_entry,
                           library.GetEntry("XOR"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i0, module->ResolveNet("i0"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i1, module->ResolveNet("i1"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i2, module->ResolveNet("i2"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef i3, module->ResolveNet("i3"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef and_o, module->ResolveNet("and_o"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef or_o, module->ResolveNet("or_o"));
  XLS_ASSERT_OK_AND_ASSIGN(rtl::NetRef xor_o, module->ResolveNet("xor_o"));

  absl::flat_hash_map<std::string, rtl::NetRef> and_params;
  and_params["A"] = i0;
  and_params["B"] = i1;
  and_params["Z"] = and_o;

  XLS_ASSERT_OK_AND_ASSIGN(
      rtl::Cell tmp_cell,
      rtl::Cell::Create(and_entry, "and", and_params, std::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto and_cell, module->AddCell(tmp_cell));
  i0->NoteConnectedCell(and_cell);
  i1->NoteConnectedCell(and_cell);
  and_o->NoteConnectedCell(and_cell);

  absl::flat_hash_map<std::string, rtl::NetRef> or_params;
  or_params["A"] = i2;
  or_params["B"] = i3;
  or_params["Z"] = or_o;

  XLS_ASSERT_OK_AND_ASSIGN(
      tmp_cell,
      rtl::Cell::Create(or_entry, "or", or_params, std::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto or_cell, module->AddCell(tmp_cell));
  i2->NoteConnectedCell(or_cell);
  i3->NoteConnectedCell(or_cell);
  or_o->NoteConnectedCell(or_cell);

  absl::flat_hash_map<std::string, rtl::NetRef> xor_params;
  xor_params["A"] = and_o;
  xor_params["B"] = or_o;
  xor_params["Z"] = xor_o;
  XLS_ASSERT_OK_AND_ASSIGN(
      tmp_cell,
      rtl::Cell::Create(xor_entry, "xor", xor_params, std::nullopt, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(auto xor_cell, module->AddCell(tmp_cell));
  and_o->NoteConnectedCell(xor_cell);
  or_o->NoteConnectedCell(xor_cell);
  xor_o->NoteConnectedCell(xor_cell);

  rtl::Netlist netlist;
  netlist.AddModule(std::move(module));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module_ptr,
                           netlist.GetModule("the_module"));
  Interpreter interpreter(&netlist);

  NetRef2Value inputs, outputs;
  // AND inputs
  inputs[module_ptr->inputs()[0]] = true;
  inputs[module_ptr->inputs()[1]] = false;

  // OR inputs
  inputs[module_ptr->inputs()[2]] = true;
  inputs[module_ptr->inputs()[3]] = false;

  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module_ptr, inputs));

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module_ptr->outputs()[0]], 1);
}

TEST(InterpreterTest, Submodules) {
  std::string module_text = R"(
module submodule_0 (i2_0, i2_1, o2_0);
  input i2_0, i2_1;
  output o2_0;

  AND and0( .A(i2_0), .B(i2_1), .Z(o2_0) );
endmodule

module submodule_1 (i2_2, i2_3, o2_1);
  input i2_2, i2_3;
  output o2_1;

  OR or0( .A(i2_2), .B(i2_3), .Z(o2_1) );
endmodule

module submodule_2 (i1_0, i1_1, i1_2, i1_3, o1_0);
  input i1_0, i1_1, i1_2, i1_3;
  output o1_0;
  wire res0, res1;

  submodule_0 and0 ( .i2_0(i1_0), .i2_1(i1_1), .o2_0(res0) );
  submodule_1 or0 ( .i2_2(i1_2), .i2_3(i1_3), .o2_1(res1) );
  XOR xor0 ( .A(res0), .B(res1), .Z(o1_0) );
endmodule

module main (i0, i1, i2, i3, o0);
  input i0, i1, i2, i3;
  output o0;

  submodule_2 bleh( .i1_0(i0), .i1_1(i1), .i1_2(i2), .i1_3(i3), .o1_0(o0) );
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  NetRef2Value inputs, outputs;
  inputs[module->inputs()[0]] = true;
  inputs[module->inputs()[1]] = false;
  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = false;

  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));

  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[module->outputs()[0]], 1);
}

// Verifies that a [combinational] StateTable can be correctly interpreted in a
// design.
TEST(InterpreterTest, StateTables) {
  std::string module_text = R"(
module main(i0, i1, i2, i3, o0);
  input i0, i1, i2, i3;
  output o0;
  wire and0_out, and1_out;

  AND and0 ( .A(i0), .B(i1), .Z(and0_out) );
  STATETABLE_AND and1 (.A(i2), .B(i3), .Z(and1_out) );
  AND and2 ( .A(and0_out), .B(and1_out), .Z(o0) );
endmodule
  )";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  NetRef2Value inputs, outputs;
  inputs[module->inputs()[0]] = true;
  inputs[module->inputs()[1]] = true;
  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = true;

  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_TRUE(outputs.begin()->second);

  // Make sure that it works on the flip side, too.
  inputs[module->inputs()[2]] = false;
  inputs[module->inputs()[3]] = true;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);

  inputs[module->inputs()[2]] = true;
  inputs[module->inputs()[3]] = false;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);

  inputs[module->inputs()[2]] = false;
  inputs[module->inputs()[3]] = false;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));
  EXPECT_FALSE(outputs.begin()->second);
}

static std::string_view liberty_src = R"lib(
library(notandor) {
  cell(and2) {
    pin(A) {
      direction : input;
    }
    pin(B) {
      direction : input;
    }
    pin(Y) {
      direction: output;
      function : "(A * B)";
    }
  }
  cell(or2) {
    pin(A) {
      direction : input;
    }
    pin(B) {
      direction : input;
    }
    pin(Y) {
      direction: output;
      function : "(A + B)";
    }
  }
  cell(not1) {
    pin(A) {
      direction : input;
    }
    pin(Y) {
      direction : output;
      function : "A'";
    }
  }
}
)lib";

static std::string_view liberty_statetable_src = R"(
library (notandor) {
  cell (and2) {
    pin (A) {
      direction: input;
    }
    pin (B) {
      direction: input;
    }
    pin (Y) {
      direction: output;
      function: "X";
    }
    pin (X) {
      direction: internal;
      internal_node: "X";
    }
    statetable ("A B", "X") {
      table: "L - : - : L, \
              - L : - : L, \
              H H : - : H ";
    }
  }
  cell (or2) {
    pin (A) {
      direction: input;
    }
    pin (B) {
      direction: input;
    }
    pin (Y) {
      direction: output;
      function: "X";
    }
    pin (X) {
      direction: internal;
      internal_node: "X";
    }
    statetable ("A B", "X") {
      table: "H - : - : H, \
              - H : - : H, \
              L L : - : L ";
    }
  }
  cell (not1) {
    pin (A) {
      direction: input;
    }
    pin (Y) {
      direction: output;
      function: "X";
    }
    pin (X) {
      direction: internal;
      internal_node: "X";
    }
    statetable ("A", "X") {
      table: "H : - : L, \
              L : - : H ";
    }
  }
}
)";

static constexpr std::string_view netlist_src = R"(
module xor2(x, y, out);
  wire _0_;
  wire _1_;
  wire _2_;
  output out;
  input x;
  input y;
  or2 _3_ (
    .A(x),
    .B(y),
    .Y(_0_)
  );
  and2 _4_ (
    .A(x),
    .B(y),
    .Y(_1_)
  );
  not1 _5_ (
    .A(_1_),
    .Y(_2_)
  );
  and2 _6_ (
    .A(_0_),
    .B(_2_),
    .Y(out)
  );
endmodule
)";

template <typename ValueT>
static void TestXorUsing(
    const std::string& cell_definitions, std::function<bool(ValueT)> eval,
    const xls::netlist::rtl::CellToOutputEvalFns<ValueT>& eval_fns,
    const ValueT kFalse, const ValueT kTrue, size_t num_threads = 0) {
  rtl::Scanner scanner(netlist_src);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto stream,
      xls::netlist::cell_lib::CharStream::FromText(std::string(liberty_src)));
  XLS_ASSERT_OK_AND_ASSIGN(CellLibraryProto proto,
                           xls::netlist::function::ExtractFunctions(&stream));

  XLS_ASSERT_OK_AND_ASSIGN(
      xls::netlist::AbstractCellLibrary<ValueT> cell_library,
      xls::netlist::AbstractCellLibrary<ValueT>::FromProto(proto, kFalse,
                                                           kTrue));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<rtl::AbstractNetlist<ValueT>> n,
                           rtl::AbstractParser<ValueT>::ParseNetlist(
                               &cell_library, &scanner, kFalse, kTrue));
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::AbstractModule<ValueT>* m,
                           n->GetModule("xor2"));
  EXPECT_EQ("xor2", m->name());

  XLS_ASSERT_OK(n->AddCellEvaluationFns(eval_fns));

  xls::netlist::AbstractInterpreter<ValueT> interpreter(n.get(), kFalse, kTrue,
                                                        num_threads);

  auto test_xor = [&m, &interpreter, &eval](const ValueT& a, const ValueT& b,
                                            const ValueT& y) {
    AbstractNetRef2Value<ValueT> inputs, outputs;
    inputs.emplace(m->inputs()[0], a);
    inputs.emplace(m->inputs()[1], b);
    XLS_ASSERT_OK_AND_ASSIGN(outputs, interpreter.InterpretModule(m, inputs));
    EXPECT_EQ(outputs.size(), 1);
    EXPECT_EQ(eval(outputs.at(m->outputs()[0])), eval(y));
  };

  test_xor(kFalse, kFalse, kFalse);
  test_xor(kFalse, kTrue, kTrue);
  test_xor(kTrue, kFalse, kTrue);
  test_xor(kTrue, kTrue, kFalse);
}

TEST(NetlistParserTest, XorUsingStateTables) {
  TestXorUsing<bool>(
      std::string(liberty_statetable_src), [](bool x) -> bool { return x; }, {},
      false, true);
}

TEST(NetlistParserTest, XorUsingCellFunctions) {
  TestXorUsing<bool>(
      std::string(liberty_src), [](bool x) -> bool { return x; }, {}, false,
      true);
}

class OpaqueBoolValue {
 public:
  OpaqueBoolValue(const OpaqueBoolValue& rhs)  = default;
  OpaqueBoolValue& operator=(const OpaqueBoolValue& rhs) = default;
  OpaqueBoolValue(OpaqueBoolValue&& rhs) { value_ = rhs.value_; }
  OpaqueBoolValue& operator=(OpaqueBoolValue&& rhs) {
    value_ = rhs.value_;
    return *this;
  }
  OpaqueBoolValue operator&(const OpaqueBoolValue& rhs) const {
    return OpaqueBoolValue(value_ && rhs.value_);
  }
  OpaqueBoolValue operator|(const OpaqueBoolValue& rhs) const {
    return OpaqueBoolValue(value_ || rhs.value_);
  }
  OpaqueBoolValue operator^(const OpaqueBoolValue& rhs) const {
    int xor_result = value_ ^ rhs.value_;
    return OpaqueBoolValue(xor_result != 0);
  }
  OpaqueBoolValue operator!() const { return OpaqueBoolValue(!value_); }

  bool get() const { return value_; }
  static OpaqueBoolValue Create(bool val) { return OpaqueBoolValue(val); }

 private:
  explicit OpaqueBoolValue(bool val) : value_(val) {}
  bool value_;
};

TEST(NetlistParserTest, TestOpaqueBoolValue) {
  const OpaqueBoolValue kFalse = OpaqueBoolValue::Create(false);
  const OpaqueBoolValue kTrue = OpaqueBoolValue::Create(true);
  EXPECT_FALSE(kFalse.get());
  EXPECT_TRUE(kTrue.get());
  EXPECT_FALSE((kFalse ^ kFalse).get());
  EXPECT_TRUE((kFalse ^ kTrue).get());
  EXPECT_TRUE((kTrue ^ kFalse).get());
  EXPECT_FALSE((kTrue ^ kTrue).get());
}

TEST(NetlistParserTest, XorUsingStateTablesAndOpaqueValues) {
  const OpaqueBoolValue kFalse = OpaqueBoolValue::Create(false);
  const OpaqueBoolValue kTrue = OpaqueBoolValue::Create(true);
  TestXorUsing<OpaqueBoolValue>(
      std::string(liberty_statetable_src),
      [](auto x) -> bool { return x.get(); }, {}, kFalse, kTrue);
}

TEST(NetlistParserTest, XorUsingCellFunctionsAndOpaqueValues) {
  const OpaqueBoolValue kFalse = OpaqueBoolValue::Create(false);
  const OpaqueBoolValue kTrue = OpaqueBoolValue::Create(true);
  TestXorUsing<OpaqueBoolValue>(
      std::string(liberty_src), [](auto x) -> bool { return x.get(); }, {},
      kFalse, kTrue);
}

template <typename ValueT>
struct EvalOpCallCounter {
  std::atomic_size_t num_eval_calls = 0;
  int sleep_ms = 0;
#define IMPL1(cell, OP)                                                   \
  absl::StatusOr<ValueT> EvalOp_##cell(const std::vector<ValueT>& args) { \
    CHECK_EQ(args.size(), 1);                                             \
    absl::SleepFor(absl::Milliseconds(sleep_ms));                         \
    ValueT result = OP(args[0]);                                          \
    num_eval_calls++;                                                     \
    return ValueT{result};                                                \
  }

#define IMPL2(cell, OP)                                                   \
  absl::StatusOr<ValueT> EvalOp_##cell(const std::vector<ValueT>& args) { \
    CHECK_EQ(args.size(), 2);                                             \
    absl::SleepFor(absl::Milliseconds(sleep_ms));                         \
    ValueT result = OP(args[0], args[1]);                                 \
    num_eval_calls++;                                                     \
    return ValueT{result};                                                \
  }

  IMPL1(not1, [](auto a) { return !a; });
  IMPL2(and2, [](auto a, auto b) { return a & b; });
  IMPL2(or2, [](auto a, auto b) { return a | b; });

#undef IMPL1
#undef IMPL2
};

#define OP(ValueT, counter, name)                                            \
  {                                                                          \
#name, {                                                                 \
      {                                                                      \
        "Y",                                                                 \
            [&counter](                                                      \
                const std::vector<ValueT>& args) -> absl::StatusOr<ValueT> { \
              return counter.EvalOp_##name(args);                            \
            }                                                                \
      }                                                                      \
    }                                                                        \
  }

TEST(NetlistParserTest, XorUsingCellEvalFunctionsAndOpaqueValues) {
  const OpaqueBoolValue kFalse = OpaqueBoolValue::Create(false);
  const OpaqueBoolValue kTrue = OpaqueBoolValue::Create(true);

  EvalOpCallCounter<OpaqueBoolValue> call_counter;

  xls::netlist::rtl::CellToOutputEvalFns<OpaqueBoolValue> eval_map{
      OP(OpaqueBoolValue, call_counter, not1),
      OP(OpaqueBoolValue, call_counter, and2),
      OP(OpaqueBoolValue, call_counter, or2),
  };

  TestXorUsing<OpaqueBoolValue>(
      std::string(liberty_src), [](auto x) -> bool { return x.get(); },
      eval_map, kFalse, kTrue);

  // Evaluating the XOR circuit should trap into the eval calls four times for
  // each of the four rows of the XOR truth table.
  EXPECT_EQ(call_counter.num_eval_calls, 16);
}

TEST(NetlistParserTest, XorUsingPartialCellEvalFunctionsAndOpaqueValues) {
  const OpaqueBoolValue kFalse = OpaqueBoolValue::Create(false);
  const OpaqueBoolValue kTrue = OpaqueBoolValue::Create(true);

  EvalOpCallCounter<OpaqueBoolValue> call_counter;

  xls::netlist::rtl::CellToOutputEvalFns<OpaqueBoolValue> eval_map{
      OP(OpaqueBoolValue, call_counter, not1),
      OP(OpaqueBoolValue, call_counter, or2),
  };

  TestXorUsing<OpaqueBoolValue>(
      std::string(liberty_src), [](auto x) -> bool { return x.get(); },
      eval_map, kFalse, kTrue);

  // Evaluating the XOR circuit should trap into the eval calls twice times for
  // each of the four rows of the XOR truth table.  Twice, rather than four
  // times, because we took out the and2 eval fn.
  EXPECT_EQ(call_counter.num_eval_calls, 8);
}

TEST(NetlistParserTest, XorUsingCellEvalFunctions) {
  EvalOpCallCounter<bool> call_counter;
  xls::netlist::rtl::CellToOutputEvalFns<bool> eval_map{
      OP(bool, call_counter, not1),
      OP(bool, call_counter, and2),
      OP(bool, call_counter, or2),
  };

  TestXorUsing<bool>(
      std::string(liberty_src), [](auto x) -> bool { return x; }, eval_map,
      false, true);

  // Evaluating the XOR circuit should trap into the eval calls four times for
  // each of the four rows of the XOR truth table.
  EXPECT_EQ(call_counter.num_eval_calls, 16);
}

TEST(NetlistParserTest, XorUsingPartialCellEvalFunctions) {
  EvalOpCallCounter<bool> call_counter;
  xls::netlist::rtl::CellToOutputEvalFns<bool> eval_map{
      OP(bool, call_counter, not1),
      OP(bool, call_counter, or2),
  };

  TestXorUsing<bool>(
      std::string(liberty_src), [](auto x) -> bool { return x; }, eval_map,
      false, true);

  // Evaluating the XOR circuit should trap into the eval calls twice times for
  // each of the four rows of the XOR truth table.  Twice, rather than four
  // times, because we took out the and2 eval fn.
  EXPECT_EQ(call_counter.num_eval_calls, 8);
}

TEST(NetlistParserTest, XorUsingThreads) {
  EvalOpCallCounter<bool> call_counter;
  constexpr int DELAY_MS = 200;
  call_counter.sleep_ms = DELAY_MS;  // each op will sleep for DELAY_MS

  xls::netlist::rtl::CellToOutputEvalFns<bool> eval_map{
      OP(bool, call_counter, not1),
      OP(bool, call_counter, and2),
      OP(bool, call_counter, or2),
  };

  absl::Time start_time = absl::Now();
  // Give the intepreter four threads, one for each gate in the XOR evaluation.
  TestXorUsing<bool>(
      std::string(liberty_src), [](auto x) -> bool { return x; }, eval_map,
      false, true, 4);

  absl::Time end_time = absl::Now();
  auto duration = end_time - start_time;
  // In the best case, if all 4 gates for each of the truth table is handled by
  // one of the 4 threads, then the total runtime will be DELAY_MS seconds per
  // truth-table row times 4 rows, plus coordination overhead.
  EXPECT_GE(duration, absl::Milliseconds(4 * DELAY_MS));
  // In the worst case, all 4 gates will be processed by the main thread, with
  // total time being 4 delay units per truth-table row.  Giving the main thread
  // an extra second to complete its calculcation, we expect the total duration
  // to be less than 16 delay units + 1 second.
  EXPECT_LT(duration, absl::Milliseconds(16 * DELAY_MS + 1000));

  // Evaluating the XOR circuit should trap into the eval calls twice times for
  // each of the four rows of the XOR truth table.  Twice, rather than four
  // times, because we took out the and2 eval fn.
  EXPECT_EQ(call_counter.num_eval_calls, 16);
}

#undef OP

TEST(InterpreterTest, ComplexMixedInputAndWireAssigns) {
  std::string module_text = R"(
module main (A, B, out);
  input A;
  input B;
  wire [1:0] i0;
  wire [2:0] i1;
  wire [3:0] i2;
  wire [4:0] i3;
  output [15:0] out;
  wire [15:0] out;

  // i0 = 2'bAB;
  assign i0 = { A, B };
  // i1 = 3'b1AB
  assign i1 = { 1'b1, i0 };
  // 12'b1AB1AB1AB1AB -> 9'b1AB1AB1AB
  // i2 = 4'b1AB1
  // i3 = 5'bAB1AB
  assign { i2, i3 }  = { i1, i1, i1, i1 };
  // out = 9'bAB1AB1AB1. 7'b1001010
  //     = 16'bAB1A_B1AB_1100_1010
  assign out = { i3, i2, 7'h4a };
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  auto eval = [&module, &interpreter](bool A, bool B) {
    NetRef2Value inputs, outputs;
    inputs[module->inputs()[0]] = A;
    inputs[module->inputs()[1]] = B;
    XLS_ASSERT_OK_AND_ASSIGN(outputs,
                             interpreter.InterpretModule(module, inputs));

    EXPECT_EQ(outputs.size(), 16);
    EXPECT_EQ(outputs[module->outputs()[15]], A);
    EXPECT_EQ(outputs[module->outputs()[14]], B);
    EXPECT_EQ(outputs[module->outputs()[13]], 1);
    EXPECT_EQ(outputs[module->outputs()[12]], A);
    EXPECT_EQ(outputs[module->outputs()[11]], B);
    EXPECT_EQ(outputs[module->outputs()[10]], 1);
    EXPECT_EQ(outputs[module->outputs()[9]], A);
    EXPECT_EQ(outputs[module->outputs()[8]], B);
    EXPECT_EQ(outputs[module->outputs()[7]], 1);
    EXPECT_EQ(outputs[module->outputs()[6]], 1);
    EXPECT_EQ(outputs[module->outputs()[5]], 0);
    EXPECT_EQ(outputs[module->outputs()[4]], 0);
    EXPECT_EQ(outputs[module->outputs()[3]], 1);
    EXPECT_EQ(outputs[module->outputs()[2]], 0);
    EXPECT_EQ(outputs[module->outputs()[1]], 1);
    EXPECT_EQ(outputs[module->outputs()[0]], 0);
  };

  constexpr bool b0 = false;
  constexpr bool b1 = true;
  eval(b0, b0);
  eval(b0, b1);
  eval(b1, b0);
  eval(b1, b1);
}

TEST(InterpreterTest, ComplexWireAssigns) {
  std::string module_text = R"(
module main (out);
  wire [1:0] i0;
  wire [2:0] i1;
  wire [3:0] i2;
  wire [4:0] i3;
  output [15:0] out;
  wire [15:0] out;

  // i0 = 2'b10;
  assign i0 = 2'b10;
  // i1 = 3'b110
  assign i1 = { 1'b1, i0 };
  // 12'b110110110110 -> 9'b110110110
  // i2 = 4'b1101
  // i3 = 5'b
  assign { i2, i3 }  = { i1, i1, i1, i1 };
  // out = 9'b101101101 . 7'b1001010
  //     = 16'b1011_0110_1100_1010
  assign out = { i3, i2, 7'h4a };
endmodule
)";

  XLS_ASSERT_OK_AND_ASSIGN(CellLibrary cell_library, MakeFakeCellLibrary());
  rtl::Scanner scanner(module_text);
  XLS_ASSERT_OK_AND_ASSIGN(auto netlist,
                           rtl::Parser::ParseNetlist(&cell_library, &scanner));

  Interpreter interpreter(netlist.get());
  XLS_ASSERT_OK_AND_ASSIGN(const rtl::Module* module,
                           netlist->GetModule("main"));

  NetRef2Value inputs, outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs,
                           interpreter.InterpretModule(module, inputs));

  EXPECT_EQ(outputs.size(), 16);
  EXPECT_EQ(outputs[module->outputs()[15]], 1);
  EXPECT_EQ(outputs[module->outputs()[14]], 0);
  EXPECT_EQ(outputs[module->outputs()[13]], 1);
  EXPECT_EQ(outputs[module->outputs()[12]], 1);
  EXPECT_EQ(outputs[module->outputs()[11]], 0);
  EXPECT_EQ(outputs[module->outputs()[10]], 1);
  EXPECT_EQ(outputs[module->outputs()[9]], 1);
  EXPECT_EQ(outputs[module->outputs()[8]], 0);
  EXPECT_EQ(outputs[module->outputs()[7]], 1);
  EXPECT_EQ(outputs[module->outputs()[6]], 1);
  EXPECT_EQ(outputs[module->outputs()[5]], 0);
  EXPECT_EQ(outputs[module->outputs()[4]], 0);
  EXPECT_EQ(outputs[module->outputs()[3]], 1);
  EXPECT_EQ(outputs[module->outputs()[2]], 0);
  EXPECT_EQ(outputs[module->outputs()[1]], 1);
  EXPECT_EQ(outputs[module->outputs()[0]], 0);
}

}  // namespace
}  // namespace netlist
}  // namespace xls
