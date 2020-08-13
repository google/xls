// Copyright 2020 Google LLC
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

#include "xls/codegen/module_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

constexpr char kTestName[] = "module_builder_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

Value Make1DArray(int64 element_width, absl::Span<const int64> elements) {
  std::vector<Value> values;
  for (int64 element : elements) {
    values.push_back(Value(UBits(element, element_width)));
  }
  return Value::ArrayOrDie(values);
}

Value Make2DArray(int64 element_width,
                  absl::Span<const absl::Span<const int64>> elements) {
  std::vector<Value> rows;
  for (const auto& row : elements) {
    rows.push_back(Make1DArray(element_width, row));
  }
  return Value::ArrayOrDie(rows);
}

class ModuleBuilderTest : public VerilogTestBase {};

TEST_P(ModuleBuilderTest, AddTwoNumbers) {
  VerilogFile file;
  Package p(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x,
                           mb.AddInputPort("x", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y,
                           mb.AddInputPort("y", p.GetBitsType(32)));
  XLS_ASSERT_OK(mb.AddOutputPort("out", p.GetBitsType(32), file.Add(x, y)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, NewSections) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));

  LogicRef* a = mb.DeclareVariable("a", u32);
  XLS_ASSERT_OK(mb.Assign(a, file.Add(x, y), u32));

  mb.NewDeclarationAndAssignmentSections();
  LogicRef* b = mb.DeclareVariable("b", u32);
  XLS_ASSERT_OK(mb.Assign(b, file.Add(a, y), u32));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Negate(b)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, Registers) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* clk = mb.AddInputPort("clk", 1);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register a,
                           mb.DeclareRegister("a", u32, file.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register b,
                           mb.DeclareRegister("b", u32, y));
  XLS_ASSERT_OK(mb.AssignRegisters(clk, {a, b}));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithSynchronousReset) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* clk = mb.AddInputPort("clk", 1);
  LogicRef* rst = mb.AddInputPort("rst", 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister("a", u32, file.Add(x, y),
                         /*reset_value=*/file.Literal(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister("b", u32, y,
                         /*reset_value=*/file.Literal(UBits(0x42, 32))));
  XLS_ASSERT_OK(mb.AssignRegisters(clk, {a, b},
                                   /*load_enable=*/nullptr,
                                   Reset{.signal = rst->AsLogicRefNOrDie<1>(),
                                         .asynchronous = false,
                                         .active_low = false}));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithAsynchronousActiveLowReset) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* clk = mb.AddInputPort("clk", 1);
  LogicRef* rst = mb.AddInputPort("rst", 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister("a", u32, file.Add(x, y),
                         /*reset_value=*/file.Literal(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister("b", u32, y,
                         /*reset_value=*/file.Literal(UBits(0x42, 32))));
  XLS_ASSERT_OK(mb.AssignRegisters(clk, {a, b},
                                   /*load_enable=*/nullptr,
                                   Reset{.signal = rst->AsLogicRefNOrDie<1>(),
                                         .asynchronous = true,
                                         .active_low = true}));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithLoadEnable) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* clk = mb.AddInputPort("clk", 1);
  LogicRef* load_enable = mb.AddInputPort("le", 1);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register a,
                           mb.DeclareRegister("a", u32, file.Add(x, y)));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register b,
                           mb.DeclareRegister("b", u32, y));
  XLS_ASSERT_OK(mb.AssignRegisters(clk, {a, b}, load_enable));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithLoadEnableAndReset) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* clk = mb.AddInputPort("clk", 1);
  LogicRef* rst = mb.AddInputPort("rstn", 1);
  LogicRef* load_enable = mb.AddInputPort("le", 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister("a", u32, file.Add(x, y),
                         /*reset_value=*/file.Literal(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister("b", u32, y,
                         /*reset_value=*/file.Literal(UBits(0x42, 32))));
  XLS_ASSERT_OK(mb.AssignRegisters(clk, {a, b}, load_enable,
                                   Reset{.signal = rst->AsLogicRefNOrDie<1>(),
                                         .asynchronous = true,
                                         .active_low = true}));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ComplexComputation) {
  VerilogFile file;
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  Type* u16 = p.GetBitsType(16);
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  mb.declaration_section()->Add<Comment>("Declaration section.");
  mb.assignment_section()->Add<Comment>("Assignment section.");
  LogicRef* a = mb.DeclareVariable("a", u32);
  LogicRef* b = mb.DeclareVariable("b", u16);
  LogicRef* c = mb.DeclareVariable("c", u16);
  XLS_ASSERT_OK(mb.Assign(a, file.Shrl(x, y), u32));
  XLS_ASSERT_OK(mb.Assign(b, file.Slice(y, 16, 0), u16));
  XLS_ASSERT_OK(mb.Assign(c, file.Add(b, b), u16));
  XLS_ASSERT_OK(mb.AddOutputPort("out", u16, file.Add(b, c)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ReturnConstantArray) {
  VerilogFile file;
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  Value ar_value = Make2DArray(7, {{0x33, 0x12, 0x42}, {0x1, 0x2, 0x3}});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * ar,
                           mb.DeclareModuleConstant("ar", ar_value));
  XLS_ASSERT_OK(mb.AddOutputPort("out", package.GetTypeForValue(ar_value), ar));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, PassThroughArray) {
  VerilogFile file;
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  ArrayType* ar_type = package.GetArrayType(4, package.GetBitsType(13));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * a, mb.AddInputPort("a", ar_type));
  XLS_ASSERT_OK(mb.AddOutputPort("out", ar_type, a));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ReturnConstantTuple) {
  VerilogFile file;
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  Value tuple =
      Value::Tuple({Value(UBits(0x8, 8)), Make1DArray(24, {0x3, 0x6, 0x9}),
                    Value(UBits(0xab, 16))});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * t, mb.DeclareModuleConstant("t", tuple));
  XLS_ASSERT_OK(mb.AddOutputPort("out", package.GetTypeForValue(tuple), t));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, PassThroughTuple) {
  VerilogFile file;
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  TupleType* tuple_type = package.GetTupleType(
      {package.GetBitsType(42), package.GetArrayType(7, package.GetBitsType(6)),
       package.GetTupleType({})});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * a, mb.AddInputPort("a", tuple_type));
  XLS_ASSERT_OK(mb.AddOutputPort("out", tuple_type, a));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, SmulAsFunction) {
  VerilogFile file;
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x_smul_y = fb.SMul(fb.Param("x", u32), fb.Param("y", u32));
  BValue z_smul_z = fb.SMul(fb.Param("z", u32), fb.Param("z", u32));

  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("x_smul_y", x_smul_y.node(), {x, y}).status());
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("z_smul_z", z_smul_z.node(), {z, z}).status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, DynamicBitSliceAsFunction) {
  VerilogFile file;
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  Type* u16 = package.GetBitsType(16);
  BValue dyn_slice_x_y_5 =
      fb.DynamicBitSlice(fb.Param("x", u32), fb.Param("y", u32), 5);
  BValue dyn_slice_y_z_5 =
      fb.DynamicBitSlice(fb.Param("y", u32), fb.Param("z", u32), 5);
  BValue dyn_slice_w_z_10 =
      fb.DynamicBitSlice(fb.Param("w", u16), fb.Param("z", u32), 10);

  ModuleBuilder mb(TestBaseName(), &file,
                   /*use_system_verilog=*/UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * w, mb.AddInputPort("w", u16));

  XLS_ASSERT_OK(
      mb.EmitAsAssignment("dyn_slice_x_y_5", dyn_slice_x_y_5.node(), {x, y})
          .status());
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("dyn_slice_y_z_5", dyn_slice_y_z_5.node(), {y, z})
          .status());
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("dyn_slice_w_z_10", dyn_slice_w_z_10.node(), {w, z})
          .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}
INSTANTIATE_TEST_SUITE_P(ModuleBuilderTestInstantiation, ModuleBuilderTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ModuleBuilderTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
