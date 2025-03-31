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

#include "xls/codegen/module_builder.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr char kTestName[] = "module_builder_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

Value Make1DArray(int64_t element_width, absl::Span<const int64_t> elements) {
  std::vector<Value> values;
  for (int64_t element : elements) {
    values.push_back(Value(UBits(element, element_width)));
  }
  return Value::ArrayOrDie(values);
}

Value Make2DArray(int64_t element_width,
                  absl::Span<const absl::Span<const int64_t>> elements) {
  std::vector<Value> rows;
  for (const auto& row : elements) {
    rows.push_back(Make1DArray(element_width, row));
  }
  return Value::ArrayOrDie(rows);
}

class ModuleBuilderTest : public VerilogTestBase {};

TEST_P(ModuleBuilderTest, AddTwoNumbers) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x,
                           mb.AddInputPort("x", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y,
                           mb.AddInputPort("y", p.GetBitsType(32)));
  XLS_ASSERT_OK(
      mb.AddOutputPort("out", p.GetBitsType(32), file.Add(x, y, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, NewSections) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));

  LogicRef* a = mb.DeclareVariable("a", u32);
  XLS_ASSERT_OK(mb.Assign(a, file.Add(x, y, SourceInfo()), u32));

  mb.NewDeclarationAndAssignmentSections();
  LogicRef* b = mb.DeclareVariable("b", u32);
  XLS_ASSERT_OK(mb.Assign(b, file.Add(a, y, SourceInfo()), u32));

  XLS_ASSERT_OK(mb.AddOutputPort("out", u32, file.Negate(b, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, Registers) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk");
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister("a", u32, file.Add(x, y, SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register b,
                           mb.DeclareRegister("b", u32, y));
  XLS_ASSERT_OK(mb.AssignRegisters({a, b}));

  XLS_ASSERT_OK(
      mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, DifferentResetPassedToAssignRegisters) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk", reset);
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register has_reset,
      mb.DeclareRegister("has_reset_reg", u32, x,
                         file.Literal(UBits(0, 32), SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register no_reset,
                           mb.DeclareRegister("no_reset_reg", u32, x));
  EXPECT_THAT(
      mb.AssignRegisters({has_reset, no_reset}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("All registers passed to AssignRegisters must either have "
                    "a reset or none have a reset. Registers no_reset_reg (no "
                    "reset) and has_reset_reg (has reset) differ.")));
}

TEST_P(ModuleBuilderTest, RegisterWithSynchronousReset) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(false);
  reset.set_active_low(false);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk", reset);
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister(
          "a", u32, file.Add(x, y, SourceInfo()),
          /*reset_value=*/file.Literal(UBits(0, 32), SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister(
          "b", u32, y,
          /*reset_value=*/file.Literal(UBits(0x42, 32), SourceInfo())));
  XLS_ASSERT_OK(mb.AssignRegisters({a, b}));

  XLS_ASSERT_OK(
      mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithAsynchronousActiveLowReset) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ResetProto reset;
  reset.set_name("rst");
  reset.set_asynchronous(true);
  reset.set_active_low(true);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk", reset);
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister(
          "a", u32, file.Add(x, y, SourceInfo()),
          /*reset_value=*/file.Literal(UBits(0, 32), SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister(
          "b", u32, y,
          /*reset_value=*/file.Literal(UBits(0x42, 32), SourceInfo())));
  XLS_ASSERT_OK(mb.AssignRegisters({a, b}));

  XLS_ASSERT_OK(
      mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithLoadEnable) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk");
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* load_enable = mb.AddInputPort("le", 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister("a", u32, file.Add(x, y, SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleBuilder::Register b,
                           mb.DeclareRegister("b", u32, y));
  a.load_enable = load_enable;
  b.load_enable = load_enable;
  XLS_ASSERT_OK(mb.AssignRegisters({a, b}));

  XLS_ASSERT_OK(
      mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, RegisterWithLoadEnableAndReset) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  ResetProto reset;
  reset.set_name("rstn");
  reset.set_asynchronous(true);
  reset.set_active_low(true);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options(),
                   /*clk_name=*/"clk", reset);
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  LogicRef* load_enable = mb.AddInputPort("le", 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register a,
      mb.DeclareRegister(
          "a", u32, file.Add(x, y, SourceInfo()),
          /*reset_value=*/file.Literal(UBits(0, 32), SourceInfo())));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleBuilder::Register b,
      mb.DeclareRegister(
          "b", u32, y,
          /*reset_value=*/file.Literal(UBits(0x42, 32), SourceInfo())));
  a.load_enable = load_enable;
  b.load_enable = load_enable;

  XLS_ASSERT_OK(mb.AssignRegisters({a, b}));

  XLS_ASSERT_OK(
      mb.AddOutputPort("out", u32, file.Add(a.ref, b.ref, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ComplexComputation) {
  VerilogFile file = NewVerilogFile();
  Package p(TestBaseName());
  Type* u32 = p.GetBitsType(32);
  Type* u16 = p.GetBitsType(16);
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  mb.declaration_section()->Add<Comment>(SourceInfo(), "Declaration section.");
  mb.assignment_section()->Add<Comment>(SourceInfo(), "Assignment section.");
  LogicRef* a = mb.DeclareVariable("a", u32);
  LogicRef* b = mb.DeclareVariable("b", u16);
  LogicRef* c = mb.DeclareVariable("c", u16);
  XLS_ASSERT_OK(mb.Assign(a, file.Shrl(x, y, SourceInfo()), u32));
  XLS_ASSERT_OK(mb.Assign(b, file.Slice(y, 16, 0, SourceInfo()), u16));
  XLS_ASSERT_OK(mb.Assign(c, file.Add(b, b, SourceInfo()), u16));
  XLS_ASSERT_OK(mb.AddOutputPort("out", u16, file.Add(b, c, SourceInfo())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ReturnConstantArray) {
  VerilogFile file = NewVerilogFile();
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  Value ar_value = Make2DArray(7, {{0x33, 0x12, 0x42}, {0x1, 0x2, 0x3}});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * ar,
                           mb.DeclareModuleConstant("ar", ar_value));
  XLS_ASSERT_OK(mb.AddOutputPort("out", package.GetTypeForValue(ar_value), ar));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, PassThroughArray) {
  VerilogFile file = NewVerilogFile();
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  ArrayType* ar_type = package.GetArrayType(4, package.GetBitsType(13));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * a, mb.AddInputPort("a", ar_type));
  XLS_ASSERT_OK(mb.AddOutputPort("out", ar_type, a));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ReturnConstantTuple) {
  VerilogFile file = NewVerilogFile();
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  Value tuple =
      Value::Tuple({Value(UBits(0x8, 8)), Make1DArray(24, {0x3, 0x6, 0x9}),
                    Value(UBits(0xab, 16))});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * t, mb.DeclareModuleConstant("t", tuple));
  XLS_ASSERT_OK(mb.AddOutputPort("out", package.GetTypeForValue(tuple), t));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, PassThroughTuple) {
  VerilogFile file = NewVerilogFile();
  // The XLS IR package is just used for type management.
  Package package(TestBaseName());
  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  TupleType* tuple_type = package.GetTupleType(
      {package.GetBitsType(42), package.GetArrayType(7, package.GetBitsType(6)),
       package.GetTupleType({})});
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * a, mb.AddInputPort("a", tuple_type));
  XLS_ASSERT_OK(mb.AddOutputPort("out", tuple_type, a));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, SmulAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x_param = fb.Param("x", u32);
  BValue y_param = fb.Param("y", u32);
  BValue x_smul_y = fb.SMul(x_param, y_param);
  BValue z_param = fb.Param("z", u32);
  BValue z_smul_z = fb.SMul(z_param, z_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
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

TEST_P(ModuleBuilderTest, SmulpAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x_param = fb.Param("x", u32);
  BValue y_param = fb.Param("y", u32);
  BValue x_smulp_y = fb.SMulp(x_param, y_param);
  BValue z_param = fb.Param("z", u32);
  BValue z_smulp_z = fb.SMulp(z_param, z_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("x_smulp_y", x_smulp_y.node(), {x, y}).status());
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("z_smulp_z", z_smulp_z.node(), {z, z}).status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, UmulpAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x_param = fb.Param("x", u32);
  BValue y_param = fb.Param("y", u32);
  BValue x_umulp_y = fb.UMulp(x_param, y_param);
  BValue z_param = fb.Param("z", u32);
  BValue z_umulp_z = fb.UMulp(z_param, z_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("x_umulp_y", x_umulp_y.node(), {x, y}).status());
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("z_umulp_z", z_umulp_z.node(), {z, z}).status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, DynamicBitSliceAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  Type* u16 = package.GetBitsType(16);
  BValue y_param = fb.Param("y", u32);
  BValue z_param = fb.Param("z", u32);
  BValue dyn_slice_x_y_5 = fb.DynamicBitSlice(fb.Param("x", u32), y_param, 5);
  BValue dyn_slice_y_z_5 = fb.DynamicBitSlice(y_param, z_param, 5);
  BValue dyn_slice_w_z_10 = fb.DynamicBitSlice(fb.Param("w", u16), z_param, 10);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
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

TEST_P(ModuleBuilderTest, BitSliceUpdateAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  Type* u16 = package.GetBitsType(16);
  Type* u8 = package.GetBitsType(8);

  BValue x_param = fb.Param("x", u32);
  BValue start_param = fb.Param("start", u16);
  BValue value_param = fb.Param("value", u8);
  BValue update = fb.BitSliceUpdate(x_param, start_param, value_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * start, mb.AddInputPort("start", u16));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * value, mb.AddInputPort("value", u8));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("slice_update", update.node(), {x, start, value})
          .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ArrayIndexAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  Type* big_array = package.GetArrayType(
      6, package.GetArrayType(16, package.GetArrayType(4, u32)));
  BValue x_param = fb.Param("x", big_array);
  BValue y_param = fb.Param("y", u32);
  BValue z_param = fb.Param("z", u32);
  BValue array_index_x_y_z = fb.ArrayIndex(x_param, {y_param, z_param});
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));

  XLS_ASSERT_OK(mb.EmitAsAssignment("array_index_x_y_z",
                                    array_index_x_y_z.node(), {x, y, z})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, PrioritySelectAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u3 = package.GetBitsType(3);
  Type* u32 = package.GetBitsType(32);
  BValue s_param = fb.Param("s", u3);
  BValue x_param = fb.Param("x", u32);
  BValue y_param = fb.Param("y", u32);
  BValue z_param = fb.Param("z", u32);
  BValue d_param = fb.Param("d", u32);
  BValue priority_select =
      fb.PrioritySelect(s_param, {x_param, y_param, z_param}, d_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s, mb.AddInputPort("s", u3));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * z, mb.AddInputPort("z", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d, mb.AddInputPort("d", u32));

  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select", priority_select.node(),
                                    {s, x, y, z, d})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, TwoWayPrioritySelectAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u1 = package.GetBitsType(1);
  Type* u32 = package.GetBitsType(32);
  BValue s_param = fb.Param("s", u1);
  BValue x_param = fb.Param("x", u32);
  BValue d_param = fb.Param("d", u32);
  BValue priority_select = fb.PrioritySelect(s_param, {x_param}, d_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s, mb.AddInputPort("s", u1));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d, mb.AddInputPort("d", u32));

  XLS_ASSERT_OK(
      mb.EmitAsAssignment("priority_select", priority_select.node(), {s, x, d})
          .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, SimilarPrioritySelects) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u2 = package.GetBitsType(2);
  Type* u32 = package.GetBitsType(32);
  BValue s1_param = fb.Param("s1", u2);
  BValue s2_param = fb.Param("s2", u2);
  BValue x1_param = fb.Param("x1", u32);
  BValue x2_param = fb.Param("x2", u32);
  BValue y1_param = fb.Param("y1", u32);
  BValue y2_param = fb.Param("y2", u32);
  BValue d1_param = fb.Param("d1", u32);
  BValue d2_param = fb.Param("d2", u32);
  BValue priority_select1 =
      fb.PrioritySelect(s1_param, {x1_param, y1_param}, d1_param);
  BValue priority_select2 =
      fb.PrioritySelect(s2_param, {x2_param, y2_param}, d2_param);
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({priority_select1, priority_select2})));

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s1, mb.AddInputPort("s1", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s2, mb.AddInputPort("s2", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x1, mb.AddInputPort("x1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x2, mb.AddInputPort("x2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y1, mb.AddInputPort("y1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y2, mb.AddInputPort("y2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d1, mb.AddInputPort("d1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d2, mb.AddInputPort("d2", u32));

  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select1", priority_select1.node(),
                                    {s1, x1, y1, d1})
                    .status());
  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select2", priority_select2.node(),
                                    {s2, x2, y2, d2})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, DifferentPrioritySelects) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u1 = package.GetBitsType(1);
  Type* u2 = package.GetBitsType(2);
  Type* u32 = package.GetBitsType(32);
  BValue s1_param = fb.Param("s1", u2);
  BValue x1_param = fb.Param("x1", u32);
  BValue y1_param = fb.Param("y1", u32);
  BValue d1_param = fb.Param("d1", u32);
  BValue s2_param = fb.Param("s2", u1);
  BValue x2_param = fb.Param("x2", u32);
  BValue d2_param = fb.Param("d2", u32);
  BValue priority_select1 =
      fb.PrioritySelect(s1_param, {x1_param, y1_param}, d1_param);
  BValue priority_select2 = fb.PrioritySelect(s2_param, {x2_param}, d2_param);
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({priority_select1, priority_select2})));

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s1, mb.AddInputPort("s1", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x1, mb.AddInputPort("x1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y1, mb.AddInputPort("y1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d1, mb.AddInputPort("d1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s2, mb.AddInputPort("s2", u1));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x2, mb.AddInputPort("x2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d2, mb.AddInputPort("d2", u32));

  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select1", priority_select1.node(),
                                    {s1, x1, y1, d1})
                    .status());
  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select2", priority_select2.node(),
                                    {s2, x2, d2})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, SimilarArrayPrioritySelects) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u2 = package.GetBitsType(2);
  Type* u32 = package.GetBitsType(32);
  Type* big_array = package.GetArrayType(
      6, package.GetArrayType(16, package.GetArrayType(4, u32)));
  BValue s1_param = fb.Param("s1", u2);
  BValue s2_param = fb.Param("s2", u2);
  BValue x1_param = fb.Param("x1", big_array);
  BValue x2_param = fb.Param("x2", big_array);
  BValue y1_param = fb.Param("y1", big_array);
  BValue y2_param = fb.Param("y2", big_array);
  BValue d1_param = fb.Param("d1", big_array);
  BValue d2_param = fb.Param("d2", big_array);
  BValue priority_select1 =
      fb.PrioritySelect(s1_param, {x1_param, y1_param}, d1_param);
  BValue priority_select2 =
      fb.PrioritySelect(s2_param, {x2_param, y2_param}, d2_param);
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({priority_select1, priority_select2})));

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s1, mb.AddInputPort("s1", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s2, mb.AddInputPort("s2", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x1, mb.AddInputPort("x1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x2, mb.AddInputPort("x2", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y1, mb.AddInputPort("y1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y2, mb.AddInputPort("y2", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d1, mb.AddInputPort("d1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d2, mb.AddInputPort("d2", big_array));

  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select1", priority_select1.node(),
                                    {s1, x1, y1, d1})
                    .status());
  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select2", priority_select2.node(),
                                    {s2, x2, y2, d2})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, DifferentArrayPrioritySelects) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u1 = package.GetBitsType(1);
  Type* u2 = package.GetBitsType(2);
  Type* u32 = package.GetBitsType(32);
  Type* big_array = package.GetArrayType(
      6, package.GetArrayType(16, package.GetArrayType(4, u32)));
  BValue s1_param = fb.Param("s1", u2);
  BValue x1_param = fb.Param("x1", big_array);
  BValue y1_param = fb.Param("y1", big_array);
  BValue d1_param = fb.Param("d1", big_array);
  BValue s2_param = fb.Param("s2", u1);
  BValue x2_param = fb.Param("x2", big_array);
  BValue d2_param = fb.Param("d2", big_array);
  BValue priority_select1 =
      fb.PrioritySelect(s1_param, {x1_param, y1_param}, d1_param);
  BValue priority_select2 = fb.PrioritySelect(s2_param, {x2_param}, d2_param);
  XLS_ASSERT_OK(
      fb.BuildWithReturnValue(fb.Tuple({priority_select1, priority_select2})));

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s1, mb.AddInputPort("s1", u2));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x1, mb.AddInputPort("x1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y1, mb.AddInputPort("y1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d1, mb.AddInputPort("d1", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * s2, mb.AddInputPort("s2", u1));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x2, mb.AddInputPort("x2", big_array));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * d2, mb.AddInputPort("d2", big_array));

  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select1", priority_select1.node(),
                                    {s1, x1, y1, d1})
                    .status());
  XLS_ASSERT_OK(mb.EmitAsAssignment("priority_select2", priority_select2.node(),
                                    {s2, x2, d2})
                    .status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ShraAsFunction) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x_param = fb.Param("x", u32);
  BValue y_param = fb.Param("y", u32);
  BValue x_shra_y = fb.Shra(x_param, y_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u32));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u32));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("x_smul_y", x_shra_y.node(), {x, y}).status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ShraAsFunctionSingleBit) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  Type* u1 = package.GetBitsType(1);
  BValue x_param = fb.Param("x", u1);
  BValue y_param = fb.Param("y", u1);
  BValue x_shra_y = fb.Shra(x_param, y_param);
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * x, mb.AddInputPort("x", u1));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * y, mb.AddInputPort("y", u1));
  XLS_ASSERT_OK(
      mb.EmitAsAssignment("x_smul_y", x_shra_y.node(), {x, y}).status());

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit());
}

TEST_P(ModuleBuilderTest, ArrayUpdate1D) {
  VerilogFile file = NewVerilogFile();
  Package package(TestBaseName());
  FunctionBuilder fb(TestBaseName(), &package);
  ArrayType* array_type = package.GetArrayType(4, package.GetBitsType(32));
  BValue array = fb.Param("array", array_type);
  BValue index = fb.Param("index", package.GetBitsType(2));
  BValue value = fb.Param("value", package.GetBitsType(32));
  BValue updated_array = fb.ArrayUpdate(array, value, /*indices=*/{index});
  XLS_ASSERT_OK(fb.Build());

  ModuleBuilder mb(TestBaseName(), &file, codegen_options());
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * array_ref,
                           mb.AddInputPort("array", array_type));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * index_ref,
                           mb.AddInputPort("index", package.GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(LogicRef * value_ref,
                           mb.AddInputPort("value", package.GetBitsType(32)));
  XLS_ASSERT_OK(mb.EmitAsAssignment("updated_array", updated_array.node(),
                                    {array_ref, index_ref, value_ref})
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
