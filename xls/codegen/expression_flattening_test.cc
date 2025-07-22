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

#include "xls/codegen/expression_flattening.h"

#include <string>

#include "gtest/gtest.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"

namespace xls {
namespace {

class FlatteningTest : public IrTestBase {};

TEST_F(FlatteningTest, ExpressionFlattening) {
  Package p(TestName());
  Type* b5 = p.GetBitsType(5);
  ArrayType* a_of_b5 = p.GetArrayType(3, b5);
  ArrayType* array_2d = p.GetArrayType(2, a_of_b5);

  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());

    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.UnpackedArrayType(5, {3}, SourceInfo()),
                  SourceInfo()));
    auto* flattened = FlattenArray(foo, a_of_b5, &f, SourceInfo());
    std::string emitted = flattened->Emit(nullptr);
    EXPECT_EQ(emitted, "{foo[2], foo[1], foo[0]}");
  }

  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());

    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.UnpackedArrayType(5, {2}, SourceInfo()),
                  SourceInfo()));
    auto* flattened = FlattenArray(foo, array_2d, &f, SourceInfo());
    std::string emitted = flattened->Emit(nullptr);
    EXPECT_EQ(emitted,
              "{{foo[1][2], foo[1][1], foo[1][0]}, {foo[0][2], foo[0][1], "
              "foo[0][0]}}");
  }
}

TEST_F(FlatteningTest, ExpressionUnflattening) {
  Package p(TestName());
  Type* b5 = p.GetBitsType(5);
  ArrayType* a_of_b5 = p.GetArrayType(3, b5);
  ArrayType* array_2d = p.GetArrayType(2, a_of_b5);

  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());

    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.BitVectorType(15, SourceInfo()), SourceInfo()));
    auto* unflattened = UnflattenArray(foo, a_of_b5, &f, SourceInfo());
    std::string emitted = unflattened->Emit(nullptr);
    EXPECT_EQ(emitted, "'{foo[4:0], foo[9:5], foo[14:10]}");
  }
  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.BitVectorType(30, SourceInfo()), SourceInfo()));
    auto* unflattened = UnflattenArray(foo, array_2d, &f, SourceInfo());
    std::string emitted = unflattened->Emit(nullptr);
    EXPECT_EQ(emitted,
              "'{'{foo[4:0], foo[9:5], foo[14:10]}, '{foo[19:15], foo[24:20], "
              "foo[29:25]}}");
  }
  TupleType* tuple_type = p.GetTupleType({array_2d, b5, a_of_b5});
  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.BitVectorType(50, SourceInfo()), SourceInfo()));
    auto* unflattened =
        UnflattenArrayShapedTupleElement(foo, tuple_type, 0, &f, SourceInfo());
    std::string emitted = unflattened->Emit(nullptr);
    EXPECT_EQ(emitted,
              "'{'{foo[24:20], foo[29:25], foo[34:30]}, '{foo[39:35], "
              "foo[44:40], foo[49:45]}}");
  }
  {
    verilog::VerilogFile f(verilog::FileType::kVerilog);
    verilog::Module* m = f.AddModule(TestName(), SourceInfo());
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::LogicRef * foo,
        m->AddReg("foo", f.BitVectorType(50, SourceInfo()), SourceInfo()));
    auto* unflattened =
        UnflattenArrayShapedTupleElement(foo, tuple_type, 2, &f, SourceInfo());
    std::string emitted = unflattened->Emit(nullptr);
    EXPECT_EQ(emitted, "'{foo[4:0], foo[9:5], foo[14:10]}");
  }
}

}  // namespace
}  // namespace xls
