// Copyright 2023 The XLS Authors
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

#include "xls/dslx/ir_convert/function_converter.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace {

TEST(FunctionConverterTest, ConvertsSimpleFunctionWithoutError) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn f() -> u32 { u32:42 }", "test_module.x",
                        "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  xls::Package package("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f() -> bits[32] {
  ret literal.1: bits[32] = literal(value=42, id=1, pos=[(0,0,16)])
}
)");
}

TEST(FunctionConverterTest, ExternFunctionAttributePreservedInIR) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[extern_verilog("extern_foobar {fn} (.out({return}));")]
fn f() -> u32 { u32:42 }
)",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_TRUE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  xls::Package package("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  // We expect a single function, that contains the FFI info for "extern_foobar"
  ASSERT_FALSE(package_data.package->functions().empty());
  ASSERT_TRUE(package_data.package->functions().front()->ForeignFunctionData());
  EXPECT_EQ(package_data.package->functions()
                .front()
                ->ForeignFunctionData()
                ->code_template(),
            "extern_foobar {fn} (.out({return}));");
}

TEST(FunctionConverterTest, ConvertsLastExprAndImplicitTokenWithoutError) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
fn f() {
    let acc: u32 = u32:0;
    for (i, acc): (u32, u32) in range(u32:0, u32:8) {
        let acc = acc + i;
        trace_fmt!("Do nothing");
        acc
    }(acc);
}
)",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  xls::Package package("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));
}

TEST(FunctionConverterTest, ConvertsFunctionWithZipBuiltin) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(
          "fn f(x: u32[2], y: u64[2]) -> (u32, u64)[2] { zip(x, y) }",
          "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);

  const ConvertOptions convert_options;
  xls::Package package("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(x: bits[32][2], y: bits[64][2]) -> (bits[32], bits[64])[2] {
  literal.3: bits[32] = literal(value=0, id=3)
  literal.7: bits[32] = literal(value=1, id=7)
  array_index.4: bits[32] = array_index(x, indices=[literal.3], id=4, pos=[(0,0,49)])
  array_index.5: bits[64] = array_index(y, indices=[literal.3], id=5, pos=[(0,0,49)])
  array_index.8: bits[32] = array_index(x, indices=[literal.7], id=8, pos=[(0,0,49)])
  array_index.9: bits[64] = array_index(y, indices=[literal.7], id=9, pos=[(0,0,49)])
  tuple.6: (bits[32], bits[64]) = tuple(array_index.4, array_index.5, id=6, pos=[(0,0,49)])
  tuple.10: (bits[32], bits[64]) = tuple(array_index.8, array_index.9, id=10, pos=[(0,0,49)])
  ret array.11: (bits[32], bits[64])[2] = array(tuple.6, tuple.10, id=11, pos=[(0,0,49)])
}
)");
}

}  // namespace
}  // namespace xls::dslx
