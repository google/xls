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

#include <memory>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/package.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {
namespace {
using proto_testing::EqualsProto;

PackageConversionData MakeConversionData(std::string_view n) {
  return {.package = std::make_unique<Package>(n)};
}

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
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
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
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  result_type { type_enum: BITS bit_count: 32 }
                }
              )pb"));
}
TEST(FunctionConverterTest, ConvertsSimpleFunctionWithAsserts) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(fn f() -> () {
        assert!(u32:42 == u32:31 + u32:1, "foo");
        assert_eq(u32:42, u32:31 + u32:1);
        assert_lt(u32:41, u32:31 + u32:1);
      })",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{.conversion_info = &package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"IR(package test_module_package

file_number 0 "test_module.x"

fn __itok__test_module__f(__token: token, __activated: bits[1]) -> (token, ()) {
  literal.4: bits[32] = literal(value=31, id=4, pos=[(0,1,26)])
  literal.5: bits[32] = literal(value=1, id=5, pos=[(0,1,35)])
  literal.14: bits[32] = literal(value=31, id=14, pos=[(0,2,26)])
  literal.15: bits[32] = literal(value=1, id=15, pos=[(0,2,35)])
  literal.23: bits[32] = literal(value=31, id=23, pos=[(0,3,26)])
  literal.24: bits[32] = literal(value=1, id=24, pos=[(0,3,35)])
  literal.3: bits[32] = literal(value=42, id=3, pos=[(0,1,16)])
  add.6: bits[32] = add(literal.4, literal.5, id=6, pos=[(0,1,33)])
  literal.13: bits[32] = literal(value=42, id=13, pos=[(0,2,18)])
  add.16: bits[32] = add(literal.14, literal.15, id=16, pos=[(0,2,33)])
  literal.22: bits[32] = literal(value=41, id=22, pos=[(0,3,18)])
  add.25: bits[32] = add(literal.23, literal.24, id=25, pos=[(0,3,33)])
  not.9: bits[1] = not(__activated, id=9)
  eq.7: bits[1] = eq(literal.3, add.6, id=7, pos=[(0,1,23)])
  not.18: bits[1] = not(__activated, id=18)
  eq.17: bits[1] = eq(literal.13, add.16, id=17)
  not.27: bits[1] = not(__activated, id=27)
  ult.26: bits[1] = ult(literal.22, add.25, id=26)
  or.10: bits[1] = or(not.9, eq.7, id=10)
  or.19: bits[1] = or(not.18, eq.17, id=19)
  or.28: bits[1] = or(not.27, ult.26, id=28)
  assert.11: token = assert(__token, or.10, message="Assertion failure via assert! @ test_module.x:2:16-2:49", label="foo", id=11)
  assert.20: token = assert(__token, or.19, message="Assertion failure via assert_eq @ test_module.x:3:18-3:42", label="assert_eq(u32:42, u32:31 + u32:1)", id=20)
  assert.29: token = assert(__token, or.28, message="Assertion failure via assert_lt @ test_module.x:4:18-4:42", label="assert_lt(u32:41, u32:31 + u32:1)", id=29)
  after_all.32: token = after_all(assert.11, assert.20, assert.29, id=32)
  tuple.31: () = tuple(id=31, pos=[(0,0,13)])
  literal.8: bits[8][3] = literal(value=[102, 111, 111], id=8, pos=[(0,1,42)])
  tuple.12: () = tuple(id=12)
  tuple.21: () = tuple(id=21)
  tuple.30: () = tuple(id=30)
  ret tuple.33: (token, ()) = tuple(after_all.32, tuple.31, id=33)
}

top fn __test_module__f() -> () {
  after_all.34: token = after_all(id=34)
  literal.35: bits[1] = literal(value=1, id=35)
  invoke.36: (token, ()) = invoke(after_all.34, literal.35, to_apply=__itok__test_module__f, id=36)
  ret tuple_index.37: () = tuple_index(invoke.36, index=1, id=37)
}
)IR");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__itok__test_module__f" }
                  parameters {
                    name: "__token"
                    type { type_enum: TOKEN }
                  }
                  parameters {
                    name: "__activated"
                    type { type_enum: BITS bit_count: 1 }
                  }
                  result_type {
                    type_enum: TUPLE
                    tuple_elements { type_enum: TOKEN }
                    tuple_elements { type_enum: TUPLE }
                  }
                }
                functions { base { name: "__test_module__f" } }
              )pb"));
}

TEST(FunctionConverterTest, TracksMultipleTypeAliasSvType) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(#[sv_type("something::cool")]
                           type FooBar = u32;
                           type Baz = u32;
                           fn f(b: Baz) -> FooBar { b + u32:42 })",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(), R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(b: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=42, id=2, pos=[(0,3,56)])
  ret add.3: bits[32] = add(b, literal.2, id=3, pos=[(0,3,54)])
}
)");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "b"
                    type { type_enum: BITS bit_count: 32 }
                  }
                  result_type { type_enum: BITS bit_count: 32 }
                  sv_result_type: "something::cool"
                }
              )pb"));
}

TEST(FunctionConverterTest, TracksTypeAliasSvType) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(#[sv_type("something::cool")]
                           type FooBar = u32;
                           #[sv_type("even::cooler")]
                           type Baz = u32;
                           fn f(b: Baz) -> FooBar { b + u32:42 })",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(b: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=42, id=2, pos=[(0,4,56)])
  ret add.3: bits[32] = add(b, literal.2, id=3, pos=[(0,4,54)])
}
)");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "b"
                    type { type_enum: BITS bit_count: 32 }
                    sv_type: "even::cooler"
                  }
                  result_type { type_enum: BITS bit_count: 32 }
                  sv_result_type: "something::cool"
                }
              )pb"));
}

TEST(FunctionConverterTest, TracksTypeAliasStopsAtFirstSvType) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[sv_type("something::cool")]
type FooBar = u32;
#[sv_type("even::cooler")]
type Baz = FooBar;
fn f(b: Baz) -> FooBar { b + u32:42 })",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);
  EXPECT_FALSE(f->extern_verilog_module().has_value());

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(b: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=42, id=2, pos=[(0,5,29)])
  ret add.3: bits[32] = add(b, literal.2, id=3, pos=[(0,5,27)])
}
)");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "b"
                    type { type_enum: BITS bit_count: 32 }
                    sv_type: "even::cooler"
                  }
                  result_type { type_enum: BITS bit_count: 32 }
                  sv_result_type: "something::cool"
                }
              )pb"));
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
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  // We expect a single function, that contains the FFI info for "extern_foobar"
  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  ASSERT_TRUE(package_data.conversion_info->package->functions()
                  .front()
                  ->ForeignFunctionData());
  EXPECT_EQ(package_data.conversion_info->package->functions()
                .front()
                ->ForeignFunctionData()
                ->code_template(),
            "extern_foobar {fn} (.out({return}));");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  result_type { type_enum: BITS bit_count: 32 }
                }
              )pb"));
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
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));
  EXPECT_THAT(package.interface.functions(),
              testing::UnorderedElementsAre(
                  EqualsProto(R"pb(
                    base { top: true name: "__itok__test_module__f" }
                    parameters {
                      name: "__token"
                      type { type_enum: TOKEN }
                    }
                    parameters {
                      name: "__activated"
                      type { type_enum: BITS bit_count: 1 }
                    }
                    result_type {
                      type_enum: TUPLE
                      tuple_elements { type_enum: TOKEN }
                      tuple_elements { type_enum: TUPLE }
                    })pb"),
                  EqualsProto(R"pb(
                    base { name: "____itok__test_module__f_counted_for_0_body" }
                    parameters {
                      name: "i"
                      type { type_enum: BITS bit_count: 32 }
                    }
                    parameters {
                      name: "__token_wrapped"
                      type {
                        type_enum: TUPLE
                        tuple_elements { type_enum: TOKEN }
                        tuple_elements { type_enum: BITS bit_count: 1 }
                        tuple_elements { type_enum: BITS bit_count: 32 }
                      }
                    }
                  )pb"),
                  EqualsProto(R"pb(
                    base { name: "__test_module__f" }
                  )pb")));
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
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
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
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "x"
                    type {
                      type_enum: ARRAY
                      array_size: 2
                      array_element { type_enum: BITS bit_count: 32 }
                    }
                  }
                  parameters {
                    name: "y"
                    type {
                      type_enum: ARRAY
                      array_size: 2
                      array_element { type_enum: BITS bit_count: 64 }
                    }
                  }
                  result_type {
                    type_enum: ARRAY
                    array_size: 2
                    array_element {
                      type_enum: TUPLE
                      tuple_elements { type_enum: BITS bit_count: 32 }
                      tuple_elements { type_enum: BITS bit_count: 64 }
                    }
                  }
                }
              )pb"));
}

TEST(FunctionConverterTest, ConvertsFunctionWithUpdate2DBuiltin) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn f(a: u32[2][3]) -> u32[2][3] { update(a, (u1:1, "
                        "u32:0), u32:42) }",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{.conversion_info = &package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(a: bits[32][2][3]) -> bits[32][2][3] {
  literal.2: bits[1] = literal(value=1, id=2, pos=[(0,0,45)])
  literal.3: bits[32] = literal(value=0, id=3, pos=[(0,0,51)])
  literal.5: bits[32] = literal(value=42, id=5, pos=[(0,0,59)])
  tuple.4: (bits[1], bits[32]) = tuple(literal.2, literal.3, id=4, pos=[(0,0,44)])
  ret array_update.6: bits[32][2][3] = array_update(a, literal.5, indices=[literal.2, literal.3], id=6, pos=[(0,0,40)])
}
)");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "a"
                    type {
                      type_enum: ARRAY
                      array_size: 3
                      array_element {
                        type_enum: ARRAY
                        array_size: 2
                        array_element { type_enum: BITS bit_count: 32 }
                      }
                    }
                  }
                  result_type {
                    type_enum: ARRAY
                    array_size: 3
                    array_element {
                      type_enum: ARRAY
                      array_size: 2
                      array_element { type_enum: BITS bit_count: 32 }
                    }
                  }
                }
              )pb"));
}

TEST(FunctionConverterTest, ConvertsFunctionWithUpdate2DBuiltinEmptyTuple) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn f(a: u32[2][3]) -> u32[2][3] { update(a, (), a) }",
                        "test_module.x", "test_module", &import_data));

  Function* f = tm.module->GetFunction("f").value();
  ASSERT_NE(f, nullptr);

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{.conversion_info = &package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  EXPECT_EQ(package.DumpIr(),
            R"(package test_module_package

file_number 0 "test_module.x"

top fn __test_module__f(a: bits[32][2][3]) -> bits[32][2][3] {
  tuple.2: () = tuple(id=2, pos=[(0,0,44)])
  ret array_update.3: bits[32][2][3] = array_update(a, a, indices=[], id=3, pos=[(0,0,40)])
}
)");
  EXPECT_THAT(package.interface, EqualsProto(R"pb(
                functions {
                  base { top: true name: "__test_module__f" }
                  parameters {
                    name: "a"
                    type {
                      type_enum: ARRAY
                      array_size: 3
                      array_element {
                        type_enum: ARRAY
                        array_size: 2
                        array_element { type_enum: BITS bit_count: 32 }
                      }
                    }
                  }
                  result_type {
                    type_enum: ARRAY
                    array_size: 3
                    array_element {
                      type_enum: ARRAY
                      array_size: 2
                      array_element { type_enum: BITS bit_count: 32 }
                    }
                  }
                }
              )pb"));
}

}  // namespace
}  // namespace xls::dslx
