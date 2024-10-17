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

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
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
using ::xls::proto_testing::EqualsProto;

constexpr std::string_view kTestName = "function_converter_test";
constexpr std::string_view kTestdataPath = "xls/dslx/ir_convert/testdata";

static std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

static std::filesystem::path TestFilePath(std::string_view test_name) {
  return absl::StrFormat("%s/%s_%s.ir", kTestdataPath, kTestName, test_name);
}

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
  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
  EXPECT_EQ(package_data.ir_to_dslx.size(), 1);
  ExpectEqualToGoldenFile(TestFilePath(TestName()), package.DumpIr());
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
