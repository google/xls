// Copyright 2026 The XLS Authors
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

#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xls/common/attribute_data.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/function_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/package.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls::dslx {
namespace {
using ::xls::proto_testing::EqualsProto;

PackageConversionData MakeConversionData(std::string_view n) {
  return {.package = std::make_unique<Package>(n)};
}

TEST(FunctionConverterFuzzTestTest, BasicDomains) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..10, ()`)]
fn f(x: u32, y: u32) -> u32 { x + y }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
                parameter_domains { arbitrary: true }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, OneDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..10`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ConstantRangeDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const C = u32:1..u32:10;
#[fuzz_test(domains = `C`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData& attr = attributes[0];
  EXPECT_EQ(attr.kind(), AttributeKind::kFuzzTest);
  ASSERT_EQ(attr.args().size(), 1);
  const AttributeData::Argument& arg = attr.args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);
  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\001\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ConstantRangeReferenceDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const C = u32:1..u32:10;
const D = C;
#[fuzz_test(domains = `D`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData& attr = attributes[0];
  EXPECT_EQ(attr.kind(), AttributeKind::kFuzzTest);
  ASSERT_EQ(attr.args().size(), 1);
  const AttributeData::Argument& arg = attr.args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);
  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\001\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ConstantRangeInTuple) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const C = u32:1..u32:10;
#[fuzz_test(domains = `(C, u16:0..5)`)]
fn f(x: (u32, u16)) -> u16 { x.0 as u16 + x.1 }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData& attr = attributes[0];
  EXPECT_EQ(attr.kind(), AttributeKind::kFuzzTest);
  ASSERT_EQ(attr.args().size(), 1);
  const AttributeData::Argument& arg = attr.args()[0];

  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);
  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  tuple {
                    elements {
                      range {
                        min { bits { bit_count: 32 data: "\001\000\000\000" } }
                        max { bits { bit_count: 32 data: "\t\000\000\000" } }
                      }
                    }
                    elements {
                      range {
                        min { bits { bit_count: 16 data: "\000\000" } }
                        max { bits { bit_count: 16 data: "\004\000" } }
                      }
                    }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ConstantRangeAndLiteral) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const C = u32:1..u32:10;
#[fuzz_test(domains = `C, u16:0..5`)]
fn f(x: u32, y: u16) -> u16 { x as u16 + y }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData& attr = attributes[0];
  EXPECT_EQ(attr.kind(), AttributeKind::kFuzzTest);
  ASSERT_EQ(attr.args().size(), 1);
  const AttributeData::Argument& arg = attr.args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\001\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
                parameter_domains {
                  range {
                    min { bits { bit_count: 16 data: "\000\000" } }
                    max { bits { bit_count: 16 data: "\004\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, InclusiveRangeDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..=10`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\n\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, EmptyRangeError) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..0`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  EXPECT_THAT(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Empty ranges are unsupported as fuzztest domains")));
}

TEST(FunctionConverterFuzzTestTest, ConstantArrayDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const A = [u32:1, u32:2, u32:3];
#[fuzz_test(domains = `A`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  EXPECT_THAT(skv.second, testing::HasSubstr("element_of"));
  EXPECT_THAT(skv.second, testing::HasSubstr("data: \"\\001\\000\\000\\000\""));
}

TEST(FunctionConverterFuzzTestTest, ConstantTupleDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const T = (u32:1..u32:5, u32:2..u32:10);
#[fuzz_test(domains = `T`)]
fn f(x: (u32, u32)) -> u32 { x.0 }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  EXPECT_THAT(skv.second, testing::HasSubstr("tuple"));
  EXPECT_THAT(skv.second, testing::HasSubstr("data: \"\\001\\000\\000\\000\""));
}

TEST(FunctionConverterFuzzTestTest, ConstantArrayWithConstantEmbedded) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const A = u32:1;
const C = [A, u32:2, u32:3];
#[fuzz_test(domains = `C`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  EXPECT_THAT(skv.second, testing::HasSubstr("element_of"));
  EXPECT_THAT(skv.second, testing::HasSubstr("data: \"\\001\\000\\000\\000\""));
}

TEST(FunctionConverterFuzzTestTest, DifferentBitWidths) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u8:0..5, u64:0..100`)]
fn f(x: u8, y: u64) -> u64 { (x as u64) + y }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(
      function_proto, EqualsProto(R"pb(
        parameter_domains {
          range {
            min { bits { bit_count: 8 data: "\000" } }
            max { bits { bit_count: 8 data: "\004" } }
          }
        }
        parameter_domains {
          range {
            min {
              bits { bit_count: 64 data: "\000\000\000\000\000\000\000\000" }
            }
            max { bits { bit_count: 64 data: "c\000\000\000\000\000\000\000" } }
          }
        }
      )pb"));
}

TEST(FunctionConverterFuzzTestTest, RangeEdgeCases) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..0xFFFFFFFF`)]
fn f(y: u32) -> u32 { y }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\376\377\377\377" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, MultipleDomains) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32:0..10, u32:20..30`)]
fn f(x: u32, y: u32) -> u32 { x + y }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  ASSERT_EQ(attributes.size(), 1);
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\024\000\000\000" } }
                    max { bits { bit_count: 32 data: "\035\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, TupleDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `(u32:0..10, u32:0..20)`)]
fn f(x: (u32, u32)) -> u32 { x.0 }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  tuple {
                    elements {
                      range {
                        min { bits { bit_count: 32 data: "\000\000\000\000" } }
                        max { bits { bit_count: 32 data: "\t\000\000\000" } }
                      }
                    }
                    elements {
                      range {
                        min { bits { bit_count: 32 data: "\000\000\000\000" } }
                        max { bits { bit_count: 32 data: "\023\000\000\000" } }
                      }
                    }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ElementOfDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `[u32:1, u32:2, u32:3]`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  element_of {
                    values { bits { bit_count: 32 data: "\001\000\000\000" } }
                    values { bits { bit_count: 32 data: "\002\000\000\000" } }
                    values { bits { bit_count: 32 data: "\003\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, EnumElementOfDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
enum MyEnum : u32 {
  A = 1,
  B = 2,
}
#[fuzz_test(domains = `[MyEnum::A, MyEnum::B]`)]
fn f(x: MyEnum) -> MyEnum { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  element_of {
                    values { bits { bit_count: 32 data: "\001\000\000\000" } }
                    values { bits { bit_count: 32 data: "\002\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, NestedTupleDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
const R = u32:0..5;
#[fuzz_test(domains = `u32:0..10, ((), R)`)]
fn f(x: u32, y: ((), u32)) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);

  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  ASSERT_FALSE(package_data.conversion_info->package->functions().empty());
  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  EXPECT_TRUE(ir_fn->HasAttribute(AttributeKind::kFuzzTest));
  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const auto& skv =
      std::get<AttributeData::StringKeyValueArgument>(attributes[0].args()[0]);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  range {
                    min { bits { bit_count: 32 data: "\000\000\000\000" } }
                    max { bits { bit_count: 32 data: "\t\000\000\000" } }
                  }
                }
                parameter_domains {
                  tuple {
                    elements { arbitrary: true }
                    elements {
                      range {
                        min { bits { bit_count: 32 data: "\000\000\000\000" } }
                        max { bits { bit_count: 32 data: "\004\000\000\000" } }
                      }
                    }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ScalarConstantDomain) {
  ImportData import_data = CreateImportDataForTest();
  EXPECT_THAT(ParseAndTypecheck(R"(
const C = u32:42;
#[fuzz_test(domains = `C`)]
fn f(x: u32) -> u32 { x }
)",
                                "test_module.x", "test_module", &import_data),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Expected range or set domain for scalar "
                                     "parameter x: u32; got type ubits")));
}

TEST(FunctionConverterFuzzTestTest, EmptyArrayDomain) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
#[fuzz_test(domains = `u32[0]:[]`)]
fn f(x: u32) -> u32 { x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);
  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  EXPECT_THAT(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Empty arrays are unsupported as fuzztest domains")));
}

TEST(FunctionConverterFuzzTestTest, ArbitraryEnumBecomesElementOf) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
enum MyEnum : u32 {
  A = 1,
  B = 2,
}

#[fuzz_test(domains = `()`)]
fn f(x: MyEnum) -> bool { x == x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(function_proto, EqualsProto(R"pb(
                parameter_domains {
                  element_of {
                    values { bits { bit_count: 32 data: "\001\000\000\000" } }
                    values { bits { bit_count: 32 data: "\002\000\000\000" } }
                  }
                }
              )pb"));
}

TEST(FunctionConverterFuzzTestTest, ArbitraryEnumInTupleBecomesElementOf) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
enum MyEnum : u32 {
  A = 1,
  B = 2,
}

#[fuzz_test(domains = `()`)]
fn f(x: (MyEnum, MyEnum)) -> bool { x == x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(
      function_proto, EqualsProto(R"pb(
        parameter_domains {
          tuple {
            elements {
              element_of {
                values { bits { bit_count: 32 data: "\001\000\000\000" } }
                values { bits { bit_count: 32 data: "\002\000\000\000" } }
              }
            }
            elements {
              element_of {
                values { bits { bit_count: 32 data: "\001\000\000\000" } }
                values { bits { bit_count: 32 data: "\002\000\000\000" } }
              }
            }
          }
        }
      )pb"));
}

TEST(FunctionConverterFuzzTestTest, DeeplyNestedArbitraryEnumInTuple) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
enum MyEnum : u32 {
  A = 1,
  B = 2,
}

#[fuzz_test(domains = `()`)]
fn f(x: (u32, (MyEnum, u8))) -> bool { x.0 == x.0 }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(
      function_proto, EqualsProto(R"pb(
        parameter_domains {
          tuple {
            elements { arbitrary: true }
            elements {
              tuple {
                elements {
                  element_of {
                    values { bits { bit_count: 32 data: "\001\000\000\000" } }
                    values { bits { bit_count: 32 data: "\002\000\000\000" } }
                  }
                }
                elements { arbitrary: true }
              }
            }
          }
        }
      )pb"));
}

TEST(FunctionConverterFuzzTestTest, MixedTupleDomainWithArbitraryEnum) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
enum MyEnum : u32 {
  A = 1,
  B = 2,
}

#[fuzz_test(domains = `(u32:1..5, ())`)]
fn f(x: (u32, MyEnum)) -> bool { x.0 == x.0 }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  EXPECT_THAT(
      function_proto, EqualsProto(R"pb(
        parameter_domains {
          tuple {
            elements {
              range {
                min { bits { bit_count: 32 data: "\001\000\000\000" } }
                max { bits { bit_count: 32 data: "\004\000\000\000" } }
              }
            }
            elements {
              element_of {
                values { bits { bit_count: 32 data: "\001\000\000\000" } }
                values { bits { bit_count: 32 data: "\002\000\000\000" } }
              }
            }
          }
        }
      )pb"));
}

TEST(FunctionConverterFuzzTestTest, StructDomainBasic) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
struct MyStruct {
  x: u32,
  y: u8,
}
#[fuzz_test(domains = `MyStruct { x: u32:0..10, y: [u8:1, 2] }`)]
fn f(s: MyStruct) -> u32 { s.x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  ASSERT_EQ(function_proto.parameter_domains_size(), 1);
  const auto& domain = function_proto.parameter_domains(0);

  // DSLX struct domain lowers to an IR Tuple domain
  ASSERT_TRUE(domain.has_tuple());
  ASSERT_EQ(domain.tuple().elements_size(), 2);

  // First field: x: u32:0..10 -> range: [0, 10) in DSLX, so [0, 9] inclusive in
  // proto
  const auto& x_domain = domain.tuple().elements(0);
  ASSERT_TRUE(x_domain.has_range());
  EXPECT_EQ(x_domain.range().min().bits().bit_count(), 32);
  EXPECT_EQ(x_domain.range().min().bits().data(), std::string(4, '\0'));
  EXPECT_EQ(x_domain.range().max().bits().bit_count(), 32);
  EXPECT_EQ(x_domain.range().max().bits().data(),
            std::string("\011\000\000\000", 4));  // 9

  // Second field: y: [u8:1, 2] -> element_of: [1, 2]
  const auto& y_domain = domain.tuple().elements(1);
  ASSERT_TRUE(y_domain.has_element_of());
  ASSERT_EQ(y_domain.element_of().values_size(), 2);
  EXPECT_EQ(y_domain.element_of().values(0).bits().data(), std::string{'\001'});
  EXPECT_EQ(y_domain.element_of().values(1).bits().data(), std::string{'\002'});
}

TEST(FunctionConverterFuzzTestTest, StructDomainSparse) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
struct MyStruct {
  x: u32,
  y: u8,
}
#[fuzz_test(domains = `MyStruct { x: u32:0..10 }`)]
fn f(s: MyStruct) -> u32 { s.x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  ASSERT_EQ(function_proto.parameter_domains_size(), 1);
  const auto& domain = function_proto.parameter_domains(0);

  ASSERT_TRUE(domain.has_tuple());
  ASSERT_EQ(domain.tuple().elements_size(), 2);

  // First field: x: u32:0..10
  const auto& x_domain = domain.tuple().elements(0);
  ASSERT_TRUE(x_domain.has_range());

  // Second field: y omitted -> arbitrary: true
  const auto& y_domain = domain.tuple().elements(1);
  ASSERT_TRUE(y_domain.has_arbitrary());
  EXPECT_TRUE(y_domain.arbitrary());
}

TEST(FunctionConverterFuzzTestTest, StructDomainNested) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(R"(
struct Inner {
  a: u8,
}
struct Outer {
  x: u32,
  s: Inner,
}
#[fuzz_test(domains = `Outer { s: Inner { a: u8:1..5 } }`)]
fn f(o: Outer) -> u32 { o.x }
)",
                        "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(FuzzTestFunction * ft,
                           tm.module->GetMemberOrError<FuzzTestFunction>("f"));
  ASSERT_NE(ft, nullptr);

  Function* f = &ft->fn();

  const ConvertOptions convert_options;
  PackageConversionData package = MakeConversionData("test_module_package");
  PackageData package_data{&package};
  FunctionConverter converter(package_data, tm.module, &import_data,
                              convert_options, /*proc_data=*/nullptr,
                              /*channel_scope=*/nullptr,
                              /*is_top=*/true);
  XLS_ASSERT_OK(
      converter.HandleFunction(f, tm.type_info, /*parametric_env=*/nullptr));

  auto* ir_fn =
      package_data.conversion_info->package->functions().front().get();

  absl::Span<const AttributeData> attributes = ir_fn->attributes();
  const AttributeData::Argument& arg = attributes[0].args()[0];
  const auto& skv = std::get<AttributeData::StringKeyValueArgument>(arg);

  xls::PackageInterfaceProto::Function function_proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(skv.second, &function_proto));
  ASSERT_EQ(function_proto.parameter_domains_size(), 1);
  const auto& domain = function_proto.parameter_domains(0);

  ASSERT_TRUE(domain.has_tuple());
  ASSERT_EQ(domain.tuple().elements_size(), 2);

  // Outer.x omitted -> arbitrary
  const auto& x_domain = domain.tuple().elements(0);
  ASSERT_TRUE(x_domain.has_arbitrary());
  EXPECT_TRUE(x_domain.arbitrary());

  // Outer.s -> Tuple representing Inner struct
  const auto& s_domain = domain.tuple().elements(1);
  ASSERT_TRUE(s_domain.has_tuple());
  ASSERT_EQ(s_domain.tuple().elements_size(), 1);

  // Inner.a -> range [1, 4] (u8:1..5 in DSLX)
  const auto& a_domain = s_domain.tuple().elements(0);
  ASSERT_TRUE(a_domain.has_range());
  EXPECT_EQ(a_domain.range().min().bits().data(), std::string{'\001'});
  EXPECT_EQ(a_domain.range().max().bits().data(), std::string{'\004'});
}

}  // namespace
}  // namespace xls::dslx
