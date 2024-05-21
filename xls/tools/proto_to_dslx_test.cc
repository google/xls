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

#include "xls/tools/proto_to_dslx.h"

#include <memory>
#include <optional>
#include <string>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/module.h"

namespace xls {
namespace {

// Shotgun test to cover a bunch of areas for basic functionality.
TEST(ProtoToDslxTest, Smoke) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message SubField {
  optional uint32 sub_index = 1;
}

message Field {
  optional int32 index = 1;
  optional int32 bit_offset = 2;
  optional int32 width = 3;
  repeated int64 foo = 4;
  repeated SubField sub_fields = 5;
}

message Fields {
  repeated Field fields = 1;
  optional Field loner = 2;
}

message AnotherMessage {
  optional int64 pants = 1;
  optional int64 socks = 2;
  optional int64 shoes = 3;
  optional int64 spats = 4;
}
)";
  std::string textproto = R"(
fields {
  index: 0
  bit_offset: 0
  width: 4
  foo: 1
  foo: 2
  foo: 3
  foo: 4
  sub_fields: { sub_index: 1 }
  sub_fields: { sub_index: 2 }
  sub_fields: { sub_index: 3 }
  sub_fields: { sub_index: 4 }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(auto schema_file,
                           TempFile::CreateInDirectory(tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Fields", textproto, "foo"));

  EXPECT_EQ(module->ToString(),
            R"(pub struct SubField {
    sub_index: uN[32],
}
pub struct Field {
    index: sN[32],
    bit_offset: sN[32],
    width: sN[32],
    foo: sN[64][4],
    foo_count: u32,
    sub_fields: SubField[4],
    sub_fields_count: u32,
}
pub struct Fields {
    fields: Field[1],
    fields_count: u32,
    loner: Field,
}
pub const foo = Fields { fields: [Field { index: sN[32]:0, bit_offset: sN[32]:0, width: sN[32]:4, foo: [sN[64]:1, sN[64]:2, sN[64]:3, sN[64]:4], foo_count: u32:4, sub_fields: [SubField { sub_index: uN[32]:1 }, SubField { sub_index: uN[32]:2 }, SubField { sub_index: uN[32]:3 }, SubField { sub_index: uN[32]:4 }], sub_fields_count: u32:4 }], fields_count: u32:1, loner: Field { index: sN[32]:0, bit_offset: sN[32]:0, width: sN[32]:0, foo: [sN[64]:0, sN[64]:0, sN[64]:0, sN[64]:0], foo_count: u32:0, sub_fields: [SubField { sub_index: uN[32]:0 }, SubField { sub_index: uN[32]:0 }, SubField { sub_index: uN[32]:0 }, SubField { sub_index: uN[32]:0 }], sub_fields_count: u32:0 } };)");
}

TEST(ProtoToDslxTest, CanImportProtos) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

import "imported.proto";

message Top {
  optional int32 field_0 = 1;
  optional imported.Field imported_field = 2;
}
)";

  const std::string kImportedSchema = R"(
syntax = "proto2";

package imported;

message Field {
  optional int32 field_0 = 1;
}
)";

  const std::string kTextproto = R"(
field_0: 0xbeef
imported_field { field_0: 0xfeed }
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK(
      SetFileContents(tempdir.path() / "imported.proto", kImportedSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub struct imported_Field {
    field_0: sN[32],
}
pub struct Top {
    field_0: sN[32],
    imported_field: imported_Field,
}
pub const foo = Top { field_0: sN[32]:48879, imported_field: imported_Field { field_0: sN[32]:65261 } };)");
}

// Basic test for enum support.
TEST(ProtoToDslxTest, EnumSupport) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

enum MyEnum {
  VALUE_1 = 1;
  VALUE_2 = 2;
  VALUE_3 = 3;
  VALUE_600 = 600;
}

message Top {
  optional MyEnum my_scalar_enum = 1;
  repeated MyEnum my_repeated_enum = 2;
}
)";

  const std::string kTextproto = R"(
my_scalar_enum: VALUE_1
my_repeated_enum: VALUE_1
my_repeated_enum: VALUE_2
my_repeated_enum: VALUE_600
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub enum MyEnum : bits[11] {
    VALUE_1 = 1,
    VALUE_2 = 2,
    VALUE_3 = 3,
    VALUE_600 = 600,
}
pub struct Top {
    my_scalar_enum: MyEnum,
    my_repeated_enum: MyEnum[3],
    my_repeated_enum_count: u32,
}
pub const foo = Top { my_scalar_enum: MyEnum::VALUE_1, my_repeated_enum: [MyEnum::VALUE_1, MyEnum::VALUE_2, MyEnum::VALUE_600], my_repeated_enum_count: u32:3 };)");
}

TEST(ProtoToDslxTest, CanImportEnums) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

import "imported.proto";

message EnumHolder {
  repeated imported.Enum imported_enum = 1;
}

message Top {
  optional int32 field_0 = 1;
  repeated EnumHolder enum_holder = 2;
}
)";

  const std::string kImportedSchema = R"(
syntax = "proto2";

package imported;

enum Enum {
  VALUE_1 = 1;
  VALUE_2 = 2;
  VALUE_3 = 3;
  VALUE_600 = 600;
}
)";

  const std::string kTextproto = R"(
field_0: 0xbeef
enum_holder {
  imported_enum: VALUE_600
  imported_enum: VALUE_3
  imported_enum: VALUE_2
  imported_enum: VALUE_1
}
enum_holder {
  imported_enum: VALUE_3
  imported_enum: VALUE_2
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK(
      SetFileContents(tempdir.path() / "imported.proto", kImportedSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub enum imported_Enum : bits[11] {
    VALUE_1 = 1,
    VALUE_2 = 2,
    VALUE_3 = 3,
    VALUE_600 = 600,
}
pub struct EnumHolder {
    imported_enum: imported_Enum[4],
    imported_enum_count: u32,
}
pub struct Top {
    field_0: sN[32],
    enum_holder: EnumHolder[2],
    enum_holder_count: u32,
}
pub const foo = Top { field_0: sN[32]:48879, enum_holder: [EnumHolder { imported_enum: [imported_Enum::VALUE_600, imported_Enum::VALUE_3, imported_Enum::VALUE_2, imported_Enum::VALUE_1], imported_enum_count: u32:4 }, EnumHolder { imported_enum: [imported_Enum::VALUE_3, imported_Enum::VALUE_2, imported_Enum::VALUE_1, imported_Enum::VALUE_1], imported_enum_count: u32:2 }], enum_holder_count: u32:2 };)");
}

TEST(ProtoToDslxTest, HandlesStrings) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message Top {
  optional int64 my_int = 1;
  repeated string my_string = 2;
}
)";

  const std::string kTextproto = R"(
my_int: 0xbeef
my_string: "le boeuf"
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub struct Top {
    my_int: sN[64],
}
pub const foo = Top { my_int: sN[64]:48879 };)");
}

TEST(ProtoToDslxTest, CanHandleUnusedRepeatedFields) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message SubMessage {
  optional int64 my_int = 1;
}

message Top {
  repeated int64 my_ints = 1;
  optional int64 my_int = 2;
  repeated SubMessage my_submessages = 3;
  optional SubMessage my_submessage = 4;
}
)";

  const std::string kTextproto = R"(
my_int: 3
my_submessage { my_int: 6 }
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub struct SubMessage {
    my_int: sN[64],
}
pub struct Top {
    my_int: sN[64],
    my_submessage: SubMessage,
}
pub const foo = Top { my_int: sN[64]:3, my_submessage: SubMessage { my_int: sN[64]:6 } };)");
}

TEST(ProtoToDslxTest, CanHandleEmptyRepeatedFields) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message SubMessage {
  repeated int64 my_ints = 1;
}

message Top {
  repeated SubMessage submessage = 1;
}
)";

  const std::string kTextproto = R"(
submessage {
  my_ints: 1
  my_ints: 2
  my_ints: 3
  my_ints: 4
}
submessage {
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto tempdir, TempDirectory::Create());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schema_file,
      TempFile::CreateWithContentInDirectory(kSchema, tempdir.path()));
  XLS_ASSERT_OK(SetFileContents(schema_file.path(), kSchema));
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> module,
                           ProtoToDslx(tempdir.path(), schema_file.path(),
                                       "xls.Top", kTextproto, "foo"));
  EXPECT_EQ(module->ToString(),
            R"(pub struct SubMessage {
    my_ints: sN[64][4],
    my_ints_count: u32,
}
pub struct Top {
    submessage: SubMessage[2],
    submessage_count: u32,
}
pub const foo = Top { submessage: [SubMessage { my_ints: [sN[64]:1, sN[64]:2, sN[64]:3, sN[64]:4], my_ints_count: u32:4 }, SubMessage { my_ints: [sN[64]:0, sN[64]:0, sN[64]:0, sN[64]:0], my_ints_count: u32:0 }], submessage_count: u32:2 };)");
}

TEST(ProtoToDslxTest, MultipleTypesAndInstantiations) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message TypeA {
  optional uint32 index_a = 1;
}

message TypeB {
  optional uint64 index_b = 1;
}
)";

  std::string textproto_a1 = R"(
  index_a: 0
)";
  std::string textproto_a2 = R"(
  index_a: 1
)";
  std::string textproto_b1 = R"(
  index_b: 2
)";
  std::string textproto_b2 = R"(
  index_b: 3
)";

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::DescriptorPool> descriptor_pool,
      ProcessStringProtoSchema(kSchema));
  google::protobuf::DynamicMessageFactory factory;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_a1,
      ConstructProtoViaText(textproto_a1, "xls.TypeA", descriptor_pool.get(),
                            &factory));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_a2,
      ConstructProtoViaText(textproto_a2, "xls.TypeA", descriptor_pool.get(),
                            &factory));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_b1,
      ConstructProtoViaText(textproto_b1, "xls.TypeB", descriptor_pool.get(),
                            &factory));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_b2,
      ConstructProtoViaText(textproto_b2, "xls.TypeB", descriptor_pool.get(),
                            &factory));

  dslx::Module module("test_module", /*fs_path=*/std::nullopt);

  ProtoToDslxManager proto_to_dslx(&module);
  XLS_ASSERT_OK(
      proto_to_dslx.AddProtoInstantiationToDslxModule("a1", *message_a1));
  XLS_ASSERT_OK(
      proto_to_dslx.AddProtoInstantiationToDslxModule("a2", *message_a2));
  XLS_ASSERT_OK(
      proto_to_dslx.AddProtoInstantiationToDslxModule("b1", *message_b1));
  XLS_ASSERT_OK(
      proto_to_dslx.AddProtoInstantiationToDslxModule("b2", *message_b2));

  EXPECT_EQ(module.ToString(),
            R"(pub struct TypeA {
    index_a: uN[32],
}
pub const a1 = TypeA { index_a: uN[32]:0 };
pub const a2 = TypeA { index_a: uN[32]:1 };
pub struct TypeB {
    index_b: uN[64],
}
pub const b1 = TypeB { index_b: uN[64]:2 };
pub const b2 = TypeB { index_b: uN[64]:3 };)");
}

TEST(ProtoToDslxTest, CreateDslxFromParamsTest) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message TypeA {
  optional uint32 index_a = 1;
}
)";

  std::string textproto_a1 = R"(
  index_a: 10
)";
  std::string textproto_a2 = R"(
  index_a: 11
)";

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::DescriptorPool> descriptor_pool,
      ProcessStringProtoSchema(kSchema));
  google::protobuf::DynamicMessageFactory factory;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_a1,
      ConstructProtoViaText(textproto_a1, "xls.TypeA", descriptor_pool.get(),
                            &factory));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<google::protobuf::Message> message_a2,
      ConstructProtoViaText(textproto_a2, "xls.TypeA", descriptor_pool.get(),
                            &factory));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> module,
      CreateDslxFromParams(
          "module_test", {{"a1", message_a1.get()}, {"a2", message_a2.get()}}));

  EXPECT_EQ(module->ToString(),
            R"(pub struct TypeA {
    index_a: uN[32],
}
pub const a1 = TypeA { index_a: uN[32]:10 };
pub const a2 = TypeA { index_a: uN[32]:11 };)");
}

}  // namespace
}  // namespace xls
