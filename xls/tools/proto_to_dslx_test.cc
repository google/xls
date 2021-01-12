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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace {

// Shotgun test to cover a bunch of areas for basic functionality.
TEST(ProtoToDslxTest, Smoke) {
  const std::string kSchema = R"(
syntax = "proto2";

package xls;

message SubField {
  optional int32 sub_index = 1;
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
            R"(struct SubField {
  sub_index: bits[32],
}
struct Field {
  index: bits[32],
  bit_offset: bits[32],
  width: bits[32],
  foo: bits[64][4],
  foo_count: u32,
  sub_fields: SubField[4],
  sub_fields_count: u32,
}
struct Fields {
  fields: Field[1],
  fields_count: u32,
  loner: Field,
}
pub const foo = Fields { fields: [Field { index: bits[32]:0, bit_offset: bits[32]:0, width: bits[32]:4, foo: [bits[64]:1, bits[64]:2, bits[64]:3, bits[64]:4], foo_count: u32:4, sub_fields: [SubField { sub_index: bits[32]:1 }, SubField { sub_index: bits[32]:2 }, SubField { sub_index: bits[32]:3 }, SubField { sub_index: bits[32]:4 }], sub_fields_count: u32:4 }], fields_count: u32:1, loner: Field { index: bits[32]:0, bit_offset: bits[32]:0, width: bits[32]:0, foo: [bits[64]:0, bits[64]:0, bits[64]:0, bits[64]:0], foo_count: u32:0, sub_fields: [SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }, SubField { sub_index: bits[32]:0 }], sub_fields_count: u32:0 } };)");
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
}

}  // namespace
}  // namespace xls
