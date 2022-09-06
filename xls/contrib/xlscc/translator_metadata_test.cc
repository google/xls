// Copyright 2021 The XLS Authors
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

#include "xls/contrib/xlscc/translator.h"

#include <cstdio>
#include <memory>
#include <vector>

#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/unit_test.h"


namespace xlscc {
namespace {


class TranslatorMetadataTest : public XlsccTestBase {
 public:
  void standardizeMetadata(xlscc_metadata::MetadataOutput *meta) {
    ASSERT_NE(meta->top_func_proto().name().id(), 0);
    meta->mutable_top_func_proto()->mutable_name()->set_id(22078263808792L);

    meta->mutable_top_func_proto()
        ->mutable_return_location()
        ->mutable_begin()
        ->clear_filename();
    meta->mutable_top_func_proto()
        ->mutable_return_location()
        ->mutable_end()
        ->clear_filename();
    meta->mutable_top_func_proto()
        ->mutable_parameters_location()
        ->mutable_begin()
        ->clear_filename();
    meta->mutable_top_func_proto()
        ->mutable_parameters_location()
        ->mutable_end()
        ->clear_filename();
    meta->mutable_top_func_proto()
        ->mutable_whole_declaration_location()
        ->mutable_begin()
        ->clear_filename();
    meta->mutable_top_func_proto()
        ->mutable_whole_declaration_location()
        ->mutable_end()
        ->clear_filename();
    auto function_proto = meta->mutable_top_func_proto();
    for (int i = 0; i < function_proto->static_values_size(); i++) {
      auto function_value = function_proto->mutable_static_values(i);
      function_value->mutable_name()->set_id(22078263808792L);
    }
    ASSERT_GE(meta->all_func_protos_size(), 1);
    meta->clear_all_func_protos();
    for (int i = 0; i < meta->sources_size(); ++i) {
      meta->mutable_sources(i)->clear_path();
    }
  }
};

TEST_F(TranslatorMetadataTest, MetadataNamespaceStructArray) {
  const std::string content = R"(
    namespace foo {
      struct Blah {
        int aa;
      };
      #pragma hls_top
      Blah i_am_top(short a, short b[2]) {
        Blah x;
        x.aa = a+b[1];
        return x;
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "foo::i_am_top"
        id: 22078263808792
        xls_name: "i_am_top"
      }
      return_type {
        as_inst {
          name {
            name: "Blah"
            fully_qualified_name: "foo::Blah"
            id: 0
          }
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "b"
        type {
          as_array {
            element_type {
              as_int {
                width: 16
                is_signed: true
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
      whole_declaration_location {
        begin {
          line: 7
          column: 7
        }
        end {
          line: 11
          column: 7
        }
      }
      return_location {
        begin {
          line: 7
          column: 7
        }
        end {
          line: 7
          column: 7
        }
      }
      parameters_location {
        begin {
          line: 7
          column: 21
        }
        end {
          line: 7
          column: 39
        }
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  ASSERT_EQ(1, meta.structs_size());
  ASSERT_EQ(meta.top_func_proto().return_type().as_inst().name().id(),
            meta.structs(0).as_struct().name().as_inst().name().id());
  meta.mutable_top_func_proto()
      ->mutable_return_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(0)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, MetadataNamespaceNestedStruct) {
  const std::string content = R"(
    namespace foo {
      struct Blah {
        int aa;
        struct Something {
          int bb;
        }s;
      };
      #pragma hls_top
      short i_am_top(Blah a, short b[2]) {
        Blah x;
        x.s.bb = b[0];
        x.aa = a.aa+b[1];
        x.aa += x.s.bb;
        return x.aa;
      }
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        fields {
          name: "s"
          type {
            as_inst {
              name {
                name: "Something"
                fully_qualified_name: "foo::Blah::Something"
                id: 0
              }
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        no_tuple: false
      }
    }
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Something"
              fully_qualified_name: "foo::Blah::Something"
              id: 0
            }
          }
        }
        fields {
          name: "bb"
          type {
            as_int {
              width: 32
              is_signed: true
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "foo::i_am_top"
        id: 22078263808792
        xls_name: "i_am_top"
      }
      return_type {
        as_int {
          width: 16
          is_signed: true
        }
      }
      params {
        name: "a"
        type {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
              id: 0
            }
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "b"
        type {
          as_array {
            element_type {
              as_int {
                width: 16
                is_signed: true
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
      whole_declaration_location {
        begin {
          line: 10
          column: 7
        }
        end {
          line: 16
          column: 7
        }
      }
      return_location {
        begin {
          line: 10
          column: 7
        }
        end {
          line: 10
          column: 7
        }
      }
      parameters_location {
        begin {
          line: 10
          column: 22
        }
        end {
          line: 10
          column: 39
        }
      }
    })";

  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);

  ASSERT_EQ(2, meta.structs_size());

  const int subsidx = 1, topsidx = 0;

  // Order of structs is not deterministic, avoid protobuf equals failures
  if (meta.structs(0).as_struct().name().as_inst().name().name() ==
      "Something") {
    xlscc_metadata::Type top_struct = meta.structs(1);
    xlscc_metadata::Type sub_struct = meta.structs(0);
    *meta.mutable_structs(0) = top_struct;
    *meta.mutable_structs(1) = sub_struct;
  }

  ASSERT_EQ(1, meta.structs(subsidx).as_struct().fields_size());
  ASSERT_EQ(2, meta.structs(topsidx).as_struct().fields_size());

  ASSERT_EQ(meta.top_func_proto().params(0).type().as_inst().name().id(),
            meta.structs(topsidx).as_struct().name().as_inst().name().id());

  // Struct order gets reversed when emitting IR per commit
  // de1b6acdfbd9989c5b20b8a93a4d01b7853f2c09.
  ASSERT_EQ(
      meta.structs(topsidx).as_struct().fields(0).type().as_inst().name().id(),
      meta.structs(subsidx).as_struct().name().as_inst().name().id());

  meta.mutable_top_func_proto()
      ->mutable_params(0)
      ->mutable_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(topsidx)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(topsidx)
      ->mutable_as_struct()
      ->mutable_fields(0)
      ->mutable_type()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);
  meta.mutable_structs(subsidx)
      ->mutable_as_struct()
      ->mutable_name()
      ->mutable_as_inst()
      ->mutable_name()
      ->set_id(0);

  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, MetadataRefConstParams) {
  const std::string content = R"(
    #pragma hls_top
    void i_am_top(const short &a, short &b) {
      b += a;
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
        id: 22078263808792
        xls_name: "i_am_top"
      }
      return_type {
        as_void {
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: true
        is_const: true
      }
      params {
        name: "b"
        type {
          as_int {
            width: 16
            is_signed: true
          }
        }
        is_reference: true
        is_const: false
      }
      whole_declaration_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 5
          column: 5
        }
      }
      return_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 3
          column: 5
        }
      }
      parameters_location {
        begin {
          line: 3
          column: 19
        }
        end {
          line: 3
          column: 42
        }
      }
    })";

  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticInt) {
  const std::string content = R"(
    namespace foo {
    #pragma hls_top
    int my_package() {
      static int x = 22;
      int inner = 0;
      x += inner;
      return x;
    }
    } // namespacee foo
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "foo::my_package"
        id: 22078263808792
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
        }
      }
      whole_declaration_location {
        begin {
          line: 4
          column: 5
        }
        end {
          line: 9
          column: 5
        }
      }
      return_location {
        begin {
          line: 4
          column: 5
        }
        end {
          line: 4
          column: 5
        }
      }
      parameters_location {
        begin {
        }
        end {
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
          id: 22078263808792
        }
        type {
          as_int {
            width: 32
            is_signed: true
          }
        }
        value {
          as_int {
            signed_value: 22
          }
        }
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticBool) {
  const std::string content = R"(
    #pragma hls_top
    int my_package() {
      static bool x = true;
      int inner = 0;
      if (x) {
        inner += 1;
      }
      return inner;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        id: 22078263808792
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
        }
      }
      whole_declaration_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 10
          column: 5
        }
      }
      return_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 3
          column: 5
        }
      }
      parameters_location {
        begin {
        }
        end {
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
          id: 22078263808792
        }
        type {
          as_bool {
          }
        }
        value {
          as_bool: true
        }
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticArray) {
  const std::string content = R"(
    #pragma hls_top
    int my_package() {
      static int x[] = {0x00,0x01};
      int inner = 0;
      inner += x[1];
      return inner;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        id: 22078263808792
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
        }
      }
      whole_declaration_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 8
          column: 5
        }
      }
      return_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 3
          column: 5
        }
      }
      parameters_location {
        begin {
        }
        end {
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
          id: 22078263808792
        }
        type {
          as_array {
            element_type {
              as_int {
                width: 32
                is_signed: true
              }
            }
            size: 2
          }
        }
        value {
          as_array {
            element_values {
              as_int {
                signed_value: 0
              }
            }
            element_values {
              as_int {
                signed_value: 1
              }
            }
          }
        }
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, Static2DArray) {
  const std::string content = R"(
    #pragma hls_top
    int my_package() {
      static int x[][2] = {{0x00, 0x01}, {0x02, 0x03}};
      int inner = 0;
      inner += x[1][1];
      return inner;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    sources {
      number: 1
    }
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        id: 22078263808792
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
        }
      }
      whole_declaration_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 8
          column: 5
        }
      }
      return_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 3
          column: 5
        }
      }
      parameters_location {
        begin {
        }
        end {
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
          id: 22078263808792
        }
        type {
          as_array {
            element_type {
              as_array {
                element_type {
                  as_int {
                    width: 32
                    is_signed: true
                  }
                }
                size: 2
              }
            }
            size: 2
          }
        }
        value {
          as_array {
            element_values {
              as_array {
                element_values {
                  as_int {
                    signed_value: 0
                  }
                }
                element_values {
                  as_int {
                    signed_value: 1
                  }
                }
              }
            }
            element_values {
              as_array {
                element_values {
                  as_int {
                    signed_value: 2
                  }
                }
                element_values {
                  as_int {
                    signed_value: 3
                  }
                }
              }
            }
          }
        }
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, CharDeclarations) {
  const std::string content = R"(
    #pragma hls_top
    int my_package(signed char sc, unsigned char uc, char jc) {
      return sc + (int)uc + (int)jc;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        id: 22078263808792
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
        }
      }
      params {
        name: "sc"
        type {
          as_int {
            width: 8
            is_signed: true
            is_declared_as_char: false
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "uc"
        type {
          as_int {
            width: 8
            is_signed: false
            is_declared_as_char: false
          }
        }
        is_reference: false
        is_const: false
      }
      params {
        name: "jc"
        type {
          as_int {
            width: 8
            is_signed: true
            is_declared_as_char: true
          }
        }
        is_reference: false
        is_const: false
      }
      whole_declaration_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 5
          column: 5
        }
      }
      return_location {
        begin {
          line: 3
          column: 5
        }
        end {
          line: 3
          column: 5
        }
      }
      parameters_location {
        begin {
          line: 3
          column: 20
        }
        end {
          line: 3
          column: 59
        }
      }
    }
    sources {
      number: 1
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta);
  standardizeMetadata(&meta);
  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

}  // namespace

}  // namespace xlscc
