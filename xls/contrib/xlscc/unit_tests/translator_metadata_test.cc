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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"

namespace xlscc {
namespace {

class TranslatorMetadataTest : public XlsccTestBase {
 public:
  void clearIds(xlscc_metadata::Type* meta) {
    meta->clear_declaration_location();

    if (meta->has_as_inst()) {
      meta->mutable_as_inst()->mutable_name()->clear_id();
    } else if (meta->has_as_struct()) {
      clearIds(meta->mutable_as_struct()->mutable_name());
      for (xlscc_metadata::StructField& field :
           *meta->mutable_as_struct()->mutable_fields()) {
        clearIds(field.mutable_type());
      }
    } else if (meta->has_as_array()) {
      clearIds(meta->mutable_as_array()->mutable_element_type());
    }
  }
  void clearIds(xlscc_metadata::Value* value) {
    if (value->has_as_array()) {
      for (xlscc_metadata::Value& mutable_value :
           *value->mutable_as_array()->mutable_element_values()) {
        clearIds(&mutable_value);
      }
    } else if (value->has_as_struct()) {
      clearIds(value->mutable_as_struct()->mutable_name());
      for (xlscc_metadata::StructFieldValue& field :
           *value->mutable_as_struct()->mutable_fields()) {
        field.mutable_name()->clear_id();
        clearIds(field.mutable_type());
        clearIds(field.mutable_value());
      }
    }
  }
  void standardizeMetadata(xlscc_metadata::MetadataOutput* meta) {
    ASSERT_NE(meta->top_func_proto().name().id(), 0);
    meta->mutable_top_func_proto()->mutable_name()->clear_id();

    if (meta->mutable_top_func_proto()->has_return_type()) {
      clearIds(meta->mutable_top_func_proto()->mutable_return_type());
    }
    if (meta->mutable_top_func_proto()->has_this_type()) {
      clearIds(meta->mutable_top_func_proto()->mutable_this_type());
    }

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

    ASSERT_TRUE(meta->top_func_proto().has_whole_declaration_location());
    meta->mutable_top_func_proto()->clear_whole_declaration_location();

    ASSERT_TRUE(meta->top_func_proto().has_return_location());
    meta->mutable_top_func_proto()->clear_return_location();

    ASSERT_TRUE(meta->top_func_proto().has_parameters_location());
    meta->mutable_top_func_proto()->clear_parameters_location();

    for (xlscc_metadata::FunctionParameter& param :
         *meta->mutable_top_func_proto()->mutable_params()) {
      clearIds(param.mutable_type());
    }

    auto function_proto = meta->mutable_top_func_proto();
    for (int i = 0; i < function_proto->static_values_size(); i++) {
      auto function_value = function_proto->mutable_static_values(i);
      function_value->mutable_name()->clear_id();
      function_value->clear_declaration_location();
      clearIds(function_value->mutable_type());
      clearIds(function_value->mutable_value());
    }
    ASSERT_GE(meta->all_func_protos_size(), 1);
    meta->clear_all_func_protos();

    ASSERT_GE(meta->sources_size(), 1);
    meta->clear_sources();

    for (xlscc_metadata::Type& struct_type : *meta->mutable_structs()) {
      clearIds(&struct_type);
    }
  }
};

TEST_F(TranslatorMetadataTest, NamespaceStructArray) {
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
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
              is_synthetic: false
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
        xls_name: "i_am_top"
      }
      return_type {
        as_inst {
          name {
            name: "Blah"
            fully_qualified_name: "foo::Blah"
          }
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
            is_synthetic: false
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
                is_synthetic: false
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
    })";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, NamespaceNestedStruct) {
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
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
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
              is_synthetic: false
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
            }
          }
        }
        fields {
          name: "bb"
          type {
            as_int {
              width: 32
              is_signed: true
              is_synthetic: false
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
        xls_name: "i_am_top"
      }
      return_type {
        as_int {
          width: 16
          is_signed: true
          is_synthetic: false
        }
      }
      params {
        name: "a"
        type {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
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
                is_synthetic: false
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, ArrayOfStructs) {
  const std::string content = R"(
    namespace foo {
      struct Something {
        int bb;
      };
      struct Blah {
        int aa;
        Something b[10];
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
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "foo::Blah"
            }
          }
        }
        fields {
          name: "b"
          type {
            as_array {
              element_type {
                as_inst {
                  name {
                    name: "Something"
                    fully_qualified_name: "foo::Something"
                  }
                }
              }
              size: 10
            }
          }
        }
        fields {
          name: "aa"
          type {
            as_int {
              width: 32
              is_signed: true
              is_synthetic: false
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
              fully_qualified_name: "foo::Something"
            }
          }
        }
        fields {
          name: "bb"
          type {
            as_int {
              width: 32
              is_signed: true
              is_synthetic: false
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
        xls_name: "i_am_top"
      }
      return_type {
        as_inst {
          name {
            name: "Blah"
            fully_qualified_name: "foo::Blah"
          }
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 16
            is_signed: true
            is_synthetic: false
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
                is_synthetic: false
              }
            }
            size: 2
          }
        }
        is_reference: true
        is_const: false
      }
    }

  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, RefConstParams) {
  const std::string content = R"(
    #pragma hls_top
    void i_am_top(const short &a, short &b) {
      b += a;
    })";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  standardizeMetadata(&meta);

  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
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
            is_synthetic: false
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
            is_synthetic: false
          }
        }
        is_reference: true
        is_const: false
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

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
    } // namespace foo
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "foo::my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_int {
            width: 32
            is_signed: true
            is_synthetic: false
          }
        }
        value {
          as_int {
            big_endian_bytes: "\026\000\000\000"
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticIntNegative) {
  const std::string content = R"(
    namespace foo {
    #pragma hls_top
    int my_package() {
      static int x = -1;
      int inner = 0;
      x += inner;
      return x;
    }
    } // namespace foo
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "foo::my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_int {
            width: 32
            is_signed: true
            is_synthetic: false
          }
        }
        value {
          as_int {
            big_endian_bytes: "\377\377\377\377"
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticFloats) {
  const std::string content = R"(
    namespace foo {
    #pragma hls_top
    int my_package() {
      static float x = 1.5f;
      static double y = 3.14f;
      (void)x;(void)y;
      return 0;
    }
    } // namespace foo
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "foo::my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_float {
            is_double_precision: false
          }
        }
        value {
          as_float {
            value: 1.5
          }
        }
      }
      static_values {
        name {
          name: "y"
          fully_qualified_name: "y"
        }
        type {
          as_float {
            is_double_precision: true
          }
        }
        value {
          as_float {
            value: 3.1400001049041748
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

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
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_bool {
          }
        }
        value {
          as_bool: true
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

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
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_array {
            element_type {
              as_int {
                width: 32
                is_signed: true
                is_synthetic: false
              }
            }
            size: 2
          }
        }
        value {
          as_array {
            element_values {
              as_int {
                big_endian_bytes: "\000\000\000\000"
              }
            }
            element_values {
              as_int {
                big_endian_bytes: "\001\000\000\000"
              }
            }
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticStruct) {
  const std::string content = R"(
    struct Something {
      int x;
      char y;
    };

    #pragma hls_top
    int my_package() {
      static Something foo = {3, 11};
      return foo.x;
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Something"
              fully_qualified_name: "Something"
            }
          }
        }
        fields {
          name: "y"
          type {
            as_int {
              width: 8
              is_signed: true
              is_declared_as_char: true
              is_synthetic: false
            }
          }
        }
        fields {
          name: "x"
          type {
            as_int {
              width: 32
              is_signed: true
              is_synthetic: false
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "foo"
          fully_qualified_name: "foo"
        }
        type {
          as_inst {
            name {
              name: "Something"
              fully_qualified_name: "Something"
            }
          }
        }
        value {
          as_struct {
            name {
            }
            fields {
              name {
                name: "x"
                fully_qualified_name: "x"
              }
              type {
              }
              value {
                as_int {
                  big_endian_bytes: "\003\000\000\000"
                }
              }
            }
            fields {
              name {
                name: "y"
                fully_qualified_name: "y"
              }
              type {
              }
              value {
                as_int {
                  big_endian_bytes: "\013"
                }
              }
            }
            no_tuple: false
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

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
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "x"
          fully_qualified_name: "x"
        }
        type {
          as_array {
            element_type {
              as_array {
                element_type {
                  as_int {
                    width: 32
                    is_signed: true
                    is_synthetic: false
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
                    big_endian_bytes: "\000\000\000\000"
                  }
                }
                element_values {
                  as_int {
                    big_endian_bytes: "\001\000\000\000"
                  }
                }
              }
            }
            element_values {
              as_array {
                element_values {
                  as_int {
                    big_endian_bytes: "\002\000\000\000"
                  }
                }
                element_values {
                  as_int {
                    big_endian_bytes: "\003\000\000\000"
                  }
                }
              }
            }
          }
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

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
    top_func_proto 	 {
      name {
        name: "my_package"
        fully_qualified_name: "my_package"
        xls_name: "my_package"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      params {
        name: "sc"
        type {
          as_int {
            width: 8
            is_signed: true
            is_declared_as_char: false
            is_synthetic: false
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
            is_synthetic: false
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
            is_synthetic: false
          }
        }
        is_reference: false
        is_const: false
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, SyntheticInt) {
  const std::string content = R"(
    template<int W>
    struct Base {
      __xls_bits<W> val_;
    };

    #pragma hls_synthetic_int
    template<int W, int S>
    struct Blah : Base<W> {
    };
    #pragma hls_top
    int i_am_top(Blah<17, false> a) {
      (void)a;
      return 1;
    }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
        xls_name: "i_am_top"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      params {
        name: "a"
        type {
          as_int {
            width: 17
            is_signed: false
            is_synthetic: true
          }
        }
        is_reference: false
        is_const: false
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticSyntheticInt) {
  const std::string content = R"(
    #pragma hls_synthetic_int
    template<int W, bool S>
    struct Base {
      __xls_bits<W> val_;
    };

    #pragma hls_synthetic_int
    template<int W, bool S>
    struct Blah : Base<W, S> {
      Blah(int val) {
        asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_1_(aid): bits[i] = "
            "identity(a, pos=(loc)) }"
            : "=r"(this->val_)
            : "i"(32), "a"(val));
      }
    };
    #pragma hls_top
    int i_am_top() {
      static Blah<32, true> foo(034);
      return 1;
    }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
  const std::string ref_meta_str = R"(
    top_func_proto 	 {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
        xls_name: "i_am_top"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "foo"
          fully_qualified_name: "foo"
        }
        type {
          as_int {
            width: 32
            is_signed: true
            is_synthetic: true
          }
        }
        value {
          as_bits: "\000\000\000\034"
        }
      }
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, ReturnReference) {
  const std::string content = R"(
    struct Obj {
      short a;

    #pragma hls_top
      Obj& me() {
        return *this;
      }

      short val() const {
        return a;
      }
    };)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  standardizeMetadata(&meta);

  const std::string ref_meta_str = R"(
    structs 	 {
      as_struct {
        name {
          as_inst {
            name {
              name: "Obj"
              fully_qualified_name: "Obj"
            }
          }
        }
        fields {
          name: "a"
          type {
            as_int {
              width: 16
              is_signed: true
              is_synthetic: false
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "me"
        fully_qualified_name: "Obj::me"
        xls_name: "me"
      }
      return_type {
        as_inst {
          name {
            name: "Obj"
            fully_qualified_name: "Obj"
          }
        }
      }
      this_type {
        as_inst {
          name {
            name: "Obj"
            fully_qualified_name: "Obj"
          }
        }
      }
      is_const: false
      is_method: true
      returns_reference: true
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, StaticBits) {
  const std::string content = R"(
    #pragma hls_no_tuple
    template<int W, bool S>
    struct Base {
      __xls_bits<W> val_;
    };

    #pragma hls_no_tuple
    template<int W, bool S>
    struct Blah : Base<W, S> {
      Blah(int val) {
        asm("fn (fid)(a: bits[i]) -> bits[i] { ret op_1_(aid): bits[i] = "
            "identity(a, pos=(loc)) }"
            : "=r"(this->val_)
            : "i"(32), "a"(val));
      }
    };
    #pragma hls_top
    int i_am_top() {
      static Blah<32, true> foo(034);
      return 1;
    }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  standardizeMetadata(&meta);

  const std::string ref_meta_str = R"(
    structs      {
      as_struct {
        name {
          as_inst {
            name {
              name: "Base"
              fully_qualified_name: "Base"
            }
            args {
              as_integral: 32
            }
            args {
              as_integral: 1
            }
          }
        }
        fields {
          name: "val_"
          type {
            as_inst {
              name {
                name: "__xls_bits"
                fully_qualified_name: "__xls_bits"
              }
              args {
                as_integral: 32
              }
            }
          }
        }
        no_tuple: true
      }
    }
    structs {
      as_struct {
        name {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "Blah"
            }
            args {
              as_integral: 32
            }
            args {
              as_integral: 1
            }
          }
        }
        fields {
          name: "Base"
          type {
            as_inst {
              name {
                name: "Base"
                fully_qualified_name: "Base"
              }
              args {
                as_integral: 32
              }
              args {
                as_integral: 1
              }
            }
          }
        }
        no_tuple: true
      }
    }
    top_func_proto {
      name {
        name: "i_am_top"
        fully_qualified_name: "i_am_top"
        xls_name: "i_am_top"
      }
      return_type {
        as_int {
          width: 32
          is_signed: true
          is_synthetic: false
        }
      }
      static_values {
        name {
          name: "foo"
          fully_qualified_name: "foo"
        }
        type {
          as_inst {
            name {
              name: "Blah"
              fully_qualified_name: "Blah"
            }
            args {
              as_integral: 32
            }
            args {
              as_integral: 1
            }
          }
        }
        value {
          as_struct {
            name {
            }
            fields {
              name {
                name: "Base"
                fully_qualified_name: "Base"
              }
              type {
              }
              value {
                as_struct {
                  name {
                  }
                  fields {
                    name {
                      name: "val_"
                      fully_qualified_name: "val_"
                    }
                    type {
                    }
                    value {
                      as_bits: "\000\000\000\034"
                    }
                  }
                  no_tuple: true
                }
              }
            }
            no_tuple: true
          }
        }
      }
    }

  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, EnumMember) {
  const std::string content = R"(
    enum MyEnum {
      A = -1,
      B = 7,
      C = 100,
      D = 100,
    };
    class MyBlock {
      MyEnum x = MyEnum::C;
      #pragma hls_top
      void Run() {
        (void)x;
      }
    };
    )";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());

  const std::string ref_meta_str = R"(
    structs          {
      as_struct {
        name {
          as_inst {
            name {
              name: "MyBlock"
              fully_qualified_name: "MyBlock"
            }
          }
        }
        fields {
          name: "x"
          type {
            as_enum {
              name: "MyEnum"
              width: 8
              is_signed: true
              variants {
                name: "A"
                value: -1
              }
              variants {
                name: "B"
                value: 7
              }
              variants {
                name: "C"
                name: "D"
                value: 100
              }
            }
          }
        }
        no_tuple: false
      }
    }
    top_func_proto {
      name {
        name: "Run"
        fully_qualified_name: "MyBlock::Run"
        xls_name: "Run"
      }
      return_type {
        as_void {
        }
      }
      this_type {
        as_inst {
          name {
            name: "MyBlock"
            fully_qualified_name: "MyBlock"
          }
        }
      }
      is_const: false
      is_method: true
    }
  )";
  xlscc_metadata::MetadataOutput ref_meta;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(ref_meta_str, &ref_meta));

  standardizeMetadata(&meta);

  std::string diff;
  google::protobuf::util::MessageDifferencer differencer;
  differencer.ReportDifferencesToString(&diff);
  ASSERT_TRUE(differencer.Compare(meta, ref_meta)) << diff;
}

TEST_F(TranslatorMetadataTest, NoContextCrash) {
  const std::string content = R"(
      #pragma hls_top
      void Run(__xls_channel<int>& in,
               __xls_channel<int>& out) {
        int din = in.read();
        out.write(din);
      }
    )";

  XLS_ASSERT_OK_AND_ASSIGN(std::string ir, SourceToIr(content, nullptr));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
}

TEST_F(TranslatorMetadataTest, XlsFixedBitIndexNoCrash) {
  const std::string content = R"(
    #include "xls_fixed.h"
    long long my_package(long long a) {
      XlsFixed<16, 8, true> x = 0;
      XlsFixed<16, 8, true> y = 0;
      x[0] = y[0];
      return x.to_int();
    })";
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::string> clang_args,
                           GetClangArgForIntTest());
  std::vector<std::string_view> clang_argv(clang_args.begin(),
                                           clang_args.end());
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir,
                           SourceToIr(content, nullptr, clang_argv));

  XLS_ASSERT_OK_AND_ASSIGN(xlscc_metadata::MetadataOutput meta,
                           translator_->GenerateMetadata());
}

}  // namespace

}  // namespace xlscc
