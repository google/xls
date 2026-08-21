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

#include "xls/dslx/type_system/type_info_to_proto.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "re2/re2.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_info.pb.h"

namespace xls::dslx {
namespace {

constexpr int kLegacyNameDefTreeAstNodeKindProtoValue = 21;

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

class TypeInfoToProtoWithBothTypecheckVersionsTest : public ::testing::Test {
 public:
  void DoRun(std::string_view program, TypeInfoProto* proto_out = nullptr,
             ImportData* import_data = nullptr) {
    std::optional<ImportData> local_import_data;
    if (import_data == nullptr) {
      local_import_data.emplace(CreateImportDataForTest());
      import_data = &local_import_data.value();
    }
    XLS_ASSERT_OK_AND_ASSIGN(
        TypecheckedModule tm,
        ParseAndTypecheck(program, "fake.x", "fake", import_data, nullptr));

    XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip,
                             TypeInfoToProto(*tm.type_info, tm.module));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string nodes_text,
        ToHumanString(tip, *import_data, import_data->file_table()));

    std::string test_name(TestName());
    // Remove parametric test suite suffix.
    RE2::GlobalReplace(&test_name, R"(/\d+)", "");

    std::filesystem::path golden_file_path = absl::StrFormat(
        "xls/dslx/type_system/testdata/type_info_to_proto_test_%s.txt",
        test_name);
    ExpectEqualToGoldenFile(golden_file_path, nodes_text);

    if (proto_out != nullptr) {
      *proto_out = tip;
    }
  }
};

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, IdentityFunction) {
  std::string program = R"(fn id(x: u32) -> u32 { x })";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       ParametricIdentityFunction) {
  std::string program = R"(
fn pid<N: u32>(x: bits[N]) -> bits[N] { x }
fn id(x: u32) -> u32 { pid<u32:32>(x) }
)";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, UnitFunction) {
  std::string program = R"(fn f() -> () { () })";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, ArrayFunction) {
  std::string program = R"(fn f() -> u8[2] { u8[2]:[u8:1, u8:2] })";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, TokenFunction) {
  std::string program = R"(fn f(x: token) -> token { x })";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       MakeStructInstanceFunction) {
  std::string program = R"(
struct S { x: u32 }
fn f() -> S { S { x: u32:42 } }
)";
  TypeInfoProto tip;
  DoRun(program, &tip);
  EXPECT_THAT(
      tip.ShortDebugString(),
      ::testing::ContainsRegex(
          R"(struct_def \{ span \{ .*? \} identifier: "S" member_names: "x" is_public: false \})"));
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, MakeEnumFunction) {
  std::string program = R"(
enum E : u32 { A = 42 }
fn f() -> E { E::A }
)";
  ImportData import_data = CreateImportDataForTest();
  TypeInfoProto proto;
  DoRun(program, &proto, &import_data);

  int enum_index = -1;
  for (int i = 0; i < proto.nodes_size(); ++i) {
    const AstNodeTypeInfoProto& node = proto.nodes(i);
    if (node.type().has_enum_type()) {
      const EnumTypeProto& enum_type = node.type().enum_type();
      EXPECT_EQ(enum_type.members_size(), 0);
      enum_index = i;
    }
  }
  ASSERT_GE(enum_index, 0);

  XLS_ASSERT_OK(ToHumanString(proto, import_data, import_data.file_table()));

  TypeInfoProto populated = proto;
  InterpValueProto* member = populated.mutable_nodes(enum_index)
                                 ->mutable_type()
                                 ->mutable_enum_type()
                                 ->add_members();
  member->mutable_bits()->set_bit_count(32);
  member->mutable_bits()->set_is_signed(false);
  member->mutable_bits()->set_data(std::string("\0\0\0*", 4));
  XLS_ASSERT_OK(
      ToHumanString(populated, import_data, import_data.file_table()));

  TypeInfoProto extra_member = populated;
  EnumTypeProto* extra_enum = extra_member.mutable_nodes(enum_index)
                                  ->mutable_type()
                                  ->mutable_enum_type();
  *extra_enum->add_members() = extra_enum->members(0);
  EXPECT_THAT(
      ToHumanString(extra_member, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Enum member count mismatch")));

  TypeInfoProto wrong_signedness = populated;
  wrong_signedness.mutable_nodes(enum_index)
      ->mutable_type()
      ->mutable_enum_type()
      ->mutable_members(0)
      ->mutable_bits()
      ->set_is_signed(true);
  EXPECT_THAT(
      ToHumanString(wrong_signedness, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Enum member type mismatch")));

  TypeInfoProto wrong_value = populated;
  wrong_value.mutable_nodes(enum_index)
      ->mutable_type()
      ->mutable_enum_type()
      ->mutable_members(0)
      ->mutable_bits()
      ->set_data(std::string("\0\0\0+", 4));
  EXPECT_THAT(ToHumanString(wrong_value, import_data, import_data.file_table()),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  ::testing::HasSubstr("Enum member value mismatch")));
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       SemanticSumSchemaStoresOnlyConcreteTypeFacts) {
  EXPECT_EQ(SumTypeProto::descriptor()->field_count(), 2);
  EXPECT_EQ(SumTypeVariantProto::descriptor()->field_count(), 1);
  EXPECT_EQ(SumTypeProto::kSumDefSpanFieldNumber, 1);
  EXPECT_EQ(SumTypeProto::kVariantsFieldNumber, 2);
  EXPECT_EQ(SumTypeVariantProto::kPayloadMembersFieldNumber, 1);
  EXPECT_EQ(TypeProto::kSumTypeFieldNumber, 13);
  EXPECT_EQ(EnumTypeProto::kMembersFieldNumber, 4);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       RoundTripsSumPayloadTypesUsingCanonicalSourceDeclaration) {
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn id(x: u32) -> u32 { x }", "fake.x", "fake",
                        &import_data, nullptr));
  const Span span = tm.module->span();
  auto* sum_name = tm.module->Make<NameDef>(span, "Option", nullptr);
  auto* none_name = tm.module->Make<NameDef>(span, "None", nullptr);
  auto* some_name = tm.module->Make<NameDef>(span, "Some", nullptr);
  auto* pair_name = tm.module->Make<NameDef>(span, "Pair", nullptr);
  auto* u8_annotation = tm.module->Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      tm.module->GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* u16_annotation = tm.module->Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU16,
      tm.module->GetOrCreateBuiltinNameDef(BuiltinType::kU16));
  auto* none = tm.module->Make<SumVariant>(
      span, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* some = tm.module->Make<SumVariant>(
      span, some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8_annotation},
      std::vector<StructMemberNode*>{});
  std::vector<StructMemberNode*> pair_fields = {
      tm.module->Make<StructMemberNode>(
          span, tm.module->Make<NameDef>(span, "first", nullptr), span,
          u8_annotation),
      tm.module->Make<StructMemberNode>(
          span, tm.module->Make<NameDef>(span, "second", nullptr), span,
          u16_annotation),
  };
  auto* pair = tm.module->Make<SumVariant>(
      span, pair_name, SumVariant::PayloadShape::kStruct,
      std::vector<TypeAnnotation*>{}, pair_fields);
  auto* sum_def = tm.module->Make<SumDef>(
      span, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, some, pair}, /*is_public=*/false);
  sum_name->set_definer(sum_def);
  XLS_ASSERT_OK(tm.module->AddTop(sum_def, /*make_collision_error=*/nullptr));

  std::vector<SumTypeVariant> variants;
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> some_members;
  some_members.push_back(BitsType::MakeU8());
  variants.push_back(SumTypeVariant::MakeTuple(*some, std::move(some_members)));
  std::vector<std::unique_ptr<Type>> pair_members;
  pair_members.push_back(BitsType::MakeU8());
  pair_members.push_back(std::make_unique<BitsType>(false, 16));
  variants.push_back(
      SumTypeVariant::MakeStruct(*pair, std::move(pair_members)));
  tm.type_info->SetItem(
      sum_def, std::make_unique<SumType>(*sum_def, std::move(variants)));

  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto proto,
                           TypeInfoToProto(*tm.type_info, tm.module));
  int sum_index = -1;
  for (int i = 0; i < proto.nodes_size(); ++i) {
    if (proto.nodes(i).type().has_sum_type()) {
      sum_index = i;
      break;
    }
  }
  ASSERT_GE(sum_index, 0);

  std::string wire = proto.SerializeAsString();
  TypeInfoProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(wire));
  const SumTypeProto& sum = parsed.nodes(sum_index).type().sum_type();
  ASSERT_TRUE(sum.has_sum_def_span());
  EXPECT_EQ(sum.sum_def_span().start().filename(), "fake.x");
  ASSERT_EQ(sum.variants_size(), 3);
  EXPECT_EQ(sum.variants(0).payload_members_size(), 0);
  EXPECT_EQ(sum.variants(1).payload_members_size(), 1);
  ASSERT_EQ(sum.variants(2).payload_members_size(), 2);
  EXPECT_TRUE(sum.variants(2).payload_members(0).has_bits_type());
  EXPECT_TRUE(sum.variants(2).payload_members(1).has_bits_type());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string human,
      ToHumanString(parsed, import_data, import_data.file_table()));
  EXPECT_THAT(human,
              ::testing::HasSubstr("Pair { first: uN[8], second: uN[16] }"));

  TypeInfoProto wrong_payload_type = proto;
  *wrong_payload_type.mutable_nodes(sum_index)
       ->mutable_type()
       ->mutable_sum_type()
       ->mutable_variants(1)
       ->mutable_payload_members(0) =
      proto.nodes(sum_index).type().sum_type().variants(2).payload_members(1);
  EXPECT_THAT(
      ToHumanString(wrong_payload_type, import_data, import_data.file_table()),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("payload type mismatch")));

  TypeInfoProto missing_source_span = proto;
  missing_source_span.mutable_nodes(sum_index)
      ->mutable_type()
      ->mutable_sum_type()
      ->clear_sum_def_span();
  EXPECT_THAT(
      ToHumanString(missing_source_span, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("missing its source definition span")));

  TypeInfoProto missing_payload = proto;
  missing_payload.mutable_nodes(sum_index)
      ->mutable_type()
      ->mutable_sum_type()
      ->mutable_variants(1)
      ->clear_payload_members();
  EXPECT_THAT(
      ToHumanString(missing_payload, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Sum variant payload member count mismatch")));

  TypeInfoProto missing_variant = proto;
  missing_variant.mutable_nodes(sum_index)
      ->mutable_type()
      ->mutable_sum_type()
      ->mutable_variants()
      ->RemoveLast();
  EXPECT_THAT(
      ToHumanString(missing_variant, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Sum variant count mismatch")));

  TypeInfoProto reordered_typed_variants = proto;
  reordered_typed_variants.mutable_nodes(sum_index)
      ->mutable_type()
      ->mutable_sum_type()
      ->mutable_variants()
      ->SwapElements(0, 1);
  EXPECT_THAT(
      ToHumanString(reordered_typed_variants, import_data,
                    import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Sum variant payload member count mismatch")));

  TypeInfoProto meta_payload = proto;
  TypeProto* payload_member = meta_payload.mutable_nodes(sum_index)
                                  ->mutable_type()
                                  ->mutable_sum_type()
                                  ->mutable_variants(1)
                                  ->mutable_payload_members(0);
  TypeProto original_member = *payload_member;
  payload_member->clear_type_oneof();
  *payload_member->mutable_meta_type()->mutable_wrapped() = original_member;
  EXPECT_THAT(
      ToHumanString(meta_payload, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("invalid meta-type payload member")));
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       ImportModuleAndTypeAliasAnEnum) {
  std::string imported = R"(
pub enum Foo : u32 {
  A = 42,
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(imported, "my_imported_module.x", "my_imported_module",
                        &import_data));
  (void)tm;

  std::string program = R"(
import my_imported_module;

type MyFoo = my_imported_module::Foo;
)";
  DoRun(program, /*proto_out=*/nullptr, &import_data);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, ProcWithImpl) {
  std::string program = R"(
proc Foo { a: u32 }
)";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, BitsConstructorTypeProto) {
  std::string program = R"(
fn distinct<COUNT: u32, N: u32, S: bool>(items: xN[S][N][COUNT], valid: bool[COUNT]) -> bool { fail!("unimplemented", zero!<bool>()) }

#[test]
fn test_simple_nondistinct() {
    assert_eq(distinct(u2[2]:[1, 1], bool[2]:[true, true]), false)
}
)";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       SkipsSyntheticNoFileEntriesInHumanizedOutput) {
  std::string program = R"(
fn bool_update() -> bool[1] {
  update(bool[1]:[false], u1:0, true)
}

fn bit_update() -> u8 {
  bit_slice_update(u8:0, u3:0, true)
}
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", &import_data, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip,
                           TypeInfoToProto(*tm.type_info, tm.module));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string nodes_text,
      ToHumanString(tip, import_data, import_data.file_table()));

  EXPECT_THAT(nodes_text,
              ::testing::HasSubstr("update(bool[1]:[false], u1:0, true)"));
  EXPECT_THAT(nodes_text,
              ::testing::HasSubstr("bit_slice_update(u8:0, u3:0, true)"));
  EXPECT_THAT(nodes_text, ::testing::Not(::testing::HasSubstr("<no-file>")));
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       TuplePatternUsesDistinctAstNodeKind) {
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_TUPLE_PATTERN), 76);
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_SUM_VARIANT_PAYLOAD_PATTERN), 77);
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_STRUCT_PATTERN), 81);

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck("fn f() -> u32 { let (x, y) = (u32:1, u32:2); x }",
                        "fake.x", "fake", &import_data, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip,
                           TypeInfoToProto(*tm.type_info, tm.module));
  XLS_ASSERT_OK(ToHumanString(tip, import_data, import_data.file_table()));

  bool found_tuple_pattern = false;
  for (const AstNodeTypeInfoProto& node : tip.nodes()) {
    found_tuple_pattern |= node.kind() == AST_NODE_KIND_TUPLE_PATTERN;
    EXPECT_NE(static_cast<int>(node.kind()),
              kLegacyNameDefTreeAstNodeKindProtoValue);
  }
  EXPECT_TRUE(found_tuple_pattern);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       RejectsLegacyNameDefTreeAstNodeKind) {
  ImportData import_data = CreateImportDataForTest();
  AstNodeTypeInfoProto legacy;
  legacy.set_kind(
      static_cast<AstNodeKindProto>(kLegacyNameDefTreeAstNodeKindProtoValue));
  legacy.mutable_type()->mutable_token_type();

  EXPECT_THAT(
      ToHumanString(legacy, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Legacy NameDefTree type-info entries")));
}

}  // namespace
}  // namespace xls::dslx
