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
#include <optional>
#include <string>
#include <string_view>

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

const AstNodeTypeInfoProto* FindSumTypeInfoNode(const TypeInfoProto& tip,
                                                std::string_view identifier,
                                                ImportData& import_data) {
  for (const AstNodeTypeInfoProto& node : tip.nodes()) {
    if (!node.has_type() || !node.type().has_sum_type()) {
      continue;
    }
    const SumTypeProto& sum_type = node.type().sum_type();
    if (!sum_type.has_sum_def_span()) {
      continue;
    }
    auto sum_def = import_data.FindSumDef(
        FromProto(sum_type.sum_def_span(), import_data.file_table()));
    if (sum_def.ok() && (*sum_def)->identifier() == identifier) {
      return &node;
    }
  }
  return nullptr;
}

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

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest, MakeSumFunction) {
  std::string program = R"(
enum Option {
  None,
  Some(u32),
}
fn f() -> Option { Option::None }
)";
  DoRun(program);
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       RejectsReorderedSumVariantsInToHumanString) {
  std::string program = R"(
enum Option {
  None,
  Some(u32),
}
fn f() -> Option { Option::None }
)";

  ImportData import_data = CreateImportDataForTest();
  TypeInfoProto tip;
  DoRun(program, &tip, &import_data);

  int mutated_nodes = 0;
  for (AstNodeTypeInfoProto& node : *tip.mutable_nodes()) {
    if (!node.has_type() || !node.type().has_sum_type()) {
      continue;
    }
    SumTypeProto* sum_type = node.mutable_type()->mutable_sum_type();
    if (!sum_type->has_sum_def_span()) {
      continue;
    }
    ASSERT_EQ(sum_type->variants_size(), 2);
    sum_type->mutable_variants()->SwapElements(0, 1);
    ++mutated_nodes;
  }
  ASSERT_GT(mutated_nodes, 0);

  EXPECT_THAT(ToHumanString(tip, import_data, import_data.file_table()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       SemanticSumEmptyPayloadShapes) {
  std::string program = R"(
enum E {
  None,
  EmptyTuple(),
  EmptyStruct {},
  Some(u32),
  Point { x: u32 },
}

fn f(x: bool) -> E {
  if x { E::EmptyTuple() } else { E::EmptyStruct {} }
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", &import_data, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip,
                           TypeInfoToProto(*tm.type_info, tm.module));

  const AstNodeTypeInfoProto* sum_node =
      FindSumTypeInfoNode(tip, "E", import_data);
  ASSERT_NE(sum_node, nullptr);
  const SumTypeProto& sum_type = sum_node->type().sum_type();
  ASSERT_TRUE(sum_type.has_sum_def_span());
  ASSERT_EQ(sum_type.variants_size(), 5);
  EXPECT_EQ(sum_type.variants(1).payload_members_size(), 0);
  EXPECT_EQ(sum_type.variants(2).payload_members_size(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(
      const SumDef* sum_def,
      import_data.FindSumDef(
          FromProto(sum_type.sum_def_span(), import_data.file_table())));
  ASSERT_EQ(sum_def->variants().size(), 5);
  EXPECT_EQ(sum_def->variants().at(1)->identifier(), "EmptyTuple");
  EXPECT_TRUE(sum_def->variants().at(1)->is_tuple());
  EXPECT_EQ(sum_def->variants().at(2)->identifier(), "EmptyStruct");
  EXPECT_TRUE(sum_def->variants().at(2)->is_struct());
}

TEST_F(TypeInfoToProtoWithBothTypecheckVersionsTest,
       RejectsReorderedSumVariantsInProtoImport) {
  std::string program = R"(
enum Option {
  None,
  Some(u32),
}

fn f(x: bool) -> Option {
  if x { Option::None } else { Option::Some(u32:42) }
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", &import_data, nullptr));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto tip,
                           TypeInfoToProto(*tm.type_info, tm.module));

  const AstNodeTypeInfoProto* sum_node =
      FindSumTypeInfoNode(tip, "Option", import_data);
  ASSERT_NE(sum_node, nullptr);

  for (AstNodeTypeInfoProto& node : *tip.mutable_nodes()) {
    if (!node.has_type() || !node.type().has_sum_type()) {
      continue;
    }
    SumTypeProto* sum_type = node.mutable_type()->mutable_sum_type();
    if (!sum_type->has_sum_def_span()) {
      continue;
    }
    sum_type->mutable_variants()->SwapElements(0, 1);
  }

  EXPECT_THAT(
      ToHumanString(*sum_node, import_data, import_data.file_table()),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr("Sum variant payload member count mismatch")));
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
  std::string program = R"(
enum E {
  None,
  A(u8),
  B(u16),
  Pair { first: u8, second: u16 },
}

fn f() -> E { E::Pair { first: u8:1, second: u16:2 } }
)";
  ImportData import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(program, "fake.x", "fake", &import_data, nullptr));
  ASSERT_EQ(tm.module->GetSumDefs().size(), 1);
  std::optional<Type*> nominal_type =
      tm.type_info->GetItem(tm.module->GetSumDefs().front());
  ASSERT_TRUE(nominal_type.has_value());
  ASSERT_TRUE((*nominal_type)->IsMeta());
  ASSERT_TRUE((*nominal_type)->AsMeta().wrapped()->IsSum());

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
  ASSERT_EQ(sum.variants_size(), 4);
  EXPECT_EQ(sum.variants(0).payload_members_size(), 0);
  EXPECT_EQ(sum.variants(1).payload_members_size(), 1);
  EXPECT_EQ(sum.variants(2).payload_members_size(), 1);
  ASSERT_EQ(sum.variants(3).payload_members_size(), 2);
  EXPECT_TRUE(sum.variants(3).payload_members(0).has_bits_type());
  EXPECT_TRUE(sum.variants(3).payload_members(1).has_bits_type());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string human,
      ToHumanString(parsed, import_data, import_data.file_table()));
  EXPECT_THAT(human,
              ::testing::HasSubstr("Pair { first: uN[8], second: uN[16] }"));

  TypeInfoProto swapped_payload_types = parsed;
  swapped_payload_types.mutable_nodes(sum_index)
      ->mutable_type()
      ->mutable_sum_type()
      ->mutable_variants()
      ->SwapElements(1, 2);
  EXPECT_THAT(
      ToHumanString(swapped_payload_types, import_data,
                    import_data.file_table()),
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
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_SUM_DEF), 78);
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_SUM_VARIANT), 79);
  EXPECT_EQ(static_cast<int>(AST_NODE_KIND_SUM_INSTANCE), 80);
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
