// Copyright 2024 The XLS Authors
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

#include "xls/dslx/diagnostics/format_type_mismatch.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

// Macro definitions so we can use C style string concatenation where we can
// write/see the escapes more easily.
#define ANSI_RESET "\33[0m"
#define ANSI_RED "\33[31m"
#define ANSI_BOLD "\33[1m"
#define ANSI_UNBOLD "\33[22m"

TEST(FormatTypeMismatchTest, SumPayloadMismatch) {
  const Span kFakeSpan = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* option_name = module.Make<NameDef>(kFakeSpan, "Option", nullptr);
  auto* none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* some_name = module.Make<NameDef>(kFakeSpan, "Some", nullptr);
  auto* u8_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8));
  auto* u16_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU16,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU16));

  auto* none = module.Make<SumVariant>(
      kFakeSpan, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* some = module.Make<SumVariant>(
      kFakeSpan, some_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8_type, u16_type},
      std::vector<StructMemberNode*>{});
  auto* option = module.Make<SumDef>(
      kFakeSpan, option_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, some}, /*is_public=*/false);
  option_name->set_definer(option);

  std::vector<SumTypeVariant> lhs_variants;
  lhs_variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> lhs_some_members;
  lhs_some_members.push_back(BitsType::MakeU8());
  lhs_some_members.push_back(std::make_unique<BitsType>(/*is_signed=*/false,
                                                        /*size=*/16));
  lhs_variants.push_back(
      SumTypeVariant::MakeTuple(*some, std::move(lhs_some_members)));

  std::vector<SumTypeVariant> rhs_variants;
  rhs_variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> rhs_some_members;
  rhs_some_members.push_back(BitsType::MakeU8());
  rhs_some_members.push_back(BitsType::MakeU32());
  rhs_variants.push_back(
      SumTypeVariant::MakeTuple(*some, std::move(rhs_some_members)));

  SumType lhs(*option, std::move(lhs_variants));
  SumType rhs(*option, std::move(rhs_variants));
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(lhs, rhs, file_table));

  EXPECT_EQ(got,
            ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                       " type:\n"                                 //
                       "   uN[16]\n"                              //
                       "vs uN[32]\n" ANSI_BOLD                    //
                       "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   Option { None | Some(uN[8], " ANSI_RED
                       "uN[16]" ANSI_RESET
                       ") }\n"  //
                       "vs Option { None | Some(uN[8], " ANSI_RED
                       "uN[32]" ANSI_RESET ") }");
}

TEST(FormatTypeMismatchTest, NestedAggregateSumPayloadMismatch) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* option_name = module.Make<NameDef>(span, "Option", nullptr);
  auto* some_name = module.Make<NameDef>(span, "Some", nullptr);
  auto* annotation = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* some =
      module.Make<SumVariant>(span, some_name, SumVariant::PayloadShape::kTuple,
                              std::vector<TypeAnnotation*>{annotation},
                              std::vector<StructMemberNode*>{});
  auto* option =
      module.Make<SumDef>(span, option_name, std::vector<ParametricBinding*>{},
                          std::vector<SumVariant*>{some}, /*is_public=*/false);
  option_name->set_definer(option);

  std::vector<std::unique_ptr<Type>> lhs_members;
  lhs_members.push_back(TupleType::Create2(
      BitsType::MakeU8(), std::make_unique<BitsType>(false, 16)));
  std::vector<SumTypeVariant> lhs_variants;
  lhs_variants.push_back(
      SumTypeVariant::MakeTuple(*some, std::move(lhs_members)));
  SumType lhs(*option, std::move(lhs_variants));

  std::vector<std::unique_ptr<Type>> rhs_members;
  rhs_members.push_back(
      TupleType::Create2(BitsType::MakeU8(), BitsType::MakeU32()));
  std::vector<SumTypeVariant> rhs_variants;
  rhs_variants.push_back(
      SumTypeVariant::MakeTuple(*some, std::move(rhs_members)));
  SumType rhs(*option, std::move(rhs_variants));

  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(lhs, rhs, file_table));
  EXPECT_NE(got.find("Option { Some((uN[8], "), std::string::npos);
  EXPECT_NE(got.find("uN[16]" ANSI_RESET ")) }"), std::string::npos);
  EXPECT_NE(got.find("uN[32]" ANSI_RESET ")) }"), std::string::npos);
}

TEST(FormatTypeMismatchTest,
     StructSumPayloadMismatchPreservesTrailingEmptyConstructors) {
  const Span span = Span::Fake();
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  auto* option_name = module.Make<NameDef>(span, "Option", nullptr);
  auto* pair_name = module.Make<NameDef>(span, "Pair", nullptr);
  auto* empty_tuple_name = module.Make<NameDef>(span, "EmptyTuple", nullptr);
  auto* empty_struct_name = module.Make<NameDef>(span, "EmptyStruct", nullptr);
  auto* none_name = module.Make<NameDef>(span, "None", nullptr);
  auto* u8 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU8));
  auto* u16 = module.Make<BuiltinTypeAnnotation>(
      span, BuiltinType::kU16,
      module.GetOrCreateBuiltinNameDef(BuiltinType::kU16));
  std::vector<StructMemberNode*> fields = {
      module.Make<StructMemberNode>(
          span, module.Make<NameDef>(span, "left", nullptr), span, u8),
      module.Make<StructMemberNode>(
          span, module.Make<NameDef>(span, "right", nullptr), span, u16),
  };
  auto* pair = module.Make<SumVariant>(span, pair_name,
                                       SumVariant::PayloadShape::kStruct,
                                       std::vector<TypeAnnotation*>{}, fields);
  auto* empty_tuple = module.Make<SumVariant>(
      span, empty_tuple_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* empty_struct = module.Make<SumVariant>(
      span, empty_struct_name, SumVariant::PayloadShape::kStruct,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* none = module.Make<SumVariant>(
      span, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* option = module.Make<SumDef>(
      span, option_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{pair, empty_tuple, empty_struct, none},
      /*is_public=*/false);
  option_name->set_definer(option);

  auto make_type = [&](std::unique_ptr<Type> right) {
    std::vector<std::unique_ptr<Type>> pair_members;
    pair_members.push_back(BitsType::MakeU8());
    pair_members.push_back(std::move(right));
    std::vector<SumTypeVariant> variants;
    variants.push_back(
        SumTypeVariant::MakeStruct(*pair, std::move(pair_members)));
    variants.push_back(SumTypeVariant::MakeTuple(
        *empty_tuple, std::vector<std::unique_ptr<Type>>{}));
    variants.push_back(SumTypeVariant::MakeStruct(
        *empty_struct, std::vector<std::unique_ptr<Type>>{}));
    variants.push_back(SumTypeVariant::MakeUnit(*none));
    return SumType(*option, std::move(variants));
  };
  SumType lhs = make_type(std::make_unique<BitsType>(false, 16));
  SumType rhs = make_type(BitsType::MakeU32());

  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(lhs, rhs, file_table));
  EXPECT_NE(got.find("Pair { left: uN[8], right: " ANSI_RED "uN[16]"),
            std::string::npos);
  EXPECT_NE(got.find("Pair { left: uN[8], right: " ANSI_RED "uN[32]"),
            std::string::npos);
  EXPECT_NE(got.find("EmptyTuple() | EmptyStruct {} | None }"),
            std::string::npos);
}

TEST(FormatTypeMismatchTest, ElementInTuple) {
  auto t0 = TupleType::Create3(BitsType::MakeU8(),
                               std::make_unique<BitsType>(false, 16),
                               BitsType::MakeU32());
  auto t1 = TupleType::Create3(BitsType::MakeU8(),
                               std::make_unique<BitsType>(true, 16),
                               BitsType::MakeU32());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(
      got,
      ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                 " type:\n"     //
                 "   uN[16]\n"  //
                 "vs sN[16]\n" ANSI_BOLD "Overall" ANSI_UNBOLD
                 " type mismatch:\n"  //
      ANSI_RESET "   (uN[8], " ANSI_RED "uN[16]" ANSI_RESET
                 ", uN[32])\n"                                           //
                 "vs (uN[8], " ANSI_RED "sN[16]" ANSI_RESET ", uN[32])"  //
  );
}

TEST(FormatTypeMismatchTest, NestedTuple) {
  std::unique_ptr<TupleType> t0 = TupleType::Create2(
      TupleType::Create2(BitsType::MakeU8(), BitsType::MakeU1()),
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()));
  std::unique_ptr<TupleType> t1 = TupleType::Create2(
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()),
      TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU1()));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET
            "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
            " type:\n"                                                        //
            "   uN[8]\n"                                                      //
            "vs uN[1]\n" ANSI_BOLD "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   ((" ANSI_RED "uN[8]" ANSI_RESET
            ", uN[1]), (uN[1], uN[1]))\n"
            "vs ((" ANSI_RED "uN[1]" ANSI_RESET ", uN[1]), (uN[1], uN[1]))");
}

TEST(FormatTypeMismatchTest, ElementTypeInArrayInTuple) {
  auto t0 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(4)));
  auto t1 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeS32(), TypeDim::CreateU32(4)));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                       " type:\n"                                 //
                       "   uN[32]\n"                              //
                       "vs sN[32]\n" ANSI_BOLD                    //
                       "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   (uN[1], " ANSI_RED "uN[32]" ANSI_RESET
                       "[4])\n"                                           //
                       "vs (uN[1], " ANSI_RED "sN[32]" ANSI_RESET "[4])"  //
  );
}

TEST(FormatTypeMismatchTest, MismatchedArraySizeInTuple) {
  auto t0 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(4)));
  auto t1 = TupleType::Create2(
      BitsType::MakeU1(),
      std::make_unique<ArrayType>(BitsType::MakeU32(), TypeDim::CreateU32(2)));

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                       " type:\n"                                 //
                       "   uN[32][4]\n"                           //
                       "vs uN[32][2]\n" ANSI_BOLD                 //
                       "Overall" ANSI_UNBOLD " type mismatch:\n"  //
            ANSI_RESET "   (uN[1], " ANSI_RED "uN[32][4]" ANSI_RESET
                       ")\n"                                              //
                       "vs (uN[1], " ANSI_RED "uN[32][2]" ANSI_RESET ")"  //
  );
}

TEST(FormatTypeMismatchTest, TotallyDifferentTuples) {
  auto t0 = TupleType::Create2(BitsType::MakeU8(), BitsType::MakeU32());
  auto t1 = TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU64());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   (uN[8], uN[32])\n"
            "vs (uN[1], uN[64])");
}

TEST(FormatTypeMismatchTest, TuplesWithSharedPrefixDifferentLength) {
  auto t0 = TupleType::Create3(BitsType::MakeU1(), BitsType::MakeU8(),
                               BitsType::MakeU32());
  auto t1 = TupleType::Create2(BitsType::MakeU1(), BitsType::MakeU8());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(got,
            "Tuple is missing elements:\n"
            "   uN[32] (index 2 of (uN[1], uN[8], uN[32]))\n"
            "Type mismatch:\n"
            "   (uN[1], uN[8], uN[32])\n"
            "vs (uN[1], uN[8])");

  XLS_ASSERT_OK_AND_ASSIGN(got, FormatTypeMismatch(*t1, *t0, file_table));
  EXPECT_EQ(got,
            "Tuple has extra elements:\n"
            "   uN[32] (index 2 of (uN[1], uN[8], uN[32]))\n"
            "Type mismatch:\n"
            "   (uN[1], uN[8])\n"
            "vs (uN[1], uN[8], uN[32])");
}

TEST(FormatTypeMismatchTest, ChannelTypeMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 =
      std::make_unique<ChannelType>(BitsType::MakeU32(), ChannelDirection::kIn);

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*ch0, *ch1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   chan(uN[8], dir=in)\n"
            "vs chan(uN[32], dir=in)");
}

TEST(FormatTypeMismatchTest, ChannelTypeDirectionMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kOut);

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*ch0, *ch1, file_table));

  EXPECT_EQ(got,
            "Type mismatch:\n"
            "   chan(uN[8], dir=in)\n"
            "vs chan(uN[8], dir=out)");
}

TEST(FormatTypeMismatchTest, TupleOfChannelTypesElementMismatch) {
  std::unique_ptr<ChannelType> ch0 =
      std::make_unique<ChannelType>(BitsType::MakeU8(), ChannelDirection::kIn);
  std::unique_ptr<ChannelType> ch1 = std::make_unique<ChannelType>(
      BitsType::MakeU32(), ChannelDirection::kOut);

  std::unique_ptr<TupleType> t0 = TupleType::Create3(
      ch0->CloneToUnique(), ch0->CloneToUnique(), ch1->CloneToUnique());
  std::unique_ptr<TupleType> t1 = TupleType::Create3(
      ch0->CloneToUnique(), ch1->CloneToUnique(), ch1->CloneToUnique());

  FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                           FormatTypeMismatch(*t0, *t1, file_table));

  EXPECT_EQ(
      got,
      ANSI_RESET "Mismatched elements " ANSI_BOLD "within" ANSI_UNBOLD
                 " type:\n"                  //
                 "   chan(uN[8], dir=in)\n"  //
                 "vs chan(uN[32], dir=out)\n" ANSI_BOLD "Overall" ANSI_UNBOLD
                 " type mismatch:\n"  //
      ANSI_RESET "   (chan(uN[8]), " ANSI_RED "chan(uN[8], dir=in)" ANSI_RESET
                 ", chan(uN[32]))\n"  //
                 "vs (chan(uN[8]), " ANSI_RED "chan(uN[32], dir=out)" ANSI_RESET
                 ", chan(uN[32]))"  //
  );
}

}  // namespace
}  // namespace xls::dslx
