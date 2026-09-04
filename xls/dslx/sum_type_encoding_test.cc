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

#include "xls/dslx/sum_type_encoding.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

absl::StatusOr<int64_t> GetBitCount(const Type& type) {
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  if (!bits_like.has_value()) {
    return absl::InvalidArgumentError("Expected bits-like type.");
  }
  return bits_like->size.GetAsInt64();
}

SumType MakeTuplePayloadSumType(Module& module) {
  const Span kFakeSpan = Span::Fake();

  auto* sum_name = module.Make<NameDef>(kFakeSpan, "Example", nullptr);
  auto* none_name = module.Make<NameDef>(kFakeSpan, "None", nullptr);
  auto* left_name = module.Make<NameDef>(kFakeSpan, "Left", nullptr);
  auto* pair_name = module.Make<NameDef>(kFakeSpan, "Pair", nullptr);

  auto* u8_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU8,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU8));
  auto* u16_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU16,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU16));
  auto* u32_type = module.Make<BuiltinTypeAnnotation>(
      kFakeSpan, BuiltinType::kU32,
      module.GetOrCreateBuiltinNameDef(dslx::BuiltinType::kU32));

  auto* none = module.Make<SumVariant>(
      kFakeSpan, none_name, SumVariant::PayloadShape::kUnit,
      std::vector<TypeAnnotation*>{}, std::vector<StructMemberNode*>{});
  auto* left = module.Make<SumVariant>(
      kFakeSpan, left_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u8_type}, std::vector<StructMemberNode*>{});
  auto* pair = module.Make<SumVariant>(
      kFakeSpan, pair_name, SumVariant::PayloadShape::kTuple,
      std::vector<TypeAnnotation*>{u16_type, u32_type},
      std::vector<StructMemberNode*>{});
  auto* sum_def = module.Make<SumDef>(
      kFakeSpan, sum_name, std::vector<ParametricBinding*>{},
      std::vector<SumVariant*>{none, left, pair}, /*is_public=*/false);
  sum_name->set_definer(sum_def);

  std::vector<SumTypeVariant> variants;
  variants.push_back(SumTypeVariant::MakeUnit(*none));
  std::vector<std::unique_ptr<Type>> left_members;
  left_members.push_back(BitsType::MakeU8());
  variants.push_back(SumTypeVariant::MakeTuple(*left, std::move(left_members)));
  std::vector<std::unique_ptr<Type>> pair_members;
  pair_members.push_back(std::make_unique<BitsType>(false, 16));
  pair_members.push_back(BitsType::MakeU32());
  variants.push_back(SumTypeVariant::MakeTuple(*pair, std::move(pair_members)));
  return SumType(*sum_def, std::move(variants));
}

TEST(Phase1SumTypeEncodingTest, VisitsPayloadSlotsInDeclarationOrder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeTuplePayloadSumType(module);
  Phase1SumTypeEncoding encoding(sum_type);

  std::vector<int64_t> slot_indexes;
  std::vector<int64_t> bit_counts;
  int64_t slot_index = 0;
  XLS_ASSERT_OK(
      encoding.ForEachPayloadType([&](const Type& type) -> absl::Status {
        slot_indexes.push_back(slot_index++);
        XLS_ASSIGN_OR_RETURN(int64_t bit_count, GetBitCount(type));
        bit_counts.push_back(bit_count);
        return absl::OkStatus();
      }));

  EXPECT_THAT(slot_indexes, ElementsAre(0, 1, 2));
  EXPECT_THAT(bit_counts, ElementsAre(8, 16, 32));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t tag_bit_count, encoding.tag_bit_count());
  XLS_ASSERT_OK_AND_ASSIGN(TypeDim total_bits, sum_type.GetTotalBitCount());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t total_bit_count, total_bits.GetAsInt64());
  EXPECT_EQ(total_bit_count, 58);
  EXPECT_EQ(tag_bit_count, 2);
  std::vector<int64_t> dimensions;
  for (const TypeDim& dimension : sum_type.GetAllDims()) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t bit_count, dimension.GetAsInt64());
    dimensions.push_back(bit_count);
  }
  EXPECT_THAT(dimensions, ElementsAre(2, 8, 16, 32));
}

TEST(Phase1SumTypeEncodingTest, TracksActiveSlotsForLaterVariants) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType sum_type = MakeTuplePayloadSumType(module);
  Phase1SumTypeEncoding encoding(sum_type);

  XLS_ASSERT_OK_AND_ASSIGN(Phase1SumTypeEncoding::VariantInfo pair_variant,
                           encoding.GetVariant("Pair"));
  EXPECT_EQ(pair_variant.variant_index, 2);
  EXPECT_EQ(pair_variant.payload_start, 1);
  EXPECT_EQ(pair_variant.payload_size(), 2);
  EXPECT_EQ(pair_variant.payload_end(), 3);

  std::vector<int64_t> assembly_steps;
  XLS_ASSERT_OK(encoding.VisitPayloadAssemblyOrder(
      pair_variant,
      [&](int64_t active_index) -> absl::Status {
        assembly_steps.push_back(active_index);
        return absl::OkStatus();
      },
      [&](const Type& inactive_type) -> absl::Status {
        XLS_ASSIGN_OR_RETURN(int64_t bit_count, GetBitCount(inactive_type));
        assembly_steps.push_back(-bit_count);
        return absl::OkStatus();
      }));
  EXPECT_THAT(assembly_steps, ElementsAre(-8, 0, 1));

  std::vector<int64_t> active_slot_indexes;
  std::vector<int64_t> active_indexes;
  std::vector<int64_t> active_bit_counts;
  XLS_ASSERT_OK(encoding.ForEachActivePayloadSlot(
      pair_variant,
      [&](int64_t slot_index, int64_t active_index,
          const Type& type) -> absl::Status {
        active_slot_indexes.push_back(slot_index);
        active_indexes.push_back(active_index);
        XLS_ASSIGN_OR_RETURN(int64_t bit_count, GetBitCount(type));
        active_bit_counts.push_back(bit_count);
        return absl::OkStatus();
      }));
  EXPECT_THAT(active_slot_indexes, ElementsAre(1, 2));
  EXPECT_THAT(active_indexes, ElementsAre(0, 1));
  EXPECT_THAT(active_bit_counts, ElementsAre(16, 32));
}

TEST(Phase1SumTypeEncodingTest, RejectsVariantInfoFromDifferentEncoding) {
  FileTable file_table;
  Module local_module("local", /*fs_path=*/std::nullopt, file_table);
  Module foreign_module("foreign", /*fs_path=*/std::nullopt, file_table);
  SumType local_type = MakeTuplePayloadSumType(local_module);
  SumType foreign_type = MakeTuplePayloadSumType(foreign_module);
  Phase1SumTypeEncoding local_encoding(local_type);
  Phase1SumTypeEncoding foreign_encoding(foreign_type);

  XLS_ASSERT_OK_AND_ASSIGN(Phase1SumTypeEncoding::VariantInfo foreign_pair,
                           foreign_encoding.GetVariant("Pair"));
  EXPECT_THAT(local_encoding.ForEachActivePayloadSlot(
                  foreign_pair,
                  [](int64_t, int64_t, const Type&) -> absl::Status {
                    return absl::OkStatus();
                  }),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(local_encoding.VisitPayloadAssemblyOrder(
                  foreign_pair,
                  [](int64_t) -> absl::Status { return absl::OkStatus(); },
                  [](const Type&) -> absl::Status { return absl::OkStatus(); }),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Phase1SumTypeEncodingTest, RejectsSumVariantsOutsideDeclarationOrder) {
  FileTable file_table;
  Module module("test", /*fs_path=*/std::nullopt, file_table);
  SumType valid_type = MakeTuplePayloadSumType(module);

  EXPECT_DEATH(
      {
        std::vector<SumTypeVariant> invalid_variants;
        invalid_variants.push_back(valid_type.variants().at(1).Clone());
        invalid_variants.push_back(valid_type.variants().at(0).Clone());
        invalid_variants.push_back(valid_type.variants().at(2).Clone());
        SumType invalid_type(valid_type.nominal_type(),
                             std::move(invalid_variants));
      },
      "Check failed");
}

}  // namespace
}  // namespace xls::dslx
