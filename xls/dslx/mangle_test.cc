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

#include "xls/dslx/mangle.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;

class TestTypeReferenceData : public InterpValue::TypeReferenceData {
 public:
  explicit TestTypeReferenceData(std::string name) : name_(std::move(name)) {}

  bool IsSameSpecialization(const TypeReferenceData& other) const override {
    const auto* data = dynamic_cast<const TestTypeReferenceData*>(&other);
    return data != nullptr && data->name_ == name_;
  }

  absl::StatusOr<std::string> GetSpecializationName() const override {
    return name_;
  }

 private:
  std::string name_;
};

TEST(MangleTest, SimpleModuleFunction) {
  EXPECT_THAT(MangleDslxName("my_mod", "f", CallingConvention::kTypical, {},
                             ParametricEnv{}),
              IsOkAndHolds("__my_mod__f"));
  EXPECT_THAT(MangleDslxName("my_mod", "f", CallingConvention::kImplicitToken,
                             {}, ParametricEnv{}),
              IsOkAndHolds("__itok__my_mod__f"));
  EXPECT_THAT(MangleDslxName("my_mod", "f->func:0",
                             CallingConvention::kProcNext, {}, ParametricEnv{}),
              IsOkAndHolds("__my_mod__f__func_0_next"));
}

TEST(MangleTest, ModuleFunctionWithScope) {
  EXPECT_THAT(MangleDslxName("my_mod", "f", CallingConvention::kTypical, {},
                             ParametricEnv{}, "scope"),
              IsOkAndHolds("__my_mod__scope__f"));
  EXPECT_THAT(MangleDslxName("my_mod", "f", CallingConvention::kImplicitToken,
                             {}, ParametricEnv{}, "scope"),
              IsOkAndHolds("__itok__my_mod__scope__f"));
}

TEST(MangleTest, ModuleFunctionWithScopeContainingRange) {
  EXPECT_THAT(
      MangleDslxName("my_mod", "f", CallingConvention::kTypical, {},
                     ParametricEnv{},
                     "lambda_capture_struct_at_test_module.x:8:30-8:49"),
      IsOkAndHolds(
          "__my_mod__lambda_capture_struct_at_test_module_x_8_30_8_49__f"));
}

TEST(MangleTest, ModuleFunctionWithScopeContainingCommaAndSpac) {
  EXPECT_THAT(
      MangleDslxName(
          "my_mod", "f", CallingConvention::kTypical, {}, ParametricEnv{},
          "lambda_capture_struct_at_test_module.x:8:30-8:49_48_BITS, ELEMS"),
      IsOkAndHolds("__my_mod__lambda_capture_struct_at_test_module_x_8_30_8_49_"
                   "48_BITS__ELEMS__f"));
}

TEST(MangleTest, SingleFreeKey) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"x", InterpValue::MakeU32(42)}};
  ParametricEnv parametric_env(bindings);
  EXPECT_THAT(MangleDslxName("my_mod", "p", CallingConvention::kTypical, {"x"},
                             parametric_env),
              IsOkAndHolds("__my_mod__p__42"));
}

TEST(MangleTest, TwoFreeKeys) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"x", InterpValue::MakeU32(42)}, {"y", InterpValue::MakeU32(64)}};
  ParametricEnv parametric_env(bindings);
  EXPECT_THAT(MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                             {"x", "y"}, parametric_env),
              IsOkAndHolds("__my_mod__p__42_64"));
}

// Verifies: Array hashes respect interpreter equality and never expand ranges.
// Catches: Hashing a computed length or materializing a large symbolic range.
TEST(MangleTest, ParametricEnvHashesSymbolicRangesWithoutExpansion) {
  auto env = [](InterpValue value) {
    std::vector<std::pair<std::string, InterpValue>> bindings = {
        {"R", std::move(value)}};
    return ParametricEnv(bindings);
  };

  ParametricEnv unsigned_range = env(InterpValue::MakeSymbolicRange(
      InterpValue::MakeU8(128), InterpValue::MakeU8(127)));
  ParametricEnv signed_range = env(InterpValue::MakeSymbolicRange(
      InterpValue::MakeSBits(8, -128), InterpValue::MakeSBits(8, 127)));
  ParametricEnv empty_array = env(InterpValue::MakeArray({}).value());
  ASSERT_EQ(unsigned_range.bindings()[0].value.GetLength().value(), 0);
  ASSERT_EQ(signed_range.bindings()[0].value.GetLength().value(), 255);
  EXPECT_EQ(unsigned_range, signed_range);
  EXPECT_EQ(unsigned_range, empty_array);
  EXPECT_EQ(absl::HashOf(unsigned_range), absl::HashOf(signed_range));
  EXPECT_EQ(absl::HashOf(unsigned_range), absl::HashOf(empty_array));

  ParametricEnv huge_range = env(InterpValue::MakeSymbolicRange(
      InterpValue::MakeU64(0), InterpValue::MakeMaxValue(false, 64)));
  ParametricEnv equivalent_huge_range = huge_range;
  ASSERT_FALSE(huge_range.bindings()[0].value.GetLength().ok());
  EXPECT_EQ(huge_range, equivalent_huge_range);
  EXPECT_EQ(absl::HashOf(huge_range), absl::HashOf(equivalent_huge_range));
  EXPECT_TRUE(huge_range.bindings()[0].value.GetRangeData().has_value());
}

TEST(MangleTest, TypeNamesEncodeDimensionsAndChildrenDirectly) {
  ArrayType array(BitsType::MakeU32(), TypeDim::CreateU32(4));
  EXPECT_THAT(MangleType(array.element_type()), IsOkAndHolds("u32_"));
  EXPECT_THAT(MangleType(array), IsOkAndHolds("a4_u32_"));

  auto tuple = TupleType::Create2(
      array.CloneToUnique(), std::make_unique<BitsType>(/*is_signed=*/true, 8));
  EXPECT_THAT(MangleType(*tuple), IsOkAndHolds("q2_a4_u32_s8_"));
}

class TypedMangleTest : public ::testing::Test {
 protected:
  InterpValue TypeReference(std::string name) {
    auto* annotation = module_.Make<BuiltinTypeAnnotation>(
        Span::Fake(), BuiltinType::kU32,
        module_.GetOrCreateBuiltinNameDef(BuiltinType::kU32));
    return InterpValue::MakeTypeReference(
        annotation, std::make_shared<TestTypeReferenceData>(std::move(name)));
  }

  FileTable file_table_;
  Module module_{"my_mod", /*fs_path=*/std::nullopt, file_table_};
};

TEST_F(TypedMangleTest, NominalNameFramesIdentifiersAndSortsBindings) {
  auto* def = module_.Make<StructDef>(
      Span::Fake(), module_.Make<NameDef>(Span::Fake(), "S", nullptr),
      std::vector<ParametricBinding*>{}, std::vector<StructMemberNode*>{},
      /*is_public=*/false);
  StructType type({}, *def);
  type.SetNominalTypeBindings(
      {{"T", TypeReference("u32_")}, {"N", InterpValue::MakeU8(255)}});
  EXPECT_THAT(MangleType(type),
              IsOkAndHolds("r8_my_5Fmod1_S2_1_Nv8_ff_1_Ttu32_"));
}

TEST_F(TypedMangleTest, DistinguishesDottedAndUnderscoredModules) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"T", TypeReference("u32_")}};
  ParametricEnv env(bindings);
  auto dotted =
      MangleDslxName("a.b", "f", CallingConvention::kTypical, {"T"}, env);
  auto underscored =
      MangleDslxName("a_b", "f", CallingConvention::kTypical, {"T"}, env);
  ASSERT_TRUE(dotted.ok()) << dotted.status();
  ASSERT_TRUE(underscored.ok()) << underscored.status();
  EXPECT_NE(*dotted, *underscored);
}

TEST_F(TypedMangleTest, FramesRawScopeAndProcNameAfterEscaping) {
  std::vector<std::pair<std::string, InterpValue>> bindings = {
      {"T", TypeReference("u32_")}};
  EXPECT_THAT(
      MangleDslxName("m", "p->q:0", CallingConvention::kProcNext, {"T"},
                     ParametricEnv(bindings), "a b"),
      IsOkAndHolds("_0T__pnext__1_m_5_a_20b_12_p_2D_3Eq_3A0__T4_u32__next"));
}

// Verifies: Type arguments use distinct framed symbols and compatible hashes.
// Catches: Ambiguous boundaries or unequal hashes for equivalent types.
TEST_F(TypedMangleTest, TypeArgumentsUseFramedCompilerEncoding) {
  std::vector<std::pair<std::string, InterpValue>> mixed = {
      {"T", TypeReference("a_b")}, {"V", InterpValue::MakeU8(255)}};
  std::vector<std::pair<std::string, InterpValue>> equivalent = {
      {"T", TypeReference("a_b")}, {"V", InterpValue::MakeSBits(8, -1)}};
  EXPECT_EQ(ParametricEnv(mixed), ParametricEnv(equivalent));
  EXPECT_EQ(absl::HashOf(ParametricEnv(mixed)),
            absl::HashOf(ParametricEnv(equivalent)));
  auto mixed_name = MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                                   {"T", "V"}, ParametricEnv(mixed));
  EXPECT_THAT(mixed_name,
              IsOkAndHolds("_0T__8_my_5Fmod_0__1_p__T3_a_b_V3_255"));

  auto* enum_name = module_.Make<NameDef>(Span::Fake(), "E", nullptr);
  auto* enum_def =
      module_.Make<EnumDef>(Span::Fake(), enum_name,
                            /*type=*/nullptr, std::vector<EnumMember>{}, false);
  equivalent[1].second = InterpValue::MakeEnum(
      InterpValue::MakeU8(255).GetBitsOrDie(), /*is_signed=*/false, enum_def);
  EXPECT_EQ(ParametricEnv(mixed), ParametricEnv(equivalent));
  EXPECT_EQ(absl::HashOf(ParametricEnv(mixed)),
            absl::HashOf(ParametricEnv(equivalent)));

  mixed[1].second = InterpValue::MakeSymbolicRange(InterpValue::MakeU8(128),
                                                   InterpValue::MakeU8(127));
  equivalent[1].second = InterpValue::MakeSymbolicRange(
      InterpValue::MakeSBits(8, -128), InterpValue::MakeSBits(8, 127));
  ParametricEnv unsigned_range(mixed);
  ParametricEnv signed_range(equivalent);
  equivalent[1].second = InterpValue::MakeArray({}).value();
  ParametricEnv empty_array(equivalent);
  EXPECT_NE(unsigned_range, signed_range);
  EXPECT_EQ(unsigned_range, empty_array);
  auto unsigned_name = MangleDslxName(
      "my_mod", "p", CallingConvention::kTypical, {"T", "V"}, unsigned_range);
  auto signed_name = MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                                    {"T", "V"}, signed_range);
  auto empty_name = MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                                   {"T", "V"}, empty_array);
  ASSERT_TRUE(unsigned_name.ok()) << unsigned_name.status();
  ASSERT_TRUE(signed_name.ok()) << signed_name.status();
  ASSERT_TRUE(empty_name.ok()) << empty_name.status();
  EXPECT_NE(*unsigned_name, *signed_name);
  EXPECT_EQ(*unsigned_name, *empty_name);

  mixed[1].second = InterpValue::MakeSymbolicRange(InterpValue::MakeU8(0),
                                                   InterpValue::MakeU8(2));
  equivalent[1].second =
      InterpValue::MakeArray({InterpValue::MakeU8(0), InterpValue::MakeU8(1)})
          .value();
  EXPECT_EQ(ParametricEnv(mixed), ParametricEnv(equivalent));
  EXPECT_EQ(absl::HashOf(ParametricEnv(mixed)),
            absl::HashOf(ParametricEnv(equivalent)));

  mixed[1].second = InterpValue::MakeTuple(
      {InterpValue::MakeArray({InterpValue::MakeU8(255)}).value()});
  equivalent[1].second = InterpValue::MakeTuple(
      {InterpValue::MakeArray({InterpValue::MakeSBits(8, -1)}).value()});
  EXPECT_EQ(ParametricEnv(mixed), ParametricEnv(equivalent));
  EXPECT_EQ(absl::HashOf(ParametricEnv(mixed)),
            absl::HashOf(ParametricEnv(equivalent)));

  std::vector<std::pair<std::string, InterpValue>> first = {
      {"A", TypeReference("a_b")}, {"B", TypeReference("c")}};
  std::vector<std::pair<std::string, InterpValue>> second = {
      {"A", TypeReference("a")}, {"B", TypeReference("b_c")}};
  auto first_name = MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                                   {"A", "B"}, ParametricEnv(first));
  auto second_name = MangleDslxName("my_mod", "p", CallingConvention::kTypical,
                                    {"A", "B"}, ParametricEnv(second));
  ASSERT_TRUE(first_name.ok());
  ASSERT_TRUE(second_name.ok());
  EXPECT_NE(*first_name, *second_name);

  auto explicit_name =
      MangleDslxName("my_mod", "p__T3_a_b_V3_255", CallingConvention::kTypical,
                     {}, ParametricEnv{});
  ASSERT_TRUE(explicit_name.ok());
  ASSERT_TRUE(mixed_name.ok());
  EXPECT_NE(*mixed_name, *explicit_name);
}

}  // namespace
}  // namespace xls::dslx
