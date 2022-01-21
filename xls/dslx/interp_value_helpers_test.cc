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
#include "xls/dslx/interp_value_helpers.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/pos.h"

namespace xls::dslx {
namespace {

TEST(InterpValueHelpersTest, CastBitsToArray) {
  InterpValue input(InterpValue::MakeU32(0xa5a5a5a5));

  ArrayType array_type(BitsType::MakeU8(), ConcreteTypeDim::CreateU32(4));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           CastBitsToArray(input, array_type));
  ASSERT_TRUE(converted.IsArray());
  XLS_ASSERT_OK_AND_ASSIGN(int64_t length, converted.GetLength());
  ASSERT_EQ(length, 4);
  for (int i = 0; i < 4; i++) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue value, converted.Index(i));
    ASSERT_TRUE(value.IsBits());
    XLS_ASSERT_OK_AND_ASSIGN(int64_t int_value, value.GetBitValueUint64());
    ASSERT_EQ(int_value, 0xa5);
  }
}

TEST(InterpValueHelpersTest, CastBitsToEnum) {
  constexpr int kBitCount = 13;
  constexpr int kNumMembers = 16;
  Module module("my_test_module");

  std::vector<EnumMember> members;
  std::vector<InterpValue> member_values;
  TypeAnnotation* element_type =
      module.Make<BuiltinTypeAnnotation>(Span::Fake(), BuiltinType::kU13);
  for (int i = 0; i < kNumMembers; i++) {
    NameDef* name_def =
        module.Make<NameDef>(Span::Fake(), absl::StrCat("member_", i), nullptr);
    Number* number = module.Make<Number>(Span::Fake(), absl::StrCat(i),
                                         NumberKind::kOther, element_type);
    name_def->set_definer(number);
    members.push_back(EnumMember{name_def, number});
    member_values.push_back(InterpValue::MakeUBits(kBitCount, i));
  }

  NameDef* name_def =
      module.Make<NameDef>(Span::Fake(), "my_test_enum", nullptr);
  EnumDef* enum_def = module.Make<EnumDef>(Span::Fake(), name_def, element_type,
                                           members, /*is_public=*/true);

  EnumType enum_type(*enum_def, ConcreteTypeDim::CreateU32(kBitCount),
                     member_values);

  InterpValue bits_value(InterpValue::MakeUBits(kBitCount, 11));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue converted,
                           CastBitsToEnum(bits_value, enum_type));
  ASSERT_TRUE(converted.IsEnum());
  ASSERT_EQ(converted.type(), enum_def);
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t int_value, converted.GetBitValueUint64());
  ASSERT_EQ(int_value, 11);
}

}  // namespace
}  // namespace xls::dslx
