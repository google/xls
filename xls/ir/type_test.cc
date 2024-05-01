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

#include "xls/ir/type.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TypeTest, TestVariousTypes) {
  BitsType b42(42);
  BitsType b42_2(42);
  BitsType b123(123);

  EXPECT_TRUE(b42.IsEqualTo(&b42));
  EXPECT_TRUE(b42.IsEqualTo(&b42_2));
  EXPECT_FALSE(b42.IsEqualTo(&b123));

  EXPECT_EQ(b42.leaf_count(), 1);
  EXPECT_EQ(b42_2.leaf_count(), 1);
  EXPECT_EQ(b123.leaf_count(), 1);

  TupleType t_empty({});
  TupleType t1({&b42, &b42});
  TupleType t2({&b42, &b42});
  TupleType t3({&b42, &b42_2});
  TupleType t4({&b42, &b42, &b42});

  EXPECT_TRUE(t_empty.IsEqualTo(&t_empty));
  EXPECT_FALSE(t_empty.IsEqualTo(&t1));
  EXPECT_FALSE(t_empty.IsEqualTo(&b42));
  EXPECT_TRUE(t1.IsEqualTo(&t1));
  EXPECT_TRUE(t1.IsEqualTo(&t2));
  EXPECT_TRUE(t1.IsEqualTo(&t3));
  EXPECT_FALSE(t1.IsEqualTo(&t4));

  EXPECT_EQ(t_empty.leaf_count(), 0);
  EXPECT_EQ(t4.leaf_count(), 3);

  TupleType t_nested_empty({&t_empty});
  TupleType t_nested1({&t1, &t2});
  TupleType t_nested2({&t2, &t1});
  TupleType t_nested3({&t1, &t3});
  TupleType t_nested4({&t1, &t4});

  EXPECT_TRUE(t_nested_empty.IsEqualTo(&t_nested_empty));
  EXPECT_FALSE(t_nested_empty.IsEqualTo(&t_empty));
  EXPECT_TRUE(t_nested1.IsEqualTo(&t_nested2));
  EXPECT_TRUE(t_nested1.IsEqualTo(&t_nested3));
  EXPECT_FALSE(t_nested1.IsEqualTo(&t_nested4));

  EXPECT_EQ(t_nested_empty.leaf_count(), 0);
  EXPECT_EQ(t_nested3.leaf_count(), 4);

  ArrayType a1(7, &b42);
  ArrayType a2(7, &b42_2);
  ArrayType a3(3, &b42);
  ArrayType a4(7, &b123);

  EXPECT_TRUE(a1.IsEqualTo(&a1));
  EXPECT_TRUE(a1.IsEqualTo(&a2));
  EXPECT_FALSE(a1.IsEqualTo(&a3));
  EXPECT_FALSE(a1.IsEqualTo(&a4));

  EXPECT_EQ(a1.leaf_count(), 7);
  EXPECT_EQ(a3.leaf_count(), 3);

  // Arrays-of-tuples.
  ArrayType a_of_t1(42, &t1);
  ArrayType a_of_t2(42, &t2);
  ArrayType a_of_t3(42, &t4);

  EXPECT_TRUE(a_of_t1.IsEqualTo(&a_of_t2));
  EXPECT_FALSE(a_of_t1.IsEqualTo(&a_of_t3));

  EXPECT_EQ(a_of_t1.leaf_count(), 84);
  EXPECT_EQ(a_of_t3.leaf_count(), 126);

  // Tuple-of-Arrays.
  TupleType t_of_a1({&a1, &a1, &a2});
  TupleType t_of_a2({&a1, &a1, &a1});
  TupleType t_of_a3({&a1, &a2, &a3});

  EXPECT_TRUE(t_of_a1.IsEqualTo(&t_of_a2));
  EXPECT_FALSE(t_of_a1.IsEqualTo(&t_of_a3));
  EXPECT_FALSE(t_of_a1.IsEqualTo(&b42));

  EXPECT_EQ(t_of_a1.leaf_count(), 21);
  EXPECT_EQ(t_of_a3.leaf_count(), 17);

  // Token types.
  TokenType token_0;
  TokenType token_1;

  EXPECT_TRUE(token_0.IsEqualTo(&token_0));
  EXPECT_TRUE(token_0.IsEqualTo(&token_1));
  EXPECT_FALSE(token_0.IsEqualTo(&b42_2));

  EXPECT_EQ(token_0.leaf_count(), 1);

  // Function types.
  FunctionType f_type1({&b42, &a1}, &b42);
  FunctionType f_type2({&b42, &a2}, &b42);
  FunctionType f_type3({&b42}, &b42);
  FunctionType f_type4({}, &b42);
  FunctionType f_type5({&b42, &a1}, &b123);

  EXPECT_TRUE(f_type1.IsEqualTo(&f_type2));
  EXPECT_FALSE(f_type1.IsEqualTo(&f_type3));
  EXPECT_FALSE(f_type1.IsEqualTo(&f_type4));
  EXPECT_FALSE(f_type1.IsEqualTo(&f_type5));
}

TEST(TypeTest, ArrayDimensionAndIndex) {
  BitsType b32(32);
  TokenType token;
  ArrayType a_1d(7, &b32);
  ArrayType a_2d(123, &a_1d);
  ArrayType a_3d(1, &a_2d);
  TupleType t({&b32, &a_2d, &b32});
  ArrayType a_of_tuple(22, &t);
  ArrayType a_2d_of_tuple(22, &a_of_tuple);

  EXPECT_EQ(GetArrayDimensionCount(&b32), 0);
  EXPECT_EQ(GetArrayDimensionCount(&token), 0);
  EXPECT_EQ(GetArrayDimensionCount(&a_1d), 1);
  EXPECT_EQ(GetArrayDimensionCount(&a_2d), 2);
  EXPECT_EQ(GetArrayDimensionCount(&a_3d), 3);
  EXPECT_EQ(GetArrayDimensionCount(&t), 0);
  EXPECT_EQ(GetArrayDimensionCount(&a_of_tuple), 1);
  EXPECT_EQ(GetArrayDimensionCount(&a_2d_of_tuple), 2);

  EXPECT_THAT(GetIndexedElementType(&b32, 0), IsOkAndHolds(&b32));
  EXPECT_THAT(GetIndexedElementType(&b32, 1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index has more elements (1) than type "
                                 "bits[32] has array dimensions (0)")));

  EXPECT_THAT(GetIndexedElementType(&a_1d, 0), IsOkAndHolds(&a_1d));
  EXPECT_THAT(GetIndexedElementType(&a_1d, 1), IsOkAndHolds(&b32));
  EXPECT_THAT(GetIndexedElementType(&a_1d, 2),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index has more elements (2) than type "
                                 "bits[32][7] has array dimensions (1)")));

  EXPECT_THAT(GetIndexedElementType(&a_3d, 3), IsOkAndHolds(&b32));
}

TEST(TypeTest, AsXTypeCallsWork) {
  BitsType b32(32);
  TupleType t_empty({});
  TupleType t1({&b32, &b32});
  ArrayType a1(7, &b32);

  XLS_EXPECT_OK(b32.AsBits());
  XLS_EXPECT_OK(t_empty.AsTuple());
  XLS_EXPECT_OK(t1.AsTuple());
  XLS_EXPECT_OK(a1.AsArray());

  EXPECT_THAT(b32.AsArray(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not an array: bits[32]")));
  EXPECT_THAT(b32.AsTuple(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not a tuple: bits[32]")));

  EXPECT_THAT(t_empty.AsBits(), StatusIs(absl::StatusCode::kInvalidArgument,
                                         HasSubstr("Type is not 'bits': ()")));
  EXPECT_THAT(t_empty.AsArray(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not an array: ()")));

  EXPECT_THAT(t1.AsBits(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not 'bits': (bits[32], bits[32])")));
  EXPECT_THAT(
      t1.AsArray(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Type is not an array: (bits[32], bits[32])")));

  EXPECT_THAT(a1.AsBits(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not 'bits': bits[32][7]")));
  EXPECT_THAT(a1.AsTuple(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type is not a tuple: bits[32][7]")));
}

}  // namespace
}  // namespace xls
