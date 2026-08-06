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

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/type_manager.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;

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

TEST(TypeTest, InstantiationType) {
  TypeManager man;
  InstantiationType it1(/*input_types=*/{{"foo", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)}});
  InstantiationType it2(/*input_types=*/{{"foo", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)}});
  EXPECT_EQ(it1, it2);
  InstantiationType it3(/*input_types=*/{{"bar", man.GetBitsType(32)}},
                        /*output_types=*/{{"foo", man.GetBitsType(32)}});
  EXPECT_NE(it1, it3);
  InstantiationType it4(/*input_types=*/{{"fooooooo", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)}});
  EXPECT_NE(it1, it4);
  InstantiationType it5(/*input_types=*/{{"foo", man.GetBitsType(32)}},
                        /*output_types=*/{{"baaaaar", man.GetBitsType(32)}});
  EXPECT_NE(it1, it5);
  InstantiationType it6(/*input_types=*/{{"foo", man.GetBitsType(32)},
                                         {"more", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)}});
  EXPECT_NE(it1, it6);
  InstantiationType it7(/*input_types=*/{{"foo", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)},
                                          {"more", man.GetBitsType(32)}});
  EXPECT_NE(it1, it7);
  InstantiationType it8(/*input_types=*/{{"foo", man.GetBitsType(32)}},
                        /*output_types=*/{{"bar", man.GetBitsType(3)}});
  EXPECT_NE(it1, it8);
  InstantiationType it9(/*input_types=*/{{"foo", man.GetBitsType(3)}},
                        /*output_types=*/{{"bar", man.GetBitsType(32)}});
  EXPECT_NE(it1, it9);
}

TEST(TypeTest, StructuralMetadataBitsAndToken) {
  BitsType b32(32);
  TokenType token;

  EXPECT_THAT(b32.leaf_types(), ElementsAre(&b32));
  EXPECT_EQ(b32.leaf_type(0), &b32);
  EXPECT_THAT(b32.tree_index_vectors(), ElementsAre(ElementsAre()));
  EXPECT_THAT(b32.tree_index(0), ElementsAre());
  EXPECT_EQ(b32.GetLinearOffset({}), 0);

  EXPECT_THAT(token.leaf_types(), ElementsAre(&token));
  EXPECT_EQ(token.leaf_type(0), &token);
  EXPECT_THAT(token.tree_index_vectors(), ElementsAre(ElementsAre()));
  EXPECT_THAT(token.tree_index(0), ElementsAre());
  EXPECT_EQ(token.GetLinearOffset({}), 0);
}

TEST(TypeTest, StructuralMetadataArrayOfTuples) {
  BitsType b8(8);
  BitsType b16(16);
  TupleType tuple({&b8, &b16});
  ArrayType array_of_tuples(2, &tuple);

  EXPECT_EQ(array_of_tuples.leaf_count(), 4);
  EXPECT_THAT(array_of_tuples.leaf_types(), ElementsAre(&b8, &b16, &b8, &b16));

  EXPECT_THAT(array_of_tuples.tree_index(0), ElementsAre(0, 0));
  EXPECT_THAT(array_of_tuples.tree_index(1), ElementsAre(0, 1));
  EXPECT_THAT(array_of_tuples.tree_index(2), ElementsAre(1, 0));
  EXPECT_THAT(array_of_tuples.tree_index(3), ElementsAre(1, 1));

  EXPECT_EQ(array_of_tuples.GetLinearOffset({0, 0}), 0);
  EXPECT_EQ(array_of_tuples.GetLinearOffset({0, 1}), 1);
  EXPECT_EQ(array_of_tuples.GetLinearOffset({1, 0}), 2);
  EXPECT_EQ(array_of_tuples.GetLinearOffset({1, 1}), 3);
}

TEST(TypeTest, StructuralMetadataTupleOfTuplesWithEmptyTuples) {
  BitsType b8(8);
  BitsType b16(16);
  BitsType b32(32);
  TupleType empty_tuple({});
  TupleType inner1({&b8, &b16});
  TupleType inner2({&b32});
  TupleType tuple({&empty_tuple, &inner1, &empty_tuple, &inner2});

  EXPECT_EQ(tuple.leaf_count(), 3);
  EXPECT_THAT(tuple.leaf_types(), ElementsAre(&b8, &b16, &b32));
  EXPECT_THAT(tuple.member_leaf_offsets(), ElementsAre(0, 0, 2, 2));

  EXPECT_EQ(tuple.member_leaf_offset(0), 0);
  EXPECT_EQ(tuple.member_leaf_offset(1), 0);
  EXPECT_EQ(tuple.member_leaf_offset(2), 2);
  EXPECT_EQ(tuple.member_leaf_offset(3), 2);

  EXPECT_THAT(tuple.tree_index(0), ElementsAre(1, 0));
  EXPECT_THAT(tuple.tree_index(1), ElementsAre(1, 1));
  EXPECT_THAT(tuple.tree_index(2), ElementsAre(3, 0));

  EXPECT_EQ(tuple.GetLinearOffset({1, 0}), 0);
  EXPECT_EQ(tuple.GetLinearOffset({1, 1}), 1);
  EXPECT_EQ(tuple.GetLinearOffset({3, 0}), 2);
}

TEST(TypeTest, StructuralMetadataTupleOfArrays) {
  BitsType b8(8);
  BitsType b16(16);
  ArrayType a1(2, &b8);
  ArrayType a2(3, &b16);
  TupleType tuple({&a1, &a2});

  EXPECT_EQ(tuple.leaf_count(), 5);
  EXPECT_THAT(tuple.leaf_types(), ElementsAre(&b8, &b8, &b16, &b16, &b16));
  EXPECT_THAT(tuple.member_leaf_offsets(), ElementsAre(0, 2));

  EXPECT_THAT(tuple.tree_index(0), ElementsAre(0, 0));
  EXPECT_THAT(tuple.tree_index(1), ElementsAre(0, 1));
  EXPECT_THAT(tuple.tree_index(2), ElementsAre(1, 0));
  EXPECT_THAT(tuple.tree_index(3), ElementsAre(1, 1));
  EXPECT_THAT(tuple.tree_index(4), ElementsAre(1, 2));

  EXPECT_EQ(tuple.GetLinearOffset({0, 0}), 0);
  EXPECT_EQ(tuple.GetLinearOffset({0, 1}), 1);
  EXPECT_EQ(tuple.GetLinearOffset({1, 0}), 2);
  EXPECT_EQ(tuple.GetLinearOffset({1, 1}), 3);
  EXPECT_EQ(tuple.GetLinearOffset({1, 2}), 4);
}

TEST(TypeTest, StructuralMetadataDeeplyNestedType) {
  BitsType b8(8);
  BitsType b16(16);
  ArrayType inner_arr(2, &b8);
  TupleType inner_tuple({&inner_arr, &b16});
  ArrayType deep_type(2, &inner_tuple);

  EXPECT_EQ(deep_type.leaf_count(), 6);
  EXPECT_THAT(deep_type.leaf_types(),
              ElementsAre(&b8, &b8, &b16, &b8, &b8, &b16));

  EXPECT_THAT(deep_type.tree_index(0), ElementsAre(0, 0, 0));
  EXPECT_THAT(deep_type.tree_index(1), ElementsAre(0, 0, 1));
  EXPECT_THAT(deep_type.tree_index(2), ElementsAre(0, 1));
  EXPECT_THAT(deep_type.tree_index(3), ElementsAre(1, 0, 0));
  EXPECT_THAT(deep_type.tree_index(4), ElementsAre(1, 0, 1));
  EXPECT_THAT(deep_type.tree_index(5), ElementsAre(1, 1));

  EXPECT_EQ(deep_type.GetLinearOffset({0, 0, 0}), 0);
  EXPECT_EQ(deep_type.GetLinearOffset({0, 0, 1}), 1);
  EXPECT_EQ(deep_type.GetLinearOffset({0, 1}), 2);
  EXPECT_EQ(deep_type.GetLinearOffset({1, 0, 0}), 3);
  EXPECT_EQ(deep_type.GetLinearOffset({1, 0, 1}), 4);
  EXPECT_EQ(deep_type.GetLinearOffset({1, 1}), 5);

  EXPECT_EQ(deep_type.GetSubtype({0}), &inner_tuple);
  EXPECT_EQ(deep_type.GetSubtype({0, 0}), &inner_arr);
  EXPECT_EQ(deep_type.GetSubtype({0, 1}), &b16);

  EXPECT_THAT(deep_type.GetSubtypeAndOffset({0}),
              Pair(&inner_tuple, int64_t{0}));
  EXPECT_THAT(deep_type.GetSubtypeAndOffset({1, 0}),
              Pair(&inner_arr, int64_t{3}));
}

}  // namespace
}  // namespace xls
