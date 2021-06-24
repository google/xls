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

#include "xls/ir/package.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/type.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::ElementsAre;
using testing::HasSubstr;

class PackageTest : public IrTestBase {};

TEST_F(PackageTest, GetBitsTypes) {
  Package p(TestName());
  EXPECT_FALSE(p.IsOwnedType(nullptr));

  BitsType* bits42 = p.GetBitsType(42);
  EXPECT_TRUE(bits42->IsBits());
  EXPECT_EQ(bits42->bit_count(), 42);
  EXPECT_TRUE(p.IsOwnedType(bits42));
  EXPECT_EQ(bits42, p.GetBitsType(42));
  EXPECT_EQ("bits[42]", bits42->ToString());

  BitsType imposter(42);
  EXPECT_FALSE(p.IsOwnedType(&imposter));

  BitsType* bits77 = p.GetBitsType(77);
  EXPECT_TRUE(p.IsOwnedType(bits77));
  EXPECT_NE(bits77, bits42);

  TypeProto bits77_proto = bits77->ToProto();
  EXPECT_EQ(bits77_proto.type_enum(), TypeProto::BITS);
  EXPECT_EQ(bits77_proto.bit_count(), 77);
  EXPECT_THAT(p.GetTypeFromProto(bits77_proto), IsOkAndHolds(bits77));
}

TEST_F(PackageTest, GetArrayTypes) {
  Package p(TestName());
  BitsType* bits42 = p.GetBitsType(42);

  ArrayType* array_bits42 = p.GetArrayType(123, bits42);
  EXPECT_TRUE(array_bits42->IsArray());
  EXPECT_EQ(array_bits42->size(), 123);
  EXPECT_EQ(array_bits42, p.GetArrayType(123, bits42));
  EXPECT_EQ(array_bits42->element_type(), p.GetBitsType(42));
  EXPECT_EQ("bits[42][123]", array_bits42->ToString());

  ArrayType* array_array_bits42 = p.GetArrayType(444, array_bits42);
  EXPECT_TRUE(array_array_bits42->IsArray());
  EXPECT_EQ(array_array_bits42->size(), 444);
  EXPECT_EQ(array_array_bits42->element_type(), array_bits42);
  EXPECT_EQ(array_array_bits42, p.GetArrayType(444, array_bits42));

  EXPECT_EQ("bits[42][123][444]", array_array_bits42->ToString());

  EXPECT_THAT(p.GetTypeFromProto(array_array_bits42->ToProto()),
              IsOkAndHolds(array_array_bits42));
}

TEST_F(PackageTest, GetTupleTypes) {
  Package p(TestName());
  BitsType* bits42 = p.GetBitsType(42);
  BitsType* bits86 = p.GetBitsType(86);

  TupleType* tuple1 = p.GetTupleType({bits42});
  EXPECT_EQ("(bits[42])", tuple1->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple1));

  TupleType* tuple2 = p.GetTupleType({bits42, bits86});
  EXPECT_EQ("(bits[42], bits[86])", tuple2->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple2));

  TupleType* empty_tuple = p.GetTupleType({});
  EXPECT_EQ("()", empty_tuple->ToString());
  EXPECT_TRUE(p.IsOwnedType(empty_tuple));

  TupleType* tuple_of_arrays =
      p.GetTupleType({p.GetArrayType(2, bits42), p.GetArrayType(4, bits86)});
  EXPECT_EQ("(bits[42][2], bits[86][4])", tuple_of_arrays->ToString());
  EXPECT_TRUE(p.IsOwnedType(tuple_of_arrays));

  TupleType* nested_tuple = p.GetTupleType({empty_tuple, tuple2, bits86});
  EXPECT_EQ("((), (bits[42], bits[86]), bits[86])", nested_tuple->ToString());
  EXPECT_TRUE(p.IsOwnedType(nested_tuple));

  EXPECT_THAT(p.GetTypeFromProto(nested_tuple->ToProto()),
              IsOkAndHolds(nested_tuple));
}

TEST_F(PackageTest, GetTokenType) {
  Package p(TestName());

  TokenType* my_token_type = p.GetTokenType();
  EXPECT_TRUE(my_token_type->IsToken());
  EXPECT_TRUE(p.IsOwnedType(my_token_type));
  EXPECT_EQ(my_token_type, p.GetTokenType());
  EXPECT_EQ("token", my_token_type->ToString());
}

TEST_F(PackageTest, MapTypeFromOtherPackageBitsTypes) {
  Package p(TestName());
  Package other_package("other_package");

  BitsType* bits42_p = p.GetBitsType(42);
  XLS_ASSERT_OK_AND_ASSIGN(Type * mapped_bits42_p,
                           p.MapTypeFromOtherPackage(bits42_p));
  EXPECT_EQ(bits42_p, mapped_bits42_p);
  XLS_ASSERT_OK_AND_ASSIGN(Type * bits42_generic,
                           other_package.MapTypeFromOtherPackage(bits42_p));
  BitsType* bits42 = bits42_generic->AsBitsOrDie();
  EXPECT_FALSE(p.IsOwnedType(bits42));
  EXPECT_TRUE(other_package.IsOwnedType(bits42));
  EXPECT_TRUE(bits42->IsBits());
  EXPECT_EQ(bits42->bit_count(), 42);
  EXPECT_EQ(bits42, other_package.GetBitsType(42));
  EXPECT_EQ("bits[42]", bits42->ToString());

  BitsType imposter(42);
  EXPECT_FALSE(p.IsOwnedType(&imposter));
  EXPECT_FALSE(other_package.IsOwnedType(&imposter));

  BitsType* bits77_p = p.GetBitsType(77);
  XLS_ASSERT_OK_AND_ASSIGN(Type * bits77_generic,
                           other_package.MapTypeFromOtherPackage(bits77_p));
  BitsType* bits77 = bits77_generic->AsBitsOrDie();
  EXPECT_FALSE(p.IsOwnedType(bits77));
  EXPECT_TRUE(other_package.IsOwnedType(bits77));
  EXPECT_NE(bits77, bits42);

  TypeProto bits77_proto = bits77->ToProto();
  EXPECT_EQ(bits77_proto.type_enum(), TypeProto::BITS);
  EXPECT_EQ(bits77_proto.bit_count(), 77);
  EXPECT_THAT(other_package.GetTypeFromProto(bits77_proto),
              IsOkAndHolds(bits77));
}

TEST_F(PackageTest, MapTypeFromOtherPackageArrayTypes) {
  Package p(TestName());
  BitsType* bits42 = p.GetBitsType(42);
  Package other_package("other_package");

  ArrayType* array_bits42_p = p.GetArrayType(123, bits42);
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * array_bits42_generic,
      other_package.MapTypeFromOtherPackage(array_bits42_p));
  ArrayType* array_bits42 = array_bits42_generic->AsArrayOrDie();
  EXPECT_TRUE(array_bits42->IsArray());
  EXPECT_EQ(array_bits42->size(), 123);
  BitsType* bits42_other = other_package.GetBitsType(42);
  EXPECT_EQ(array_bits42, other_package.GetArrayType(123, bits42_other));
  EXPECT_EQ(array_bits42->element_type(), other_package.GetBitsType(42));
  EXPECT_EQ("bits[42][123]", array_bits42->ToString());

  ArrayType* array_array_bits42_p = p.GetArrayType(444, array_bits42_p);
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * array_array_bits42_generic,
      other_package.MapTypeFromOtherPackage(array_array_bits42_p));
  ArrayType* array_array_bits42 = array_array_bits42_generic->AsArrayOrDie();
  EXPECT_TRUE(array_array_bits42->IsArray());
  EXPECT_EQ(array_array_bits42->size(), 444);
  EXPECT_EQ(array_array_bits42->element_type(), array_bits42);
  EXPECT_EQ(array_array_bits42, other_package.GetArrayType(444, array_bits42));

  EXPECT_EQ("bits[42][123][444]", array_array_bits42->ToString());
}

TEST_F(PackageTest, MapTypeFromOtherPackageTupleTypes) {
  Package p(TestName());
  Package other_package("other_package");
  BitsType* bits42_p = p.GetBitsType(42);
  BitsType* bits86_p = p.GetBitsType(86);

  TupleType* tuple1_p = p.GetTupleType({bits42_p});
  XLS_ASSERT_OK_AND_ASSIGN(Type * tuple1_generic,
                           other_package.MapTypeFromOtherPackage(tuple1_p));
  TupleType* tuple1 = tuple1_generic->AsTupleOrDie();
  EXPECT_EQ("(bits[42])", tuple1->ToString());
  EXPECT_TRUE(other_package.IsOwnedType(tuple1));

  TupleType* tuple2_p = p.GetTupleType({bits42_p, bits86_p});
  XLS_ASSERT_OK_AND_ASSIGN(Type * tuple2_generic,
                           other_package.MapTypeFromOtherPackage(tuple2_p));
  TupleType* tuple2 = tuple2_generic->AsTupleOrDie();
  EXPECT_EQ("(bits[42], bits[86])", tuple2->ToString());
  EXPECT_TRUE(other_package.IsOwnedType(tuple2));

  TupleType* empty_tuple_p = p.GetTupleType({});
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * empty_tuple_generic,
      other_package.MapTypeFromOtherPackage(empty_tuple_p));
  TupleType* empty_tuple = empty_tuple_generic->AsTupleOrDie();
  EXPECT_EQ("()", empty_tuple->ToString());
  EXPECT_TRUE(other_package.IsOwnedType(empty_tuple));

  TupleType* tuple_of_arrays_p = p.GetTupleType(
      {p.GetArrayType(2, bits42_p), p.GetArrayType(4, bits86_p)});
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * tuple_of_arrays_generic,
      other_package.MapTypeFromOtherPackage(tuple_of_arrays_p));
  TupleType* tuple_of_arrays = tuple_of_arrays_generic->AsTupleOrDie();
  EXPECT_EQ("(bits[42][2], bits[86][4])", tuple_of_arrays->ToString());
  EXPECT_TRUE(other_package.IsOwnedType(tuple_of_arrays));

  TupleType* nested_tuple_p =
      p.GetTupleType({empty_tuple_p, tuple2_p, bits86_p});
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * nested_tuple_generic,
      other_package.MapTypeFromOtherPackage(nested_tuple_p));
  TupleType* nested_tuple = nested_tuple_generic->AsTupleOrDie();
  EXPECT_EQ("((), (bits[42], bits[86]), bits[86])", nested_tuple->ToString());
  EXPECT_TRUE(other_package.IsOwnedType(nested_tuple));
  EXPECT_THAT(other_package.GetTypeFromProto(nested_tuple->ToProto()),
              IsOkAndHolds(nested_tuple));
}

TEST_F(PackageTest, MapTypeFromOtherPackageTokenType) {
  Package p(TestName());
  Package other_package("other_package");

  TokenType* my_token_type_p = p.GetTokenType();
  XLS_ASSERT_OK_AND_ASSIGN(
      Type * my_token_type_generic,
      other_package.MapTypeFromOtherPackage(my_token_type_p));
  TokenType* my_token_type = my_token_type_generic->AsTokenOrDie();
  EXPECT_TRUE(my_token_type->IsToken());
  EXPECT_TRUE(other_package.IsOwnedType(my_token_type));
  EXPECT_EQ(my_token_type, other_package.GetTokenType());
  EXPECT_EQ("token", my_token_type->ToString());
}

TEST_F(PackageTest, IsDefinitelyEqualTo) {
  const char text1[] = R"(
package package1

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn main(a: bits[32]) -> bits[32] {
  ret invoke.5: bits[32] = invoke(a, a, to_apply=f)
}
)";

  // This package uses subtract instead of add.
  const char text2[] = R"(
package package1

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.1: bits[32] = sub(x, y)
}

fn main(a: bits[32]) -> bits[32] {
  ret invoke.5: bits[32] = invoke(a, a, to_apply=f)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto p1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto p1_2, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto p2, ParsePackage(text2));

  EXPECT_TRUE(p1->IsDefinitelyEqualTo(p1.get()));
  EXPECT_TRUE(p1->IsDefinitelyEqualTo(p1_2.get()));
  EXPECT_FALSE(p1->IsDefinitelyEqualTo(p2.get()));
}

TEST_F(PackageTest, CreateStreamingChannel) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));
  EXPECT_EQ(ch0->name(), "ch0");
  EXPECT_EQ(ch0->id(), 0);
  EXPECT_EQ(ch0->type(), p.GetBitsType(32));
  EXPECT_EQ(ch0->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_THAT(p.GetChannel(0), IsOkAndHolds(ch0));

  EXPECT_THAT(p.channels(), ElementsAre(ch0));

  // Next channel should get ID 1.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kReceiveOnly,
                                              p.GetBitsType(444)));
  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_EQ(ch1->id(), 1);
  EXPECT_EQ(ch1->supported_ops(), ChannelOps::kReceiveOnly);
  EXPECT_THAT(p.GetChannel(1), IsOkAndHolds(ch1));

  // Create a channel with a specific ID.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch42, p.CreateStreamingChannel(
                          "ch42", ChannelOps::kReceiveOnly, p.GetBitsType(44),
                          /*initial_values=*/{}, FlowControl::kReadyValid,
                          ChannelMetadataProto(), 42));
  EXPECT_EQ(ch42->id(), 42);
  EXPECT_THAT(p.GetChannel(42), IsOkAndHolds(ch42));

  // Next channel should get next sequential ID after highest ID so far.
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch43, p.CreateStreamingChannel("ch43", ChannelOps::kReceiveOnly,
                                               p.GetBitsType(44)));
  EXPECT_EQ(ch43->id(), 43);

  // Creating a channel with a duplicate ID is an error.
  EXPECT_THAT(p.CreateStreamingChannel(
                   "ch1_dup", ChannelOps::kReceiveOnly, p.GetBitsType(44),
                   /*initial_values=*/{}, FlowControl::kReadyValid,
                   ChannelMetadataProto(), 1)
                  .status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Channel already exists with id 1")));

  // Fetching a non-existent channel ID is an error.
  EXPECT_THAT(p.GetChannel(123), StatusIs(absl::StatusCode::kNotFound,
                                          HasSubstr("No channel with id 123")));

  EXPECT_THAT(p.channels(), ElementsAre(ch0, ch1, ch42, ch43));
}

TEST_F(PackageTest, ChannelRemoval) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1, p.CreateStreamingChannel("ch1", ChannelOps::kSendOnly,
                                              p.GetBitsType(32)));
  TokenlessProcBuilder b(TestName(), Value::Tuple({}), "tkn", "st", &p);
  b.Send(ch0, b.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK(b.Build(b.GetStateParam()).status());

  Package other_p("other");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * other_ch,
      other_p.CreateStreamingChannel("other", ChannelOps::kSendOnly,
                                     other_p.GetBitsType(32)));

  EXPECT_EQ(p.channels().size(), 2);

  XLS_ASSERT_OK(p.RemoveChannel(ch1));
  EXPECT_EQ(p.channels().size(), 1);

  // Removing a channel not owned by the package.
  EXPECT_THAT(p.RemoveChannel(other_ch),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Channel not owned by package")));

  // Removing a channel in use should be an error.
  EXPECT_THAT(p.RemoveChannel(ch0),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("cannot be removed because it is used")));
}

}  // namespace
}  // namespace xls
