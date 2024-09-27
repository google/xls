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

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::Contains;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::IsEmpty;
using testing::Not;
using testing::Property;

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

top fn main(a: bits[32]) -> bits[32] {
  ret invoke.5: bits[32] = invoke(a, a, to_apply=f)
}
)";

  // This package uses subtract instead of add.
  const char text2[] = R"(
package package1

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.1: bits[32] = sub(x, y)
}

top fn main(a: bits[32]) -> bits[32] {
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
      Channel * ch42,
      p.CreateStreamingChannel(
          "ch42", ChannelOps::kReceiveOnly, p.GetBitsType(44),
          /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
          FlowControl::kReadyValid, ChannelStrictness::kProvenMutuallyExclusive,
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
                   /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
                   FlowControl::kReadyValid,
                   ChannelStrictness::kProvenMutuallyExclusive,
                   ChannelMetadataProto(), 1)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
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
  TokenlessProcBuilder b(TestName(), "tkn", &p);
  b.Send(ch0, b.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK(b.Build({}).status());

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

TEST_F(PackageTest, CloneSingleValueChannelSamePackage) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateSingleValueChannel("ch0", ChannelOps::kSendOnly,
                                                p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch1, p.CloneChannel(ch0, "ch1"));

  EXPECT_EQ(p.channels().size(), 2);
  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_EQ(ch1->kind(), ChannelKind::kSingleValue);
  EXPECT_THAT(ch1, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_NE(ch0->id(), ch1->id());
  EXPECT_EQ(ch0->supported_ops(), ch1->supported_ops());
}

TEST_F(PackageTest, CloneSingleValueChannelSetParams) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateSingleValueChannel("ch0", ChannelOps::kSendOnly,
                                                p.GetBitsType(32)));
  std::vector<Value> initial_values_override;
  EXPECT_THAT(
      p.CloneChannel(ch0, "ch1",
                     Package::CloneChannelOverrides().OverrideInitialValues(
                         initial_values_override)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot clone single value channel with "
                         "streaming channel parameter overrides")));
  EXPECT_THAT(
      p.CloneChannel(
          ch0, "ch2",
          Package::CloneChannelOverrides().OverrideFifoConfig(std::nullopt)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot clone single value channel with "
                         "streaming channel parameter overrides")));
  EXPECT_THAT(
      p.CloneChannel(ch0, "ch3",
                     Package::CloneChannelOverrides().OverrideFlowControl(
                         FlowControl::kReadyValid)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot clone single value channel with "
                         "streaming channel parameter overrides")));

  ChannelMetadataProto new_metadata;
  new_metadata.add_block_ports()->set_block_name("new_block_name");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch4,
      p.CloneChannel(
          ch0, "ch4",
          Package::CloneChannelOverrides().OverrideMetadata(new_metadata)));
  EXPECT_THAT(ch4->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "new_block_name")));
  EXPECT_THAT(ch0->metadata().block_ports(),
              Not(Contains(Property(&BlockPortMappingProto::block_name,
                                    "new_block_name"))));
}

TEST_F(PackageTest, CloneSingleValueChannelSamePackageButDifferentOps) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p.CreateSingleValueChannel("ch0", ChannelOps::kSendOnly,
                                                p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1,
      p.CloneChannel(ch0, "ch1",
                     Package::CloneChannelOverrides().OverrideSupportedOps(
                         ChannelOps::kSendReceive)));

  EXPECT_EQ(p.channels().size(), 2);
  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_EQ(ch1->kind(), ChannelKind::kSingleValue);
  EXPECT_THAT(ch1, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_NE(ch0->id(), ch1->id());
  EXPECT_NE(ch0->supported_ops(), ch1->supported_ops());
  EXPECT_EQ(ch1->supported_ops(), ChannelOps::kSendReceive);
}

TEST_F(PackageTest, CloneSingleValueChannelToDifferentPackage) {
  Package p0(TestName());
  Package p1(absl::StrCat(TestName(), "_clone"));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0, p0.CreateSingleValueChannel("ch0", ChannelOps::kSendOnly,
                                                 p0.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch0_clone, p1.CloneChannel(ch0, "ch0"));

  EXPECT_EQ(p0.channels().size(), 1);
  EXPECT_EQ(p1.channels().size(), 1);
  EXPECT_EQ(ch0_clone->name(), "ch0");
  EXPECT_EQ(ch0_clone->kind(), ChannelKind::kSingleValue);
  EXPECT_THAT(ch0_clone, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch0->supported_ops(), ch0_clone->supported_ops());
}

TEST_F(PackageTest, CloneStreamingChannelSamePackage) {
  Package p(TestName());

  std::vector<Value> initial_values = {Value(UBits(1, 32)),
                                       Value(UBits(2, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0,
      p.CreateStreamingChannel("ch0", ChannelOps::kSendOnly, p.GetBitsType(32),
                               /*initial_values=*/initial_values,
                               /*fifo_config=*/
                               FifoConfig(/*depth=*/3, /*bypass=*/true,
                                          /*register_push_outputs=*/true,
                                          /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch1, p.CloneChannel(ch0, "ch1"));

  EXPECT_EQ(p.channels().size(), 2);
  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_EQ(ch1->kind(), ChannelKind::kStreaming);
  EXPECT_THAT(ch1, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_NE(ch0->id(), ch1->id());
  EXPECT_EQ(ch0->supported_ops(), ch1->supported_ops());

  // Check streaming-specific fields.
  StreamingChannel* streaming_ch1 = down_cast<StreamingChannel*>(ch1);
  EXPECT_EQ(streaming_ch1->initial_values(), initial_values);
  EXPECT_EQ(streaming_ch1->GetFifoDepth(), 3);
  EXPECT_EQ(streaming_ch1->GetFlowControl(), FlowControl::kReadyValid);
}

TEST_F(PackageTest, CloneStreamingChannelSetParams) {
  Package p(TestName());

  ChannelMetadataProto original_metadata;
  original_metadata.add_block_ports()->set_block_name("original_name");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0,
      p.CreateStreamingChannel(
          "ch0", ChannelOps::kSendOnly, p.GetBitsType(32),
          /*initial_values*/ std::vector<Value>{Value(UBits(0, 32))},
          /*fifo_config=*/
          FifoConfig(/*depth=*/3, /*bypass=*/true,
                     /*register_push_outputs=*/true,
                     /*register_pop_outputs=*/false),
          /*flow_control=*/FlowControl::kNone,
          /*strictness=*/ChannelStrictness::kProvenMutuallyExclusive,
          /*metadata=*/original_metadata));
  std::vector<Value> initial_values_override;
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1_base,
      p.CloneChannel(ch0, "ch1",
                     Package::CloneChannelOverrides().OverrideInitialValues(
                         initial_values_override)));
  StreamingChannel* ch1 = down_cast<StreamingChannel*>(ch1_base);
  ASSERT_NE(ch1, nullptr);

  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_THAT(ch1, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch1->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch1->GetFifoDepth(), 3);
  EXPECT_EQ(ch1->GetFlowControl(), FlowControl::kNone);
  EXPECT_EQ(ch1->GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
  EXPECT_THAT(ch1->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "original_name")));

  EXPECT_EQ(ch1->initial_values(), initial_values_override);

  std::optional<int64_t> fifo_depth = std::nullopt;
  std::optional<std::optional<int64_t>> fifo_depth_override =
      std::make_optional(fifo_depth);
  EXPECT_TRUE(fifo_depth_override.has_value());
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch2_base,
      p.CloneChannel(
          ch0, "ch2",
          Package::CloneChannelOverrides().OverrideFifoConfig(std::nullopt)));
  StreamingChannel* ch2 = down_cast<StreamingChannel*>(ch2_base);
  ASSERT_NE(ch2, nullptr);

  EXPECT_EQ(ch2->name(), "ch2");
  EXPECT_THAT(ch2, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch2->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch2->initial_values(), std::vector<Value>{Value(UBits(0, 32))});
  EXPECT_EQ(ch2->GetFlowControl(), FlowControl::kNone);
  EXPECT_EQ(ch2->GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
  EXPECT_THAT(ch2->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "original_name")));

  EXPECT_EQ(ch2->GetFifoDepth(), std::nullopt);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch3_base,
      p.CloneChannel(ch0, "ch3",
                     Package::CloneChannelOverrides().OverrideFlowControl(
                         FlowControl::kReadyValid)));
  StreamingChannel* ch3 = down_cast<StreamingChannel*>(ch3_base);
  ASSERT_NE(ch3, nullptr);

  EXPECT_EQ(ch3->name(), "ch3");
  EXPECT_THAT(ch3, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch3->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch3->initial_values(), std::vector<Value>{Value(UBits(0, 32))});
  EXPECT_EQ(ch3->GetFifoDepth(), 3);
  EXPECT_EQ(ch3->GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
  EXPECT_THAT(ch3->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "original_name")));

  EXPECT_EQ(ch3->GetFlowControl(), FlowControl::kReadyValid);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch4_base,
      p.CloneChannel(ch0, "ch4",
                     Package::CloneChannelOverrides().OverrideMetadata(
                         ChannelMetadataProto())));
  StreamingChannel* ch4 = down_cast<StreamingChannel*>(ch4_base);
  ASSERT_NE(ch4, nullptr);

  EXPECT_EQ(ch4->name(), "ch4");
  EXPECT_THAT(ch4, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch4->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch4->initial_values(), std::vector<Value>{Value(UBits(0, 32))});
  EXPECT_EQ(ch4->GetFifoDepth(), 3);
  EXPECT_EQ(ch4->GetFlowControl(), FlowControl::kNone);
  EXPECT_EQ(ch4->GetStrictness(), ChannelStrictness::kProvenMutuallyExclusive);
  EXPECT_THAT(ch0->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "original_name")));
  EXPECT_THAT(ch4->metadata().block_ports(), IsEmpty());

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch5_base,
      p.CloneChannel(ch0, "ch5",
                     Package::CloneChannelOverrides().OverrideStrictness(
                         ChannelStrictness::kArbitraryStaticOrder)));
  StreamingChannel* ch5 = down_cast<StreamingChannel*>(ch5_base);
  ASSERT_NE(ch5, nullptr);

  EXPECT_EQ(ch5->name(), "ch5");
  EXPECT_THAT(ch5, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch5->supported_ops(), ChannelOps::kSendOnly);
  EXPECT_EQ(ch5->initial_values(), std::vector<Value>{Value(UBits(0, 32))});
  EXPECT_EQ(ch5->GetFifoDepth(), 3);
  EXPECT_EQ(ch5->GetFlowControl(), FlowControl::kNone);
  EXPECT_THAT(ch5->metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::block_name,
                                   "original_name")));
  EXPECT_EQ(ch5->GetStrictness(), ChannelStrictness::kArbitraryStaticOrder);
}

TEST_F(PackageTest, CloneStreamingChannelSamePackageButDifferentOps) {
  Package p(TestName());

  std::vector<Value> initial_values = {Value(UBits(1, 32)),
                                       Value(UBits(2, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch0,
      p.CreateStreamingChannel("ch0", ChannelOps::kSendOnly, p.GetBitsType(32),
                               /*initial_values=*/initial_values,
                               /*fifo_config=*/
                               FifoConfig(/*depth=*/3, /*bypass=*/true,
                                          /*register_push_outputs=*/true,
                                          /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch1,
      p.CloneChannel(ch0, "ch1",
                     Package::CloneChannelOverrides().OverrideSupportedOps(
                         ChannelOps::kSendReceive)));

  EXPECT_EQ(p.channels().size(), 2);
  EXPECT_EQ(ch1->name(), "ch1");
  EXPECT_EQ(ch1->kind(), ChannelKind::kStreaming);
  EXPECT_THAT(ch1, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_NE(ch0->id(), ch1->id());
  EXPECT_NE(ch0->supported_ops(), ch1->supported_ops());
  EXPECT_EQ(ch1->supported_ops(), ChannelOps::kSendReceive);

  // Check streaming-specific fields.
  StreamingChannel* streaming_ch1 = down_cast<StreamingChannel*>(ch1);
  EXPECT_EQ(streaming_ch1->initial_values(), initial_values);
  EXPECT_EQ(streaming_ch1->GetFifoDepth(), 3);
  EXPECT_EQ(streaming_ch1->GetFlowControl(), FlowControl::kReadyValid);
}

TEST_F(PackageTest, CloneStreamingChannelDifferentPackage) {
  Package p0(TestName());
  Package p1(absl::StrCat(TestName(), "_clone"));

  std::vector<Value> initial_values = {Value(UBits(1, 32)),
                                       Value(UBits(2, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch0,
                           p0.CreateStreamingChannel(
                               "ch0", ChannelOps::kSendOnly, p0.GetBitsType(32),
                               /*initial_values=*/initial_values,
                               /*fifo_config=*/
                               FifoConfig(/*depth=*/3, /*bypass=*/true,
                                          /*register_push_outputs=*/true,
                                          /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch0_clone,
                           p1.CloneChannel(ch0, "ch0_clone"));

  EXPECT_EQ(p1.channels().size(), 1);
  EXPECT_EQ(ch0_clone->name(), "ch0_clone");
  EXPECT_EQ(ch0_clone->kind(), ChannelKind::kStreaming);
  EXPECT_THAT(ch0_clone, op_matchers::ChannelWithType("bits[32]"));
  EXPECT_EQ(ch0->supported_ops(), ch0_clone->supported_ops());

  // Check streaming-specific fields.
  StreamingChannel* streaming_ch0_clone =
      down_cast<StreamingChannel*>(ch0_clone);
  EXPECT_EQ(streaming_ch0_clone->initial_values(), initial_values);
  EXPECT_EQ(streaming_ch0_clone->GetFifoDepth(), 3);
  EXPECT_EQ(streaming_ch0_clone->GetFlowControl(), FlowControl::kReadyValid);
}

TEST_F(PackageTest, Top) {
  const char text[] = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}

)";
  std::optional<FunctionBase*> top;
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  Package dummy_package("dummy_package");
  Block dummy_block("dummy_block", &dummy_package);
  // Test FunctionBase owned by different package.
  EXPECT_THAT(pkg->SetTop(&dummy_block),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot set the top entity of the package:")));
  // Test non-existent FunctionBase.
  EXPECT_THAT(
      pkg->SetTopByName("top_not_present"),
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr(
              R"available("my_function", "my_proc", "my_block")available")));
  // Set function as top.
  EXPECT_FALSE(pkg->GetTop().has_value());
  XLS_EXPECT_OK(pkg->SetTopByName("my_function"));
  top = pkg->GetTop();
  EXPECT_TRUE(top.has_value());
  EXPECT_TRUE(top.value()->IsFunction());
  Function* func = top.value()->AsFunctionOrDie();
  XLS_EXPECT_OK(pkg->SetTop(func));
  EXPECT_EQ(func->name(), "my_function");
  // Can't remove function as top.
  EXPECT_THAT(pkg->RemoveFunction(func),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot remove function:")));
  XLS_EXPECT_OK(pkg->SetTop(std::nullopt));
  XLS_EXPECT_OK(pkg->RemoveFunction(func));
  // Set proc as top.
  EXPECT_FALSE(pkg->GetTop().has_value());
  XLS_EXPECT_OK(pkg->SetTopByName("my_proc"));
  top = pkg->GetTop();
  EXPECT_TRUE(top.has_value());
  EXPECT_TRUE(top.value()->IsProc());
  Proc* proc = top.value()->AsProcOrDie();
  XLS_EXPECT_OK(pkg->SetTop(proc));
  EXPECT_EQ(proc->name(), "my_proc");
  // Can't remove proc as top.
  EXPECT_THAT(pkg->RemoveProc(proc),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot remove proc:")));
  XLS_EXPECT_OK(pkg->SetTop(std::nullopt));
  XLS_EXPECT_OK(pkg->RemoveProc(proc));
  // Set block as top.
  EXPECT_FALSE(pkg->GetTop().has_value());
  XLS_EXPECT_OK(pkg->SetTopByName("my_block"));
  top = pkg->GetTop();
  EXPECT_TRUE(top.has_value());
  EXPECT_TRUE(top.value()->IsBlock());
  Block* block = top.value()->AsBlockOrDie();
  XLS_EXPECT_OK(pkg->SetTop(block));
  EXPECT_EQ(block->name(), "my_block");
  // Can't remove block as top.
  EXPECT_THAT(pkg->RemoveBlock(block),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot remove block:")));
  XLS_EXPECT_OK(pkg->SetTop(std::nullopt));
  XLS_EXPECT_OK(pkg->RemoveBlock(block));
  EXPECT_EQ(pkg->GetNodeCount(), 0);
}

TEST_F(PackageTest, NodeCount) {
  const char text[] = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}

fn my_function2(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.9: bits[32] = add(x, y)
}

proc my_proc2(st: bits[32], init={42}) {
  literal.10: bits[32] = literal(value=1, id=10)
  add.11: bits[32] = add(literal.10, st, id=11)
  next (add.11)
}

block my_block2(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=12)
  b: bits[32] = input_port(name=b, id=13)
  add.14: bits[32] = add(a, b, id=14)
  out: () = output_port(add.14, name=out, id=15)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  EXPECT_EQ(pkg->GetNodeCount(), 20);
  EXPECT_EQ(pkg->GetFunctionNodeCount(), 6);
  EXPECT_EQ(pkg->GetProcNodeCount(), 6);
  EXPECT_EQ(pkg->GetBlockNodeCount(), 8);
}

TEST_F(PackageTest, FunctionAsTop) {
  const char text[] = R"(
package my_package

top fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  EXPECT_THAT(pkg->GetTopAsProc(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a proc for package:")));
  EXPECT_THAT(pkg->GetTopAsBlock(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a block for package:")));
  EXPECT_EQ(pkg->GetTopAsFunction(), pkg->GetFunction("my_function"));
}

TEST_F(PackageTest, ProcAsTop) {
  const char text[] = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

top proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  EXPECT_THAT(pkg->GetTopAsFunction(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a function for package:")));
  EXPECT_THAT(pkg->GetTopAsBlock(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a block for package:")));
  EXPECT_EQ(pkg->GetTopAsProc(), pkg->GetProc("my_proc"));
}

TEST_F(PackageTest, BlockAsTop) {
  const char text[] = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

top block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  EXPECT_THAT(pkg->GetTopAsFunction(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a function for package:")));
  EXPECT_THAT(pkg->GetTopAsProc(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity is not a proc for package:")));
  EXPECT_EQ(pkg->GetTopAsBlock(), pkg->GetBlock("my_block"));
}

TEST_F(PackageTest, TopNotSet) {
  const char text[] = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

proc my_proc(st: bits[32], init={42}) {
  literal.3: bits[32] = literal(value=1, id=3)
  add.4: bits[32] = add(literal.3, st, id=4)
  next (add.4)
}

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg, ParsePackage(text));
  EXPECT_THAT(pkg->GetTopAsFunction(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity not set for package:")));
  EXPECT_THAT(pkg->GetTopAsProc(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity not set for package:")));
  EXPECT_THAT(pkg->GetTopAsBlock(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Top entity not set for package:")));
}

TEST_F(PackageTest, LinkFunctionsSimple) {
  constexpr std::string_view text1 = R"(
package my_package

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
  )";
  constexpr std::string_view text2 = R"(
package my_package2

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.1: bits[32] = sub(x, y)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto pkg1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg2, ParsePackage(text2));
  XLS_ASSERT_OK(pkg1->AddPackage(pkg2.get()).status());
  EXPECT_EQ(pkg1->functions().size(), 2);
  XLS_EXPECT_OK(pkg1->GetFunction("my_function"));
  XLS_EXPECT_OK(pkg1->GetFunction("my_function_1"));
}

TEST_F(PackageTest, LinkFunctionsWithInvokes) {
  constexpr std::string_view text1 = R"(
package my_package

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=f)
}
  )";
  constexpr std::string_view text2 = R"(
package my_package2

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.1: bits[32] = sub(x, y)
}

fn my_function(x: bits[32], y: bits[32]) -> bits[32] {
  ret invoke.2: bits[32] = invoke(x, y, to_apply=f)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto pkg1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg2, ParsePackage(text2));
  XLS_ASSERT_OK(pkg1->AddPackage(pkg2.get()).status());

  EXPECT_EQ(pkg1->functions().size(), 4);
  XLS_EXPECT_OK(pkg1->GetFunction("my_function"));
  XLS_EXPECT_OK(pkg1->GetFunction("my_function_1"));
  XLS_EXPECT_OK(pkg1->GetFunction("f"));
  XLS_EXPECT_OK(pkg1->GetFunction("f_1"));
}

TEST_F(PackageTest, LinkChannelsAndProcs) {
  constexpr std::string_view text1 = R"(
package my_package

chan test_channel(
  bits[32], id=0, kind=streaming, ops=send_receive,
  flow_control=ready_valid, metadata="""""")

top proc main(__state: (), init={()}) {
  __token: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(__token, tuple_index.3, channel=test_channel)
  tuple.5: () = tuple()
  next (tuple.5)
}
  )";
  constexpr std::string_view text2 = R"(
package my_package2

chan another_test_channel(
  bits[32], id=0, kind=streaming, ops=send_receive,
  flow_control=ready_valid, metadata="""""")


top proc another_main(__state: (), init={()}) {
  __token: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(__token, channel=another_test_channel)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  send.4: token = send(__token, tuple_index.3, channel=another_test_channel)
  tuple.5: () = tuple()
  next (tuple.5)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto pkg1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg2, ParsePackage(text2));
  XLS_ASSERT_OK(pkg1->AddPackage(pkg2.get()).status());
  EXPECT_EQ(pkg1->channels().size(), 2);
  EXPECT_EQ(pkg1->procs().size(), 2);
  XLS_EXPECT_OK(pkg1->GetProc("main"));
  XLS_EXPECT_OK(pkg1->GetProc("another_main"));
  XLS_EXPECT_OK(pkg1->GetChannel("test_channel"));
  XLS_EXPECT_OK(pkg1->GetChannel("another_test_channel"));
}

TEST_F(PackageTest, LinkChannelsAndProcsWithInvokes) {
  constexpr std::string_view text1 = R"(
package my_package

chan test_channel(
  bits[32], id=0, kind=streaming, ops=send_receive,
  flow_control=ready_valid, metadata="""""")

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.15: bits[32] = add(x, y)
}

top proc main(__state: (), init={()}) {
  __token: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  invoke.4: bits[32] = invoke(tuple_index.3, tuple_index.3, to_apply=f)
  send.5: token = send(__token, invoke.4, channel=test_channel)
  tuple.6: () = tuple()
  next (tuple.6)
}
  )";
  constexpr std::string_view text2 = R"(
package my_package2

chan another_test_channel(
  bits[32], id=0, kind=streaming, ops=send_receive,
  flow_control=ready_valid, metadata="""""")

fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret sub.15: bits[32] = sub(x, y)
}

top proc another_main(__state: (), init={()}) {
  __token: token = literal(value=token, id=1000)
  receive.1: (token, bits[32]) = receive(__token, channel=another_test_channel)
  tuple_index.2: token = tuple_index(receive.1, index=0)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
  invoke.4: bits[32] = invoke(tuple_index.3, tuple_index.3, to_apply=f)
  send.5: token = send(__token, invoke.4, channel=another_test_channel)
  tuple.6: () = tuple()
  next (tuple.6)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto pkg1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg2, ParsePackage(text2));
  XLS_ASSERT_OK(pkg1->AddPackage(pkg2.get()).status());

  EXPECT_EQ(pkg1->channels().size(), 2);
  EXPECT_EQ(pkg1->procs().size(), 2);
  EXPECT_EQ(pkg1->functions().size(), 2);
  XLS_EXPECT_OK(pkg1->GetProc("main"));
  XLS_EXPECT_OK(pkg1->GetProc("another_main"));
  XLS_EXPECT_OK(pkg1->GetChannel("test_channel"));
  XLS_EXPECT_OK(pkg1->GetChannel("another_test_channel"));
  XLS_EXPECT_OK(pkg1->GetFunction("f"));
  XLS_EXPECT_OK(pkg1->GetFunction("f_1"));
}

TEST_F(PackageTest, LinkBlocks) {
  constexpr std::string_view text1 = R"(
package my_package

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  add.7: bits[32] = add(a, b, id=7)
  out: () = output_port(add.7, name=out, id=8)
}
  )";
  constexpr std::string_view text2 = R"(
package my_package2

block my_block(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=5)
  b: bits[32] = input_port(name=b, id=6)
  sub.7: bits[32] = sub(a, b, id=7)
  out: () = output_port(sub.7, name=out, id=8)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(auto pkg1, ParsePackage(text1));
  XLS_ASSERT_OK_AND_ASSIGN(auto pkg2, ParsePackage(text2));
  XLS_ASSERT_OK(pkg1->AddPackage(pkg2.get()).status());

  EXPECT_EQ(pkg1->blocks().size(), 2);
  XLS_EXPECT_OK(pkg1->GetBlock("my_block"));
  XLS_EXPECT_OK(pkg1->GetBlock("my_block_1"));
}

// TODO(google/xls#917): Remove this test when empty arrays are supported.
TEST_F(PackageTest, EmptyArrayIsError) {
  constexpr std::string_view text = R"(
package my_package


top fn main(a: bits[32]) -> bits[32][0] {
  ret array: bits[32][0] = literal(value=[])
}
)";
  EXPECT_THAT(ParsePackage(text).status(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Empty array Values are not supported.")));
}

TEST_F(PackageTest, TransformMetrics) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto literal0 = fb.Literal(UBits(123, 32));
  auto literal1 = fb.Literal(UBits(444, 32));
  auto add = fb.Add(literal0, literal1);
  auto sub = fb.Subtract(literal0, literal1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add));

  EXPECT_EQ(p->transform_metrics().nodes_added, 4);
  EXPECT_EQ(p->transform_metrics().nodes_removed, 0);
  EXPECT_EQ(p->transform_metrics().nodes_replaced, 0);
  EXPECT_EQ(p->transform_metrics().operands_replaced, 0);

  XLS_ASSERT_OK(
      literal0.node()->ReplaceUsesWithNew<Literal>(Value(UBits(42, 32))));

  EXPECT_EQ(p->transform_metrics().nodes_added, 5);
  EXPECT_EQ(p->transform_metrics().nodes_removed, 0);
  EXPECT_EQ(p->transform_metrics().nodes_replaced, 1);
  EXPECT_EQ(p->transform_metrics().operands_replaced, 2);

  XLS_ASSERT_OK(f->RemoveNode(sub.node()));

  EXPECT_EQ(p->transform_metrics().nodes_added, 5);
  EXPECT_EQ(p->transform_metrics().nodes_removed, 1);
  EXPECT_EQ(p->transform_metrics().nodes_replaced, 1);
  EXPECT_EQ(p->transform_metrics().operands_replaced, 2);
}

}  // namespace
}  // namespace xls
