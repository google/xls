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

#include "xls/common/proto_test_utils.h"

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "xls/common/testdata/matcher_test_messages.pb.h"

namespace xls::proto_testing {
namespace {
using ::testing::Not;

Foo MakeFoo(std::string_view sv) {
  Foo foo;
  EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(sv, &foo));
  return foo;
}

TEST(ProtoTestUtilsTest, EqualsProto) {
  Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb"));
  EXPECT_THAT(foo, EqualsProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c"
              )pb"));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foobar" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo,
              Not(EqualsProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c" r3: "d"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foo" i2: 32 r3: "a" r3: "b" r3: "c"
              )pb")));
}

TEST(ProtoTestUtilsTest, Partially) {
  Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                s1: "foo"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                r3: "a" r3: "b" r3: "c"
              )pb")));
  // bad order
  EXPECT_THAT(foo,
              Not(Partially(EqualsProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a"
              )pb"))));
  // new value
  EXPECT_THAT(foo, Not(Partially(EqualsProto(R"pb(
                s1: "foo"
                i2: 10
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
}

TEST(ProtoTestUtilsTest, IgnoringOrder) {
  Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "c"
                r3: "b"
              )pb")));
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "a"
                r3: "c"
                s1: "foo"
                r3: "b"
              )pb")));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foobar"
                r3: "b"
                r3: "a"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                s1: "foo"
                r3: "c"
                r3: "d"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                i2: 32
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
}

TEST(ProtoTestUtilsTest, PartiallyIgnoringOrder) {
  Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
              )pb"))));
  // bad order
  EXPECT_THAT(foo, Not(Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "bar"
                r3: "b"
                r3: "c"
                r3: "a"
              )pb")))));
  // new value
  EXPECT_THAT(foo, Not(Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                i2: 10
                r3: "b"
                r3: "a"
                r3: "c"
              )pb")))));
}

}  // namespace
}  // namespace xls::proto_testing
