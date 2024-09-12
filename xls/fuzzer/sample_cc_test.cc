// Copyright 2023 The XLS Authors
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

#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/interp_value.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"

namespace xls {
namespace {

const char kDslxFunction[] = R"(fn foo(x: u32) -> u32 {
  x
})";

const char kDslxProc[] = R"(proc foo {
  config() {
    ()
  }
  init { () }
  next(s: ()) {
    s
  }
})";

TEST(SampleCcTest, SerializeDeserializeFunction) {
  SampleOptions options;
  Sample sample(
      kDslxFunction, options,
      {{dslx::InterpValue::MakeU32(42)}, {dslx::InterpValue::MakeU32(123)}});
  XLS_ASSERT_OK_AND_ASSIGN(Sample copy,
                           Sample::Deserialize(sample.Serialize()));

  VLOG(1) << sample.Serialize();
  EXPECT_EQ(sample, copy);

  EXPECT_EQ(sample.options().input_is_dslx(), copy.options().input_is_dslx());
  EXPECT_EQ(sample.options().sample_type(), copy.options().sample_type());
  EXPECT_EQ(sample.options().ir_converter_args(),
            copy.options().ir_converter_args());
  EXPECT_EQ(sample.options().convert_to_ir(), copy.options().convert_to_ir());
  EXPECT_EQ(sample.options().optimize_ir(), copy.options().optimize_ir());

  EXPECT_EQ(sample.args_batch(), copy.args_batch());
  EXPECT_EQ(sample.args_batch().size(), 2);
  EXPECT_EQ(sample.args_batch()[0].size(), 1);
  EXPECT_EQ(sample.args_batch()[0][0].ToString(), "u32:42");
  EXPECT_EQ(sample.args_batch()[1].size(), 1);
  EXPECT_EQ(sample.args_batch()[1][0].ToString(), "u32:123");
}

TEST(SampleCcTest, DeserializationCanHandleNewlinesInStringLiterals) {
  // Due to file-formatting, it can happen that strings are separated
  // like in the following args example
  static constexpr std::string_view kNewlinedConfig = R"""(
// BEGIN_CONFIG
// # proto-message: xls.fuzzer.CrasherConfigurationProto
// issue: "Foo"
// inputs {
//   function_args {
//     args: "(bits[32]:0x01,
//             bits[32]:0x02,
//             bits[32]:0x03)"
//   }
// }
// END_CONFIG
)""";
  XLS_ASSERT_OK_AND_ASSIGN(Sample parsed, Sample::Deserialize(kNewlinedConfig));

  EXPECT_EQ(parsed.args_batch().size(), 1);
  EXPECT_EQ(parsed.args_batch()[0].size(), 1);
  EXPECT_EQ(parsed.args_batch()[0][0].ToString(), "(u32:1, u32:2, u32:3)");
}

TEST(SampleCcTest, SerializeDeserializeProc) {
  SampleOptions options;
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_ir_converter_args({"--foo", "--bar"});
  options.set_codegen(true);
  options.set_codegen_args({"--quux"});

  Sample sample(
      kDslxProc, options,
      {{dslx::InterpValue::MakeU32(42), dslx::InterpValue::MakeU32(123)}},
      std::vector<std::string>({"channel0", "channel1"}));

  XLS_ASSERT_OK_AND_ASSIGN(Sample copy,
                           Sample::Deserialize(sample.Serialize()));

  VLOG(1) << sample.Serialize();
  EXPECT_EQ(sample, copy);

  EXPECT_EQ(sample.options().sample_type(), copy.options().sample_type());
  EXPECT_EQ(sample.options().codegen(), copy.options().codegen());
  EXPECT_EQ(sample.options().codegen_args(), copy.options().codegen_args());

  EXPECT_EQ(sample.args_batch(), copy.args_batch());
  EXPECT_EQ(sample.args_batch().size(), 1);
  EXPECT_EQ(sample.args_batch()[0].size(), 2);
  EXPECT_EQ(sample.args_batch()[0][0].ToString(), "u32:42");
  EXPECT_EQ(sample.args_batch()[0][1].ToString(), "u32:123");

  EXPECT_THAT(sample.ir_channel_names(),
              testing::ElementsAre("channel0", "channel1"));
}

}  // namespace
}  // namespace xls
