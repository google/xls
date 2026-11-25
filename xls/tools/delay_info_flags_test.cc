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

#include "xls/tools/delay_info_flags.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
#include "xls/common/proto_test_utils.h"
#include "xls/tools/delay_info_flags.pb.h"

namespace xls {
namespace {

using ::xls::proto_testing::EqualsProto;

void ParseFlagsForTest(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  std::string prog = "delay_info_flags_test";
  argv.push_back(prog.data());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  absl::ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

TEST(DelayInfoFlagsTest, ProtoDefaultInstanceMatchesFlagDefaults) {
  const DelayInfoFlagsProto& defaults = DelayInfoFlagsProto::default_instance();
  EXPECT_EQ(defaults.synthesis_server(), "ipv4:///0.0.0.0:10000");

  DelayInfoFlagsProto empty_proto;
  EXPECT_EQ(empty_proto.synthesis_server(), "ipv4:///0.0.0.0:10000");

  DelayInfoFlagsProto from_flags = GetDelayInfoFlagsProto("input.ir");
  EXPECT_THAT(from_flags, EqualsProto(R"pb(input_path: "input.ir")pb"));
  EXPECT_EQ(from_flags.synthesis_server(), defaults.synthesis_server());
}

TEST(DelayInfoFlagsTest, SynthesisServerOverrideFromFlag) {
  absl::FlagSaver flag_saver;
  ParseFlagsForTest({"--synthesis_server=ipv4:///127.0.0.1:12345"});

  EXPECT_THAT(GetDelayInfoFlagsProto("input.ir"), EqualsProto(R"pb(
                input_path: "input.ir"
                synthesis_server: "ipv4:///127.0.0.1:12345"
              )pb"));
}

}  // namespace
}  // namespace xls
