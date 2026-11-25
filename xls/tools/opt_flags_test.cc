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

#include "xls/tools/opt_flags.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/commandlineflag.h"
#include "absl/flags/parse.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/passes/optimization_pass.h"
#include "xls/tools/opt_flags.pb.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

void ParseFlagsForTest(std::vector<std::string> args) {
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  std::string prog = "opt_flags_test";
  argv.push_back(prog.data());
  for (std::string& arg : args) {
    argv.push_back(arg.data());
  }
  absl::ParseCommandLine(static_cast<int>(argv.size()), argv.data());
}

std::string FlagDefault(std::string_view name) {
  absl::CommandLineFlag* flag = absl::FindCommandLineFlag(name);
  if (flag == nullptr) {
    return "<missing>";
  }
  return flag->DefaultValue();
}

TEST(OptFlagsTest, FlagDefaultsMatchOptFlagsProtoDefaultInstance) {
  const OptFlagsProto& defaults = OptFlagsProto::default_instance();

  EXPECT_EQ(FlagDefault("top"), defaults.top());
  EXPECT_EQ(FlagDefault("ir_dump_path"), defaults.ir_dump_path());
  EXPECT_EQ(FlagDefault("opt_level"), std::to_string(defaults.opt_level()));
  EXPECT_EQ(kMaxOptLevel, defaults.opt_level());
  EXPECT_EQ(FlagDefault("split_next_value_selects"),
            std::to_string(defaults.split_next_value_selects()));
  // TODO(rigge): Remove once kDefaultSplitNextValueSelects is replaced by
  // OptFlagsProto::default_instance().split_next_value_selects().
  EXPECT_EQ(kDefaultSplitNextValueSelects, defaults.split_next_value_selects());
  EXPECT_EQ(OptimizationPassOptions().split_next_value_selects,
            defaults.split_next_value_selects());
  EXPECT_EQ(FlagDefault("use_context_narrowing_analysis"),
            defaults.use_context_narrowing_analysis() ? "true" : "false");
  EXPECT_EQ(FlagDefault("optimize_for_best_case_throughput"),
            defaults.optimize_for_best_case_throughput() ? "true" : "false");
  EXPECT_EQ(FlagDefault("decouple_state_access"),
            defaults.decouple_state_access() ? "true" : "false");
  EXPECT_EQ(FlagDefault("enable_resource_sharing"),
            defaults.enable_resource_sharing() ? "true" : "false");
  EXPECT_EQ(FlagDefault("force_resource_sharing"),
            defaults.force_resource_sharing() ? "true" : "false");
  EXPECT_EQ(FlagDefault("passes_bisect_limit_is_error"),
            defaults.passes_bisect_limit_is_error() ? "true" : "false");
  EXPECT_EQ(FlagDefault("debug_optimizations"),
            defaults.debug_optimizations() ? "true" : "false");
}

TEST(OptFlagsTest, RejectsCombiningOptOptionsProtoAndRamRewritesPb) {
  absl::FlagSaver flag_saver;
  XLS_ASSERT_OK_AND_ASSIGN(TempFile opt_proto_file,
                           TempFile::CreateWithContent("opt_level: 2\n"));
  XLS_ASSERT_OK_AND_ASSIGN(TempFile ram_rewrites_file,
                           TempFile::CreateWithContent(""));

  ParseFlagsForTest(
      {absl::StrCat("--opt_options_proto=", opt_proto_file.path().string()),
       absl::StrCat("--ram_rewrites_pb=", ram_rewrites_file.path().string())});

  EXPECT_THAT(GetOptFlags(std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot combine 'opt_options_proto'")));
}

}  // namespace
}  // namespace xls
