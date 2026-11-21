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

#include "xls/tools/typecheck_flags.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/tools/typecheck_flags.pb.h"

ABSL_DECLARE_FLAG(std::string, typecheck_trace_out_dir);
ABSL_DECLARE_FLAG(bool, typecheck_dump_traces);

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TypecheckFlagsTest, ProtoDefaultInstanceMatchesFlagDefaults) {
  const TypecheckFlagsProto& defaults = TypecheckFlagsProto::default_instance();
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckFlagsProto from_flags,
                           GetTypecheckFlagsProto());

  EXPECT_FALSE(from_flags.has_trace_out_dir());
  EXPECT_EQ(from_flags.trace_out_dir(), defaults.trace_out_dir());
  EXPECT_EQ(from_flags.dump_inference_table(), defaults.dump_inference_table());
  EXPECT_EQ(from_flags.dump_traces(), defaults.dump_traces());
  EXPECT_EQ(from_flags.time_every_action(), defaults.time_every_action());
}

TEST(TypecheckFlagsTest, DumpTracesWithoutOutDirReturnsError) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_typecheck_trace_out_dir, "");
  absl::SetFlag(&FLAGS_typecheck_dump_traces, true);

  EXPECT_THAT(GetTypecheckFlagsProto(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--typecheck_trace_out_dir")));
}

}  // namespace
}  // namespace xls
