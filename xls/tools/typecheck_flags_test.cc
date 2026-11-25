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
#include "xls/common/proto_test_utils.h"
#include "xls/tools/typecheck_flags.pb.h"

ABSL_DECLARE_FLAG(std::string, typecheck_trace_out_dir);
ABSL_DECLARE_FLAG(bool, typecheck_dump_traces);

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::xls::proto_testing::EqualsProto;

TEST(TypecheckFlagsTest, ProtoDefaultInstanceMatchesFlagDefaults) {
  EXPECT_THAT(GetTypecheckFlagsProto(), IsOkAndHolds(EqualsProto(R"pb(
                dump_inference_table: false
                dump_traces: false
                time_every_action: false
              )pb")));
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
