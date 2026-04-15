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

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/vtrace_fmt_block_jit_wrapper.h"

namespace xls {
namespace examples {
namespace {

TEST(VtraceFmtJitTest, VerbosityTest) {
  // The maximum vtrace verbosity cannot be greater than that set by the
  // `max_trace_verbosity` flag for the `vtrace_fmt_verilog` target on which
  // this test depends. You will not get logs with greater verbosity even after
  // setting the evaluator option because the logs are filtered from the
  // generated block IR file which does not contain such logs.
  EvaluatorOptions opts = EvaluatorOptions().set_max_trace_verbosity(16);
  XLS_ASSERT_OK_AND_ASSIGN(auto vtrace_block_jit, VtraceFmtBlock::Create(opts));
  auto cont = vtrace_block_jit->NewContinuation();
  XLS_ASSERT_OK(vtrace_block_jit->RunOneCycle(*cont));
  EXPECT_THAT(cont->interpreter_events().GetAssertMessages(),
              testing::IsEmpty());
  EXPECT_THAT(
      cont->interpreter_events().GetTraceMessageStrings(),
      testing::ElementsAre("Verbosity level 0", "Verbosity level 2",
                           "Verbosity level 4", "Verbosity level 8",
                           "Verbosity level 16", "Trace verification."));
}

}  // namespace
}  // namespace examples
}  // namespace xls
