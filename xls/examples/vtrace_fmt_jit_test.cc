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
#include "xls/examples/vtrace_fmt_jit_wrapper.h"

namespace xls {
namespace examples {
namespace {

TEST(VtraceFmtJitTest, VerbosityTest) {
  EvaluatorOptions opts = EvaluatorOptions().set_max_trace_verbosity(20);
  XLS_ASSERT_OK_AND_ASSIGN(auto vtrace_fmt_jit, VtraceFmt::Create(opts));
  FunctionJit* func_vtrace = vtrace_fmt_jit->jit();

  std::vector<Value> args = {Value(UBits(1, 32)), Value(UBits(2, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(auto result, func_vtrace->Run(args));

  EXPECT_EQ(result.value, Value(UBits(3, 32)));
  EXPECT_THAT(
      result.events.GetTraceMessageStrings(),
      testing::ElementsAre("Verbosity level 0", "Verbosity level 2",
                           "Verbosity level 4", "Verbosity level 8",
                           "Verbosity level 16", "Trace verification."));
}

}  // namespace
}  // namespace examples
}  // namespace xls
