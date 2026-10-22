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
#include "xls/examples/vtrace_fmt_proc_jit_wrapper.h"

namespace xls {
namespace examples {
namespace {

TEST(VtraceFmtProcJitTest, VerbosityTest) {
  EvaluatorOptions opts = EvaluatorOptions().set_max_trace_verbosity(10);
  XLS_ASSERT_OK_AND_ASSIGN(auto proc_vtrace_jit, VtraceFmtProc::Create(opts));
  XLS_EXPECT_OK(proc_vtrace_jit->SendToChannel(
      "_trigger", "__vtrace_fmt__Vprinter_0_next", Value(UBits(1, 1))));
  XLS_ASSERT_OK(proc_vtrace_jit->Tick());

  ProcRuntime* rt = proc_vtrace_jit->runtime();
  const auto& instances = rt->elaboration().proc_instances();
  ASSERT_EQ(instances.size(), 1);
  xls::ProcInstance* instance = instances[0];
  const InterpreterEvents& events = rt->GetInterpreterEvents(instance);

  EXPECT_THAT(events.GetTraceMessageStrings(),
              testing::ElementsAre("Verbosity level 0", "Verbosity level 8",
                                   "Trace verification."));
}

}  // namespace
}  // namespace examples
}  // namespace xls
