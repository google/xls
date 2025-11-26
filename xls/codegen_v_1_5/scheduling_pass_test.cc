// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/scheduling_pass.h"

#include <memory>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/codegen.h"

namespace xls::codegen {
namespace {

class SchedulingPassTest : public IrTestBase {
 protected:
  absl::StatusOr<std::string> RunPass(std::string_view input,
                                      int stage_count = 2) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                         Parser::ParsePackage(input));

    TestDelayEstimator delay_estimator;

    XLS_ASSIGN_OR_RETURN(
        SchedulingResult scheduling_result,
        Schedule(package.get(),
                 SchedulingOptions().pipeline_stages(stage_count),
                 &delay_estimator));

    SchedulingPass pass;
    PassResults results;
    BlockConversionPassOptions options;
    options.package_schedule = scheduling_result.package_schedule;
    XLS_ASSIGN_OR_RETURN(bool result,
                         pass.Run(package.get(), options, &results));
    XLS_RET_CHECK(result);
    return package->DumpIr();
  }

  TestDelayEstimator delay_estimator_;
};

TEST_F(SchedulingPassTest, SimpleFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string output, RunPass(R"(
package test

top fn __test__f(a: bits[32] id=1, b: bits[32] id=2, c: bits[32] id=3, d: bits[32] id=4) -> bits[32] {
  umul.5: bits[32] = umul(a, b, id=5)
  umul.6: bits[32] = umul(c, d, id=6)
  ret add.7: bits[32] = add(umul.5, umul.6, id=7)
}
)"));

  EXPECT_EQ(output, R"(package test

top scheduled_fn __test__f(a: bits[32] id=1, b: bits[32] id=2, c: bits[32] id=3, d: bits[32] id=4) -> bits[32] {
  stage {
    umul.5: bits[32] = umul(a, b, id=5)
    umul.6: bits[32] = umul(c, d, id=6)
  }
  stage {
    ret add.7: bits[32] = add(umul.5, umul.6, id=7)
  }
}
)");
}

TEST_F(SchedulingPassTest, SimpleProcWithProcScopedChannels) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string output, RunPass(R"(
package test

top proc __test__P_0_next<a: bits[32] in, b: bits[32] in, result: bits[32] out>(__state: bits[32], init={0}) {
  chan_interface a(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface b(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface result(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  after_all.5: token = after_all(id=5)
  literal.3: bits[1] = literal(value=1, id=3)
  receive.6: (token, bits[32]) = receive(after_all.5, predicate=literal.3, channel=a, id=6)
  tok: token = tuple_index(receive.6, index=0, id=8)
  receive.10: (token, bits[32]) = receive(tok, predicate=literal.3, channel=b, id=10)
  a_value: bits[32] = tuple_index(receive.6, index=1, id=9)
  b_value: bits[32] = tuple_index(receive.10, index=1, id=13)
  umul.14: bits[32] = umul(a_value, b_value, id=14)
  __state: bits[32] = state_read(state_element=__state, id=2)
  tok__1: token = tuple_index(receive.10, index=0, id=12)
  result_value: bits[32] = add(umul.14, __state, id=15)
  __token: token = literal(value=token, id=1)
  tuple.4: () = tuple(id=4)
  tuple_index.7: token = tuple_index(receive.6, index=0, id=7)
  tuple_index.11: token = tuple_index(receive.10, index=0, id=11)
  send.16: token = send(tok__1, result_value, predicate=literal.3, channel=result, id=16)
  next_value.17: () = next_value(param=__state, value=result_value, id=17)
}

)"));

  EXPECT_EQ(output, R"(package test

top scheduled_proc __test__P_0_next<a: bits[32] in, b: bits[32] in, result: bits[32] out>(__state: bits[32], init={0}) {
  chan_interface a(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface b(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  chan_interface result(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  stage {
    after_all.5: token = after_all(id=5)
    receive.6: (token, bits[32]) = receive(after_all.5, predicate=literal.3, channel=a, id=6)
    tok: token = tuple_index(receive.6, index=0, id=8)
    receive.10: (token, bits[32]) = receive(tok, predicate=literal.3, channel=b, id=10)
    a_value: bits[32] = tuple_index(receive.6, index=1, id=9)
    b_value: bits[32] = tuple_index(receive.10, index=1, id=13)
    umul.14: bits[32] = umul(a_value, b_value, id=14)
    tok__1: token = tuple_index(receive.10, index=0, id=12)
  }
  stage {
    __state: bits[32] = state_read(state_element=__state, id=2)
    result_value: bits[32] = add(umul.14, __state, id=15)
    send.16: token = send(tok__1, result_value, predicate=literal.3, channel=result, id=16)
    next_value.17: () = next_value(param=__state, value=result_value, id=17)
  }
}
)");
}

TEST_F(SchedulingPassTest, SimpleProcWithGlobalChannels) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string output, RunPass(R"(
package test

chan test__a(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan test__b(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan test__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)

top proc __test__P_0_next(__state: bits[32], init={0}) {
  after_all.4: token = after_all(id=4)
  literal.3: bits[1] = literal(value=1, id=3)
  receive.5: (token, bits[32]) = receive(after_all.4, predicate=literal.3, channel=test__a, id=5)
  tok: token = tuple_index(receive.5, index=0, id=7)
  receive.9: (token, bits[32]) = receive(tok, predicate=literal.3, channel=test__b, id=9)
  a_value: bits[32] = tuple_index(receive.5, index=1, id=8)
  b_value: bits[32] = tuple_index(receive.9, index=1, id=12)
  umul.13: bits[32] = umul(a_value, b_value, id=13)
  __state: bits[32] = state_read(state_element=__state, id=2)
  tok__1: token = tuple_index(receive.9, index=0, id=11)
  result_value: bits[32] = add(umul.13, __state, id=14)
  __token: token = literal(value=token, id=1)
  tuple_index.6: token = tuple_index(receive.5, index=0, id=6)
  tuple_index.10: token = tuple_index(receive.9, index=0, id=10)
  send.15: token = send(tok__1, result_value, predicate=literal.3, channel=test__result, id=15)
  next_value.16: () = next_value(param=__state, value=result_value, id=16)
}

)"));

  EXPECT_EQ(output, R"(package test

chan test__a(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan test__b(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan test__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive)

top scheduled_proc __test__P_0_next(__state: bits[32], init={0}) {
  stage {
    after_all.4: token = after_all(id=4)
    receive.5: (token, bits[32]) = receive(after_all.4, predicate=literal.3, channel=test__a, id=5)
    tok: token = tuple_index(receive.5, index=0, id=7)
    receive.9: (token, bits[32]) = receive(tok, predicate=literal.3, channel=test__b, id=9)
    a_value: bits[32] = tuple_index(receive.5, index=1, id=8)
    b_value: bits[32] = tuple_index(receive.9, index=1, id=12)
    umul.13: bits[32] = umul(a_value, b_value, id=13)
    tok__1: token = tuple_index(receive.9, index=0, id=11)
  }
  stage {
    __state: bits[32] = state_read(state_element=__state, id=2)
    result_value: bits[32] = add(umul.13, __state, id=14)
    send.15: token = send(tok__1, result_value, predicate=literal.3, channel=test__result, id=15)
    next_value.16: () = next_value(param=__state, value=result_value, id=16)
  }
}
)");
}

}  // namespace
}  // namespace xls::codegen
