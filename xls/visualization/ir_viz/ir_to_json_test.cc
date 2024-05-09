// Copyright 2021 The XLS Authors
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

#include "xls/visualization/ir_viz/ir_to_json.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

using ::testing::HasSubstr;

using IrToJsonTest = IrTestBase;

TEST_F(IrToJsonTest, SimpleFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string json, IrToJson(p.get(), *delay_estimator,
                                                      /*schedule=*/nullptr,
                                                      /*entry_name=*/"main"));
  // We can't compare the JSON to a golden file because the underlying proto to
  // JSON library is nondeterministic in the order of fields so just check a
  // couple things.
  VLOG(1) << json;
  EXPECT_THAT(json, HasSubstr(R"("name": "main")"));
  EXPECT_THAT(json, HasSubstr(R"("id": "f0")"));

  EXPECT_THAT(json, HasSubstr(R"("edges": [)"));
  EXPECT_THAT(json, HasSubstr(R"("nodes": [)"));
  EXPECT_THAT(json, HasSubstr(R"("id": "f0_p1")"));
  EXPECT_THAT(
      json,
      HasSubstr(
          "\"ir\": \"add.1: bits[32] = add(x: bits[32], y: bits[32], id=1)\""));
  EXPECT_THAT(json, HasSubstr(R"("name": "add.1")"));
  EXPECT_THAT(json, HasSubstr(R"("opcode": "add")"));
  EXPECT_THAT(
      json,
      HasSubstr(
          R"("known_bits": "0bXXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX_XXXX")"));
}

TEST_F(IrToJsonTest, SimpleFunctionWithSchedule) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y);
  BValue negate = fb.Negate(add);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScheduleCycleMap cycle_map;
  cycle_map[x.node()] = 0;
  cycle_map[y.node()] = 0;
  cycle_map[add.node()] = 1;
  cycle_map[negate.node()] = 2;
  PipelineSchedule schedule(f, cycle_map);
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));

  XLS_ASSERT_OK_AND_ASSIGN(std::string json, IrToJson(p.get(), *delay_estimator,
                                                      /*schedule=*/&schedule,
                                                      /*entry_name=*/"main"));
  VLOG(1) << json;
}

TEST_F(IrToJsonTest, MultipleFunctions) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

fn other(z: bits[32]) -> bits[32] {
  ret neg.1: bits[32] = neg(z, id=1)
}

fn main(x: bits[32], xx: bits[32]) -> bits[32] {
  add1: bits[32] = add(x, x)
  zero: bits[32] = literal(value=0)
  neg_x: bits[32] = invoke(x, to_apply=other)
  ret sub: bits[32] = sub(x, xx)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string json, IrToJson(p.get(), *delay_estimator,
                                                      /*schedule=*/nullptr,
                                                      /*entry_name=*/"main"));
  VLOG(1) << json;
}

TEST_F(IrToJsonTest, SimpleProc) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc the_proc(x: bits[32], y: bits[64], init={0, 42}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=in)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  next_x: bits[32] = add(x, rcv_data)
  not_y: bits[64] = not(y)
  send: token = send(rcv_token, next_x, channel=out)
  next (next_x, not_y)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string json, IrToJson(p.get(), *delay_estimator,
                                                      /*schedule=*/nullptr,
                                                      /*entry_name=*/"main"));
  VLOG(1) << json;
}

}  // namespace
}  // namespace xls
