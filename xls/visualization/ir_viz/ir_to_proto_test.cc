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

#include "xls/visualization/ir_viz/ir_to_proto.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/visualization/ir_viz/visualization.pb.h"

namespace xls {
namespace {

constexpr char kTestdataPath[] = "xls/visualization/ir_viz/testdata";

class IrToProtoTest : public IrTestBase {
 protected:
  std::filesystem::path GoldenFilePath(std::string_view file_ext) {
    return absl::StrFormat("%s/ir_to_html_test_%s.%s", kTestdataPath,
                           TestName(), file_ext);
  }
};

TEST_F(IrToProtoTest, SimpleFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(viz::Package proto,
                           IrToProto(p.get(), *delay_estimator,
                                     /*schedule=*/nullptr,
                                     /*entry_name=*/"main"));

  VLOG(1) << proto.ir_html();
  ExpectEqualToGoldenFile(GoldenFilePath("htmltext"), proto.ir_html());
}

TEST_F(IrToProtoTest, SimpleFunctionWithSchedule) {
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

  XLS_ASSERT_OK_AND_ASSIGN(viz::Package proto,
                           IrToProto(p.get(), *delay_estimator,
                                     /*schedule=*/&schedule,
                                     /*entry_name=*/"main"));

  VLOG(1) << proto.ir_html();
  ExpectEqualToGoldenFile(GoldenFilePath("htmltext"), proto.ir_html());
}

TEST_F(IrToProtoTest, MultipleFunctions) {
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
  XLS_ASSERT_OK_AND_ASSIGN(viz::Package proto,
                           IrToProto(p.get(), *delay_estimator,
                                     /*schedule=*/nullptr,
                                     /*entry_name=*/"main"));

  VLOG(1) << proto.ir_html();
  ExpectEqualToGoldenFile(GoldenFilePath("htmltext"), proto.ir_html());
}

TEST_F(IrToProtoTest, SimpleProc) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

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
  XLS_ASSERT_OK_AND_ASSIGN(viz::Package proto,
                           IrToProto(p.get(), *delay_estimator,
                                     /*schedule=*/nullptr,
                                     /*entry_name=*/"main"));

  VLOG(1) << proto.ir_html();
  ExpectEqualToGoldenFile(GoldenFilePath("htmltext"), proto.ir_html());
}

TEST_F(IrToProtoTest, SimpleProcWithNextValue) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)

top proc the_proc(x: bits[32], y: bits[64], init={0, 42}) {
  tkn: token = literal(value=token)
  rcv: (token, bits[32]) = receive(tkn, channel=in)
  rcv_token: token = tuple_index(rcv, index=0)
  rcv_data: bits[32] = tuple_index(rcv, index=1)
  next_x: bits[32] = add(x, rcv_data)
  not_y: bits[64] = not(y)
  send: token = send(rcv_token, next_x, channel=out)
  one: bits[1] = literal(value=1)
  next_value_x: () = next_value(param=x, value=next_x, predicate=one)
  next_value_y: () = next_value(param=y, value=not_y, predicate=one)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(viz::Package proto,
                           IrToProto(p.get(), *delay_estimator,
                                     /*schedule=*/nullptr,
                                     /*entry_name=*/"main"));

  VLOG(1) << proto.ir_html();
  ExpectEqualToGoldenFile(GoldenFilePath("htmltext"), proto.ir_html());
}

}  // namespace
}  // namespace xls
