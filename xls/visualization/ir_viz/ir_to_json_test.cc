// Copyright 2020 The XLS Authors
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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::HasSubstr;

class IrToJsonTest : public IrTestBase {};

TEST_F(IrToJsonTest, SimpleFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(R"(
package test

fn other(z: bits[32]) -> bits[32] {
  ret result: bits[32] = neg(z)
}

fn main(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * entry, p->EntryFunction());
  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string json, IrToJson(p.get(), *delay_estimator,
                                                      nullptr, entry->name()));
  XLS_VLOG(1) << json;
  // Match several substrings in the JSON. Avoid a full string match because
  // then this test becomes a change detector for the sources of the metadata
  // included in the graph (delay estimation, etc).
  EXPECT_THAT(json, HasSubstr(R"("name": "other")"));
  EXPECT_THAT(json, HasSubstr(R"("name": "main")"));

  EXPECT_THAT(json, HasSubstr(R"("edges": [)"));
  EXPECT_THAT(json, HasSubstr(R"("nodes": [)"));
  EXPECT_THAT(json, HasSubstr(R"("id": "add_1")"));
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
                                                      &schedule, f->name()));

  XLS_VLOG(1) << json;
  // Match several substrings in the JSON. Avoid a full string match because
  // then this test becomes a change detector for the sources of the metadata
  // included in the graph (delay estimation, etc).
  EXPECT_THAT(json, HasSubstr(R"("name": "x")"));
  EXPECT_THAT(json, HasSubstr(R"("name": "y")"));
  EXPECT_THAT(json, HasSubstr(R"("name": "add.3")"));
  EXPECT_THAT(json, HasSubstr(R"("name": "neg.4")"));

  EXPECT_THAT(json, HasSubstr(R"("cycle": 0)"));
  EXPECT_THAT(json, HasSubstr(R"("cycle": 1)"));
  EXPECT_THAT(json, HasSubstr(R"("cycle": 2)"));
}

}  // namespace
}  // namespace xls
