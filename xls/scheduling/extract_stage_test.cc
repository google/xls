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

#include "xls/scheduling/extract_stage.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

namespace m = xls::op_matchers;

class ExtractStageTest : public IrTestBase {};

// Smoke test.
TEST_F(ExtractStageTest, SimpleExtraction) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  sub.2: bits[3] = sub(add.1, i1)
  or.3: bits[3] = or(sub.2, add.1)
  ret and.4: bits[3] = and(or.3, sub.2)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));
  ScheduleCycleMap cycle_map;
  cycle_map[FindNode("i0", function)] = 0;
  cycle_map[FindNode("i1", function)] = 0;
  cycle_map[FindNode("add.1", function)] = 0;
  cycle_map[FindNode("sub.2", function)] = 1;
  cycle_map[FindNode("or.3", function)] = 2;
  cycle_map[FindNode("and.4", function)] = 3;

  PipelineSchedule schedule(function, cycle_map, cycle_map.size());
  for (int i = 0; i < 4; i++) {
    XLS_ASSERT_OK_AND_ASSIGN(Function * stage_fn,
                             ExtractStage(function, schedule, i));
    EXPECT_EQ(stage_fn->name(),
              absl::StrFormat("%s_stage_%d", function->name(), i));
    std::string expected;

    switch (i) {
      case 0:
        EXPECT_THAT(
            stage_fn->return_value(),
            m::Tuple(m::Param("i1"), m::Add(m::Param("i0"), m::Param("i1"))));
        break;
      case 1:
        EXPECT_THAT(stage_fn->return_value(),
                    m::Tuple(m::Sub(m::Param(), m::Param())));
        break;
      case 2:
        EXPECT_THAT(stage_fn->return_value(),
                    m::Tuple(m::Or(m::Param(), m::Param())));
        break;
      case 3:
        EXPECT_THAT(stage_fn->return_value(), m::And(m::Param(), m::Param()));
        break;
      default:
        FAIL() << "Should never get here!";
    }

    bool found = false;
    for (const Node* node : function->nodes()) {
      if (node->GetName().find(expected) != std::string::npos) {
        found = true;
      }
    }
    EXPECT_TRUE(found);
  }
}

// Verifies that stages w/multiple outputs have them grouped into a tuple.
TEST_F(ExtractStageTest, TuplizesStageOutputs) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[1], i2: bits[8], i3: bits[23]) -> (bits[1], bits[8], bits[23]) {
  // Stage 0: Multiply each float component (i1-i3) by i0.
  umul.1: bits[1] = umul(i0, i1)
  umul.2: bits[8] = umul(i0, i2)
  umul.3: bits[23] = umul(i0, i3)

  // Stage 1: Do an smul, then tuple up the outputs.
  ret tuple.4: (bits[1], bits[8], bits[23]) = tuple(umul.1, umul.2, umul.3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));
  ScheduleCycleMap cycle_map;
  for (Node* node : function->nodes()) {
    int stage = 0;
    if (node->Is<Param>() || (node->Is<ArithOp>() && node->op() == Op::kUMul)) {
      stage = 0;
    } else {
      stage = 1;
    }
    cycle_map[node] = stage;
  }

  PipelineSchedule schedule(function, cycle_map, cycle_map.size());
  for (int stage = 0; stage < 2; stage++) {
    XLS_ASSERT_OK_AND_ASSIGN(Function * stage_fn,
                             ExtractStage(function, schedule, stage));
    EXPECT_EQ(stage_fn->name(),
              absl::StrFormat("%s_stage_%d", function->name(), stage));

    // If stage == 0, make sure the output is a 3-tuple. Otherwise...well, it
    // should actually be the same thing :)
    Node* output = stage_fn->return_value();
    Type* type = output->GetType();
    ASSERT_TRUE(type->IsTuple());
    TupleType* tuple_type = type->AsTupleOrDie();
    ASSERT_EQ(tuple_type->size(), 3);
    EXPECT_THAT(stage_fn->return_value(),
                AllOf(m::Tuple(), m::Type("(bits[1], bits[8], bits[23])")));
  }
}

}  // namespace
}  // namespace xls
