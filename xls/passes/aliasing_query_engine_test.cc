// Copyright 2024 The XLS Authors
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

#include "xls/passes/aliasing_query_engine.h"

#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "cppitertools/combinations.hpp"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {
namespace {

using ::absl_testing::IsOk;
using ::testing::Optional;

class AliasingQueryEngineTest : public IrTestBase {};
TEST_F(AliasingQueryEngineTest, AliasFollowed) {
  std::unique_ptr<Package> p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Literal(UBits(12, 12), SourceInfo(), "a");
  BValue param = fb.Param("param", p->GetBitsType(12));
  BValue b = fb.Add(a, fb.Subtract(param, param), SourceInfo(), "b");
  BValue c = fb.Add(b, fb.Subtract(param, param), SourceInfo(), "c");
  BValue d = fb.Add(c, fb.Subtract(param, param), SourceInfo(), "d");
  ASSERT_THAT(fb.Build().status(), IsOk());

  // Add aliases in each order
  for (auto&& order : iter::combinations(std::vector<BValue>{b, c, d}, 3)) {
    for (auto&& links : iter::combinations(
             std::vector<std::pair<BValue, BValue>>{
                 {order[0], order[1]}, {order[1], order[2]}, {order[2], a}},
             3)) {
      AliasingQueryEngine uqe(std::make_unique<StatelessQueryEngine>());
      for (const auto& [from, to] : links) {
        EXPECT_THAT(uqe.AddAlias(from.node(), to.node()), IsOk());
      }
      EXPECT_THAT(uqe.KnownValue(a.node()), Optional(Value(UBits(12, 12))));
      EXPECT_THAT(uqe.KnownValue(b.node()), Optional(Value(UBits(12, 12))));
      EXPECT_THAT(uqe.KnownValue(c.node()), Optional(Value(UBits(12, 12))));
      EXPECT_THAT(uqe.KnownValue(d.node()), Optional(Value(UBits(12, 12))));
    }
  }
}

}  // namespace
}  // namespace xls
