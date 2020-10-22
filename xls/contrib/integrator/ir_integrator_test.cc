// Copyright 2020 Google LLC
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

#include "xls/contrib/integrator/ir_integrator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

class IntegratorTest : public IrTestBase {};

TEST_F(IntegratorTest, NoSourceFunctions) {
  EXPECT_FALSE(IntegrationBuilder::Build({}).ok());
}

TEST_F(IntegratorTest, OneSourceFunction) {
  auto p = CreatePackage();

  FunctionBuilder fb_body("body", p.get());
  fb_body.Param("index", p->GetBitsType(2));
  fb_body.Param("acc", p->GetBitsType(2));
  fb_body.Literal(UBits(0b11, 2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * body_func, fb_body.Build());

  FunctionBuilder fb_double("double", p.get());
  auto double_in = fb_double.Param("in1", p->GetBitsType(2));
  fb_double.Add(double_in, double_in);
  XLS_ASSERT_OK_AND_ASSIGN(Function * double_func, fb_double.Build());

  FunctionBuilder fb_main("main", p.get());
  auto main_in1 = fb_main.Param("in1", p->GetBitsType(2));
  auto main_in_arr =
      fb_main.Param("in_arr", p->GetArrayType(2, p->GetBitsType(2)));
  fb_main.Map(main_in_arr, double_func, /*loc=*/std::nullopt, "map");
  fb_main.CountedFor(main_in1, /*trip_count=*/4, /*stride=*/1, body_func,
                     /*invariant_args=*/{}, /*loc=*/std::nullopt,
                     "counted_for");
  fb_main.Invoke(/*args=*/{main_in1}, double_func, /*loc=*/std::nullopt,
                 "invoke");
  XLS_ASSERT_OK_AND_ASSIGN(Function * main_func, fb_main.Build());

  auto get_function_names = [](Package* p) {
    std::vector<std::string> names;
    for (const auto& func : p->functions()) {
      names.push_back(func->name());
    }
    return names;
  };

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<IntegrationBuilder> builder,
                           IntegrationBuilder::Build({main_func}));

  // Original package is unchanged.
  EXPECT_THAT(get_function_names(p.get()),
              UnorderedElementsAre("body", "double", "main"));

  // Original CountedFor
  CountedFor* counted_original =
      FindNode("counted_for", main_func)->As<CountedFor>();
  EXPECT_EQ(counted_original->body(), body_func);
  EXPECT_EQ(counted_original->body()->name(), "body");
  EXPECT_EQ(counted_original->body()->package(), p.get());

  // Original Map
  Map* map_original = FindNode("map", main_func)->As<Map>();
  EXPECT_EQ(map_original->to_apply(), double_func);
  EXPECT_EQ(map_original->to_apply()->name(), "double");
  EXPECT_EQ(map_original->to_apply()->package(), p.get());

  // Original Invoke
  Invoke* invoke_original = FindNode("invoke", main_func)->As<Invoke>();
  EXPECT_EQ(invoke_original->to_apply(), double_func);
  EXPECT_EQ(invoke_original->to_apply()->name(), "double");
  EXPECT_EQ(invoke_original->to_apply()->package(), p.get());

  // builder package has copies of functions.
  EXPECT_THAT(get_function_names(builder->package()),
              UnorderedElementsAre("body", "double", "main"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * clone_body_func,
                           builder->package()->GetFunction("body"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * clone_double_func,
                           builder->package()->GetFunction("double"));

  // Clone CountedFor
  Function* main_clone_func = builder->integrated_function();
  CountedFor* counted_clone =
      FindNode("counted_for", main_clone_func)->As<CountedFor>();
  EXPECT_EQ(counted_clone->body(), clone_body_func);
  EXPECT_EQ(counted_clone->body()->name(), "body");
  EXPECT_EQ(counted_clone->body()->package(), builder->package());

  // Clone Map
  Map* map_clone = FindNode("map", main_clone_func)->As<Map>();
  EXPECT_EQ(map_clone->to_apply(), clone_double_func);
  EXPECT_EQ(map_clone->to_apply()->name(), "double");
  EXPECT_EQ(map_clone->to_apply()->package(), builder->package());

  // Clone Invoke
  Invoke* invoke_clone = FindNode("invoke", main_clone_func)->As<Invoke>();
  EXPECT_EQ(invoke_clone->to_apply(), clone_double_func);
  EXPECT_EQ(invoke_clone->to_apply()->name(), "double");
  EXPECT_EQ(invoke_clone->to_apply()->package(), builder->package());
}

}  // namespace
}  // namespace xls
