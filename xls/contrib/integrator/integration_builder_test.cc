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

#include "xls/contrib/integrator/integration_builder.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::testing::UnorderedElementsAre;

class IntegrationBuilderTest : public IrTestBase {};

TEST_F(IntegrationBuilderTest, IntegrationBuilderNoSourceFunctions) {
  EXPECT_FALSE(IntegrationBuilder::Build({}).ok());
}

TEST_F(IntegrationBuilderTest,
       IntegrationBuilderCopySourceFunctionsIntoPackage) {
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
  fb_main.Map(main_in_arr, double_func, SourceInfo(), "map");
  fb_main.CountedFor(main_in1, /*trip_count=*/4, /*stride=*/1, body_func,
                     /*invariant_args=*/{}, SourceInfo(), "counted_for");
  fb_main.Invoke(/*args=*/{main_in1}, double_func, SourceInfo(), "invoke");
  XLS_ASSERT_OK_AND_ASSIGN(Function * main_func, fb_main.Build());

  auto diff_package_dummy_p = CreatePackage();
  FunctionBuilder fb_diff_package_dummy("diff_package_dummy",
                                        diff_package_dummy_p.get());
  auto diff_package_dummy_in =
      fb_diff_package_dummy.Param("in", diff_package_dummy_p->GetBitsType(32));
  fb_diff_package_dummy.Identity(diff_package_dummy_in);
  XLS_ASSERT_OK_AND_ASSIGN(Function * diff_package_dummy,
                           fb_diff_package_dummy.Build());

  auto get_function_names = [](Package* p) {
    std::vector<std::string> names;
    for (const auto& func : p->functions()) {
      names.push_back(func->name());
    }
    return names;
  };

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build({main_func, diff_package_dummy}));

  // Original packages are unchanged.
  EXPECT_THAT(get_function_names(p.get()),
              UnorderedElementsAre("body", "double", "main"));
  EXPECT_THAT(get_function_names(diff_package_dummy_p.get()),
              UnorderedElementsAre("diff_package_dummy"));

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

  // Original diff_package_dummy
  EXPECT_THAT(diff_package_dummy->return_value(), m::Identity(m::Param("in")));

  // builder package has copies of functions.
  EXPECT_THAT(
      get_function_names(builder->package()),
      UnorderedElementsAre("body", "double", "main", "diff_package_dummy"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * main_clone_func,
                           builder->package()->GetFunction("main"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * clone_body_func,
                           builder->package()->GetFunction("body"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * clone_double_func,
                           builder->package()->GetFunction("double"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * clone_diff_package_dummy,
      builder->package()->GetFunction("diff_package_dummy"));

  // Clone CountedFor
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

  // Clone diff_package_dummy
  EXPECT_THAT(clone_diff_package_dummy->return_value(),
              m::Identity(m::Param("in")));
}

}  // namespace
}  // namespace xls
