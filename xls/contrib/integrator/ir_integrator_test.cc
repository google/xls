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
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::ElementsAre;
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

TEST_F(IntegratorTest, MappingTestSimple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_1,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(2)));

  // Before mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_FALSE(integration->HasMapping(external_1));
  EXPECT_FALSE(integration->IsMappingTarget(internal_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(external_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(internal_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());

  // Mapping = external_1 -> MapsTo -> internal_1
  XLS_ASSERT_OK(integration->SetNodeMapping(external_1, internal_1));

  // After mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_TRUE(integration->HasMapping(external_1));
  EXPECT_TRUE(integration->IsMappingTarget(internal_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  ASSERT_THAT(integration->GetNodeMapping(external_1),
              IsOkAndHolds(internal_1));
  auto mapped_to_internal_1 = integration->GetNodesMappedToNode(internal_1);
  EXPECT_TRUE(mapped_to_internal_1.ok());
  EXPECT_THAT(*(mapped_to_internal_1.value()),
              UnorderedElementsAre(external_1));
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());
}

TEST_F(IntegratorTest, MappingTestMultipleNodesMapToTarget) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_1,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_2,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(3)));

  // Before mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_2));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_FALSE(integration->HasMapping(external_1));
  EXPECT_FALSE(integration->HasMapping(external_2));
  EXPECT_FALSE(integration->IsMappingTarget(internal_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_2));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(external_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(external_2).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(internal_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_2).ok());

  // Mapping = external_1 && external_2 -> MapsTo -> internal_1
  XLS_ASSERT_OK(integration->SetNodeMapping(external_1, internal_1));
  XLS_ASSERT_OK(integration->SetNodeMapping(external_2, internal_1));

  // After mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_2));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_TRUE(integration->HasMapping(external_1));
  EXPECT_TRUE(integration->HasMapping(external_2));
  EXPECT_TRUE(integration->IsMappingTarget(internal_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_2));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  ASSERT_THAT(integration->GetNodeMapping(external_1),
              IsOkAndHolds(internal_1));
  ASSERT_THAT(integration->GetNodeMapping(external_2),
              IsOkAndHolds(internal_1));
  auto mapped_to_internal_1 = integration->GetNodesMappedToNode(internal_1);
  EXPECT_TRUE(mapped_to_internal_1.ok());
  EXPECT_THAT(*(mapped_to_internal_1.value()),
              UnorderedElementsAre(external_1, external_2));
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_2).ok());
}

TEST_F(IntegratorTest, MappingTestRepeatedMapping) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_1,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(3)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_2,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_2",
                                            p->GetBitsType(4)));

  // Before mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_2));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_2));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_FALSE(integration->HasMapping(internal_2));
  EXPECT_FALSE(integration->HasMapping(external_1));
  EXPECT_FALSE(integration->HasMapping(external_2));
  EXPECT_FALSE(integration->IsMappingTarget(internal_1));
  EXPECT_FALSE(integration->IsMappingTarget(internal_2));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_2));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(internal_2).ok());
  EXPECT_FALSE(integration->GetNodeMapping(external_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(external_2).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(internal_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(internal_2).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_2).ok());

  // Mapping = external_1 && external_2 -> MapsTo -> internal_1
  XLS_ASSERT_OK(integration->SetNodeMapping(external_1, internal_1));
  XLS_ASSERT_OK(integration->SetNodeMapping(external_2, internal_1));

  // Mapping = external_1 && external_2 -> MapsTo -> internal_1 -> internal_2
  XLS_ASSERT_OK(integration->SetNodeMapping(internal_1, internal_2));

  // After mapping.
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_1));
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(internal_2));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_1));
  EXPECT_FALSE(integration->IntegrationFunctionOwnsNode(external_2));
  EXPECT_FALSE(integration->HasMapping(internal_1));
  EXPECT_FALSE(integration->HasMapping(internal_2));
  EXPECT_TRUE(integration->HasMapping(external_1));
  EXPECT_TRUE(integration->HasMapping(external_2));
  EXPECT_FALSE(integration->IsMappingTarget(internal_1));
  EXPECT_TRUE(integration->IsMappingTarget(internal_2));
  EXPECT_FALSE(integration->IsMappingTarget(external_1));
  EXPECT_FALSE(integration->IsMappingTarget(external_2));
  EXPECT_FALSE(integration->GetNodeMapping(internal_1).ok());
  EXPECT_FALSE(integration->GetNodeMapping(internal_2).ok());
  ASSERT_THAT(integration->GetNodeMapping(external_1),
              IsOkAndHolds(internal_2));
  ASSERT_THAT(integration->GetNodeMapping(external_2),
              IsOkAndHolds(internal_2));
  EXPECT_FALSE(integration->GetNodesMappedToNode(internal_1).ok());
  auto mapped_to_internal_2 = integration->GetNodesMappedToNode(internal_2);
  EXPECT_TRUE(mapped_to_internal_2.ok());
  EXPECT_THAT(*(mapped_to_internal_2.value()),
              UnorderedElementsAre(external_1, external_2));
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_1).ok());
  EXPECT_FALSE(integration->GetNodesMappedToNode(external_2).ok());
}

TEST_F(IntegratorTest, MappingTestSetNodeMappingFailureCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_1,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(3)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_2,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_2",
                                            p->GetBitsType(4)));

  // Mapping = external_1 -> MapsTo -> external_1
  // Mapping target must be internal.
  EXPECT_FALSE(integration->SetNodeMapping(external_1, external_1).ok());

  // Mapping = external_1 -> MapsTo -> external_2
  // Mapping target must be internal.
  EXPECT_FALSE(integration->SetNodeMapping(external_1, external_2).ok());

  // Mapping = internal_1 -> MapsTo -> external_1
  // Mapping target must be internal.
  EXPECT_FALSE(integration->SetNodeMapping(internal_1, external_1).ok());

  // Mapping = internal_1 -> MapsTo -> internal_1
  // Cannot map to self.
  EXPECT_FALSE(integration->SetNodeMapping(internal_1, internal_1).ok());

  // Mapping = internal_2 -> MapsTo -> internal_1
  // Cannot map internal nodes that are not mapping targets.
  EXPECT_FALSE(integration->SetNodeMapping(internal_2, internal_1).ok());
}

TEST_F(IntegratorTest, ParamterPacking) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  fb_a.Param("a1", p->GetBitsType(2));
  fb_a.Param("a2", p->GetBitsType(4));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  FunctionBuilder fb_b("func_b", p.get());
  fb_b.Param("b1", p->GetBitsType(6));
  fb_b.Param("b2", p->GetBitsType(8));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b}));

  auto GetTupleIndexWithNumBits = [&](int64 num_bits) {
    for (Node* node : integration->function()->nodes()) {
      if (node->op() == Op::kTupleIndex) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return absl::optional<Node*>(node);
        }
      }
    }
    return absl::optional<Node*>(std::nullopt);
  };
  auto GetParamWithNumBits = [&p](Function* function, int64 num_bits) {
    for (Node* node : function->nodes()) {
      if (node->op() == Op::kParam) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return absl::optional<Node*>(node);
        }
      }
    }
    return absl::optional<Node*>(std::nullopt);
  };

  auto a1_index = GetTupleIndexWithNumBits(2);
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_THAT(a1_index.value(), m::TupleIndex(m::Param("func_aParamTuple"), 0));
  auto a1_source = GetParamWithNumBits(func_a, 2);
  EXPECT_TRUE(a1_source.has_value());
  EXPECT_TRUE(integration->HasMapping(a1_source.value()));
  EXPECT_EQ(integration->GetNodeMapping(a1_source.value()).value(),
            a1_index.value());
  EXPECT_TRUE(integration->IsMappingTarget(a1_index.value()));
  EXPECT_THAT(*(integration->GetNodesMappedToNode(a1_index.value()).value()),
              UnorderedElementsAre(a1_source.value()));

  auto a2_index = GetTupleIndexWithNumBits(4);
  EXPECT_TRUE(a2_index.has_value());
  EXPECT_THAT(a2_index.value(), m::TupleIndex(m::Param("func_aParamTuple"), 1));
  auto a2_source = GetParamWithNumBits(func_a, 4);
  EXPECT_TRUE(a2_source.has_value());
  EXPECT_TRUE(integration->HasMapping(a2_source.value()));
  EXPECT_EQ(integration->GetNodeMapping(a2_source.value()).value(),
            a2_index.value());
  EXPECT_TRUE(integration->IsMappingTarget(a2_index.value()));
  EXPECT_THAT(*(integration->GetNodesMappedToNode(a2_index.value()).value()),
              UnorderedElementsAre(a2_source.value()));

  auto b1_index = GetTupleIndexWithNumBits(6);
  EXPECT_TRUE(b1_index.has_value());
  EXPECT_THAT(b1_index.value(), m::TupleIndex(m::Param("func_bParamTuple"), 0));
  auto b1_source = GetParamWithNumBits(func_b, 6);
  EXPECT_TRUE(b1_source.has_value());
  EXPECT_TRUE(integration->HasMapping(b1_source.value()));
  EXPECT_EQ(integration->GetNodeMapping(b1_source.value()).value(),
            b1_index.value());
  EXPECT_TRUE(integration->IsMappingTarget(b1_index.value()));
  EXPECT_THAT(*(integration->GetNodesMappedToNode(b1_index.value()).value()),
              UnorderedElementsAre(b1_source.value()));

  auto b2_index = GetTupleIndexWithNumBits(8);
  EXPECT_TRUE(b2_index.has_value());
  EXPECT_THAT(b2_index.value(), m::TupleIndex(m::Param("func_bParamTuple"), 1));
  auto b2_source = GetParamWithNumBits(func_b, 8);
  EXPECT_TRUE(b2_source.has_value());
  EXPECT_TRUE(integration->HasMapping(b2_source.value()));
  EXPECT_EQ(integration->GetNodeMapping(b2_source.value()).value(),
            b2_index.value());
  EXPECT_TRUE(integration->IsMappingTarget(b2_index.value()));
  EXPECT_THAT(*(integration->GetNodesMappedToNode(b2_index.value()).value()),
              UnorderedElementsAre(b2_source.value()));

  EXPECT_EQ(integration->function()->node_count(), 6);
}

TEST_F(IntegratorTest, GetIntegratedOperandsExternalNode) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a1 = fb_a.Param("a1", p->GetBitsType(2));
  auto a2 = fb_a.Param("a2", p->GetBitsType(2));
  fb_a.Add(a1, a2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(),
                                                                  {func_a}));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a1_node, func_a->GetNode("a1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a2_node, func_a->GetNode("a2"));
  Node* add_node = func_a->return_value();

  XLS_ASSERT_OK_AND_ASSIGN(Node * a1_map_target,
                           integration->GetNodeMapping(a1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a2_map_target,
                           integration->GetNodeMapping(a2_node));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> operand_mappings,
                           integration->GetIntegratedOperands(add_node));
  EXPECT_THAT(operand_mappings, ElementsAre(a1_map_target, a2_map_target));
}

TEST_F(IntegratorTest, GetIntegratedOperandsInternalNode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * cat,
      internal_func.MakeNode<Concat>(
          /*loc=*/std::nullopt, std::vector<Node*>({internal_1, internal_2})));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> operand_mappings,
                           integration->GetIntegratedOperands(cat));
  EXPECT_THAT(operand_mappings, ElementsAre(internal_1, internal_2));
}

TEST_F(IntegratorTest, UnifyIntegrationNodesSameNode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));

  int64 initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_1));
  EXPECT_EQ(unity.node, internal_1);
  EXPECT_EQ(initial_node_count, integration->function()->node_count());
}

TEST_F(IntegratorTest, UnifyIntegrationNodesSameNodeCheckMuxAdded) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));

  int64 initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_1));
  EXPECT_FALSE(unity.new_mux_added);
  EXPECT_EQ(unity.node, internal_1);
  EXPECT_EQ(initial_node_count, integration->function()->node_count());
}

TEST_F(IntegratorTest, UnifyIntegrationNodesDifferentNodes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));

  int64 initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  std::string select_name =
      internal_1->GetName() + "_" + internal_2->GetName() + "_mux_sel";
  XLS_ASSERT_OK_AND_ASSIGN(Node * mux_sel,
                           integration->function()->GetNode(select_name));
  EXPECT_EQ(mux_sel->users().size(), 1);
  Node* mux = mux_sel->users().at(0);
  EXPECT_EQ(unity.node, mux);
  EXPECT_THAT(unity.node,
              m::Select(m::Param(select_name),
                        {m::Param("internal_1"), m::Param("internal_2")}));
  EXPECT_EQ(initial_node_count + 2, integration->function()->node_count());
}

TEST_F(IntegratorTest, UnifyIntegrationNodesDifferentNodesRepeated) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));

  int64 initial_node_count = integration->function()->node_count();
  for (int64 i = 0; i < 10; ++i) {
    XLS_ASSERT_OK(integration->UnifyIntegrationNodes(internal_1, internal_2));
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  std::string select_name =
      internal_1->GetName() + "_" + internal_2->GetName() + "_mux_sel";
  XLS_ASSERT_OK_AND_ASSIGN(Node * mux_sel,
                           integration->function()->GetNode(select_name));
  EXPECT_EQ(mux_sel->users().size(), 1);
  Node* mux = mux_sel->users().at(0);
  EXPECT_EQ(unity.node, mux);
  EXPECT_THAT(unity.node,
              m::Select(m::Param(select_name),
                        {m::Param("internal_1"), m::Param("internal_2")}));
  EXPECT_EQ(initial_node_count + 2, integration->function()->node_count());
}

TEST_F(IntegratorTest,
       UnifyIntegrationNodesDifferentNodesRepeatedCheckMuxAdded) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));

  int64 initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  EXPECT_TRUE(unity.new_mux_added);

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity2,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  EXPECT_FALSE(unity2.new_mux_added);

  std::string select_name =
      internal_1->GetName() + "_" + internal_2->GetName() + "_mux_sel";
  XLS_ASSERT_OK_AND_ASSIGN(Node * mux_sel,
                           integration->function()->GetNode(select_name));
  EXPECT_EQ(mux_sel->users().size(), 1);
  Node* mux = mux_sel->users().at(0);
  EXPECT_EQ(unity.node, mux);
  EXPECT_THAT(unity.node,
              m::Select(m::Param(select_name),
                        {m::Param("internal_1"), m::Param("internal_2")}));
  EXPECT_EQ(initial_node_count + 2, integration->function()->node_count());
}

TEST_F(IntegratorTest, UnifyIntegrationNodesDifferentNodesFailureCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_1,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_1",
                                            p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_2,
      internal_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "internal_2",
                                            p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_1,
      external_func.MakeNodeWithName<Param>(/*loc=*/std::nullopt, "external_1",
                                            p->GetBitsType(2)));

  // Can't unify with external node.
  EXPECT_FALSE(integration->UnifyIntegrationNodes(internal_1, external_1).ok());
  EXPECT_FALSE(integration->UnifyIntegrationNodes(external_1, internal_1).ok());

  // Unifying types must match.
  EXPECT_FALSE(integration->UnifyIntegrationNodes(internal_2, internal_1).ok());
}

TEST_F(IntegratorTest, InsertNodeSimple) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a1 = fb_a.Param("a1", p->GetBitsType(2));
  auto a2 = fb_a.Param("a2", p->GetBitsType(2));
  fb_a.Add(a1, a2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(),
                                                                  {func_a}));

  Node* a1_node = FindNode("a1", func_a);
  Node* a2_node = FindNode("a2", func_a);
  Node* add_node = func_a->return_value();

  XLS_ASSERT_OK_AND_ASSIGN(Node * integrated_add,
                           integration->InsertNode(add_node));

  EXPECT_TRUE(integration->HasMapping(add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * add_node_map,
                           integration->GetNodeMapping(add_node));
  EXPECT_EQ(add_node_map, integrated_add);
  EXPECT_TRUE(integration->IntegrationFunctionOwnsNode(integrated_add));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a1_map_target,
                           integration->GetNodeMapping(a1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a2_map_target,
                           integration->GetNodeMapping(a2_node));
  EXPECT_THAT(integrated_add->operands(),
              ElementsAre(a1_map_target, a2_map_target));
}

TEST_F(IntegratorTest, InsertNodeRepeatedly) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a1 = fb_a.Param("a1", p->GetBitsType(2));
  auto a2 = fb_a.Param("a2", p->GetBitsType(2));
  fb_a.Add(a1, a2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(),
                                                                  {func_a}));
  Node* add_node = func_a->return_value();

  XLS_ASSERT_OK(integration->InsertNode(add_node));
  auto repeat_result = integration->InsertNode(add_node);
  EXPECT_FALSE(repeat_result.ok());
}

}  // namespace
}  // namespace xls
