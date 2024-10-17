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

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class IntegratorTest : public IrTestBase {};

TEST_F(IntegratorTest, MappingTestSimple) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(p.get(), {}));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(1), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "external_1"));

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

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(1), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "external_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_2,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(3), "external_1"));

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

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(1), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(3), "external_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_2,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(4), "external_2"));

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

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(1), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(3), "external_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_2,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(4), "external_2"));

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

TEST_F(IntegratorTest, ParameterPacking) {
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
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  auto GetTupleIndexWithNumBits = [&](int64_t num_bits) {
    for (Node* node : integration->function()->nodes()) {
      if (node->op() == Op::kTupleIndex) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return std::optional<Node*>(node);
        }
      }
    }
    return std::optional<Node*>(std::nullopt);
  };
  auto GetParamWithNumBits = [&p](Function* function, int64_t num_bits) {
    for (Node* node : function->nodes()) {
      if (node->op() == Op::kParam) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return std::optional<Node*>(node);
        }
      }
    }
    return std::optional<Node*>(std::nullopt);
  };

  auto a1_index = GetTupleIndexWithNumBits(2);
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_THAT(a1_index.value(),
              m::TupleIndex(m::Param("func_a_ParamTuple"), 0));
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
  EXPECT_THAT(a2_index.value(),
              m::TupleIndex(m::Param("func_a_ParamTuple"), 1));
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
  EXPECT_THAT(b1_index.value(),
              m::TupleIndex(m::Param("func_b_ParamTuple"), 0));
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
  EXPECT_THAT(b2_index.value(),
              m::TupleIndex(m::Param("func_b_ParamTuple"), 1));
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

TEST_F(IntegratorTest, ParameterPackingUniversalMuxSelect) {
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
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  auto GetTupleIndexWithNumBits = [&](int64_t num_bits) {
    for (Node* node : integration->function()->nodes()) {
      if (node->op() == Op::kTupleIndex) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return std::optional<Node*>(node);
        }
      }
    }
    return std::optional<Node*>(std::nullopt);
  };
  auto GetParamWithNumBits = [&p](Function* function, int64_t num_bits) {
    for (Node* node : function->nodes()) {
      if (node->op() == Op::kParam) {
        if (node->GetType() == p->GetBitsType(num_bits)) {
          return std::optional<Node*>(node);
        }
      }
    }
    return std::optional<Node*>(std::nullopt);
  };

  auto a1_index = GetTupleIndexWithNumBits(2);
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_TRUE(a1_index.has_value());
  EXPECT_THAT(a1_index.value(),
              m::TupleIndex(m::Param("func_a_ParamTuple"), 0));
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
  EXPECT_THAT(a2_index.value(),
              m::TupleIndex(m::Param("func_a_ParamTuple"), 1));
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
  EXPECT_THAT(b1_index.value(),
              m::TupleIndex(m::Param("func_b_ParamTuple"), 0));

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
  EXPECT_THAT(b2_index.value(),
              m::TupleIndex(m::Param("func_b_ParamTuple"), 1));

  auto b2_source = GetParamWithNumBits(func_b, 8);
  EXPECT_TRUE(b2_source.has_value());
  EXPECT_TRUE(integration->HasMapping(b2_source.value()));
  EXPECT_EQ(integration->GetNodeMapping(b2_source.value()).value(),
            b2_index.value());
  EXPECT_TRUE(integration->IsMappingTarget(b2_index.value()));
  EXPECT_THAT(*(integration->GetNodesMappedToNode(b2_index.value()).value()),
              UnorderedElementsAre(b2_source.value()));

  EXPECT_EQ(integration->function()->node_count(), 7);
  auto global_mux_select =
      FindNode("global_mux_select", integration->function());
  EXPECT_EQ(global_mux_select->GetType(), p->GetBitsType(1));
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

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * cat,
      internal_func.MakeNode<Concat>(
          SourceInfo(), std::vector<Node*>({internal_1, internal_2})));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> operand_mappings,
                           integration->GetIntegratedOperands(cat));
  EXPECT_THAT(operand_mappings, ElementsAre(internal_1, internal_2));
}

TEST_F(IntegratorTest, UnifyIntegrationNodesSameNode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));

  int64_t initial_node_count = integration->function()->node_count();
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
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));

  int64_t initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_1));
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNoChange);
  EXPECT_EQ(unity.node, internal_1);
  EXPECT_EQ(initial_node_count, integration->function()->node_count());
}

TEST_F(IntegratorTest, UnifyIntegrationNodesDifferentNodes) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));

  int64_t initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  std::string select_name =
      internal_1->GetName() + "_" + internal_2->GetName() + "_mux_sel";
  XLS_ASSERT_OK_AND_ASSIGN(Node * mux_sel,
                           integration->function()->GetNode(select_name));
  EXPECT_EQ(mux_sel->users().size(), 1);
  Node* mux = *mux_sel->users().begin();
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
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));

  int64_t initial_node_count = integration->function()->node_count();
  for (int64_t i = 0; i < 10; ++i) {
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
  Node* mux = *mux_sel->users().begin();
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
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));

  int64_t initial_node_count = integration->function()->node_count();
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNewMuxAdded);

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity2,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  EXPECT_EQ(unity2.change, IntegrationFunction::UnificationChange::kNoChange);

  std::string select_name =
      internal_1->GetName() + "_" + internal_2->GetName() + "_mux_sel";
  XLS_ASSERT_OK_AND_ASSIGN(Node * mux_sel,
                           integration->function()->GetNode(select_name));
  EXPECT_EQ(mux_sel->users().size(), 1);
  Node* mux = *mux_sel->users().begin();
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
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(1), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "external_1"));

  // Can't unify with external node.
  EXPECT_FALSE(integration->UnifyIntegrationNodes(internal_1, external_1).ok());
  EXPECT_FALSE(integration->UnifyIntegrationNodes(external_1, internal_1).ok());

  // Unifying types must match.
  EXPECT_FALSE(integration->UnifyIntegrationNodes(internal_2, internal_1).ok());
}

TEST_F(IntegratorTest, UnifyIntegrationGlobalSelectCreateMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* a_add_node = FindNode("add", func_a);
  Node* c_add_node = FindNode("add", func_c);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));

  EXPECT_EQ(integration->function()->node_count(), 9);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));

  // Check added mux.
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(integration->function()->node_count(), 10);
  EXPECT_EQ(unity.node->op(), Op::kSel);
  Select* mux = unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), a_add_internal);
  EXPECT_THAT(mux->cases(),
              ElementsAre(a_add_internal, a_add_internal, c_add_internal));

  // Check bookkeeping.
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* occupied,
                           integration->GetGlobalMuxOccupiedCaseIndexes(mux));
  EXPECT_THAT(*occupied, ElementsAre(0, 2));
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* last_assigned,
                           integration->GetGlobalMuxLastCaseIndexesAdded(mux));
  EXPECT_THAT(*last_assigned, ElementsAre(0, 2));
}

TEST_F(IntegratorTest, UnifyIntegrationGlobalSelectModifyMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_add_node = FindNode("add", func_d);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add_internal,
                           integration->InsertNode(b_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * d_add_internal,
                           integration->InsertNode(d_add_node));

  EXPECT_EQ(integration->function()->node_count(), 13);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(
      unity, integration->UnifyIntegrationNodes(unity.node, d_add_internal));

  // Create another mux separate from the one we're modifying.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode other,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));

  XLS_ASSERT_OK_AND_ASSIGN(
      unity, integration->UnifyIntegrationNodes(unity.node, b_add_internal));

  // Check modified mux.
  EXPECT_EQ(unity.change,
            IntegrationFunction::UnificationChange::kExistingMuxCasesModified);
  EXPECT_EQ(integration->function()->node_count(), 15);
  // Modified mux.
  EXPECT_EQ(unity.node->op(), Op::kSel);
  Select* mux = unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, b_add_internal,
                                        c_add_internal, d_add_internal));
  // Other mux.
  EXPECT_EQ(other.node->op(), Op::kSel);
  Select* other_mux = other.node->As<Select>();
  EXPECT_EQ(other_mux->selector(), integration->global_mux_select());
  EXPECT_EQ(other_mux->default_value(), std::nullopt);
  EXPECT_THAT(other_mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                              c_add_internal, a_add_internal));

  // Check bookkeeping.
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);
  // Modified mux.
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* occupied,
                           integration->GetGlobalMuxOccupiedCaseIndexes(mux));
  EXPECT_THAT(*occupied, ElementsAre(0, 1, 2, 3));
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* last_assigned,
                           integration->GetGlobalMuxLastCaseIndexesAdded(mux));
  EXPECT_THAT(*last_assigned, ElementsAre(1));
  // Other mux.
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* other_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(other_mux));
  EXPECT_THAT(*other_occupied, ElementsAre(0, 2));
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* other_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(other_mux));
  EXPECT_THAT(*other_last_assigned, ElementsAre(0, 2));
}

TEST_F(IntegratorTest, UnifyIntegrationGlobalSelectUnifyMultiMappedNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_add_node = FindNode("add", func_d);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_add_node, a_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(d_add_node, c_add_internal));

  EXPECT_EQ(integration->function()->node_count(), 11);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));

  // Check added mux.
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(integration->function()->node_count(), 12);
  EXPECT_EQ(unity.node->op(), Op::kSel);
  Select* mux = unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        c_add_internal, c_add_internal));

  // Check bookkeeping.
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* occupied,
                           integration->GetGlobalMuxOccupiedCaseIndexes(mux));
  EXPECT_THAT(*occupied, ElementsAre(0, 1, 2, 3));
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* last_assigned,
                           integration->GetGlobalMuxLastCaseIndexesAdded(mux));
  EXPECT_THAT(*last_assigned, ElementsAre(0, 1, 2, 3));
}

TEST_F(IntegratorTest, UnifyIntegrationGlobalSelectErrorCases) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  fb_a.Add(in1, in1, SourceInfo(), "add2");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_add_node = FindNode("add", func_a);
  Node* a_add2_node = FindNode("add2", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_add_node = FindNode("add", func_d);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add2_internal,
                           integration->InsertNode(a_add2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add_internal,
                           integration->InsertNode(b_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * d_add_internal,
                           integration->InsertNode(d_add_node));

  // Try to unify two muxes.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity1,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity2,
      integration->UnifyIntegrationNodes(b_add_internal, d_add_internal));
  auto unify_mux_result =
      integration->UnifyIntegrationNodes(unity1.node, unity2.node);
  EXPECT_FALSE(unify_mux_result.ok());

  // Conflicting case assignments
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity3,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  auto conflicitng_case_mux_result =
      integration->UnifyIntegrationNodes(unity3.node, a_add2_internal);
  EXPECT_FALSE(conflicitng_case_mux_result.ok());

  // Try to unify w/ node that is not a map target.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity4,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           integration->function()->MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  auto no_map_mux_result =
      integration->UnifyIntegrationNodes(unity4.node, internal_1);
  EXPECT_FALSE(no_map_mux_result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationGlobalSelectRemoveMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add_internal,
                           integration->InsertNode(b_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));

  EXPECT_EQ(integration->function()->node_count(), 10);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode inner_unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  EXPECT_EQ(inner_unity.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(integration->function()->node_count(), 11);
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode outer_unity,
      integration->UnifyIntegrationNodes(a_add_internal, b_add_internal));
  EXPECT_EQ(outer_unity.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(integration->function()->node_count(), 12);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);

  // Check inner mux.
  EXPECT_EQ(inner_unity.node->op(), Op::kSel);
  Select* mux = inner_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), a_add_internal);
  EXPECT_THAT(mux->cases(),
              ElementsAre(a_add_internal, a_add_internal, c_add_internal));

  // Check outer mux.
  EXPECT_EQ(outer_unity.node->op(), Op::kSel);
  mux = outer_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), a_add_internal);
  EXPECT_THAT(mux->cases(),
              ElementsAre(a_add_internal, b_add_internal, a_add_internal));

  // Check inner bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* inner_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(inner_unity.node));
  EXPECT_THAT(*inner_occupied, ElementsAre(0, 2));
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* inner_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(inner_unity.node));
  EXPECT_THAT(*inner_last_assigned, ElementsAre(0, 2));

  // Check outer bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* outer_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(outer_unity.node));
  EXPECT_THAT(*outer_occupied, ElementsAre(0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* outer_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(outer_unity.node));
  EXPECT_THAT(*outer_last_assigned, ElementsAre(0, 1));

  // Remove inner mux.
  auto inner_deunify_result =
      integration->DeUnifyIntegrationNodes(inner_unity.node);
  EXPECT_TRUE(inner_deunify_result.ok());
  EXPECT_EQ(inner_deunify_result.value(), static_cast<Node*>(nullptr));
  EXPECT_EQ(integration->function()->node_count(), 11);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);

  // Check outer mux.
  EXPECT_EQ(outer_unity.node->op(), Op::kSel);
  mux = outer_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), a_add_internal);
  EXPECT_THAT(mux->cases(),
              ElementsAre(a_add_internal, b_add_internal, a_add_internal));

  // Check outer bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(outer_unity.node));
  EXPECT_THAT(*outer_occupied, ElementsAre(0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(outer_unity.node));
  EXPECT_THAT(*outer_last_assigned, ElementsAre(0, 1));

  // Remove outer mux.
  auto outer_deunify_result =
      integration->DeUnifyIntegrationNodes(outer_unity.node);
  EXPECT_TRUE(outer_deunify_result.ok());
  EXPECT_EQ(outer_deunify_result.value(), static_cast<Node*>(nullptr));
  EXPECT_EQ(integration->function()->node_count(), 10);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 0);
}

TEST_F(IntegratorTest, DeUnifyIntegrationGlobalSelectRevertMuxCases) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_add_node = FindNode("add", func_d);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add_internal,
                           integration->InsertNode(b_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * d_add_internal,
                           integration->InsertNode(d_add_node));

  EXPECT_EQ(integration->function()->node_count(), 13);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode outer_unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(outer_unity, integration->UnifyIntegrationNodes(
                                            outer_unity.node, d_add_internal));
  EXPECT_EQ(outer_unity.change,
            IntegrationFunction::UnificationChange::kExistingMuxCasesModified);
  EXPECT_EQ(integration->function()->node_count(), 14);
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode inner_unity,
      integration->UnifyIntegrationNodes(a_add_internal, d_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(inner_unity, integration->UnifyIntegrationNodes(
                                            inner_unity.node, b_add_internal));
  EXPECT_EQ(inner_unity.change,
            IntegrationFunction::UnificationChange::kExistingMuxCasesModified);
  EXPECT_EQ(integration->function()->node_count(), 15);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);

  // Check inner mux.
  EXPECT_EQ(inner_unity.node->op(), Op::kSel);
  Select* mux = inner_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, b_add_internal,
                                        a_add_internal, d_add_internal));

  // Check outer mux.
  EXPECT_EQ(outer_unity.node->op(), Op::kSel);
  mux = outer_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        c_add_internal, d_add_internal));

  // Check inner bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* inner_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(inner_unity.node));
  EXPECT_THAT(*inner_occupied, ElementsAre(0, 1, 3));
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* inner_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(inner_unity.node));
  EXPECT_THAT(*inner_last_assigned, ElementsAre(1));

  // Check outer bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* outer_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(outer_unity.node));
  EXPECT_THAT(*outer_occupied, ElementsAre(0, 2, 3));
  XLS_ASSERT_OK_AND_ASSIGN(
      const std::set<int64_t>* outer_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(outer_unity.node));
  EXPECT_THAT(*outer_last_assigned, ElementsAre(3));

  // Revert inner mux.
  XLS_ASSERT_OK_AND_ASSIGN(
      inner_unity.node, integration->DeUnifyIntegrationNodes(inner_unity.node));
  EXPECT_EQ(integration->function()->node_count(), 15);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);

  // Check inner mux.
  EXPECT_EQ(inner_unity.node->op(), Op::kSel);
  mux = inner_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        a_add_internal, d_add_internal));

  // Check outer mux.
  EXPECT_EQ(outer_unity.node->op(), Op::kSel);
  mux = outer_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        c_add_internal, d_add_internal));

  // Check inner bookkeeping.
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);
  XLS_ASSERT_OK_AND_ASSIGN(
      inner_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(inner_unity.node));
  EXPECT_THAT(*inner_occupied, ElementsAre(0, 3));
  XLS_ASSERT_OK_AND_ASSIGN(
      inner_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(inner_unity.node));
  EXPECT_TRUE(inner_last_assigned->empty());

  // Check outer bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(outer_unity.node));
  EXPECT_THAT(*outer_occupied, ElementsAre(0, 2, 3));
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(outer_unity.node));
  EXPECT_THAT(*outer_last_assigned, ElementsAre(3));

  // Revert outer mux.
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_unity.node, integration->DeUnifyIntegrationNodes(outer_unity.node));
  EXPECT_EQ(integration->function()->node_count(), 15);
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);

  // Check inner mux.
  EXPECT_EQ(inner_unity.node->op(), Op::kSel);
  mux = inner_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        a_add_internal, d_add_internal));

  // Check outer mux.
  EXPECT_EQ(outer_unity.node->op(), Op::kSel);
  mux = outer_unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        c_add_internal, a_add_internal));

  // Check inner bookkeeping.
  muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 2);
  XLS_ASSERT_OK_AND_ASSIGN(
      inner_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(inner_unity.node));
  EXPECT_THAT(*inner_occupied, ElementsAre(0, 3));
  XLS_ASSERT_OK_AND_ASSIGN(
      inner_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(inner_unity.node));
  EXPECT_TRUE(inner_last_assigned->empty());

  // Check outer bookkeeping.
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_occupied,
      integration->GetGlobalMuxOccupiedCaseIndexes(outer_unity.node));
  EXPECT_THAT(*outer_occupied, ElementsAre(0, 2));
  XLS_ASSERT_OK_AND_ASSIGN(
      outer_last_assigned,
      integration->GetGlobalMuxLastCaseIndexesAdded(outer_unity.node));
  EXPECT_TRUE(outer_last_assigned->empty());

  // Can't do repeated reversions.
  auto inner_unity_repeated_revert_result =
      integration->DeUnifyIntegrationNodes(inner_unity.node);
  EXPECT_FALSE(inner_unity_repeated_revert_result.ok());
  auto outer_unity_repeated_revert_result =
      integration->DeUnifyIntegrationNodes(outer_unity.node);
  EXPECT_FALSE(outer_unity_repeated_revert_result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationGlobalSelectMultiMappedNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_add_node = FindNode("add", func_d);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_add_node, a_add_internal));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(d_add_node, c_add_internal));

  EXPECT_EQ(integration->function()->node_count(), 11);
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(a_add_internal, c_add_internal));

  // Check added mux.
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(integration->function()->node_count(), 12);
  EXPECT_EQ(unity.node->op(), Op::kSel);
  Select* mux = unity.node->As<Select>();
  EXPECT_EQ(mux->selector(), integration->global_mux_select());
  EXPECT_EQ(mux->default_value(), std::nullopt);
  EXPECT_THAT(mux->cases(), ElementsAre(a_add_internal, a_add_internal,
                                        c_add_internal, c_add_internal));

  // Check bookkeeping.
  int64_t muxes_tracked = integration->GetNumberOfGlobalMuxesTracked();
  EXPECT_EQ(muxes_tracked, 1);
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* occupied,
                           integration->GetGlobalMuxOccupiedCaseIndexes(mux));
  EXPECT_THAT(*occupied, ElementsAre(0, 1, 2, 3));
  XLS_ASSERT_OK_AND_ASSIGN(const std::set<int64_t>* last_assigned,
                           integration->GetGlobalMuxLastCaseIndexesAdded(mux));
  EXPECT_THAT(*last_assigned, ElementsAre(0, 1, 2, 3));

  // Remove mux.
  XLS_ASSERT_OK(integration->DeUnifyIntegrationNodes(unity.node));
  EXPECT_EQ(integration->function()->node_count(), 11);
  EXPECT_EQ(integration->GetNumberOfGlobalMuxesTracked(), 0);
}

TEST_F(IntegratorTest, DeUnifyIntegrationGlobalSelectMuxNotCreatedByUnifyCall) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(in1, in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* a_add_node = FindNode("add", func_a);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_add_node = FindNode("add", func_c);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_internal,
                           integration->InsertNode(a_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add_internal,
                           integration->InsertNode(b_add_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_add_internal,
                           integration->InsertNode(c_add_node));

  std::vector<Node*> cases = {a_add_internal, b_add_internal, c_add_internal};
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * new_mux, integration->function()->MakeNode<Select>(
                          SourceInfo(), integration->global_mux_select(), cases,
                          /*default_value=*/a_add_internal));

  // Remove mux. Shouldn't be able to deunify mux not created
  // by a unify call.
  auto deunify_result = integration->DeUnifyIntegrationNodes(new_mux);
  EXPECT_FALSE(deunify_result.ok());
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

TEST_F(IntegratorTest, InsertNodeIntegrationNode) {
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

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           integration->function()->MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_cat,
      integration->function()->MakeNode<Concat>(
          SourceInfo(), std::vector<Node*>({internal_1, internal_1})));

  auto unmapped_cost_result = integration->GetInsertNodeCost(internal_cat);
  EXPECT_FALSE(unmapped_cost_result.ok());
  auto unmapped_merge_result = integration->InsertNode(internal_cat);
  EXPECT_FALSE(unmapped_merge_result.ok());

  XLS_ASSERT_OK_AND_ASSIGN(Node * map_target,
                           integration->InsertNode(add_node));
  auto mapped_cost_result = integration->GetInsertNodeCost(map_target);
  EXPECT_FALSE(mapped_cost_result.ok());
  auto mapped_merge_result = integration->InsertNode(map_target);
  EXPECT_FALSE(mapped_merge_result.ok());
}

TEST_F(IntegratorTest, GetNodeCost) {
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
  XLS_ASSERT_OK_AND_ASSIGN(float cost,
                           integration->GetInsertNodeCost(add_node));
  EXPECT_EQ(cost, integration->GetNodeCost(add_node));
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesExternalNode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function external_func("external", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * external_1,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "external_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * external_2,
                           external_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "external_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_sel,
      external_func.MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(1),
                                            "external_sel"));
  std::vector<Node*> elements = {external_1, external_2};
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * external_mux,
      external_func.MakeNode<Select>(SourceInfo(), external_sel, elements,
                                     /*default_value=*/std::nullopt));

  // Can't DeUnify external node.
  auto result = integration->DeUnifyIntegrationNodes(external_mux);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesNonUnifyNonMux) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_cat,
      internal_func.MakeNode<Concat>(
          SourceInfo(), std::vector<Node*>({internal_1, internal_1})));

  auto result = integration->DeUnifyIntegrationNodes(internal_cat);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesNonUnifyMuxDoesNotHaveTwoCases) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_sel,
      internal_func.MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(1),
                                            "internal_sel"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_default,
      internal_func.MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(2),
                                            "internal_default"));
  std::vector<Node*> elements = {internal_1};
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_mux,
                           integration->function()->MakeNode<Select>(
                               SourceInfo(), internal_sel, elements,
                               /*default_value=*/internal_default));

  auto result = integration->DeUnifyIntegrationNodes(internal_mux);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesNonUnifyMuxSameOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_sel,
      internal_func.MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(1),
                                            "internal_sel"));
  XLS_ASSERT_OK(integration->UnifyIntegrationNodes(internal_1, internal_2));
  std::vector<Node*> elements = {internal_1, internal_2};
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_mux,
      internal_func.MakeNode<Select>(SourceInfo(), internal_sel, elements,
                                     /*default_value=*/std::nullopt));

  auto result = integration->DeUnifyIntegrationNodes(internal_mux);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesNonUnifyMuxDifferentOperands) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_3,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_3"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_4,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_4"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_sel,
      internal_func.MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(1),
                                            "internal_sel"));
  XLS_ASSERT_OK(integration->UnifyIntegrationNodes(internal_1, internal_2));
  std::vector<Node*> elements = {internal_3, internal_4};
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * internal_mux,
      internal_func.MakeNode<Select>(SourceInfo(), internal_sel, elements,
                                     /*default_value=*/std::nullopt));

  auto result = integration->DeUnifyIntegrationNodes(internal_mux);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesMuxHasUsers) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unify_mux,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  XLS_ASSERT_OK(internal_func.MakeNode<Concat>(
      SourceInfo(), std::vector<Node*>({unify_mux.node, unify_mux.node})));

  auto result = integration->DeUnifyIntegrationNodes(unify_mux.node);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesRemoveMuxAndParam) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unify_mux,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  Node* sel = unify_mux.node->As<Select>()->selector();

  EXPECT_EQ(integration->function()->node_count(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * result, integration->DeUnifyIntegrationNodes(unify_mux.node));
  EXPECT_EQ(result, static_cast<Node*>(nullptr));

  auto integration_contains = [&integration](Node* target_node) {
    for (Node* node : integration->function()->nodes()) {
      if (target_node == node) {
        return true;
      }
    }
    return false;
  };
  EXPECT_EQ(integration->function()->node_count(), 2);
  EXPECT_FALSE(integration_contains(unify_mux.node));
  EXPECT_FALSE(integration_contains(sel));

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unity,
      integration->UnifyIntegrationNodes(internal_1, internal_2));
  EXPECT_EQ(unity.change, IntegrationFunction::UnificationChange::kNewMuxAdded);
}

TEST_F(IntegratorTest, DeUnifyIntegrationNodesParamHasUsers) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  Function& internal_func = *integration->function();

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           internal_func.MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedNode unify_mux,
      integration->UnifyIntegrationNodes(internal_1, internal_2));

  Node* sel = unify_mux.node->As<Select>()->selector();
  XLS_ASSERT_OK(internal_func.MakeNode<Concat>(SourceInfo(),
                                               std::vector<Node*>({sel, sel})));

  EXPECT_EQ(integration->function()->node_count(), 5);
  auto result = integration->DeUnifyIntegrationNodes(unify_mux.node);
  EXPECT_FALSE(result.ok());
}

TEST_F(IntegratorTest, UnifyNodeOperands) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("clone"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in2 = FindNode("in2", func_a);
  Node* a_add = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_in2 = FindNode("in2", func_b);
  Node* b_add = *b_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map, integration->InsertNode(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_map, integration->InsertNode(a_in2));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1, a_in1_map));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_map, integration->InsertNode(b_in2));

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add, b_add));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 1);
  IntegrationFunction::UnifiedNode& unified_node =
      unified_operands.changed_muxes.at(0);
  EXPECT_EQ(unified_node.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  Select* mux = unified_node.node->As<Select>();
  EXPECT_THAT(mux->cases(), ElementsAre(a_in2_map, b_in2_map));
  EXPECT_THAT(unified_operands.operands, ElementsAre(a_in1_map, mux));
}

TEST_F(IntegratorTest, UnifyNodeOperandsMultipleMux) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("clone"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in2 = FindNode("in2", func_a);
  Node* a_add = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_in2 = FindNode("in2", func_b);
  Node* b_add = *b_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map, integration->InsertNode(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_map, integration->InsertNode(a_in2));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in1_map, integration->InsertNode(b_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_map, integration->InsertNode(b_in2));

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add, b_add));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 2);
  IntegrationFunction::UnifiedNode& unified_node_1 =
      unified_operands.changed_muxes.at(0);
  EXPECT_EQ(unified_node_1.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  IntegrationFunction::UnifiedNode& unified_node_2 =
      unified_operands.changed_muxes.at(1);
  EXPECT_EQ(unified_node_2.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  Select* mux1 = unified_node_1.node->As<Select>();
  EXPECT_THAT(mux1->cases(), ElementsAre(a_in1_map, b_in1_map));
  Select* mux2 = unified_node_2.node->As<Select>();
  EXPECT_THAT(mux2->cases(), ElementsAre(a_in2_map, b_in2_map));
  EXPECT_THAT(unified_operands.operands, ElementsAre(mux1, mux2));
}

TEST_F(IntegratorTest, UnifyNodeOperandsRepeatedMux) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  fb.Add(in1, in1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("clone"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_add = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_add = *b_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map, integration->InsertNode(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in1_map, integration->InsertNode(b_in1));

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add, b_add));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 1);
  IntegrationFunction::UnifiedNode& unified_node =
      unified_operands.changed_muxes.at(0);
  EXPECT_EQ(unified_node.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  Select* mux = unified_node.node->As<Select>();
  EXPECT_THAT(mux->cases(), ElementsAre(a_in1_map, b_in1_map));
  EXPECT_THAT(unified_operands.operands, ElementsAre(mux, mux));
}

TEST_F(IntegratorTest, UnifyNodeOperandsNoAddedMuxes) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("clone"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in2 = FindNode("in2", func_a);
  Node* a_add = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_in2 = FindNode("in2", func_b);
  Node* b_add = *b_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map, integration->InsertNode(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_map, integration->InsertNode(a_in2));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1, a_in1_map));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in2, a_in2_map));

  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add, b_add));
  EXPECT_EQ(a_in1_map->users().size(), 0);
  EXPECT_EQ(a_in2_map->users().size(), 0);
  EXPECT_EQ(unified_operands.changed_muxes.size(), 0);
  EXPECT_THAT(unified_operands.operands, ElementsAre(a_in1_map, a_in2_map));
}

TEST_F(IntegratorTest, UnifyNodeOperandsGlobalMuxSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb("func_a", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Concat({in1, in2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in2 = FindNode("in2", func_a);
  Node* a_cat = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_in2 = FindNode("in2", func_b);
  Node* b_cat = *b_in1->users().begin();
  Node* c_in1 = FindNode("in1", func_c);
  Node* c_in2 = FindNode("in2", func_c);
  Node* c_cat = *c_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map,
                           integration->GetNodeMapping(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_map,
                           integration->GetNodeMapping(a_in2));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1, a_in1_map));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_map,
                           integration->GetNodeMapping(b_in2));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_in1_map,
                           integration->GetNodeMapping(c_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * c_in2_map,
                           integration->GetNodeMapping(c_in2));

  // One operand already matched, second operand requires mux insert.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(b_cat, a_cat));
  // Check inserted mux.
  EXPECT_EQ(unified_operands.changed_muxes.size(), 1);
  IntegrationFunction::UnifiedNode& mux1_info =
      unified_operands.changed_muxes.at(0);
  EXPECT_EQ(mux1_info.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  Select* mux1 = mux1_info.node->As<Select>();
  EXPECT_EQ(mux1->selector(), integration->global_mux_select());
  EXPECT_EQ(mux1->default_value(), a_in2_map);
  EXPECT_THAT(mux1->cases(), ElementsAre(a_in2_map, b_in2_map, a_in2_map));
  // Check operands.
  EXPECT_THAT(unified_operands.operands,
              ElementsAre(a_in1_map, mux1_info.node));

  // One new mux inserted, one mux modified.
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * cat_with_mux,
      integration->function()->MakeNode<Concat>(
          SourceInfo(), std::vector<Node*>({a_in1_map, mux1_info.node})));
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands_repeated,
      integration->UnifyNodeOperands(c_cat, cat_with_mux));
  EXPECT_EQ(unified_operands_repeated.changed_muxes.size(), 2);
  // Check inserted mux.
  IntegrationFunction::UnifiedNode& mux2_info =
      unified_operands_repeated.changed_muxes.at(0);
  EXPECT_EQ(mux2_info.change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  Select* mux2 = mux2_info.node->As<Select>();
  EXPECT_EQ(mux2->selector(), integration->global_mux_select());
  EXPECT_EQ(mux2->default_value(), a_in1_map);
  EXPECT_THAT(mux2->cases(), ElementsAre(a_in1_map, a_in1_map, c_in1_map));
  // Check modified mux.
  IntegrationFunction::UnifiedNode& mux3_info =
      unified_operands_repeated.changed_muxes.at(1);
  EXPECT_EQ(mux3_info.change,
            IntegrationFunction::UnificationChange::kExistingMuxCasesModified);
  Select* mux3 = mux3_info.node->As<Select>();
  EXPECT_EQ(mux3->selector(), integration->global_mux_select());
  EXPECT_EQ(mux3->default_value(), a_in2_map);
  EXPECT_THAT(mux3->cases(), ElementsAre(a_in2_map, b_in2_map, c_in2_map));
  // Check operands.
  EXPECT_THAT(unified_operands_repeated.operands,
              ElementsAre(mux2_info.node, mux3_info.node));
}

TEST_F(IntegratorTest, MergeBackendErrorNonMappedInternal) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in2 = FindNode("in2", func_a);
  Node* a_add = *a_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target, integration->InsertNode(a_in1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target, integration->InsertNode(a_in2));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_add,
                           a_add->CloneInNewFunction(
                               std::vector<Node*>({a_in1_target, a_in2_target}),
                               integration->function()));

  auto result1 = integration->GetMergeNodesCost(a_add, internal_add);
  EXPECT_THAT(
      result1,
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("Trying to merge non-mapping-target integration node")));
  auto result2 = integration->GetMergeNodesCost(internal_add, a_add);
  EXPECT_THAT(
      result1,
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("Trying to merge non-mapping-target integration node")));
}

TEST_F(IntegratorTest, MergeBackendErrorMappedExternal) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("clone"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_add = *a_in1->users().begin();
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_add = *b_in1->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));
  XLS_ASSERT_OK(integration->InsertNode(a_add));

  auto result1 = integration->GetMergeNodesCost(a_add, b_add);
  EXPECT_THAT(result1, StatusIs(absl::StatusCode::kFailedPrecondition,
                                HasSubstr("Trying to merge non-integration "
                                          "node that already has mapping")));
  auto result2 = integration->GetMergeNodesCost(b_add, a_add);
  EXPECT_THAT(result2, StatusIs(absl::StatusCode::kFailedPrecondition,
                                HasSubstr("Trying to merge non-integration "
                                          "node that already has mapping")));
}

TEST_F(IntegratorTest, MergeBackendNodeSourceFunctionsCollide) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  auto in3 = fb.Param("in3", p->GetBitsType(2));
  fb.Add(in1, in2);
  fb.Add(in1, in3);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_in3 = FindNode("in3", func_a);
  Node* a_add1 = *a_in1->users().begin();
  Node* a_add2 = *a_in3->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(auto optional_cost,
                           integration->GetMergeNodesCost(a_add1, a_add2));
  EXPECT_FALSE(optional_cost.has_value());
}

TEST_F(IntegratorTest, MergeBackendDoNotMergeIncompatible) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("in2", p->GetBitsType(2));
  fb_b.Concat({b_in1, b_in2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("in1", func_a);
  Node* a_add_node = *a_in1_node->users().begin();
  Node* b_in1_node = FindNode("in1", func_b);
  Node* b_cat_node = *b_in1_node->users().begin();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  // Cost frontend.
  XLS_ASSERT_OK_AND_ASSIGN(auto optional_cost, integration->GetMergeNodesCost(
                                                   a_add_node, b_cat_node));
  EXPECT_FALSE(optional_cost.has_value());

  // Merge frontend.
  EXPECT_FALSE(integration->MergeNodes(a_add_node, b_cat_node).ok());
}

TEST_F(IntegratorTest, MergeCostExternalExternalTwoMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("in2", p->GetBitsType(2));
  auto b_sel = fb_b.Param("sel", p->GetBitsType(1));
  fb_b.Add(b_in1, b_in2, SourceInfo(), "add");
  fb_b.Select(b_sel, {b_in1, b_in2},
              /*default_value=*/std::nullopt, SourceInfo(), "mux");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("in1", func_a);
  Node* a_in2_node = FindNode("in2", func_a);
  Node* a_add_node = FindNode("add", func_a);
  Node* b_in1_node = FindNode("in1", func_b);
  Node* b_in2_node = FindNode("in2", func_b);
  Node* b_add_node = FindNode("add", func_b);
  Node* ref_mux_node = FindNode("mux", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target,
                           integration->GetNodeMapping(a_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target,
                           integration->GetNodeMapping(a_in2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in1_target,
                           integration->GetNodeMapping(b_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_target,
                           integration->GetNodeMapping(b_in2_node));
  int64_t init_node_count = integration->function()->node_count();

  // Get cost.
  XLS_ASSERT_OK_AND_ASSIGN(auto optional_cost, integration->GetMergeNodesCost(
                                                   a_add_node, b_add_node));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check cost.
  EXPECT_TRUE(optional_cost.has_value());
  float expected_cost = integration->GetNodeCost(a_add_node) +
                        2 * integration->GetNodeCost(ref_mux_node);
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);

  // Reverse order of merged nodes.
  XLS_ASSERT_OK_AND_ASSIGN(
      optional_cost, integration->GetMergeNodesCost(b_add_node, a_add_node));
  EXPECT_TRUE(optional_cost.has_value());
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);

  // Check function un-altered and mappings preserved.
  EXPECT_EQ(integration->function()->node_count(), init_node_count);
  absl::flat_hash_set<Node*> found_nodes;
  for (auto* node : integration->function()->nodes()) {
    found_nodes.insert(node);
  }
  auto check_param = [&](const Node* src, const Node* target) {
    XLS_ASSERT_OK_AND_ASSIGN(Node * observed_target,
                             integration->GetNodeMapping(src));
    EXPECT_EQ(target, observed_target);
    EXPECT_TRUE(found_nodes.contains(target));
    EXPECT_TRUE(target->users().empty());
    XLS_ASSERT_OK_AND_ASSIGN(auto nodes_map_to_target,
                             integration->GetNodesMappedToNode(target));
    EXPECT_EQ(nodes_map_to_target->size(), 1);
    EXPECT_TRUE(nodes_map_to_target->contains(src));
  };
  check_param(a_in1_node, a_in1_target);
  check_param(a_in2_node, a_in2_target);
  check_param(b_in1_node, b_in1_target);
  check_param(b_in2_node, b_in2_target);

  // Check that mux tracking was not updated.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add_node, b_add_node));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 2);
  EXPECT_EQ(unified_operands.changed_muxes.at(0).change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
  EXPECT_EQ(unified_operands.changed_muxes.at(1).change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
}

TEST_F(IntegratorTest, MergeCostInternalExternalOneMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("in2", p->GetBitsType(2));
  auto b_sel = fb_b.Param("sel", p->GetBitsType(1));
  fb_b.Add(b_in1, b_in2, SourceInfo(), "add");
  fb_b.Select(b_sel, {b_in1, b_in2},
              /*default_value=*/std::nullopt, SourceInfo(), "mux");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("in1", func_a);
  Node* a_in2_node = FindNode("in2", func_a);
  Node* a_add_node = FindNode("add", func_a);
  Node* b_in1_node = FindNode("in1", func_b);
  Node* b_in2_node = FindNode("in2", func_b);
  Node* b_add_node = FindNode("add", func_b);
  Node* ref_mux_node = FindNode("mux", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target,
                           integration->GetNodeMapping(a_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target,
                           integration->GetNodeMapping(a_in2_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1_node, a_in1_target));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_target,
                           integration->GetNodeMapping(b_in2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_target,
                           integration->InsertNode(a_add_node));
  int64_t init_node_count = integration->function()->node_count();

  // Get cost.
  XLS_ASSERT_OK_AND_ASSIGN(auto optional_cost, integration->GetMergeNodesCost(
                                                   a_add_target, b_add_node));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check cost.
  EXPECT_TRUE(optional_cost.has_value());
  float expected_cost = integration->GetNodeCost(ref_mux_node);
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);

  // Reverse order of merged nodes.
  XLS_ASSERT_OK_AND_ASSIGN(
      optional_cost, integration->GetMergeNodesCost(b_add_node, a_add_target));
  EXPECT_TRUE(optional_cost.has_value());
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);

  // Check function un-altered and mappings preserved.
  EXPECT_EQ(integration->function()->node_count(), init_node_count);
  absl::flat_hash_set<Node*> found_nodes;
  for (auto* node : integration->function()->nodes()) {
    found_nodes.insert(node);
  }
  auto check_param = [&](std::vector<const Node*> srcs, const Node* target,
                         bool used_by_add) {
    EXPECT_TRUE(found_nodes.contains(target));
    if (used_by_add) {
      EXPECT_EQ(target->users().size(), 1);
      EXPECT_EQ(*target->users().begin(), a_add_target);
    } else {
      EXPECT_TRUE(target->users().empty());
    }
    XLS_ASSERT_OK_AND_ASSIGN(auto nodes_map_to_target,
                             integration->GetNodesMappedToNode(target));
    for (auto src : srcs) {
      XLS_ASSERT_OK_AND_ASSIGN(Node * observed_target,
                               integration->GetNodeMapping(src));
      EXPECT_EQ(target, observed_target);
      EXPECT_TRUE(nodes_map_to_target->contains(src));
    }
    EXPECT_EQ(nodes_map_to_target->size(), srcs.size());
  };
  check_param({a_in1_node, b_in1_node}, a_in1_target, true);
  check_param({a_in2_node}, a_in2_target, true);
  check_param({b_in2_node}, b_in2_target, false);
  EXPECT_TRUE(found_nodes.contains(a_add_target));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_observed_target,
                           integration->GetNodeMapping(a_add_node));
  EXPECT_EQ(a_add_observed_target, a_add_target);
  XLS_ASSERT_OK_AND_ASSIGN(auto nodes_mapped_to_add,
                           integration->GetNodesMappedToNode(a_add_target));
  EXPECT_EQ(nodes_mapped_to_add->size(), 1);
  EXPECT_TRUE(nodes_mapped_to_add->contains(a_add_node));

  // Check that mux tracking was not updated.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add_node, b_add_node));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 1);
  EXPECT_EQ(unified_operands.changed_muxes.at(0).change,
            IntegrationFunction::UnificationChange::kNewMuxAdded);
}

TEST_F(IntegratorTest, MergeCostGlobalMuxSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb("func_a", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  auto in3 = fb.Param("in3", p->GetBitsType(2));
  auto in_sel = fb.Param("in_sel", p->GetBitsType(2));
  fb.Select(in_sel, {in1, in2, in3},
            /*default_value=*/in1, SourceInfo(), "three_input_mux");
  fb.Add(in1, in2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* ref_3_input_mux_node = FindNode("three_input_mux", func_b);
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_add = FindNode("add", func_a);
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_add = FindNode("add", func_b);
  Node* c_add = FindNode("add", func_c);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map,
                           integration->GetNodeMapping(a_in1));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1, a_in1_map));
  int64_t init_node_count = integration->function()->node_count();

  // Get cost.
  XLS_ASSERT_OK_AND_ASSIGN(auto optional_cost,
                           integration->GetMergeNodesCost(a_add, b_add));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check cost. One 'add' and one new mux added.
  EXPECT_TRUE(optional_cost.has_value());
  float expected_cost = integration->GetNodeCost(ref_3_input_mux_node) +
                        integration->GetNodeCost(a_add);
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);

  // Reverse order of merged nodes.
  XLS_ASSERT_OK_AND_ASSIGN(optional_cost,
                           integration->GetMergeNodesCost(b_add, a_add));
  EXPECT_TRUE(optional_cost.has_value());
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);
  EXPECT_EQ(init_node_count, integration->function()->node_count());

  // Add node for 2nd merge cost estimate.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> merged_nodes,
                           integration->MergeNodes(a_add, b_add));
  EXPECT_EQ(merged_nodes.size(), 1);
  Node* merged_add = merged_nodes.front();
  int64_t modified_node_count = integration->function()->node_count();

  // Get cost.
  XLS_ASSERT_OK_AND_ASSIGN(optional_cost,
                           integration->GetMergeNodesCost(c_add, merged_add));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check cost. One mux added, one mux modified (no cost).
  EXPECT_TRUE(optional_cost.has_value());
  expected_cost = integration->GetNodeCost(ref_3_input_mux_node);
  EXPECT_FLOAT_EQ(optional_cost.value(), expected_cost);
  EXPECT_EQ(modified_node_count, integration->function()->node_count());
}

TEST_F(IntegratorTest, MergeNodesExternalExternaTwoMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("a_in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("a_in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2, SourceInfo(), "a_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("b_in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("b_in2", p->GetBitsType(2));
  fb_b.Add(b_in1, b_in2, SourceInfo(), "b_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("a_in1", func_a);
  Node* a_in2_node = FindNode("a_in2", func_a);
  Node* a_add_node = FindNode("a_add", func_a);
  Node* b_in1_node = FindNode("b_in1", func_b);
  Node* b_in2_node = FindNode("b_in2", func_b);
  Node* b_add_node = FindNode("b_add", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target,
                           integration->GetNodeMapping(a_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target,
                           integration->GetNodeMapping(a_in2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in1_target,
                           integration->GetNodeMapping(b_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_target,
                           integration->GetNodeMapping(b_in2_node));
  int64_t init_node_count = integration->function()->node_count();

  // Merge.
  XLS_ASSERT_OK_AND_ASSIGN(auto generated_nodes,
                           integration->MergeNodes(a_add_node, b_add_node));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check merge.
  EXPECT_EQ(integration->function()->node_count(), init_node_count + 5);
  EXPECT_EQ(generated_nodes.size(), 1);
  Node* add_target = generated_nodes.front();
  EXPECT_THAT(
      add_target,
      m::Add(m::Select(m::Param("tuple_index_8_tuple_index_11_mux_sel"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                        m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
             m::Select(m::Param("tuple_index_9_tuple_index_12_mux_sel"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})));

  // Check mapping.
  auto check_mapping = [&](std::vector<const Node*> srcs, const Node* target) {
    XLS_ASSERT_OK_AND_ASSIGN(auto nodes_map_to_target,
                             integration->GetNodesMappedToNode(target));
    EXPECT_EQ(nodes_map_to_target->size(), srcs.size());
    for (auto src : srcs) {
      XLS_ASSERT_OK_AND_ASSIGN(Node * observed_target,
                               integration->GetNodeMapping(src));
      EXPECT_EQ(target, observed_target);
      EXPECT_TRUE(nodes_map_to_target->contains(src));
    }
  };
  check_mapping({a_in1_node}, a_in1_target);
  check_mapping({a_in2_node}, a_in2_target);
  check_mapping({b_in1_node}, b_in1_target);
  check_mapping({b_in2_node}, b_in2_target);
  check_mapping({a_add_node, b_add_node}, add_target);

  // Check that mux tracking was updated.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add_node, b_add_node));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 0);
}

TEST_F(IntegratorTest, MergeNodesInternalExternalOneMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("a_in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("a_in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2, SourceInfo(), "a_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("b_in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("b_in2", p->GetBitsType(2));
  fb_b.Add(b_in1, b_in2, SourceInfo(), "b_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("a_in1", func_a);
  Node* a_in2_node = FindNode("a_in2", func_a);
  Node* a_add_node = FindNode("a_add", func_a);
  Node* b_in1_node = FindNode("b_in1", func_b);
  Node* b_in2_node = FindNode("b_in2", func_b);
  Node* b_add_node = FindNode("b_add", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target,
                           integration->GetNodeMapping(a_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target,
                           integration->GetNodeMapping(a_in2_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1_node, a_in1_target));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_target,
                           integration->GetNodeMapping(b_in2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_initial_target,
                           integration->InsertNode(a_add_node));
  int64_t init_node_count = integration->function()->node_count();

  // Merge.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto generated_nodes,
      integration->MergeNodes(a_add_initial_target, b_add_node));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check merge.
  EXPECT_EQ(integration->function()->node_count(), init_node_count + 2);
  EXPECT_EQ(generated_nodes.size(), 1);
  Node* add_target = generated_nodes.front();
  EXPECT_THAT(
      add_target,
      m::Add(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
             m::Select(m::Param("tuple_index_9_tuple_index_12_mux_sel"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})));

  // Check mapping.
  auto check_mapping = [&](std::vector<const Node*> srcs, const Node* target) {
    XLS_ASSERT_OK_AND_ASSIGN(auto nodes_map_to_target,
                             integration->GetNodesMappedToNode(target));
    EXPECT_EQ(nodes_map_to_target->size(), srcs.size());
    for (auto src : srcs) {
      XLS_ASSERT_OK_AND_ASSIGN(Node * observed_target,
                               integration->GetNodeMapping(src));
      EXPECT_EQ(target, observed_target);
      EXPECT_TRUE(nodes_map_to_target->contains(src));
    }
  };
  check_mapping({a_in1_node, b_in1_node}, a_in1_target);
  check_mapping({a_in2_node}, a_in2_target);
  check_mapping({b_in2_node}, b_in2_target);
  check_mapping({a_add_node, b_add_node}, add_target);
  EXPECT_FALSE(integration->HasMapping(a_add_initial_target));
  EXPECT_FALSE(integration->IsMappingTarget(a_add_initial_target));

  // Check that mux tracking was updated.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(a_add_node, b_add_node));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 0);
}

TEST_F(IntegratorTest, MergeNodesExternalInternalOneMux) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("a_in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("a_in2", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in2, SourceInfo(), "a_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("b_in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("b_in2", p->GetBitsType(2));
  fb_b.Add(b_in1, b_in2, SourceInfo(), "b_add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* a_in1_node = FindNode("a_in1", func_a);
  Node* a_in2_node = FindNode("a_in2", func_a);
  Node* a_add_node = FindNode("a_add", func_a);
  Node* b_in1_node = FindNode("b_in1", func_b);
  Node* b_in2_node = FindNode("b_in2", func_b);
  Node* b_add_node = FindNode("b_add", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(true)));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_target,
                           integration->GetNodeMapping(a_in1_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in2_target,
                           integration->GetNodeMapping(a_in2_node));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1_node, a_in1_target));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_in2_target,
                           integration->GetNodeMapping(b_in2_node));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add_initial_target,
                           integration->InsertNode(a_add_node));
  int64_t init_node_count = integration->function()->node_count();

  // Merge.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto generated_nodes,
      integration->MergeNodes(b_add_node, a_add_initial_target));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));

  // Check merge.
  EXPECT_EQ(integration->function()->node_count(), init_node_count + 2);
  EXPECT_EQ(generated_nodes.size(), 1);
  Node* add_target = generated_nodes.front();
  EXPECT_THAT(
      add_target,
      m::Add(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
             m::Select(m::Param("tuple_index_12_tuple_index_9_mux_sel"),
                       {m::TupleIndex(m::Param("func_b_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_a_ParamTuple"), 1)})));

  // Check mapping.
  auto check_mapping = [&](std::vector<const Node*> srcs, const Node* target) {
    XLS_ASSERT_OK_AND_ASSIGN(auto nodes_map_to_target,
                             integration->GetNodesMappedToNode(target));
    EXPECT_EQ(nodes_map_to_target->size(), srcs.size());
    for (auto src : srcs) {
      XLS_ASSERT_OK_AND_ASSIGN(Node * observed_target,
                               integration->GetNodeMapping(src));
      EXPECT_EQ(target, observed_target);
      EXPECT_TRUE(nodes_map_to_target->contains(src));
    }
  };
  check_mapping({a_in1_node, b_in1_node}, a_in1_target);
  check_mapping({a_in2_node}, a_in2_target);
  check_mapping({b_in2_node}, b_in2_target);
  check_mapping({a_add_node, b_add_node}, add_target);
  EXPECT_FALSE(integration->HasMapping(a_add_initial_target));
  EXPECT_FALSE(integration->IsMappingTarget(a_add_initial_target));

  // Check that mux tracking was updated.
  XLS_ASSERT_OK_AND_ASSIGN(
      IntegrationFunction::UnifiedOperands unified_operands,
      integration->UnifyNodeOperands(b_add_node, a_add_node));
  EXPECT_EQ(unified_operands.changed_muxes.size(), 0);
}

TEST_F(IntegratorTest, MergeNodesGlobalMuxSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb("func_a", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  fb.Add(in1, in2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  Node* a_in1 = FindNode("in1", func_a);
  Node* a_add = FindNode("add", func_a);
  Node* b_in1 = FindNode("in1", func_b);
  Node* b_add = FindNode("add", func_b);
  Node* c_add = FindNode("add", func_c);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c},
          IntegrationOptions().unique_select_signal_per_mux(false)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_in1_map,
                           integration->GetNodeMapping(a_in1));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1, a_in1_map));
  int64_t init_node_count = integration->function()->node_count();

  // Merge adds with one new mux.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> merged_nodes,
                           integration->MergeNodes(a_add, b_add));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));
  EXPECT_EQ(merged_nodes.size(), 1);
  Node* merged_add = merged_nodes.front();
  EXPECT_THAT(
      merged_add,
      m::Add(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
             m::Select(m::Param("global_mux_select"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_b_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_a_ParamTuple"), 1)},
                       m::TupleIndex(m::Param("func_a_ParamTuple"), 1))));
  EXPECT_EQ(integration->function()->node_count(), init_node_count + 2);

  // Merge adds with one new mux and one modified mux.
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> merged_nodes_repeated,
                           integration->MergeNodes(c_add, merged_add));
  XLS_ASSERT_OK(VerifyFunction(integration->function()));
  EXPECT_EQ(merged_nodes_repeated.size(), 1);
  Node* merged_add_repeated = merged_nodes_repeated.front();
  EXPECT_THAT(
      merged_add_repeated,
      m::Add(m::Select(m::Param("global_mux_select"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                        m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                        m::TupleIndex(m::Param("func_c_ParamTuple"), 0)},
                       m::TupleIndex(m::Param("func_a_ParamTuple"), 0)),
             m::Select(m::Param("global_mux_select"),
                       {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_b_ParamTuple"), 1),
                        m::TupleIndex(m::Param("func_c_ParamTuple"), 1)},
                       m::TupleIndex(m::Param("func_a_ParamTuple"), 1))));
  EXPECT_EQ(integration->function()->node_count(), init_node_count + 3);
}

TEST_F(IntegratorTest, GetSourceFunctionIndexOfNodeTest) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_in1_node = FindNode("in1", func_a);
  Node* a_add_node = FindNode("add", func_a);
  Node* b_in1_node = FindNode("in1", func_b);
  Node* b_add_node = FindNode("add", func_b);
  Node* c_in1_node = FindNode("in1", func_c);
  Node* c_add_node = FindNode("add", func_c);
  Node* d_in1_node = FindNode("in1", func_d);
  Node* d_add_node = FindNode("add", func_d);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c}));

  auto test_idx = [&integration](Node* node, int64_t expected_index) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t idx,
                             integration->GetSourceFunctionIndexOfNode(node));
    EXPECT_EQ(idx, expected_index);
  };

  test_idx(a_in1_node, 0);
  test_idx(a_add_node, 0);
  test_idx(b_in1_node, 1);
  test_idx(b_add_node, 1);
  test_idx(c_in1_node, 2);
  test_idx(c_add_node, 2);
  auto non_source_result =
      integration->GetSourceFunctionIndexOfNode(d_in1_node);
  EXPECT_FALSE(non_source_result.ok());
  non_source_result = integration->GetSourceFunctionIndexOfNode(d_add_node);
  EXPECT_FALSE(non_source_result.ok());
}

TEST_F(IntegratorTest, GetSourceFunctionIndexesOfNodesMappedToNodeTest) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  fb_a.Param("in1", p->GetBitsType(2));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_c, func_a->Clone("func_c"));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_d, func_a->Clone("func_d"));
  Node* a_in1_node = FindNode("in1", func_a);
  Node* b_in1_node = FindNode("in1", func_b);
  Node* c_in1_node = FindNode("in1", func_c);
  Node* d_in1_node = FindNode("in1", func_d);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b, func_c, func_d}));

  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_1,
                           integration->function()->MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_1"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_2,
                           integration->function()->MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_2"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * internal_3,
                           integration->function()->MakeNodeWithName<Param>(
                               SourceInfo(), p->GetBitsType(2), "internal_3"));

  XLS_ASSERT_OK(integration->SetNodeMapping(a_in1_node, internal_1));
  XLS_ASSERT_OK(integration->SetNodeMapping(c_in1_node, internal_1));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_in1_node, internal_2));
  XLS_ASSERT_OK(integration->SetNodeMapping(d_in1_node, internal_2));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::set<int64_t> internal_1_source_indexes,
      integration->GetSourceFunctionIndexesOfNodesMappedToNode(internal_1));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::set<int64_t> internal_2_source_indexes,
      integration->GetSourceFunctionIndexesOfNodesMappedToNode(internal_2));
  EXPECT_THAT(internal_1_source_indexes, ElementsAre(0, 2));
  EXPECT_THAT(internal_2_source_indexes, ElementsAre(1, 3));

  auto non_map_target_result =
      integration->GetSourceFunctionIndexesOfNodesMappedToNode(internal_3);
  EXPECT_FALSE(non_map_target_result.ok());
}

TEST_F(IntegratorTest, MakeTupleReturnValueTest) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  Node* a_add_node = FindNode("add", func_a);

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  fb_b.UMul(b_in1, b_in1, SourceInfo(), "mul");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* b_add_node = FindNode("mul", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(false)));
  XLS_ASSERT_OK(integration->InsertNode(a_add_node).status());
  XLS_ASSERT_OK(integration->InsertNode(b_add_node).status());
  XLS_ASSERT_OK_AND_ASSIGN(Node * ret_value,
                           integration->MakeTupleReturnValue());

  EXPECT_EQ(ret_value, integration->function()->return_value());
  EXPECT_THAT(
      ret_value,
      m::Tuple(m::Add(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                      m::TupleIndex(m::Param("func_a_ParamTuple"), 0)),
               m::UMul(m::TupleIndex(m::Param("func_b_ParamTuple"), 0),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 0))));
}

TEST_F(IntegratorTest, MakeTupleReturnValueErrors) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  fb_a.Add(a_in1, a_in1, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  Node* a_add_node = FindNode("add", func_a);

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  fb_b.UMul(b_in1, b_in1, SourceInfo(), "mul");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());
  Node* b_add_node = FindNode("mul", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b},
          IntegrationOptions().unique_select_signal_per_mux(false)));
  XLS_ASSERT_OK(integration->InsertNode(a_add_node).status());

  // Can't make tuple output if not all source function return values have a
  // mapping in the integration function.
  auto missing_mapping_result = integration->MakeTupleReturnValue();
  EXPECT_FALSE(missing_mapping_result.ok());

  XLS_ASSERT_OK(integration->InsertNode(b_add_node).status());
  XLS_ASSERT_OK_AND_ASSIGN(Node * ret_value,
                           integration->MakeTupleReturnValue());
  EXPECT_EQ(ret_value, integration->function()->return_value());
  EXPECT_THAT(
      ret_value,
      m::Tuple(m::Add(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                      m::TupleIndex(m::Param("func_a_ParamTuple"), 0)),
               m::UMul(m::TupleIndex(m::Param("func_b_ParamTuple"), 0),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 0))));

  // Can only call MakeTupleReturnValue once.
  auto repeate_result = integration->MakeTupleReturnValue();
  EXPECT_FALSE(repeate_result.ok());
}

TEST_F(IntegratorTest, AllOperandsHaveMappingTest) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto op1 = fb_a.Add(in1, in1, SourceInfo(), "op1");
  auto op2 = fb_a.Add(in1, in1, SourceInfo(), "op2");
  fb_a.Add(op1, op2, SourceInfo(), "add");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());
  Node* op1_node = FindNode("op1", func_a);
  Node* op2_node = FindNode("op2", func_a);
  Node* add_node = FindNode("add", func_a);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a},
          IntegrationOptions().unique_select_signal_per_mux(false)));

  EXPECT_FALSE(integration->AllOperandsHaveMapping(add_node));
  XLS_ASSERT_OK(integration->InsertNode(op1_node).status());
  EXPECT_FALSE(integration->AllOperandsHaveMapping(add_node));
  XLS_ASSERT_OK(integration->InsertNode(op2_node).status());
  EXPECT_TRUE(integration->AllOperandsHaveMapping(add_node));
}

TEST_F(IntegratorTest, NodeSourceFunctionsCollideTest) {
  auto p = CreatePackage();
  FunctionBuilder fb("func", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  fb.Add(in1, in1, SourceInfo(), "add1");
  fb.Add(in1, in1, SourceInfo(), "add2");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));
  Node* a_add1 = FindNode("add1", func_a);
  Node* a_add2 = FindNode("add2", func_a);
  Node* b_add1 = FindNode("add1", func_b);
  Node* b_add2 = FindNode("add2", func_b);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationFunction> integration,
      IntegrationFunction::MakeIntegrationFunctionWithParamTuples(
          p.get(), {func_a, func_b}));

  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add1_internal,
                           integration->InsertNode(a_add1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * a_add2_internal,
                           integration->InsertNode(a_add2));
  XLS_ASSERT_OK_AND_ASSIGN(Node * b_add1_internal,
                           integration->InsertNode(b_add1));
  XLS_ASSERT_OK(integration->SetNodeMapping(b_add2, a_add2_internal));

  // Two external nodes.
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1, b_add1),
              IsOkAndHolds(false));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1, b_add1),
              IsOkAndHolds(false));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1, a_add2),
              IsOkAndHolds(true));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add2, a_add1),
              IsOkAndHolds(true));

  // One internal, one external node.
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1_internal, b_add1),
              IsOkAndHolds(false));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(b_add1, a_add1_internal),
              IsOkAndHolds(false));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1_internal, a_add1),
              IsOkAndHolds(true));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add1, a_add1_internal),
              IsOkAndHolds(true));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(a_add2_internal, b_add1),
              IsOkAndHolds(true));
  EXPECT_THAT(integration->NodeSourceFunctionsCollide(b_add1, a_add2_internal),
              IsOkAndHolds(true));

  // Two internal nodes.
  EXPECT_THAT(
      integration->NodeSourceFunctionsCollide(a_add1_internal, b_add1_internal),
      IsOkAndHolds(false));
  EXPECT_THAT(
      integration->NodeSourceFunctionsCollide(b_add1_internal, a_add1_internal),
      IsOkAndHolds(false));
  EXPECT_THAT(
      integration->NodeSourceFunctionsCollide(a_add1_internal, a_add2_internal),
      IsOkAndHolds(true));
  EXPECT_THAT(
      integration->NodeSourceFunctionsCollide(a_add2_internal, a_add1_internal),
      IsOkAndHolds(true));
}

}  // namespace
}  // namespace xls
