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

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/integrator/integration_builder.h"
#include "xls/contrib/integrator/integration_options.h"
#include "xls/contrib/integrator/ir_integrator.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

class BasicIntegrationAlgorithmTest : public IrTestBase {};

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationNodesNotCompatible) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("a_in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("a_in2", p->GetBitsType(2));

  fb_a.Or(a_in1, a_in2, SourceInfo(), "a_or");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("b_in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("b_in2", p->GetBitsType(2));

  fb_b.And(b_in1, b_in2, SourceInfo(), "b_and");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 10);
  EXPECT_THAT(
      builder->integrated_function()->function()->return_value(),
      m::Tuple(m::Or(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                     m::TupleIndex(m::Param("func_a_ParamTuple"), 1)),
               m::And(m::TupleIndex(m::Param("func_b_ParamTuple"), 0),
                      m::TupleIndex(m::Param("func_b_ParamTuple"), 1))));
}

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationMergeNotProfitable) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("a_in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("a_in2", p->GetBitsType(2));
  auto a_in3 = fb_a.Param("a_in3", p->GetBitsType(2));
  auto a_in4 = fb_a.Param("a_in4", p->GetBitsType(2));

  fb_a.Or({a_in1, a_in2, a_in3, a_in4}, SourceInfo(), "a_or");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("b_in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("b_in2", p->GetBitsType(2));
  auto b_in3 = fb_b.Param("b_in3", p->GetBitsType(2));
  auto b_in4 = fb_b.Param("b_in4", p->GetBitsType(2));

  fb_b.Or({b_in1, b_in2, b_in3, b_in4}, SourceInfo(), "b_or");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 14);
  EXPECT_THAT(builder->integrated_function()->function()->return_value(),
              m::Tuple(m::Or(m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_a_ParamTuple"), 3)),
                       m::Or(m::TupleIndex(m::Param("func_b_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 3))));
}

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationIdentical) {
  auto p = CreatePackage();
  FunctionBuilder fb("func_a", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto in2 = fb.Param("in2", p->GetBitsType(2));
  auto in3 = fb.Param("in3", p->GetBitsType(2));
  auto in4 = fb.Param("in4", p->GetBitsType(2));
  auto add1 = fb.Add(in1, in2, SourceInfo(), "add1");
  auto add2 = fb.Add(in3, in4, SourceInfo(), "add2");
  fb.UMul(add1, add2, SourceInfo(), "mul");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 19);
  EXPECT_THAT(
      builder->integrated_function()->function()->return_value(),
      m::Tuple(
          m::UMul(
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})),
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 2)}),
                  m::Select(
                      m::Param("global_mux_select"),
                      {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 3)}))),
          // Duplicate the above.
          m::UMul(
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})),
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 2)}),
                  m::Select(
                      m::Param("global_mux_select"),
                      {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 3)})))));
}

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationDifferentOutput) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("in2", p->GetBitsType(2));
  auto a_in3 = fb_a.Param("in3", p->GetBitsType(2));
  auto a_in4 = fb_a.Param("in4", p->GetBitsType(2));
  auto a_add1 = fb_a.Add(a_in1, a_in2, SourceInfo(), "add1");
  auto a_add2 = fb_a.Add(a_in3, a_in4, SourceInfo(), "add2");
  fb_a.And(a_add1, a_add2, SourceInfo(), "and");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("in2", p->GetBitsType(2));
  auto b_in3 = fb_b.Param("in3", p->GetBitsType(2));
  auto b_in4 = fb_b.Param("in4", p->GetBitsType(2));
  auto b_add1 = fb_b.Add(b_in1, b_in2, SourceInfo(), "add1");
  auto b_add2 = fb_b.Add(b_in3, b_in4, SourceInfo(), "add2");
  fb_b.Or(b_add1, b_add2, SourceInfo(), "or");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 20);
  EXPECT_THAT(
      builder->integrated_function()->function()->return_value(),
      m::Tuple(
          m::And(
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})),
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 2)}),
                  m::Select(
                      m::Param("global_mux_select"),
                      {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 3)}))),
          m::Or(
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 1)})),
              m::Add(
                  m::Select(m::Param("global_mux_select"),
                            {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                             m::TupleIndex(m::Param("func_b_ParamTuple"), 2)}),
                  m::Select(
                      m::Param("global_mux_select"),
                      {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                       m::TupleIndex(m::Param("func_b_ParamTuple"), 3)})))));
}

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationDifferentIntermediate) {
  auto p = CreatePackage();
  FunctionBuilder fb_a("func_a", p.get());
  auto a_in1 = fb_a.Param("in1", p->GetBitsType(2));
  auto a_in2 = fb_a.Param("in2", p->GetBitsType(2));
  auto a_in3 = fb_a.Param("in3", p->GetBitsType(2));
  auto a_in4 = fb_a.Param("in4", p->GetBitsType(2));
  auto a_in5 = fb_a.Param("in5", p->GetBitsType(2));
  auto a_add1 = fb_a.Add(a_in1, a_in2, SourceInfo(), "add1");
  auto a_add2 = fb_a.Add(a_in3, a_in4, SourceInfo(), "add2");
  auto a_and = fb_a.And(a_add1, a_add2, SourceInfo(), "and");
  fb_a.UMul(a_in5, a_and, SourceInfo(), "mul");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb_a.Build());

  FunctionBuilder fb_b("func_b", p.get());
  auto b_in1 = fb_b.Param("in1", p->GetBitsType(2));
  auto b_in2 = fb_b.Param("in2", p->GetBitsType(2));
  auto b_in3 = fb_b.Param("in3", p->GetBitsType(2));
  auto b_in4 = fb_b.Param("in4", p->GetBitsType(2));
  auto b_in5 = fb_b.Param("in5", p->GetBitsType(2));
  auto b_add1 = fb_b.Add(b_in1, b_in2, SourceInfo(), "add1");
  auto b_add2 = fb_b.Add(b_in3, b_in4, SourceInfo(), "add2");
  auto b_or = fb_b.Or(b_add1, b_add2, SourceInfo(), "or");
  fb_b.UMul(b_in5, b_or, SourceInfo(), "mul");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, fb_b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 25);
  EXPECT_THAT(
      builder->integrated_function()->function()->return_value(),
      m::Tuple(
          m::UMul(
              m::Select(m::Param("global_mux_select"),
                        {m::TupleIndex(m::Param("func_a_ParamTuple"), 4),
                         m::TupleIndex(m::Param("func_b_ParamTuple"), 4)}),
              m::Select(
                  m::Param("global_mux_select"),
                  {m::And(
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              0)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              1)})),
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              2)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              3)}))),
                   m::Or(
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              0)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              1)})),
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              2)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              3)})))})),
          // Duplicate the above.
          m::UMul(
              m::Select(m::Param("global_mux_select"),
                        {m::TupleIndex(m::Param("func_a_ParamTuple"), 4),
                         m::TupleIndex(m::Param("func_b_ParamTuple"), 4)}),
              m::Select(
                  m::Param("global_mux_select"),
                  {m::And(
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              0)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              1)})),
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              2)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              3)}))),
                   m::Or(
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              0)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 1),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              1)})),
                       m::Add(
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 2),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              2)}),
                           m::Select(
                               m::Param("global_mux_select"),
                               {m::TupleIndex(m::Param("func_a_ParamTuple"), 3),
                                m::TupleIndex(m::Param("func_b_ParamTuple"),
                                              3)})))}))));
}

TEST_F(BasicIntegrationAlgorithmTest, BasicIntegrationLiterals) {
  auto p = CreatePackage();
  FunctionBuilder fb("func_a", p.get());
  auto in1 = fb.Param("in1", p->GetBitsType(2));
  auto literal1 = fb.Literal(UBits(2, 2));
  fb.Add(in1, literal1, SourceInfo(), "add1");
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_a, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * func_b, func_a->Clone("func_b"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<IntegrationBuilder> builder,
      IntegrationBuilder::Build(
          {func_a, func_b},
          IntegrationOptions().algorithm(
              IntegrationOptions::Algorithm::kBasicIntegrationAlgorithm)));

  Function* function = builder->integrated_function()->function();
  EXPECT_EQ(function->node_count(), 9);
  EXPECT_THAT(
      builder->integrated_function()->function()->return_value(),
      m::Tuple(
          m::Add(m::Select(m::Param("global_mux_select"),
                           {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                            m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                 m::Literal(UBits(2, 2))),
          // Duplicate the above.
          m::Add(m::Select(m::Param("global_mux_select"),
                           {m::TupleIndex(m::Param("func_a_ParamTuple"), 0),
                            m::TupleIndex(m::Param("func_b_ParamTuple"), 0)}),
                 m::Literal(UBits(2, 2)))));
}

}  // namespace
}  // namespace xls
