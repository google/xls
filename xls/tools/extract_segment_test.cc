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

#include "xls/tools/extract_segment.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

class ExtractSegmentTest : public IrTestBase {};

using testing::UnorderedElementsAre;

TEST_F(ExtractSegmentTest, FromSource) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto ab = fb.Add(a, fb.Not(b));
  auto nab = fb.Add(fb.Literal(UBits(32, 32)), ab);
  auto cd = fb.Add(c, d);
  fb.Add(nab, cd);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  RecordProperty("source", f->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto* f2,
                           ExtractSegment(f, /*source_nodes=*/{a.node()},
                                          /*sink_nodes=*/{}, "extracted"));
  RecordProperty("res", f2->DumpIr());
  EXPECT_THAT(f2->return_value(),
              m::Add(m::Add(m::Literal(UBits(32, 32)),
                            m::Add(m::Param("a"), m::Param())),
                     m::Param()));
}

TEST_F(ExtractSegmentTest, ToSink) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto ab = fb.Add(a, fb.Not(b));
  auto nab = fb.Add(fb.Literal(UBits(32, 32)), ab);
  auto cd = fb.Add(c, d);
  fb.Add(nab, cd);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  RecordProperty("source", f->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* f2, ExtractSegment(f, /*source_nodes=*/{},
                               /*sink_nodes=*/{nab.node()}, "extracted"));
  RecordProperty("res", f2->DumpIr());
  EXPECT_THAT(f2->return_value(),
              m::Add(m::Literal(UBits(32, 32)),
                     m::Add(m::Param("a"), m::Not(m::Param("b")))));
  EXPECT_EQ(f2->params().size(), 2);
}

TEST_F(ExtractSegmentTest, SourceAndSink) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto ab = fb.Add(a, fb.Not(b));
  auto nab = fb.Add(fb.Literal(UBits(32, 32)), ab);
  auto cd = fb.Add(c, d);
  fb.Add(nab, cd);
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, fb.Build());
  RecordProperty("source", f->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* f2, ExtractSegment(f, /*source_nodes=*/{a.node()},
                               /*sink_nodes=*/{nab.node()}, "extracted"));
  RecordProperty("res", f2->DumpIr());
  EXPECT_THAT(f2->return_value(), m::Add(m::Literal(UBits(32, 32)),
                                         m::Add(m::Param("a"), m::Param())));
  EXPECT_THAT(f2->params(), UnorderedElementsAre(m::Param("a"), m::Param()));
}

TEST_F(ExtractSegmentTest, NextNodesAsTuples) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  auto a = pb.StateElement("a", UBits(0, 32));
  auto b = pb.StateElement("b", UBits(0, 32));
  auto c = pb.StateElement("c", UBits(0, 32));
  auto d = pb.StateElement("d", UBits(0, 32));
  auto n = pb.Next(a, pb.Add(a, b), pb.Eq(b, c));
  pb.Next(b, pb.Add(c, d));
  pb.Next(c, pb.Add(a, d));
  pb.Next(d, pb.Add(b, c));
  XLS_ASSERT_OK_AND_ASSIGN(auto* f, pb.Build());
  RecordProperty("source", f->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto* f2, ExtractSegment(f, /*source_nodes=*/{},
                               /*sink_nodes=*/{n.node()}, "extracted"));
  RecordProperty("res", f2->DumpIr());
  EXPECT_THAT(f2->return_value(),
              m::Tuple(m::Param("a"), m::Add(m::Param("a"), m::Param("b")),
                       m::Eq(m::Param("b"), m::Param("c"))));
  EXPECT_THAT(f2->params(), UnorderedElementsAre(m::Param("a"), m::Param("b"),
                                                 m::Param("c")));
}

}  // namespace
}  // namespace xls
