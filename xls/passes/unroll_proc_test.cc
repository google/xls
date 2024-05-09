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

#include "xls/passes/unroll_proc.h"

#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

class UnrollProcTest : public IrTestBase {};

TEST_F(UnrollProcTest, UnrollSimple) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(__state: bits[10], init={0}) {
       literal.1: bits[10] = literal(value=1)
       add.2: bits[10] = add(literal.1, __state)
       next (add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  XLS_ASSERT_OK_AND_ASSIGN(ProcUnrollInfo unroll_info, UnrollProc(proc, 3));
  EXPECT_EQ(proc->NextState().size(), 1);
  EXPECT_THAT(
      proc->NextState().front(),
      m::Add(m::Literal(UBits(1, 10)),
             m::Add(m::Literal(UBits(1, 10)),
                    m::Add(m::Literal(UBits(1, 10)), m::Param("__state")))));
  auto get_original = [&](const std::string& name) -> std::string {
    return unroll_info.original_node_map.at(*(proc->GetNode(name)))->GetName();
  };
  auto get_iteration = [&](const std::string& name) -> int64_t {
    return unroll_info.iteration_map.at(*(proc->GetNode(name)));
  };
  EXPECT_EQ(get_original("literal_1_iter_0"), "literal_1_iter_0");
  EXPECT_EQ(get_original("literal_1_iter_1"), "literal_1_iter_0");
  EXPECT_EQ(get_original("literal_1_iter_2"), "literal_1_iter_0");
  EXPECT_EQ(get_original("add_2_iter_0"), "add_2_iter_0");
  EXPECT_EQ(get_original("add_2_iter_1"), "add_2_iter_0");
  EXPECT_EQ(get_original("add_2_iter_2"), "add_2_iter_0");
  EXPECT_EQ(get_iteration("literal_1_iter_0"), 0);
  EXPECT_EQ(get_iteration("literal_1_iter_1"), 1);
  EXPECT_EQ(get_iteration("literal_1_iter_2"), 2);
  EXPECT_EQ(get_iteration("add_2_iter_0"), 0);
  EXPECT_EQ(get_iteration("add_2_iter_1"), 1);
  EXPECT_EQ(get_iteration("add_2_iter_2"), 2);
}

TEST_F(UnrollProcTest, UnrollWithIO) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[10], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__token: token, __state: bits[10], init={token, 0}) {
       literal.1: bits[10] = literal(value=1)
       add.2: bits[10] = add(literal.1, __state)
       send.3: token = send(__token, add.2, channel=test_channel)
       next (send.3, add.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  XLS_ASSERT_OK_AND_ASSIGN(ProcUnrollInfo unroll_info, UnrollProc(proc, 3));
  EXPECT_EQ(proc->NextState().size(), 2);
  EXPECT_THAT(proc->GetNextStateElement(0),
              m::Send(m::Send(m::Send(m::Param("__token"), m::Add()), m::Add()),
                      m::Add()));
  EXPECT_THAT(
      proc->GetNextStateElement(1),
      m::Add(m::Literal(UBits(1, 10)),
             m::Add(m::Literal(UBits(1, 10)),
                    m::Add(m::Literal(UBits(1, 10)), m::Param("__state")))));
  auto get_original = [&](const std::string& name) -> std::string {
    return unroll_info.original_node_map.at(*(proc->GetNode(name)))->GetName();
  };
  auto get_iteration = [&](const std::string& name) -> int64_t {
    return unroll_info.iteration_map.at(*(proc->GetNode(name)));
  };
  EXPECT_EQ(get_original("literal_1_iter_0"), "literal_1_iter_0");
  EXPECT_EQ(get_original("literal_1_iter_1"), "literal_1_iter_0");
  EXPECT_EQ(get_original("literal_1_iter_2"), "literal_1_iter_0");
  EXPECT_EQ(get_original("add_2_iter_0"), "add_2_iter_0");
  EXPECT_EQ(get_original("add_2_iter_1"), "add_2_iter_0");
  EXPECT_EQ(get_original("add_2_iter_2"), "add_2_iter_0");
  EXPECT_EQ(get_original("send_3_iter_0"), "send_3_iter_0");
  EXPECT_EQ(get_original("send_3_iter_1"), "send_3_iter_0");
  EXPECT_EQ(get_original("send_3_iter_2"), "send_3_iter_0");
  EXPECT_EQ(get_iteration("literal_1_iter_0"), 0);
  EXPECT_EQ(get_iteration("literal_1_iter_1"), 1);
  EXPECT_EQ(get_iteration("literal_1_iter_2"), 2);
  EXPECT_EQ(get_iteration("add_2_iter_0"), 0);
  EXPECT_EQ(get_iteration("add_2_iter_1"), 1);
  EXPECT_EQ(get_iteration("add_2_iter_2"), 2);
  EXPECT_EQ(get_iteration("send_3_iter_0"), 0);
  EXPECT_EQ(get_iteration("send_3_iter_1"), 1);
  EXPECT_EQ(get_iteration("send_3_iter_2"), 2);
}

}  // namespace
}  // namespace xls
