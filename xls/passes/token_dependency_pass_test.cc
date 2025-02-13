// Copyright 2022 The XLS Authors
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

#include "xls/passes/token_dependency_pass.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class TokenDependencyPassTest : public IrTestBase {
 protected:
  TokenDependencyPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TokenDependencyPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results));
    return changed;
  }
};

TEST_F(TokenDependencyPassTest, Simple) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: (), init={()}) {
       __token: token = literal(value=token, id=1000)
       receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
       tuple_index.2: token = tuple_index(receive.1, index=0)
       tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
       send.4: token = send(__token, tuple_index.3, channel=test_channel)
       after_all.5: token = after_all(send.4, tuple_index.2)
       tuple.6: () = tuple()
       next (tuple.6)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->GetNode("after_all.5"),
              IsOkAndHolds(m::AfterAll(
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::TupleIndex())));
}

TEST_F(TokenDependencyPassTest, MultipleSends) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: (), init={()}) {
       __token: token = literal(value=token, id=1000)
       receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
       tuple_index.2: token = tuple_index(receive.1, index=0)
       tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
       send.4: token = send(__token, tuple_index.3, channel=test_channel)
       send.5: token = send(__token, tuple_index.3, channel=test_channel)
       send.6: token = send(__token, tuple_index.3, channel=test_channel)
       after_all.7: token = after_all(send.4, send.5, send.6, tuple_index.2)
       tuple.8: () = tuple()
       next (tuple.8)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->GetNode("after_all.7"),
              IsOkAndHolds(m::AfterAll(
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::TupleIndex())));
}

TEST_F(TokenDependencyPassTest, DependentSends) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: (), init={()}) {
       __token: token = literal(value=token, id=1000)
       receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
       tuple_index.2: token = tuple_index(receive.1, index=0)
       tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
       send.4: token = send(__token, tuple_index.3, channel=test_channel)
       send.5: token = send(send.4, tuple_index.3, channel=test_channel)
       send.6: token = send(__token, tuple_index.3, channel=test_channel)
       after_all.7: token = after_all(send.4, send.5, send.6, tuple_index.2)
       tuple.8: () = tuple()
       next (tuple.8)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->GetNode("after_all.7"),
              IsOkAndHolds(m::AfterAll(
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::Send(m::Send(), m::TupleIndex()),
                  m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                                      m::Literal(Value::Token())),
                          m::TupleIndex()),
                  m::TupleIndex())));
}

TEST_F(TokenDependencyPassTest, DependentSendsMultipleReceives) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_receive,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: (), init={()}) {
       __token: token = literal(value=token, id=1000)
       receive.1: (token, bits[32]) = receive(__token, channel=test_channel)
       tuple_index.2: token = tuple_index(receive.1, index=0)
       tuple_index.3: bits[32] = tuple_index(receive.1, index=1)
       receive.4: (token, bits[32]) = receive(__token, channel=test_channel)
       tuple_index.5: token = tuple_index(receive.4, index=0)
       tuple_index.6: bits[32] = tuple_index(receive.4, index=1)
       send.7: token = send(__token, tuple_index.3, channel=test_channel)
       add.8: bits[32] = add(tuple_index.3, tuple_index.6)
       send.9: token = send(send.7, add.8, channel=test_channel)
       after_all.10: token = after_all(send.7, send.9, tuple_index.2, tuple_index.5)
       tuple.11: () = tuple()
       next (tuple.11)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->GetNode("after_all.10"),
      IsOkAndHolds(m::AfterAll(
          m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0),
                              m::Literal(Value::Token())),
                  m::TupleIndex(m::Receive(), 1)),
          m::Send(m::AfterAll(m::TupleIndex(m::Receive(), 0), m::Send()),
                  m::Add()),
          m::TupleIndex(m::Receive(), 0), m::TupleIndex(m::Receive(), 0))));
}

TEST_F(TokenDependencyPassTest, SideEffectingNontokenOps) {
  // Regression test for https://github.com/google/xls/issues/776
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=receive_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(init={}) {
       __token: token = literal(value=token, id=1000)
       rcv: (token, bits[32]) = receive(__token, channel=test_channel)
       tkn: token = tuple_index(rcv, index=0)
       data: bits[32] = tuple_index(rcv, index=1)
       one: bits[1] = literal(value=1)
       g: bits[32] = gate(one, data)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  // Should not crash.
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(TokenDependencyPassTest, SupportsCrossActivationTokens) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid, metadata="""""")

     top proc main(__state: (bits[32], token, bits[32]), init={(1, token, 1)}) {
       a: bits[32] = tuple_index(__state, index=0)
       b: bits[32] = tuple_index(__state, index=2)
       tok: token = tuple_index(__state, index=1)
       c: bits[32] = add(a, b)
       new_tok: token = literal(value=token)
       snd: token = send(new_tok, c, channel=test_channel)
       next_tok: token = after_all(tok, snd)
       next_state: (bits[32], token, bits[32]) = tuple(b, next_tok, c)
       next (next_state)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  // No changes required; in particular, we don't need the send to depend on the
  // cross-activation token as written.
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
