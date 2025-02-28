// Copyright 2023 The XLS Authors
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

#include "xls/passes/token_simplification_pass.h"

#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

class TokenSimplificationPassTest : public IrTestBase {
 protected:
  TokenSimplificationPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    OptimizationContext context;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         TokenSimplificationPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results, context));
    return changed;
  }
};

TEST_F(TokenSimplificationPassTest, SingleArgument) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       after_all.1: token = after_all(tok)
       tuple.2: () = tuple()
       next (after_all.1, tuple.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                                  m::StateRead("tok"))));
}

TEST_F(TokenSimplificationPassTest, DuplicatedArgument) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       after_all.1: token = after_all(tok, tok, tok)
       tuple.2: () = tuple()
       next (after_all.1, tuple.2)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                                  m::StateRead("tok"))));
}

TEST_F(TokenSimplificationPassTest, NestedAfterAll) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       after_all.1: token = after_all(tok, tok, tok)
       after_all.2: token = after_all(after_all.1, tok, tok)
       tuple.3: () = tuple()
       next (after_all.2, tuple.3)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                                  m::StateRead("tok"))));
}

TEST_F(TokenSimplificationPassTest, DelayZero) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       min_delay.1: token = min_delay(tok, delay=0)
       tuple.3: () = tuple()
       next (min_delay.1, tuple.3)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                                  m::StateRead("tok"))));
}

TEST_F(TokenSimplificationPassTest, NestedDelay) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       min_delay.1: token = min_delay(tok, delay=1)
       min_delay.2: token = min_delay(min_delay.1, delay=2)
       tuple.3: () = tuple()
       next (min_delay.2, tuple.3)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                          m::MinDelay(m::StateRead("tok"), /*delay=*/3))));
}

TEST_F(TokenSimplificationPassTest, AfterAllWithCommonDelay) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     top proc main(tok: token, state: (), init={token, ()}) {
       min_delay.1: token = min_delay(tok, delay=1)
       min_delay.2: token = min_delay(tok, delay=2)
       after_all.3: token = after_all(min_delay.1, min_delay.2)
       min_delay.4: token = min_delay(after_all.3, delay=4)
       tuple.5: () = tuple()
       next (min_delay.4, tuple.5)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                          m::MinDelay(m::StateRead("tok"), /*delay=*/6))));
}

TEST_F(TokenSimplificationPassTest, DuplicatedArgument2) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid)

     top proc main(tok: token, state: (), init={token, ()}) {
       literal.1: bits[32] = literal(value=10)
       send.2: token = send(tok, literal.1, channel=test_channel)
       send.3: token = send(send.2, literal.1, channel=test_channel)
       send.4: token = send(tok, literal.1, channel=test_channel)
       after_all.5: token = after_all(send.2, send.3, send.4)
       tuple.6: () = tuple()
       next (after_all.5, tuple.6)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      ElementsAre(m::Next(
          proc->GetStateRead(int64_t{0}),
          m::AfterAll(
              m::Send(m::Send(m::StateRead("tok"), m::Literal()), m::Literal()),
              m::Send(m::StateRead("tok"), m::Literal())))));
}

TEST_F(TokenSimplificationPassTest, UnrelatedArguments) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid)

     top proc main(tok: token, state: (), init={token, ()}) {
       literal.1: bits[32] = literal(value=10)
       send.2: token = send(tok, literal.1, channel=test_channel)
       send.3: token = send(tok, literal.1, channel=test_channel)
       send.4: token = send(tok, literal.1, channel=test_channel)
       after_all.5: token = after_all(send.2, send.3, send.4)
       tuple.6: () = tuple()
       next (after_all.5, tuple.6)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(
                  proc->GetStateRead(int64_t{0}),
                  m::AfterAll(m::Send(m::StateRead("tok"), m::Literal()),
                              m::Send(m::StateRead("tok"), m::Literal()),
                              m::Send(m::StateRead("tok"), m::Literal())))));
}

TEST_F(TokenSimplificationPassTest, ArgumentsWithDependencies) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid)

     top proc main(tok: token, state: (), init={token, ()}) {
       literal.1: bits[32] = literal(value=10)
       send.2: token = send(tok, literal.1, channel=test_channel)
       send.3: token = send(send.2, literal.1, channel=test_channel)
       after_all.4: token = after_all(tok, send.2, send.3)
       tuple.5: () = tuple()
       next (after_all.4, tuple.5)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(proc->GetStateRead(int64_t{0})),
              ElementsAre(m::Next(proc->GetStateRead(int64_t{0}), m::Send())));
}

TEST_F(TokenSimplificationPassTest, DoNotRelyOnInvokeForDependencies) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(R"(
     package test_module

     chan test_channel(
       bits[32], id=0, kind=streaming, ops=send_only,
       flow_control=ready_valid)

     fn test_fn(tok1: token, tok2: token) -> token {
       ret tok: token = param(name=tok2)
     }

     top proc main(tok: token, state: (), init={token, ()}) {
       literal.1: bits[32] = literal(value=10)
       send.2: token = send(tok, literal.1, channel=test_channel)
       send.3: token = send(tok, literal.1, channel=test_channel)
       invoke.4: token = invoke(send.2, send.3, to_apply=test_fn)
       after_all.5: token = after_all(tok, send.2, send.3, invoke.4)
       tuple.6: () = tuple()
       next (after_all.5, tuple.6)
     }
  )"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, p->GetTopAsProc());
  EXPECT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(int64_t{0})),
      ElementsAre(m::Next(proc->GetStateRead(int64_t{0}),
                          m::AfterAll(m::Send(), m::Send(), m::Invoke()))));
}

}  // namespace
}  // namespace xls
