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

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/error_test_utils.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {
namespace {

TEST(TypecheckTest, ConfigSpawnTerminatingSemicolonNoMembers) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { }
    config() { }
    next(tok: token, state: ()) { }
}

proc entry {
    init { () }
    config() { spawn foo(); }
    next (tok: token, state: ()) { () }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, RecvIfDefaultValueWrongType) {
  constexpr std::string_view kProgram = R"(
proc foo {
    c : chan<u32> in;
    init {
        u32:0
    }
    config(c: chan<u32> in) {
        (c,)
    }
    next(tok: token, state: u32) {
        let (tok, x) = recv_if(tok, c, true, u42:1234);
        (state + x,)
    }
}

proc entry {
    c: chan<u32> out;
    init { () }
    config() {
        let (p, c) = chan<u32>;
        spawn foo(p);
        (c,)
    }
    next (tok: token, state: ()) { () }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Want argument 3 to 'recv_if' to have type uN[32]; got uN[42]")));
}

TEST(TypecheckTest, InitDoesntMatchStateParam) {
  constexpr std::string_view kProgram = R"(
proc oopsie {
    init { u32:0xbeef }
    config() { () }
    next(tok: token, state: u33) {
      state
    }
})";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("'next' state param and 'init' types differ")));
}

TEST(TypecheckTest, NextReturnDoesntMatchState) {
  constexpr std::string_view kProgram = R"(
proc oopsie {
    init { u32:0xbeef }
    config() { () }
    next(tok: token, state: u32) {
      state as u33
    }
})";

  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("input and output state types differ")));
}

TEST(TypecheckTest, CantSendOnNonMember) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { () }

    config() {
        ()
    }

    next(tok: token, state: ()) {
        let foo = u32:0;
        let tok = send(tok, foo, u32:0x0);
        ()
    }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Want argument 1 to 'send' to be a channel; got uN[32]")));
}

TEST(TypecheckTest, CantSendOnNonChannel) {
  constexpr std::string_view kProgram = R"(
proc foo {
    bar: u32;
    init { () }
    config() {
        (u32:0,)
    }
    next(tok: token, state: ()) {
        let tok = send(tok, bar, u32:0x0);
        ()
    }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Want argument 1 to 'send' to be a channel; got uN[32]")));
}

TEST(TypecheckTest, CantRecvOnOutputChannel) {
  constexpr std::string_view kProgram = R"(
proc foo {
    c : chan<u32> out;
    init {
        u32:0
    }
    config(c: chan<u32> out) {
        (c,)
    }
    next(tok: token, state: u32) {
        let (tok, x) = recv(tok, c);
        (state + x,)
    }
}

proc entry {
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>;
        spawn foo(c);
        (p,)
    }
    next (tok: token, state: ()) { () }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Want argument 1 to 'recv' to be an 'in' (recv) "
                             "channel; got chan(uN[32], dir=out)")));
}

TEST(TypecheckTest, CantSendOnOutputChannel) {
  constexpr std::string_view kProgram = R"(
proc entry {
    p: chan<u32> out;
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>;
        (p, c)
    }
    next (tok: token, state: ()) {
        let tok = send(tok, c, u32:0);
        ()
    }
}
)";
  EXPECT_THAT(
      Typecheck(kProgram),
      status_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Want argument 1 to 'send' to be an 'out' (send) "
                             "channel; got chan(uN[32], dir=in)")));
}

TEST(TypecheckTest, CanUseZeroMacroInInitIssue943) {
  constexpr std::string_view kProgram = R"(
struct bar_t {
  f: u32
}

proc foo {
  config() { () }

  init { zero!<bar_t>()  }

  next(tok: token, state: bar_t) {
    state
  }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, SendWithBadTokenType) {
  constexpr std::string_view kProgram = R"(
proc entry {
    p: chan<u32> out;
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>;
        (p, c)
    }
    next (tok: token, state: ()) {
        let tok = send(u32:42, p, u32:0);
        ()
    }
}
)";
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Want argument 0 to 'send' to be a token; got uN[32]")));
}

TEST(TypecheckTest, SimpleProducer) {
  constexpr std::string_view kProgram = R"(
proc producer {
    s: chan<u32> out;

    init { true }

    config(s: chan<u32> out) {
        (s,)
    }

    next(tok: token, do_send: bool) {
        let tok = send_if(tok, s, do_send, ((do_send) as u32));
        !do_send
    }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithParametricExpr) {
  constexpr std::string_view kProgram = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

pub proc MyProc<X:u32, Y:u32={id(X)}> {
  config() { () }
  init { () }
  next(tok: token, state: ()) { state }
}

#[test_proc]
proc MyTestProc {
  init { () }

  config (terminator: chan<bool> out) {
    spawn MyProc<u32:7>();
  }

  next(tok: token, st:()) { }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckTest, ProcWithArrayOfChannels) {
  constexpr std::string_view kProgram = R"(
proc p {
  result: chan<u32>[2] out;

  config(result: chan<u32>[2] out) { (result,) }
  init { () }
  next(tok: token, state: ()) { send(tok, result[0], u32:0); }
}
)";
  XLS_EXPECT_OK(Typecheck(kProgram));
}

TEST(TypecheckErrorTest, ProcWithArrayOfChannelsIndexOob) {
  constexpr std::string_view kProgram = R"(
proc p {
  result: chan<u32>[2] out;

  config(result: chan<u32>[2] out) { (result,) }
  init { () }
  next(tok: token, state: ()) { send(tok, result[2], u32:0); }
}
)";
  EXPECT_THAT(Typecheck(kProgram).status(),
              IsPosError("TypeInferenceError",
                         testing::HasSubstr(
                             "Index has a compile-time constant value 2 that "
                             "is out of bounds of the array type.")));
}

TEST(TypecheckErrorTest, ProcWithArrayOfChannelsSendWrongType) {
  constexpr std::string_view kProgram = R"(
proc p {
  result: chan<u32>[2] out;

  config(result: chan<u32>[2] out) { (result,) }
  init { () }
  next(tok: token, state: ()) { send(tok, result[1], u64:0); }
}
)";
  EXPECT_THAT(Typecheck(kProgram).status(),
              IsPosError("XlsTypeError",
                         testing::HasSubstr(
                             "Want argument 2 to 'send' to have type uN[32]")));
}

}  // namespace
}  // namespace xls::dslx
