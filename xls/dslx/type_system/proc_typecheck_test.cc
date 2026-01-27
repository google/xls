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
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/error_test_utils.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"

namespace xls::dslx {
namespace {

// Previously this would cause an internal error due to token not having a
// signedness.
TEST(TypecheckTest, TokenArrayTypeAnnotation) {
  constexpr std::string_view kProgram = R"(proc t{e:token[0]})";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

// Scenario where entry proc has a config with a "spawn" statement and a
// trailing semicolon, with no members to configure for itself.
TEST(TypecheckTest, ConfigSpawnTerminatingSemicolonNoMembers) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { }
    config() { }
    next(state: ()) { }
}

proc entry {
    init { () }
    config() { spawn foo(); }
    next (state: ()) { () }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

// As above, but with an explicit empty tuple to configure the zero members.
TEST(TypecheckTest, ConfigSpawnExplicitNilTupleNoMembers) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { }
    config() { }
    next(state: ()) { }
}

proc entry {
    init { () }
    config() { spawn foo(); () }
    next (state: ()) { () }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckErrorTest, ConfigTooManyElementsGiven) {
  constexpr std::string_view kProgram = R"(
proc entry {
    init { () }
    config() {
      let (_p, c) = chan<u32>("my_chan");
      (c,)
    }
    next (state: ()) { () }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Out-of-bounds tuple index specified: 0")));
}

TEST(TypecheckErrorTest, ConfigTooFewElementsGiven) {
  constexpr std::string_view kProgram = R"(
proc entry {
    c : chan<u32> in;
    init { () }
    config() {
      ()
    }
    next (state: ()) { () }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Cannot match a 1-element tuple to 0 values.")));
}

TEST(TypecheckErrorTest, ConfigNonTupleGiven) {
  constexpr std::string_view kProgram = R"(
proc entry {
    c : chan<u32> in;
    init { () }
    config() {
      u32:42
    }
    next (state: ()) { () }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  HasTypeMismatch("(chan<u32> in,)", "u32")));
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
    next(state: u32) {
        let (tok, x) = recv_if(join(), c, true, u42:1234);
        (state + x,)
    }
}

proc entry {
    c: chan<u32> out;
    init { () }
    config() {
        let (p, c) = chan<u32>("my_chan");
        spawn foo(p);
        (c,)
    }
    next (state: ()) { () }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSizeMismatch("u32", "u42")));
}

TEST(TypecheckTest, InitDoesntMatchStateParam) {
  constexpr std::string_view kProgram = R"(
proc oopsie {
    init { u32:0xbeef }
    config() { () }
    next(state: u33) {
      state
    }
})";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSizeMismatch("u32", "u33")));
}

TEST(TypecheckTest, NextReturnDoesntMatchState) {
  constexpr std::string_view kProgram = R"(
proc oopsie {
    init { u32:0xbeef }
    config() { () }
    next(state: u32) {
      state as u33
    }
})";

  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSizeMismatch("u33", "u32")));
}

TEST(TypecheckTest, CantSendOnNonMember) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { () }

    config() {
        ()
    }

    next(state: ()) {
        let foo = u32:0;
        let tok = send(join(), foo, u32:0x0);
        ()
    }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasTypeMismatch("u32", "chan<Any> out")));
}

TEST(TypecheckTest, CantSendOnNonChannel) {
  constexpr std::string_view kProgram = R"(
proc foo {
    bar: u32;
    init { () }
    config() {
        (u32:0,)
    }
    next(state: ()) {
        let tok = send(join(), bar, u32:0x0);
        ()
    }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasTypeMismatch("u32", "chan<Any> out")));
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
    next(state: u32) {
        let (tok, x) = recv(join(), c);
        (state + x,)
    }
}

proc entry {
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>("my_chan");
        spawn foo(c);
        (p,)
    }
    next (state: ()) { () }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasTypeMismatch("chan<u32> out", "chan<u32> in")));
}

TEST(TypecheckTest, CantSendOnOutputChannel) {
  constexpr std::string_view kProgram = R"(
proc entry {
    p: chan<u32> out;
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>("my_chan");
        (p, c)
    }
    next (state: ()) {
        let tok = send(join(), c, u32:0);
        ()
    }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasTypeMismatch("chan<u32> in", "chan<u32> out")));
}

TEST(TypecheckTest, CanUseZeroMacroInInitIssue943) {
  constexpr std::string_view kProgram = R"(
struct bar_t {
  f: u32
}

proc foo {
  config() { () }

  init { zero!<bar_t>()  }

  next(state: bar_t) {
    state
  }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, SendWithBadTokenType) {
  constexpr std::string_view kProgram = R"(
proc entry {
    p: chan<u32> out;
    c: chan<u32> in;
    init { () }
    config() {
        let (p, c) = chan<u32>("my_chan");
        (p, c)
    }
    next (state: ()) {
        let tok = send(u32:42, p, u32:0);
        ()
    }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasTypeMismatch("token", "u32")));
}

TEST(TypecheckTest, SimpleProducer) {
  constexpr std::string_view kProgram = R"(
proc producer {
    s: chan<u32> out;

    init { true }

    config(s: chan<u32> out) {
        (s,)
    }

    next(do_send: bool) {
        let tok = send_if(join(), s, do_send, ((do_send) as u32));
        !do_send
    }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckErrorTest, ProcWithConstant) {
  XLS_EXPECT_OK(TypecheckV2(R"(
proc const_proc {
  const MY_CONST = u32:2;
  input_s: chan<u32> out;
  terminator_s: chan<bool> out;

  init { u32:0 }

  config(terminator_s: chan<bool> out) {
    let (s, r) = chan<u32>("c");
    (s, terminator_s)
  }

  // Run for two iterations then exit.
  next(iter: u32) {
    let tok = send(join(), input_s, u32:0);
    let tok = send_if(tok, terminator_s, iter == MY_CONST, true);
    iter + u32:1
  }
}
)"));
}

TEST(TypecheckErrorTest, ProcWithConstantRefFunction) {
  XLS_EXPECT_OK(TypecheckV2(R"(
fn double(x: u32) -> u32 {
  x * u32:2
}

proc const_proc {
  const MY_CONST = double(u32:4);
  input_s: chan<u32> out;
  terminator_s: chan<bool> out;

  init { u32:0 }

  config(terminator_s: chan<bool> out) {
    let (s, r) = chan<u32>("c");
    (s, terminator_s)
  }

  // Run for two iterations then exit.
  next(iter: u32) {
    let tok = send(join(), input_s, u32:0);
    let tok = send_if(tok, terminator_s, iter == MY_CONST, true);
    iter + u32:1
  }
}
)"));
}

TEST(TypecheckTest, ProcWithConstantInType) {
  constexpr std::string_view kProgram = R"(
proc my_proc<VAL: u32 = {37}> {
    c: chan<uN[VAL]> in;
    s: chan<uN[VAL]> out;

    config(c: chan<uN[VAL]> in, s: chan<uN[VAL]> out) { (c, s) }

    init { () }

    next(state: ()) {
        let (tok, input) = recv(join(), c);
        let output = ((input as uN[VAL]) * uN[VAL]:2) as uN[VAL];
        let tok = send(tok, s, output);
    }
}

#[test_proc]
proc test_proc {
    terminator: chan<bool> out;
    output_c: chan<u37> in;
    input_p: chan<u37> out;

    config(terminator: chan<bool> out) {
        let (input_p, input_c) = chan<u37>("input");
        let (output_p, output_c) = chan<u37>("output");
        spawn my_proc(input_c, output_p);
        (terminator, output_c, input_p)
    }

    init { () }

    next(state: ()) {
        let tok = send(join(), input_p, u37:1);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:2);

        let tok = send(tok, input_p, u37:8);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:16);

        let tok = send(tok, terminator, true);
    }
}

)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ParametricProcWithConstantRefParameter) {
  constexpr std::string_view kProgram = R"(
proc parametric<N: u32, VAL: u32 = {2 * N}> {
    c: chan<uN[VAL]> in;
    s: chan<uN[VAL]> out;

    config(c: chan<uN[VAL]> in, s: chan<uN[VAL]> out) { (c, s) }

    init { () }

    next(state: ()) {
        let (tok, input) = recv(join(), c);
        let output = ((input as uN[VAL]) * uN[VAL]:2) as uN[VAL];
        let tok = send(tok, s, output);
    }
}

#[test_proc]
proc test_proc {
    terminator: chan<bool> out;
    output_c: chan<u64> in;
    input_p: chan<u64> out;

    config(terminator: chan<bool> out) {
        let (input_p, input_c) = chan<u64>("input");
        let (output_p, output_c) = chan<u64>("output");
        spawn parametric<u32:32>(input_c, output_p);
        (terminator, output_c, input_p)
    }

    init { () }

    next(state: ()) {
        let tok = send(join(), input_p, 1);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, 2);

        let tok = send(tok, input_p, 8);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, 16);

        let tok = send(tok, terminator, true);
    }
}

)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ParametricProcWithConstant) {
  constexpr std::string_view kProgram = R"(
proc parametric<N: u32, VAL: u32 = {37}> {
    c: chan<uN[VAL]> in;
    s: chan<uN[VAL]> out;

    config(c: chan<uN[VAL]> in, s: chan<uN[VAL]> out) { (c, s) }

    init { () }

    next(state: ()) {
        let (tok, input) = recv(join(), c);
        let output = ((input as uN[VAL]) * uN[VAL]:2) as uN[VAL];
        let tok = send(tok, s, output);
    }
}

#[test_proc]
proc test_proc {
    terminator: chan<bool> out;
    output_c: chan<u37> in;
    input_p: chan<u37> out;

    config(terminator: chan<bool> out) {
        let (input_p, input_c) = chan<u37>("input");
        let (output_p, output_c) = chan<u37>("output");
        spawn parametric<u32:32>(input_c, output_p);
        (terminator, output_c, input_p)
    }

    init { () }

    next(state: ()) {
        let tok = send(join(), input_p, u37:1);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:2);

        let tok = send(tok, input_p, u37:8);
        let (tok, result) = recv(tok, output_c);
        assert_eq(result, u37:16);

        let tok = send(tok, terminator, true);
    }
}

)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ProcWithParametricExpr) {
  constexpr std::string_view kProgram = R"(
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

pub proc MyProc<X:u32, Y:u32={id(X)}> {
  config() { () }
  init { () }
  next(state: ()) { state }
}

#[test_proc]
proc MyTestProc {
  init { () }

  config (terminator: chan<bool> out) {
    spawn MyProc<u32:7>();
  }

  next(st:()) { }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ProcWithArrayOfChannels) {
  constexpr std::string_view kProgram = R"(
proc p {
  result: chan<u32>[2] out;

  config(result: chan<u32>[2] out) { (result,) }
  init { () }
  next(state: ()) { send(join(), result[0], u32:0); }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckErrorTest, ProcWithArrayOfChannelsIndexOob) {
  constexpr std::string_view kProgram = R"(
proc p {
  result: chan<u32>[2] out;

  config(result: chan<u32>[2] out) { (result,) }
  init { () }
  next(state: ()) { send(join(), result[2], u32:0); }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram).status(),
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
  next(state: ()) { send(join(), result[1], u64:0); }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram).status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSizeMismatch("u32", "u64")));
}

// See also https://github.com/google/xls/issues/1359
TEST(TypecheckTest, ZeroMacroOnProcLevelTypeAlias) {
  constexpr std::string_view kProgram = R"(
proc p {
  type MyU32 = u32;
  config() { let _ = zero!<MyU32>(); () }
  init { zero!<MyU32>() }
  next(state: MyU32) { zero!<MyU32>() }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ProcLevelUselessExpression) {
  constexpr std::string_view kProgram = R"(
const N = u32:42;
proc MyProc {
  N;
  config() { () }
  init { () }
  next(state: ()) { () }
}
)";
  EXPECT_THAT(TypecheckV2(kProgram),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Expected either a proc member, type alias, or "
                      "`const_assert!` at proc scope; got identifier: `N`")));
}

TEST(TypecheckTest, ProcLevelConstAssertUnparameterized) {
  constexpr std::string_view kProgram = R"(
const N = u32:42;
proc MyProc {
  const_assert!(N == u32:42);
  config() { () }
  init { () }
  next(state: ()) { () }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ProcLevelConstAssertThatFails) {
  constexpr std::string_view kProgram = R"(
proc MyProc<X: u32, Y: u32> {
  const_assert!(X == Y);
  config() { () }
  init { () }
  next(state: ()) { () }
}

proc Instantiator {
  config() {
      spawn MyProc<u32:42, u32:64>();
  }
  init { () }
  next(state: ()) { () }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("const_assert! failure: `X == Y` constexpr "
                             "environment: {X: u32:42, Y: u32:64}")));
}

TEST(TypecheckTest, ProcLevelConstAssertThatPasses) {
  constexpr std::string_view kProgram = R"(
proc MyProc<X: u32, Y: u32> {
  const_assert!(X == Y);
  config() { () }
  init { () }
  next(state: ()) { () }
}

proc Instantiator {
  config() {
      spawn MyProc<u32:42, u32:42>();
  }
  init { () }
  next(state: ()) { () }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckTest, ProcLevelConstAssertThatPassesThenFails) {
  constexpr std::string_view kProgram = R"(
proc MyProc<X: u32, Y: u32> {
  const_assert!(X == Y);
  config() { () }
  init { () }
  next(state: ()) { () }
}

proc Instantiator {
  config() {
      spawn MyProc<u32:42, u32:42>();
      spawn MyProc<u32:42, u32:64>();
  }
  init { () }
  next(state: ()) { () }
}
)";
  EXPECT_THAT(
      TypecheckV2(kProgram),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("const_assert! failure: `X == Y` constexpr "
                             "environment: {X: u32:42, Y: u32:64}")));
}

}  // namespace
}  // namespace xls::dslx
