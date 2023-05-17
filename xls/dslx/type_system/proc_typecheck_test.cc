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
#include "xls/common/status/matchers.h"
#include "xls/dslx/type_system/typecheck_test_helpers.h"

namespace xls::dslx {
namespace {

TEST(TypecheckTest, ConfigSpawnTerminatingSemicolonNoMembers) {
  constexpr std::string_view kProgram = R"(
proc foo {
    init { }
    config() {
    }
    next(tok: token, state: ()) {
    }
}

proc entry {
    init { () }
    config() {
        spawn foo();
    }
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
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Default value type does not match channel type")));
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
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("send requires a channel argument")));
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
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("send requires a channel argument")));
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
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Cannot recv on an output channel.")));
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
  EXPECT_THAT(Typecheck(kProgram),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Cannot send on an input channel.")));
}

}  // namespace
}  // namespace xls::dslx
