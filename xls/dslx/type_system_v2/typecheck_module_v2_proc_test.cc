// Copyright 2026 The XLS Authors
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
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"

// Tests for `proc` definitions, spawning, channels, and state access.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2Test, SpawnBasicProc) {
  EXPECT_THAT(R"(
proc Counter {
  c: chan<u32> out;
  max: u32;
  init { 0 }
  config(c: chan<u32> out, max: u32) {
    (c, max)
  }
  next(i: u32) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}

proc main {
  c: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn Counter(p, 50);
    (c,)
  }
  next(state: (token, u32)) {
    recv(state.0, c)
  }
}
)",
              TypecheckSucceeds(HasNodeWithType("spawn Counter(p, 50)", "()")));
}

TEST(TypecheckV2Test, SpawnParametricProc) {
  EXPECT_THAT(R"(
proc Counter<N: u32> {
  c: chan<uN[N]> out;
  max: uN[N];
  init { 0 }
  config(c: chan<uN[N]> out, max: uN[N]) {
    (c, max)
  }
  next(i: uN[N]) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}

proc main {
  c16: chan<u16> in;
  c32: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p16, c16) = chan<u16>("my_chan16");
    let (p32, c32) = chan<u32>("my_chan32");
    spawn Counter<16>(p16, 50);
    spawn Counter<32>(p32, 50);
    (c16,c32)
  }
  next(state: (token, u48)) {
    let (tok16, v16) = recv(state.0, c16);
    let (tok32, v32) = recv(tok16, c32);
    (tok32, v32 ++ v16)
  }
}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("spawn Counter<16>(p16, 50)", "()"),
                        HasNodeWithType("spawn Counter<32>(p32, 50)", "()"))));
}

TEST(TypecheckV2Test, ParametricProcWithTypeAlias) {
  EXPECT_THAT(R"(
proc Counter<N: u32> {
  type value_t = uN[N];

  c: chan<value_t> out;
  max: value_t;
  init { 0 }
  config(c: chan<value_t> out, max: value_t) {
    (c, max)
  }
  next(i: value_t) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}

proc main {
  c16: chan<u16> in;
  c32: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p16, c16) = chan<u16>("my_chan16");
    let (p32, c32) = chan<u32>("my_chan32");
    spawn Counter<16>(p16, 50);
    spawn Counter<32>(p32, 50);
    (c16,c32)
  }
  next(state: (token, u48)) {
    let (tok16, v16) = recv(state.0, c16);
    let (tok32, v32) = recv(tok16, c32);
    (tok32, v32 ++ v16)
  }
}
)",
              TypecheckSucceeds(
                  AllOf(HasNodeWithType("spawn Counter<16>(p16, 50)", "()"),
                        HasNodeWithType("spawn Counter<32>(p32, 50)", "()"))));
}

TEST(TypecheckV2Test, ParametricProcValueCloning) {
  // This is the version of matmul in https://github.com/google/xls/issues/2706
  // but reduced to a minimal repro. Note that it is too minimized to be valid
  // for IR conversion. The original error was due to not cloning parametric
  // values when replacing references to them in `NameRefMapper`.
  EXPECT_THAT(
      R"(
type F32 = u32;

struct Command<ROWS: u32, COLS: u32> {
  weights: F32[COLS][ROWS]
}

proc A<ROWS: u32, COLS: u32> {
  commands: chan<Command<ROWS, COLS>> in;
  from_norths: chan<F32>[COLS][ROWS + u32:1] in;
  to_souths: chan<F32>[COLS][ROWS + u32:1] out;

  config(commands: chan<Command<ROWS, COLS>> in) {
    let (to_souths, from_norths) = chan<F32>[COLS][ROWS + u32:1]("north_south");
    (commands, from_norths, to_souths)
  }

  init { () }
  next(state: ()) { () }
}

proc main {
  commands: chan<Command<u32:4, u32:4>> out;

  config() {
    let (commands_out, commands_in) = chan<Command<u32:4, u32:4>>("commands");
    spawn A<4, 4>(commands_in);
    (commands_out,)
  }

  init { () }
  next(state: ()) { () }
}
)",
      TypecheckSucceeds(HasNodeWithType("spawn A<4, 4>(commands_in)", "()")));
}

TEST(TypecheckV2Test, BadChannelDeclAssignmentFails) {
  EXPECT_THAT(
      R"(
proc main {
  init { () }
  config() {
    let p: u32 = chan<u32>("my_chan");
    ()
  }
  next(state: ()) { () }
}
)",
      TypecheckFails(HasTypeMismatch("u32", "(chan<u32> out, chan<u32> in)")));
}

TEST(TypecheckV2Test, ProcConfigTooFewChannels) {
  EXPECT_THAT(
      R"(
proc Proc {
  input: chan<()> in;
  output: chan<()> out;
  config(input: chan<()> in) {
    (input,)
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasSubstr("Cannot match a 2-element tuple to 1 values.")));
}

TEST(TypecheckV2Test, ProcConfigTooManyChannels) {
  EXPECT_THAT(
      R"(
proc Proc {
  req: chan<()> in;
  resp: chan<()> out;
  config(data_in: chan<()> in) {
    let (resp, req) = chan<()>("io");
    (req, resp, data_in)
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasSubstr("Out-of-bounds tuple index specified: 2")));
}

TEST(TypecheckV2Test, ProcWithBranchedFinalExpression) {
  EXPECT_THAT(
      R"(
const A = u32:5;
proc Proc {
  input: chan<()> in;
  output: chan<()> out;
  config() {
    const if A == u32:1 {
      let (first_output, first_input) = chan<()>("first");
      (first_input, first_output)
    } else {
      let (second_output, second_input) = chan<()>("second");
      (second_input, second_output)
    }
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckSucceeds(
          AllOf(HasNodeWithType("second_input", "chan((), dir=in)"),
                HasNodeWithType("second_output", "chan((), dir=out)"))));
}

TEST(TypecheckV2Test, ProcConfigFailedBranchedFinalExpression) {
  EXPECT_THAT(
      R"(
const A = u32:5;
proc Proc {
  input: chan<()> in;
  output: chan<()> out;
  config() {
    const if A == u32:5 {
      let (first_output, first_input) = chan<()>("first");
      (first_input,)
    } else {
      let (second_output, second_input) = chan<()>("second");
      (second_input, second_output)
    }
  }
  init { () }
  next(state: ()) { () }
}

)",
      TypecheckFails(HasSubstr("Cannot match a 2-element tuple to 1 values.")));
}

TEST(TypecheckV2Test, SpawnImportedProc) {
  constexpr std::string_view kImported = R"(
pub proc Counter {
  c: chan<u32> out;
  max: u32;
  init { 0 }
  config(c: chan<u32> out, max: u32) {
    (c, max)
  }
  next(i: u32) {
    send(join(), c, i);
    if i == max { i } else { i + 1 }
  }
}
)";

  constexpr std::string_view kProgram = R"(
import imported;

proc main {
  c: chan<u32> in;
  init { (join(), 0) }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn imported::Counter(p, 50);
    (c,)
  }
  next(state: (token, u32)) {
    recv(state.0, c)
  }
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("spawn imported::Counter(p, 50)", "()"))));
}

TEST(TypecheckV2Test, ExplicitStateAccessFailOnInteractingWithState) {
  EXPECT_THAT(R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let sum = state + u32:1;
  }
}
)",
              TypecheckFails(HasTypeMismatch("State<u32>", "u32")));
}

TEST(TypecheckV2Test, ExplicitStateAccessAddingStateToState) {
  EXPECT_THAT(R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let sum = state + state;
  }
}
)",
              TypecheckFails(HasSubstr("State {} Binary operations can only be "
                                       "applied to bits-typed operands.")));
}

TEST(TypecheckV2Test, ProcWithImpl) {
  constexpr std::string_view kProcType =
      "P { input: chan(uN[32], dir=in), output: "
      "chan(uN[32], dir=out), state: State {} }";

  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc P {
  input: chan<u32> in,
  output: chan<u32> out,
  state: u32,
}

impl P {
    fn new(input: chan<u32> in, output: chan<u32> out) -> Self {
      P { input, output, state: 5 }
    }

    fn next(self) {
      let s = read(self.state);
      let (tok, val) = recv(join(), self.input);
      let new_val = val + s;
      let tok = send(tok, self.output, new_val);
      write(self.state, new_val);
    }
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("P", absl::Substitute("typeof($0)", kProcType)),
          HasNodeWithType(
              "new", absl::Substitute(
                         "(chan(uN[32], dir=in), chan(uN[32], dir=out)) -> $0",
                         kProcType)),
          HasNodeWithType("next", absl::Substitute("($0) -> ()", kProcType)))));
}

TEST(TypecheckV2Test, ProcWithImplNextWithExtraParamFails) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc P {
  state: u32,
}

impl P {
    fn new() -> Self {
      P { state: 5 }
    }

    fn next(self, a: u32) {}
}
)",
      TypecheckFails(HasSubstr("The next() function of a `proc` with an `impl` "
                               "must have a single parameter")));
}

TEST(TypecheckV2Test, ProcWithImplNextWithNoParamsFails) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc P {
  state: u32,
}

impl P {
    fn new() -> Self {
      P { state: 5 }
    }

    fn next() {}
}
)",
      TypecheckFails(HasSubstr("The next() function of a `proc` with an `impl` "
                               "must have a single parameter")));
}

TEST(TypecheckV2Test, ProcWithImplNextWithReturnTypeFails) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc P {
  state: u32,
}

impl P {
    fn new() -> Self {
      P { state: 5 }
    }

    fn next() -> u32 { 0 }
}
)",
      TypecheckFails(HasSubstr("The next() function of a `proc` with an `impl` "
                               "must not return anything")));
}

TEST(TypecheckV2Test, SpawnProcWithImpl) {
  std::string_view kProgram = R"(
#![feature(explicit_state_access)]

// The derive attribute here is unnecessary, but having it proves it does no
// harm.
#[derive(Spawn)]
proc P {
    c_out: chan<u32> out,
    i: u32,
}

impl P {
    fn new(c_out: chan<u32> out) -> Self {
        P { c_out: c_out, i: 0 }
    }

    fn next(self) {
        let last_i = read(self.i);
        send(join(), self.c_out, last_i);
        write(self.i, last_i + 1);
    }
}

proc C {
    c_in: chan<u32> in,
    i: u32,
}

impl C {
    fn new(c_in: chan<u32> in) -> Self {
        C { c_in: c_in, i: 0 }
    }
    fn next(self) {
        let last_i = read(self.i);
        let (tok1, e) = recv(join(), self.c_in);
        write(self.i, e + last_i);
    }
}

proc Main {}

impl Main {
    fn new() -> Self {
        let (c_out, c_in) = chan<u32>("my_chan");
        P::new(c_out).spawn();
        let c = C::new(c_in);
        c.spawn();
        Main {}
    }
}
)";

  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckV2Test, ExplicitStateAccessSimpleU32) {
  EXPECT_THAT(
      R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let x = read(state);
    let y = x + u32:1;
    write(state, y);
  }
}
)",
      TypecheckSucceeds(AllOf(HasNodeWithType("read(state)", "uN[32]"),
                              HasNodeWithType("y", "uN[32]"),
                              HasNodeWithType("write(state, y)", "()"))));
}

TEST(TypecheckV2Test, ExplicitStateAccessLabeledSimpleU32) {
  EXPECT_THAT(
      R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let x = labeled_read(state, "counter_read");
    let y = x + u32:1;
    labeled_write(state, y, "counter_write");
  }
}
)",
      TypecheckSucceeds(AllOf(
          HasNodeWithType("labeled_read(state, \"counter_read\")", "uN[32]"),
          HasNodeWithType("y", "uN[32]"),
          HasNodeWithType("labeled_write(state, y, \"counter_write\")",
                          "()"))));
}

TEST(TypecheckV2Test, ProcWithImplIntegerParamInTopProcNewFails) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc P {
  state: u32,
}

impl P {
    fn new(init_val: u32) -> Self {
      P { state: init_val }
    }

    fn next() -> u32 { 0 }
}
)",
      TypecheckFails(
          HasSubstr("Initializer for member `state` of proc `P` must be "
                    "possible to evaluate at compile time.")));
}

TEST(TypecheckV2Test, ParametricProcDef) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

proc Loopback<N: u32> {
  c_in: chan<uN[N]> in,
  c_out: chan<uN[N]> out,
}

impl Loopback<N> {
  fn new(c_in: chan<uN[N]> in, c_out: chan<uN[N]> out) -> Self {
    Loopback { c_in, c_out }
  }

  fn next(self) {
    let (t, val) = recv(join(), self.c_in);
    send(t, self.c_out, val);
  }
}

proc Main {
  c_in: chan<u32> in,
  c_out: chan<u32> out,
  c_in_from_loopback: chan<u32> in,
  c_out_to_loopback: chan<u32> out,
  i: u32,
}

impl Main {
  fn new(c_in: chan<u32> in, c_out: chan<u32> out) -> Self {
    let (out_to_loopback, loopback_in) = chan<u32>("main_to_loopback");
    let (loopback_out, in_from_loopback) = chan<u32>("loopback_to_main");
    Loopback<32>::new(loopback_in, loopback_out).spawn();

    Main {
      c_in: c_in,
      c_out: c_out,
      c_in_from_loopback: in_from_loopback,
      c_out_to_loopback: out_to_loopback,
      i: 1
    }
  }

  fn next(self) {
    let i_val = read(self.i);
    let (_, j) = recv(join(), self.c_in);
    let loopback_tok = send(join(), self.c_out_to_loopback, j);
    let (_, loopback_val) = recv(loopback_tok, self.c_in_from_loopback);
    send(join(), self.c_out, i_val + loopback_val);
    write(self.i, i_val + loopback_val);
  }
}
)",
      TypecheckSucceeds(
          HasNodeWithType("Loopback<32>::new(loopback_in, loopback_out)",
                          "Loopback { c_in: chan(uN[32], dir=in), c_out: "
                          "chan(uN[32], dir=out) }")));
}

}  // namespace
}  // namespace xls::dslx
