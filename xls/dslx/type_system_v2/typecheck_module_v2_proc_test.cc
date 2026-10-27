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

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/channel_manager.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_system_test_utils.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"

// Tests for `proc` definitions, spawning, channels, and state access.
namespace xls::dslx {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;
using ::testing::HasSubstr;

TEST(TypecheckV2ProcTest, SpawnBasicProc) {
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

TEST(TypecheckV2ProcTest, SpawnParametricProc) {
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

TEST(TypecheckV2ProcTest, ParametricProcWithTypeAlias) {
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

TEST(TypecheckV2ProcTest, ParametricProcValueCloning) {
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

TEST(TypecheckV2ProcTest, BadChannelDeclAssignmentFails) {
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

TEST(TypecheckV2ProcTest, ProcConfigTooFewChannels) {
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

TEST(TypecheckV2ProcTest, ProcConfigTooManyChannels) {
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

TEST(TypecheckV2ProcTest, ProcWithBranchedFinalExpression) {
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

TEST(TypecheckV2ProcTest, ProcConfigFailedBranchedFinalExpression) {
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

TEST(TypecheckV2ProcTest, SpawnImportedProc) {
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

TEST(TypecheckV2ProcTest, ExplicitStateAccessFailOnInteractingWithState) {
  EXPECT_THAT(R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let sum = state + u32:1;
  }
}
)",
              TypecheckFails(HasTypeMismatch("BuiltinProcState<u32>", "u32")));
}

TEST(TypecheckV2ProcTest, ExplicitStateAccessAddingStateToState) {
  EXPECT_THAT(R"(#![feature(explicit_state_access)]
proc Counter {
  init { u32:0 }
  config() { }
  next(state: u32) {
    let sum = state + state;
  }
}
)",
              TypecheckFails(
                  HasSubstr("BuiltinProcState {} Binary operations can only be "
                            "applied to bits-typed operands.")));
}

TEST(TypecheckV2ProcTest, ProcWithImpl) {
  constexpr std::string_view kProcType =
      "P { input: chan(uN[32], dir=in), output: "
      "chan(uN[32], dir=out), state: BuiltinProcState {} }";

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

TEST(TypecheckV2ProcTest, ProcWithImplNextWithExtraParamFails) {
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

TEST(TypecheckV2ProcTest, ProcWithImplNextWithNoParamsFails) {
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

TEST(TypecheckV2ProcTest, ProcWithImplNextWithReturnTypeFails) {
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

TEST(TypecheckV2ProcTest, SpawnProcWithImpl) {
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

TEST(TypecheckV2ProcTest, ExplicitStateAccessSimpleU32) {
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

TEST(TypecheckV2ProcTest, ProcWithImplIntegerParamInTopProcNewFails) {
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

TEST(TypecheckV2ProcTest, ParametricProcDef) {
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

TEST(TypecheckV2ProcTest, ProcDefWithoutNextFunction) {
  EXPECT_THAT(
      R"(
proc Loopback {
  c_in: chan<u32> in,
  c_out: chan<u32> out,
}

impl Loopback {
  fn new(c_in: chan<u32> in, c_out: chan<u32> out) -> Self {
    Loopback { c_in, c_out }
  }
}
)",
      TypecheckSucceeds(
          HasNodeWithType("fn next(self) {}",
                          "(Loopback { c_in: chan(uN[32], dir=in), c_out: "
                          "chan(uN[32], dir=out) }) -> ()")));
}

TEST(TypecheckV2ProcTest, SpawnImportedProcDef) {
  std::string_view kImported = R"(
#![feature(explicit_state_access)]

pub proc P {
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
)";

  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

import imported;
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
        imported::P::new(c_out).spawn();
        let c = C::new(c_in);
        c.spawn();
        Main {}
    }
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2ProcTest, LegacyProcWithSelfReferenceFails) {
  EXPECT_THAT(R"(
proc Foo {
  c: chan<u32> in;
  a: uN[Self::LIMIT];

  init {
    ()
  }

  config(c: chan<u32> in) {
    (c,)
  }

  next(state: ()) {
    ()
  }
}
  )",
              TypecheckFails(HasSubstr(
                  "Use of `Self` inside legacy procs is not supported.")));
}

TEST(TypecheckV2ProcTest, SpawnProcWithImplInTestFunctionSucceeds) {
  EXPECT_THAT(R"(
proc PassThrough {
  c_in: chan<u32> in,
  c_out: chan<u32> out,
}

impl PassThrough {
  fn new(c_in: chan<u32> in, c_out: chan<u32> out) -> Self {
    PassThrough { c_in, c_out }
  }
  fn next(self) {
    let (tok, val) = recv(join(), self.c_in);
    send(tok, self.c_out, val);
  }
}

#[test]
fn test_pass_through() {
  let (in_w, in_r) = chan<u32>("in");
  let (out_w, out_r) = chan<u32>("out");
  PassThrough::new(in_r, out_w).spawn();
  let tok = send(join(), in_w, u32:42);
  let (tok, val) = recv(tok, out_r);
  assert_eq(val, u32:42)
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, ChannelOpsInTestFunctionSucceeds) {
  EXPECT_THAT(R"(
#[test]
fn my_test_fn() {
  let (p, c) = chan<u32>("test_chan");
  let tok = send(join(), p, u32:42);
  let (tok, val) = recv(tok, c);
  assert_eq(val, u32:42)
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, ChannelDeclInNormalFunctionFails) {
  EXPECT_THAT(R"(
fn bad_function() {
  let (p, c) = chan<u32>("test_chan");
  ()
}
)",
              TypecheckFails(HasSubstr(
                  "Channels can only be declared in a proc or test")));
}

TEST(TypecheckV2ProcTest, ChannelSendInNormalFunctionFails) {
  EXPECT_THAT(R"(
fn bad_function(p: chan<u32> out) {
  let tok = send(join(), p, u32:42);
  ()
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `send` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, NewSpawnInNormalFunctionFails) {
  EXPECT_THAT(
      R"(
proc Dummy {
  c: chan<u32> out,
}
impl Dummy {
  fn new(c: chan<u32> out) -> Self {
    Dummy { c }
  }
  fn next(self) {}
}

fn bad_function(p: chan<u32> out) {
  Dummy::new(p).spawn();
  ()
}
)",
      TypecheckFails(HasSubstr("Cannot spawn outside a proc or test.")));
}

TEST(TypecheckV2ProcTest, ChannelOpsInTestUtilityFails) {
  EXPECT_THAT(R"(
#[cfg(test)]
fn helper(p: chan<u32> out, c: chan<u32> in) -> u32 {
  let tok = send(join(), p, u32:42);
  let (tok, val) = recv(tok, c);
  val
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `send` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, SpawnInTestUtilityFails) {
  EXPECT_THAT(
      R"(
proc Dummy {
  c: chan<u32> out,
}
impl Dummy {
  fn new(c: chan<u32> out) -> Self {
    Dummy { c }
  }
  fn next(self) {}
}

#[cfg(test)]
fn helper(p: chan<u32> out) {
  Dummy::new(p).spawn();
  ()
}
)",
      TypecheckFails(HasSubstr("Cannot spawn outside a proc or test.")));
}

TEST(TypecheckV2ProcTest, ChannelRecvInNormalFunctionFails) {
  EXPECT_THAT(R"(
fn bad_function(c: chan<u32> in) {
  let (tok, val) = recv(join(), c);
  ()
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `recv` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, ChannelJoinInNormalFunctionFails) {
  EXPECT_THAT(R"(
fn bad_function() {
  let tok = join();
  ()
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `join` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, ChannelRecvInTestUtilityFails) {
  EXPECT_THAT(R"(
#[cfg(test)]
fn helper(c: chan<u32> in) -> u32 {
  let (tok, val) = recv(join(), c);
  val
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `recv` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, ChannelJoinInTestUtilityFails) {
  EXPECT_THAT(R"(
#[cfg(test)]
fn helper() -> token {
  join()
}
)",
              TypecheckFails(HasSubstr(
                  "Cannot call `join` outside a `proc` or proc test")));
}

TEST(TypecheckV2ProcTest, SpawnImportedProcDefInTestFunctionSucceeds) {
  std::string_view kImported = R"(
pub proc P {
    c_out: chan<u32> out,
}
impl P {
    fn new(c_out: chan<u32> out) -> Self {
        P { c_out: c_out }
    }
    fn next(self) {}
}
)";

  constexpr std::string_view kProgram = R"(
import imported;

#[test]
fn my_test() {
  let (c_out, c_in) = chan<u32>("my_chan");
  imported::P::new(c_out).spawn();
  ()
}
)";

  ImportData import_data = CreateImportDataForTest();
  XLS_EXPECT_OK(TypecheckV2(kImported, "imported", &import_data).status());
  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2ProcTest, CallTestUtilityFromTestProcConfigSucceeds) {
  EXPECT_THAT(R"(
#[cfg(test)]
fn helper() -> u32 { u32:42 }

#[test_proc]
proc MyTestProc {
  terminator: chan<bool> out;
  config(terminator: chan<bool> out) {
    let val = helper();
    (terminator,)
  }
  init {}
  next(state: ()) {
    send(join(), terminator, true);
  }
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, ProcDefWithImportedStructMember) {
  constexpr std::string_view kImported = R"(
pub struct Foo {
  bar: u32,
}
)";

  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

import imported;

proc Main {
  c: chan<u32> in,
  foo: imported::Foo,
}

impl Main {
  fn new(c: chan<u32> in) -> Self {
    Main { c: c, foo: imported::Foo { bar: 0 } }
  }
}
)";

  absl::flat_hash_map<std::filesystem::path, std::string> files;
  files[std::filesystem::path("/imported.x")] = std::string(kImported);
  auto vfs = std::make_unique<FakeFilesystem>(std::move(files),
                                              std::filesystem::path("/"));

  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{std::filesystem::path("/")},
                       kAllWarningsSet, std::move(vfs));

  EXPECT_THAT(
      TypecheckV2(kProgram, "main", &import_data),
      IsOkAndHolds(HasTypeInfo(HasNodeWithType(
          "foo: BuiltinProcState<imported::Foo>", "BuiltinProcState {}"))));
}

TEST(TypecheckV2ProcTest, ProcDefWithImportedProcMember) {
  constexpr std::string_view kImported = R"(
pub proc Foo {}
impl Foo {
  fn new() -> Self { Foo {} }
}
)";

  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

import imported;

proc Main {
  c: chan<u32> in,
  foo: imported::Foo,
}

impl Main {
  fn new(c: chan<u32> in) -> Self {
    Main { c: c, foo: imported::Foo::new() }
  }
}
)";

  absl::flat_hash_map<std::filesystem::path, std::string> files;
  files[std::filesystem::path("/imported.x")] = std::string(kImported);
  auto vfs = std::make_unique<FakeFilesystem>(std::move(files),
                                              std::filesystem::path("/"));

  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{std::filesystem::path("/")},
                       kAllWarningsSet, std::move(vfs));

  EXPECT_THAT(TypecheckV2(kProgram, "main", &import_data),
              IsOkAndHolds(HasTypeInfo(
                  HasNodeWithType("foo: imported::Foo", "Foo {}"))));
}

TEST(TypecheckV2ProcTest, UserDefinedStateStructDoesNotClash) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

struct State {
  x: u32,
}

proc Counter {
  state: State,
}

impl Counter {
  fn new() -> Self {
    Counter { state: State { x: u32:0 } }
  }

  fn next(self) {
    let s = read(self.state);
    write(self.state, State { x: s.x + u32:1 });
  }
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckV2ProcTest, ParametricProcDefWithSyntheticNext) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

proc Inner<N: u32> {
  c_in: chan<uN[N]> in,
  c_out: chan<uN[N]> out,
}

impl Inner<N> {
  fn new(c_in: chan<uN[N]> in, c_out: chan<uN[N]> out) -> Self {
    Inner { c_in, c_out }
  }
}

proc Main {
  c_in: chan<u32> in,
  c_out: chan<u32> out,
}

impl Main {
  fn new(c_in: chan<u32> in, c_out: chan<u32> out) -> Self {
    Inner<32>::new(c_in, c_out).spawn();
    Main { c_in, c_out }
  }

  fn next(self) {}
}
)";
  XLS_EXPECT_OK(TypecheckV2(kProgram));
}

TEST(TypecheckV2ProcTest, InvokeTestProcDefFromNonTestProcDefFails) {
  EXPECT_THAT(
      R"(
#![feature(explicit_state_access)]

#[test]
proc TestProc {
  terminator: chan<bool> out,
}

impl TestProc {
  fn new(terminator: chan<bool> out) -> Self {
    TestProc { terminator }
  }
}

proc NormalProc {
  terminator: chan<bool> out,
}

impl NormalProc {
  fn new(terminator: chan<bool> out) -> Self {
    TestProc::new(terminator).spawn();
    NormalProc { terminator }
  }
}
)",
      TypecheckFails(HasSubstr(
          "Test proc `TestProc` can only be instantiated in a test context.")));
}

TEST(TypecheckV2ProcTest, ProcDefWithGenericTypeParamUsedInNew) {
  XLS_EXPECT_OK(TypecheckV2(
      R"(
#![feature(explicit_state_access)]
#![feature(generics)]

proc Worker<W: u32, Data: type = bits[W]> {
  c_in: chan<Data> in,
  state: Data,
}

impl Worker<W, Data> {
  fn new(c_in: chan<Data> in) -> Self {
    Worker { c_in, state: Data:0 }
  }
}

proc Top {}

impl Top {
  fn new(c_in: chan<u32> in) -> Self {
    Worker<32>::new(c_in).spawn();
    Top {}
  }
}
)"));
}

TEST(TypecheckV2ProcTest, SourceSinkChannelSyntaxSucceeds) {
  EXPECT_THAT(
      R"(
#![feature(io_objects)]
proc Loopback<N: u32> {
    c_in: Source<u32>,
    c_out: Sink<u32>,
}
impl Loopback<N> {
    fn new() -> Self { Loopback {} }
    fn next(self) {}
}
)",
      TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, ImportedModuleWithSourceAndSinkSucceeds) {
  constexpr std::string_view kImported = R"(
#![feature(io_objects)]
pub proc Loopback<N: u32> {
  c_in: Source<u32>,
  c_out: Sink<u32>,
}
impl Loopback<N> {
  pub fn new() -> Self { Loopback {} }
  fn next(self) {}
}
)";

  constexpr std::string_view kProgram = R"(
import imported;

fn f() -> u32 { u32:0 }
)";

  absl::flat_hash_map<std::filesystem::path, std::string> files;
  files[std::filesystem::path("/imported.x")] = std::string(kImported);
  auto vfs = std::make_unique<FakeFilesystem>(std::move(files),
                                              std::filesystem::path("/"));

  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{std::filesystem::path("/")},
                       kAllWarningsSet, std::move(vfs));

  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2ProcTest, SourceImplMethodIsCallableOnProcMember) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P {
    input: Source<u32>,
}
impl P {
    fn new() -> Self { P {} }
    fn next(self) { self.input.recv(); }
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, SinkImplMethodIsCallableOnProcMember) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P {
    output: Sink<u32>,
}
impl P {
    fn new() -> Self { P {} }
    fn next(self) { self.output.send(u32:0); }
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, UndefinedImplMethodOnProcMemberIsRejected) {
  EXPECT_THAT(
      R"(
#![feature(io_objects)]
proc P {
    input: Source<u32>,
}
impl P {
    fn new() -> Self { P {} }
    fn next(self) { self.input.no_such_method(); }
}
)",
      TypecheckFails(HasSubstr("Name 'no_such_method' is not defined by the "
                               "impl for struct 'Source'")));
}

TEST(TypecheckV2ProcTest, SinkMethodOnSourceMemberIsRejected) {
  EXPECT_THAT(
      R"(
#![feature(io_objects)]
proc P {
    input: Source<u32>,
}
impl P {
    fn new() -> Self { P {} }
    fn next(self) { self.input.send(u32:0); }
}
)",
      TypecheckFails(
          HasSubstr("Name 'send' is not defined by the impl for struct "
                    "'Source'")));
}

// Returns the only canonical initializer of the proc named `proc_name`.
absl::StatusOr<InterpValue> GetCanonicalInitializer(
    const TypecheckedModule& tm, std::string_view proc_name) {
  XLS_ASSIGN_OR_RETURN(ProcDef * proc,
                       tm.module->GetMemberOrError<ProcDef>(proc_name));
  XLS_ASSIGN_OR_RETURN(std::vector<ProcInitializerWithTypeInfo> initializers,
                       tm.type_info->GetCanonicalProcInitializers(proc));
  XLS_RET_CHECK_EQ(initializers.size(), 1);
  return initializers[0].initializer;
}

// Returns the value of `member_name` in the proc initializer `proc`.
absl::StatusOr<InterpValue> GetMemberValue(const InterpValue& proc,
                                           std::string_view member_name) {
  const InterpValue::ProcInitializer& initializer =
      proc.GetProcInitializerOrDie();
  std::optional<StructMemberNode*> member =
      initializer.proc_def()->GetMemberByName(member_name);
  XLS_RET_CHECK(member.has_value());
  return initializer.GetMemberValue(*member);
}

// Returns the kind of node that defined the channel `value`.
AstNodeKind GetChannelDefinerKind(const InterpValue& value) {
  return (*value.GetChannelReferenceOrDie().GetDefiner())->kind();
}

TEST(TypecheckV2ProcTest, OmittedIoObjectMembersAreDefinedByMember) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(io_objects)]
proc P { input: Source<u32>, output: Sink<u32> }
impl P { fn new() -> Self { P {} } }
)"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue p,
                           GetCanonicalInitializer(result.tm, "P"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue input, GetMemberValue(p, "input"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue output, GetMemberValue(p, "output"));
  EXPECT_EQ(GetChannelDefinerKind(input), AstNodeKind::kStructMember);
  EXPECT_EQ(GetChannelDefinerKind(output), AstNodeKind::kStructMember);
}

TEST(TypecheckV2ProcTest, IoObjectConstructorParamIsRejected) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P { input: Source<u32> }
impl P { fn new(c: Source<u32>) -> Self { P { input: c } } }
)",
              TypecheckFails(HasSubstr(
                  "Proc constructor parameter `c: Source<u32>` contains a "
                  "Source or Sink, which are default-initialized and cannot "
                  "be passed to a proc constructor.")));
}

TEST(TypecheckV2ProcTest, IoObjectArrayConstructorParamIsRejected) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P { outputs: Sink<u32>[2] }
impl P { fn new(c: Sink<u32>[2]) -> Self { P { outputs: c } } }
)",
              TypecheckFails(HasSubstr(
                  "Proc constructor parameter `c: Sink<u32>[2]` contains a "
                  "Source or Sink")));
}

TEST(TypecheckV2ProcTest, IoObjectInTupleConstructorParamIsRejected) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P { input: Source<u32> }
impl P { fn new(t: (Source<u32>, u32)) -> Self { P {} } }
)",
              TypecheckFails(HasSubstr(
                  "Proc constructor parameter `t: (Source<u32>, u32)` contains "
                  "a Source or Sink")));
}

TEST(TypecheckV2ProcTest, LegacyChannelConstructorParamIsAllowed) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
proc P { input: chan<u32> in, output: Sink<u32> }
impl P { fn new(c: chan<u32> in) -> Self { P { input: c } } }
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, IoObjectHelperFunctionParamIsAllowed) {
  EXPECT_THAT(R"(
#![feature(io_objects)]
fn helper(s: Sink<u32>, t: (Source<u32>, u32)) {}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, DefaultIoObjectMembersAreDistinctPerInstance) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(io_objects)]
proc Child { input: Source<u32> }
impl Child { fn new() -> Self { Child {} } }
proc Parent { a: Child, b: Child }
impl Parent { fn new() -> Self { Parent { a: Child::new(), b: Child::new() } } }
)"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue parent,
                           GetCanonicalInitializer(result.tm, "Parent"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue a, GetMemberValue(parent, "a"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue b, GetMemberValue(parent, "b"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue a_input, GetMemberValue(a, "input"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue b_input, GetMemberValue(b, "input"));
  EXPECT_NE(a_input.GetChannelReferenceOrDie().GetChannelId(),
            b_input.GetChannelReferenceOrDie().GetChannelId());
}

TEST(TypecheckV2ProcTest, ParametricProcWithOmittedIoObjectMembers) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result, TypecheckV2(R"(
#![feature(io_objects)]
proc P<N: u32> { input: Source<uN[N]>, output: Sink<uN[N]> }
impl P<N> { fn new() -> Self { P<N> {} } fn next(self) {} }
proc Parent {}
impl Parent {
  fn new() -> Self {
    P<8>::new().spawn();
    P<16>::new().spawn();
    Parent {}
  }
  fn next(self) {}
}
)"));
  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * p,
                           result.tm.module->GetMemberOrError<ProcDef>("P"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ProcInitializerWithTypeInfo> inits,
      result.tm.type_info->GetCanonicalProcInitializers(p));
  ASSERT_EQ(inits.size(), 2);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue input_1,
                           GetMemberValue(inits[0].initializer, "input"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue input_2,
                           GetMemberValue(inits[1].initializer, "input"));
  EXPECT_EQ(GetChannelDefinerKind(input_1), AstNodeKind::kStructMember);
  EXPECT_NE(input_1.GetChannelReferenceOrDie().GetChannelId(),
            input_2.GetChannelReferenceOrDie().GetChannelId());

  const FileTable& file_table = *result.tm.module->file_table();
  EXPECT_THAT(TypeInfoToString(*inits[0].constructor_type_info, file_table),
              IsOkAndHolds(HasNodeWithType("uN[N]", "typeof(uN[8])")));
  EXPECT_THAT(TypeInfoToString(*inits[1].constructor_type_info, file_table),
              IsOkAndHolds(HasNodeWithType("uN[N]", "typeof(uN[16])")));
}

constexpr std::string_view kLoopbackProc = R"(
#![feature(io_objects)]
proc Loopback {
  input: Source<u32>,
  output: Sink<u32>
}
impl Loopback {
  fn new() -> Self { Loopback {} }
  fn next(self) {}
}
)";

TEST(TypecheckV2ProcTest, BindRecordsBothEndpoints) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(absl::StrCat(kLoopbackProc, R"(
proc Top { to_child: Sink<u32>, from_child: Source<u32> }
impl Top {
  fn new() -> Self {
    let top = Top {};
    let c = Loopback::new();
    top.to_child.bind(c.input);
    c.output.bind(top.from_child);
    c.spawn();
    top
  }
  fn next(self) {}
}
)")));
  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * top,
                           result.tm.module->GetMemberOrError<ProcDef>("Top"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue top_init,
                           GetCanonicalInitializer(result.tm, "Top"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<InterpValue> spawns,
                           result.tm.type_info->GetProcDefSpawnsFrom(top));
  ASSERT_EQ(spawns.size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(InterpValue to_child,
                           GetMemberValue(top_init, "to_child"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue from_child,
                           GetMemberValue(top_init, "from_child"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue child_input,
                           GetMemberValue(spawns[0], "input"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue child_output,
                           GetMemberValue(spawns[0], "output"));

  const ChannelManager& manager = result.tm.type_info->channel_manager();
  std::optional<const ImplicitChannelDecl*> to_child_decl =
      manager.GetImplicitChannelDecl(top, to_child);
  ASSERT_TRUE(to_child_decl.has_value());
  EXPECT_EQ(manager.GetImplicitChannelDecl(top, child_input), to_child_decl);
  EXPECT_EQ((*to_child_decl)->out_end, to_child);
  EXPECT_EQ((*to_child_decl)->in_end, child_input);

  std::optional<const ImplicitChannelDecl*> from_child_decl =
      manager.GetImplicitChannelDecl(top, from_child);
  ASSERT_TRUE(from_child_decl.has_value());
  EXPECT_EQ(manager.GetImplicitChannelDecl(top, child_output), from_child_decl);
  EXPECT_EQ((*from_child_decl)->out_end, child_output);
  EXPECT_EQ((*from_child_decl)->in_end, from_child);
}

TEST(TypecheckV2ProcTest, BindChildToChild) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(absl::StrCat(kLoopbackProc, R"(
proc Top {}
impl Top {
  fn new() -> Self {
    let a = Loopback::new();
    let b = Loopback::new();
    a.output.bind(b.input);
    a.spawn();
    b.spawn();
    Top {}
  }
  fn next(self) {}
}
)")));
  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * top,
                           result.tm.module->GetMemberOrError<ProcDef>("Top"));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<InterpValue> spawns,
                           result.tm.type_info->GetProcDefSpawnsFrom(top));
  ASSERT_EQ(spawns.size(), 2);
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue a_output,
                           GetMemberValue(spawns[0], "output"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpValue b_input,
                           GetMemberValue(spawns[1], "input"));

  const ChannelManager& manager = result.tm.type_info->channel_manager();
  std::optional<const ImplicitChannelDecl*> decl =
      manager.GetImplicitChannelDecl(top, a_output);
  ASSERT_TRUE(decl.has_value());
  EXPECT_EQ(manager.GetImplicitChannelDecl(top, b_input), decl);
  EXPECT_EQ((*decl)->out_end, a_output);
  EXPECT_EQ((*decl)->in_end, b_input);
}

TEST(TypecheckV2ProcTest, BindEndpointTwiceIsRejected) {
  EXPECT_THAT(absl::StrCat(kLoopbackProc, R"(
proc Top { to_child: Sink<u32> }
impl Top {
  fn new() -> Self {
    let top = Top {};
    let c = Loopback::new();
    let d = Loopback::new();
    top.to_child.bind(c.input);
    top.to_child.bind(d.input);
    c.spawn();
    d.spawn();
    top
  }
  fn next(self) {}
}
)"),
              TypecheckFails(HasSubstr("`top.to_child` is already bound by "
                                       "`top.to_child.bind(c.input)`.")));
}

TEST(TypecheckV2ProcTest, BindOutsideProcImplIsRejected) {
  EXPECT_THAT(
      R"(
#![feature(io_objects)]
fn connect(s: Sink<u32>, r: Source<u32>) { s.bind(r) }
)",
      TypecheckFails(HasSubstr("`bind` can only be called in a proc impl.")));
}

TEST(TypecheckV2ProcTest, BindInParametricProcIsRecordedPerInstantiation) {
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(absl::StrCat(kLoopbackProc, R"(
proc Top<N: u32> { to_child: Sink<u32>, from_child: Source<u32> }
impl Top<N> {
  fn new() -> Self {
    let top = Top<N> {};
    let c = Loopback::new();
    top.to_child.bind(c.input);
    c.output.bind(top.from_child);
    c.spawn();
    top
  }
  fn next(self) {}
}
proc Main {}
impl Main {
  fn new() -> Self {
    Top<1>::new().spawn();
    Top<2>::new().spawn();
    Top<1>::new().spawn();
    Main {}
  }
  fn next(self) {}
}
)")));
  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * top,
                           result.tm.module->GetMemberOrError<ProcDef>("Top"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ProcInitializerWithTypeInfo> inits,
      result.tm.type_info->GetCanonicalProcInitializers(top));
  ASSERT_EQ(inits.size(), 2);
  EXPECT_NE(inits[0].constructor_type_info, inits[1].constructor_type_info);
  for (const ProcInitializerWithTypeInfo& init : inits) {
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue to_child,
                             GetMemberValue(init.initializer, "to_child"));
    XLS_ASSERT_OK_AND_ASSIGN(InterpValue from_child,
                             GetMemberValue(init.initializer, "from_child"));
    EXPECT_TRUE(init.constructor_type_info->channel_manager()
                    .GetImplicitChannelDecl(top, to_child)
                    .has_value());
    EXPECT_TRUE(init.constructor_type_info->channel_manager()
                    .GetImplicitChannelDecl(top, from_child)
                    .has_value());
    EXPECT_FALSE(result.tm.type_info->channel_manager()
                     .GetImplicitChannelDecl(top, to_child)
                     .has_value());
  }
}

TEST(TypecheckV2ProcTest, BindsInDifferentModulesDoNotConflict) {
  // Both modules number channel ids from the same start and share one root
  // `TypeInfo`, verify that keying by `ProcDef` prevents cross-module id
  // collisions.
  constexpr std::string_view kImported = R"(#![feature(io_objects)]
proc Leaf { input: Source<u32> }
impl Leaf { fn new() -> Self { Leaf {} } fn next(self) {} }
pub proc Mid { to_leaf: Sink<u32> }
impl Mid {
  pub fn new() -> Self {
    let m = Mid {};
    let l = Leaf::new();
    m.to_leaf.bind(l.input);
    l.spawn();
    m
  }
  fn next(self) {}
}
)";
  constexpr std::string_view kProgram = R"(#![feature(io_objects)]
import imported;

proc Leaf { input: Source<u32> }
impl Leaf { fn new() -> Self { Leaf {} } fn next(self) {} }
proc Top { to_leaf: Sink<u32> }
impl Top {
  fn new() -> Self {
    let t = Top {};
    let l = Leaf::new();
    t.to_leaf.bind(l.input);
    l.spawn();
    t
  }
  fn next(self) {}
}
)";

  absl::flat_hash_map<std::filesystem::path, std::string> files;
  files[std::filesystem::path("/imported.x")] = std::string(kImported);
  auto vfs = std::make_unique<FakeFilesystem>(std::move(files),
                                              std::filesystem::path("/"));
  ImportData import_data =
      CreateImportData(xls::kDefaultDslxStdlibPath,
                       /*additional_search_paths=*/{std::filesystem::path("/")},
                       kAllWarningsSet, std::move(vfs));

  XLS_EXPECT_OK(TypecheckV2(kProgram, "main", &import_data));
}

TEST(TypecheckV2ProcTest, ProcDefWithArrayTypeAliasState) {
  EXPECT_THAT(R"(
#![feature(explicit_state_access)]

pub const MEM_SIZE = u32:256;
pub type MemWord = u32;
type State = MemWord[MEM_SIZE];

pub proc MemoryProc {
    addr_in: chan<u8> in,
    val_out: chan<MemWord> out,
    state: State,
}

impl MemoryProc {
    pub fn new(addr_in: chan<u8> in, val_out: chan<MemWord> out) -> Self {
        Self { addr_in, val_out, state: zero!<State>() }
    }

    pub fn next(self) {
        let state = read(self.state);
        let (tok, addr) = recv(join(), self.addr_in);
        let word = state[addr];
        send(tok, self.val_out, word);
        let next_state = update(state, addr, word + u32:1);
        write(self.state, next_state);
    }
}
)",
              TypecheckSucceeds(::testing::_));
}

TEST(TypecheckV2ProcTest, ProcDefWith2DChannelArrayInConstructor) {
  EXPECT_THAT(R"(
#![feature(explicit_state_access)]
#![feature(generics)]

proc Node<ROW: u32, COL: u32> {
    from_west: chan<u32> in,
    to_east: chan<u32> out,
}

impl Node<ROW, COL> {
    fn new(from_west: chan<u32> in, to_east: chan<u32> out) -> Self {
        Self { from_west, to_east }
    }
    fn next(self) {
        let (tok, val) = recv(join(), self.from_west);
        send(tok, self.to_east, val);
    }
}

proc Grid<ROWS: u32, COLS: u32> {
    from_wests: chan<u32>[COLS + u32:1][ROWS] in,
    to_easts: chan<u32>[COLS + u32:1][ROWS] out,
}

impl Grid<ROWS, COLS> {
    fn new() -> Self {
        let (to_easts, from_wests) = chan<u32>[COLS + u32:1][ROWS]("east_west");
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            unroll_for! (col, _): (u32, ()) in u32:0..COLS {
                Node<row, col>::new(from_wests[row][col], to_easts[row][col + u32:1]).spawn();
            }(());
        }(());
        Self { from_wests, to_easts }
    }
    fn next(self) {
        const FIRST_COL = u32:0;
        const LAST_COL = COLS;
        unroll_for! (row, _): (u32, ()) in u32:0..ROWS {
            send(join(), self.to_easts[row][FIRST_COL], row);
            recv(join(), self.from_wests[row][LAST_COL]);
        }(());
    }
}

proc Top {}
impl Top {
    fn new() -> Self {
        Grid<u32:2, u32:3>::new().spawn();
        Self {}
    }
}
)",
              TypecheckSucceeds(::testing::_));
}
}  // namespace
}  // namespace xls::dslx
