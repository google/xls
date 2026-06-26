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

#include "xls/dslx/fmt/legacy_proc_converter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/fmt/ast_fmt.h"
#include "xls/dslx/fmt/comments.h"
#include "xls/dslx/fmt/pretty_print.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/comment_data.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

class LegacyProcConverterTest : public testing::Test {
 public:
  void DoLegacyProcConversionFmt(std::string input, std::string_view want,
                                 int64_t text_width = kDslxDefaultTextWidth) {
    std::vector<CommentData> comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> m,
        ParseModule(input, "fake.x", "fake", file_table_, &comments_vec));
    Comments comments = Comments::Create(comments_vec);
    DocArena arena(file_table_);
    std::unique_ptr<Formatter> converter =
        CreateLegacyProcConverter(comments, arena);
    AllErrorsFilesystem vfs;
    XLS_ASSERT_OK_AND_ASSIGN(std::string got,
                             AutoFmt(vfs, *m, *converter, input, text_width));
    EXPECT_EQ(got, want);

    // Parse and typecheck the converted/formatted DSLX module to verify it's
    // valid.
    ImportData import_data = CreateImportDataForTest();
    std::vector<CommentData> dummy_comments;
    XLS_ASSERT_OK(
        ParseAndTypecheck(got, "fake.x", "fake", &import_data, &dummy_comments)
            .status());

    // Re-parse the converted DSLX module cleanly (without typechecking it)
    // so we can format the original (non-desugared) AST nodes.
    FileTable file_table_for_fmt;
    std::vector<CommentData> reparsed_comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> reparsed_m,
        ParseModule(got, "fake.x", "fake", file_table_for_fmt,
                    &reparsed_comments_vec));

    // Format the reparsed converted module using a generic Formatter.
    Comments reparsed_comments = Comments::Create(reparsed_comments_vec);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string formatted,
        AutoFmt(vfs, *reparsed_m, reparsed_comments, got, text_width));

    EXPECT_EQ(formatted, want);
  }

  void DoLegacyProcConversionFmtError(std::string input,
                                      std::string_view want_error_msg) {
    std::vector<CommentData> comments_vec;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Module> m,
        ParseModule(input, "fake.x", "fake", file_table_, &comments_vec));
    Comments comments = Comments::Create(comments_vec);
    DocArena arena(file_table_);
    std::unique_ptr<Formatter> converter =
        CreateLegacyProcConverter(comments, arena);
    AllErrorsFilesystem vfs;
    auto got_status =
        AutoFmt(vfs, *m, *converter, input, kDslxDefaultTextWidth);
    EXPECT_THAT(got_status.status(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       testing::HasSubstr(want_error_msg)));
  }

 protected:
  FileTable file_table_;
};

TEST_F(LegacyProcConverterTest, BasicProc) {
  DoLegacyProcConversionFmt(
      R"(proc Producer {
    s: chan<u32> out;
    config(input_s: chan<u32> out) {
        // A comment inside config.
        let x = u32:42;
        (input_s,)
    }
    init {
        0
    }
    next(state: u32) {
        // A comment inside next.
        send(join(), s, state);
        state + 1
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
    state: u32,
}

impl Producer {
    fn new(input_s: chan<u32> out) -> Self {
        // A comment inside config.
        let x = u32:42;
        Producer { s: input_s, state: 0 }
    }

    fn next(self) {
        let state = read(self.state);
        // A comment inside next.
        send(join(), self.s, state);
        write(self.state, state + 1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, Stateless) {
  DoLegacyProcConversionFmt(
      R"(proc Producer {
    s: chan<u32> out;
    config(s: chan<u32> out) { (s,) }
    init { () }
    next(state: ()) {
        send(join(), s, 42);
        ()
    }
}
proc Consumer {
    r: chan<u32> in;
    config(r: chan<u32> in) { (r,) }
    init { () }
    next(state: ()) {
        recv(join(), r);
        ()
    }
}
proc Main {
    config() {
        let (s, r) = chan<u32>("my_chan");
        spawn Producer(s);
        spawn Consumer(r);
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
}

impl Producer {
    fn new(s: chan<u32> out) -> Self {
        Producer { s }
    }

    fn next(self) {
        send(join(), self.s, 42);
    }
}

proc Consumer {
    r: chan<u32> in,
}

impl Consumer {
    fn new(r: chan<u32> in) -> Self {
        Consumer { r }
    }

    fn next(self) {
        recv(join(), self.r);
    }
}

proc Main {}

impl Main {
    fn new() -> Self {
        let (s, r) = chan<u32>("my_chan");
        Producer::new(s).spawn();
        Consumer::new(r).spawn();
        Main {}
    }
}
)");
}

TEST_F(LegacyProcConverterTest, Comments) {
  DoLegacyProcConversionFmt(
      R"(proc Main {
    // Channel member.
    c: chan<u32> out;

    // Config comment.
    config(c: chan<u32> out) {
        (c,)
    }

    // Init comment.
    init {
        42
    }

    // Next comment.
    next(state: u32) {
        send(join(), c, state);
        state
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Main {
    // Channel member.
    c: chan<u32> out,
    state: u32,
}

impl Main {
    // Config comment.
    // Init comment.
    fn new(c: chan<u32> out) -> Self {
        Main { c, state: 42 }
    }

    // Next comment.
    fn next(self) {
        let state = read(self.state);
        send(join(), self.c, state);
        write(self.state, state);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, HoistedStatements) {
  DoLegacyProcConversionFmt(
      R"(proc Main {
    const MY_CONST = u32:42;
    s: chan<u32> out;

    config(s: chan<u32> out) {
        (s,)
    }
    init {
        u32:0
    }
    next(state: u32) {
        let x = state + MY_CONST;
        send(join(), s, x);
        state
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Main {
    s: chan<u32> out,
    state: u32,
}

impl Main {
    const MY_CONST = u32:42;

    fn new(s: chan<u32> out) -> Self {
        Main { s, state: u32:0 }
    }

    fn next(self) {
        let state = read(self.state);
        let x = state + MY_CONST;
        send(join(), self.s, x);
        write(self.state, state);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, HoistedStatementsError) {
  DoLegacyProcConversionFmtError(
      R"(proc Main {
    const MY_CONST = u32:42;
    s: chan<u32> out;

    config(s: chan<u32> out) {
        (s,)
    }
    init {
        u32:0
    }
    next(state: u32[MY_CONST]) {
        send(join(), s, state[0]);
        state
    }
}
)",
      "Proc state parameter `state` references a constant declared inside the "
      "proc");
}

TEST_F(LegacyProcConverterTest, TypeAliasError) {
  DoLegacyProcConversionFmtError(
      R"(proc Main {
    type MyType = u32;
    s: chan<u32> out;
    config(s: chan<u32> out) { (s,) }
    init { u32:0 }
    next(state: u32) { state }
}
)",
      "Type aliases inside a proc are not supported in impl-style procs.");
}

TEST_F(LegacyProcConverterTest, NestedSpawn) {
  DoLegacyProcConversionFmt(
      R"(proc Producer {
    s: chan<u32> out;
    config(s: chan<u32> out) { (s,) }
    init { () }
    next(state: ()) {
        send(join(), s, 42);
        ()
    }
}
proc Main {
    config() {
        let (s, r) = chan<u32>("my_chan");
        let p = spawn Producer(s);
        ()
    }
    init {
        ()
    }
    next(state: ()) {
        ()
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
}

impl Producer {
    fn new(s: chan<u32> out) -> Self {
        Producer { s }
    }

    fn next(self) {
        send(join(), self.s, 42);
    }
}

proc Main {}

impl Main {
    fn new() -> Self {
        let (s, r) = chan<u32>("my_chan");
        let p = Producer::new(s).spawn();
        Main {}
    }
}
)");
}

TEST_F(LegacyProcConverterTest, Parametrics) {
  DoLegacyProcConversionFmt(
      R"(proc MyProc<X: u32, Y: u32> {
    s: chan<u32> out;
    config(s: chan<u32> out) {
        (s,)
    }
    init {
        u32:0
    }
    next(state: u32) {
        send(join(), s, state + X + Y);
        state + 1
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc MyProc<X: u32, Y: u32> {
    s: chan<u32> out,
    state: u32,
}

impl MyProc<X, Y> {
    fn new(s: chan<u32> out) -> Self {
        MyProc { s, state: u32:0 }
    }

    fn next(self) {
        let state = read(self.state);
        send(join(), self.s, state + X + Y);
        write(self.state, state + 1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, TestProc) {
  DoLegacyProcConversionFmt(
      R"(proc Producer {
    s: chan<u32> out;
    config(s: chan<u32> out) {
        (s,)
    }
    init { () }
    next(state: ()) {
        send(join(), s, u32:42);
        ()
    }
}

#[test_proc]
proc MyTestProc {
    r: chan<u32> in;
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        let (s, r) = chan<u32>("my_chan");
        spawn Producer(s);
        (r, terminator)
    }
    init { () }
    next(state: ()) {
        let (tok, val) = recv(join(), r);
        assert_eq(val, u32:42);
        send(tok, terminator, true);
        ()
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
}

impl Producer {
    fn new(s: chan<u32> out) -> Self {
        Producer { s }
    }

    fn next(self) {
        send(join(), self.s, u32:42);
    }
}

#[test]
proc MyTestProc {
    r: chan<u32> in,
    terminator: chan<bool> out,
}

impl MyTestProc {
    fn new(terminator: chan<bool> out) -> Self {
        let (s, r) = chan<u32>("my_chan");
        Producer::new(s).spawn();
        MyTestProc { r, terminator }
    }

    fn next(self) {
        let (tok, val) = recv(join(), self.r);
        assert_eq(val, u32:42);
        send(tok, self.terminator, true);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, CommentsInStatelessNext) {
  DoLegacyProcConversionFmt(
      R"(proc MyTestProc {
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {
        (terminator,)
    }
    init { () }
    next(state: ()) {
        let stok = send(join(), terminator, true);
        // comment after last non-unit statement
        ()
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc MyTestProc {
    terminator: chan<bool> out,
}

impl MyTestProc {
    fn new(terminator: chan<bool> out) -> Self {
        MyTestProc { terminator }
    }

    fn next(self) {
        let stok = send(join(), self.terminator, true);
        // comment after last non-unit statement
    }
}
)");
}

TEST_F(LegacyProcConverterTest, AlreadyImplStyle) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]

proc AlreadyImplStyle {
    in_ch: chan<u32> in,
    out_ch: chan<u32> out,
    state: u32,
}

impl AlreadyImplStyle {
    fn new(in_ch: chan<u32> in, out_ch: chan<u32> out) -> Self {
        AlreadyImplStyle { in_ch, out_ch, state: u32:42 }
    }

    fn next(self) {
        let state = read(self.state);
        let (tok, data) = recv(join(), self.in_ch);
        let tok = send(tok, self.out_ch, data + state);
        write(self.state, state + u32:1);
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc AlreadyImplStyle {
    in_ch: chan<u32> in,
    out_ch: chan<u32> out,
    state: u32,
}

impl AlreadyImplStyle {
    fn new(in_ch: chan<u32> in, out_ch: chan<u32> out) -> Self {
        AlreadyImplStyle { in_ch, out_ch, state: u32:42 }
    }

    fn next(self) {
        let state = read(self.state);
        let (tok, data) = recv(join(), self.in_ch);
        let tok = send(tok, self.out_ch, data + state);
        write(self.state, state + u32:1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, ZeroMembers) {
  DoLegacyProcConversionFmt(
      R"(proc ChildProc {
    config() { () }
    init { () }
    next(state: ()) { () }
}

proc ZeroMembers {
    config() {
        spawn ChildProc();
    }
    init { () }
    next(state: ()) { () }
}
)",
      R"(#![feature(explicit_state_access)]

proc ChildProc {}

impl ChildProc {
    fn new() -> Self {
        ChildProc {}
    }
}

proc ZeroMembers {}

impl ZeroMembers {
    fn new() -> Self {
        ChildProc::new().spawn();
        ZeroMembers {}
    }
}
)");
}

TEST_F(LegacyProcConverterTest, StatelessNextWithCommentsBefore) {
  DoLegacyProcConversionFmt(
      R"(proc MyProc {
    config() { () }
    init { () }
    // comment before next
    next(state: ()) { () }
}
)",
      R"(#![feature(explicit_state_access)]

proc MyProc {}

impl MyProc {
    fn new() -> Self {
        MyProc {}
    }

    // comment before next
    fn next(self) {}
}
)");
}

TEST_F(LegacyProcConverterTest, SingleCustomStateName) {
  DoLegacyProcConversionFmt(
      R"(proc Producer {
    s: chan<u32> out;
    config(input_s: chan<u32> out) {
        (input_s,)
    }
    init {
        u32:42
    }
    next(foo: u32) {
        send(join(), s, foo);
        foo + u32:1
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
    foo: u32,
}

impl Producer {
    fn new(input_s: chan<u32> out) -> Self {
        Producer { s: input_s, foo: u32:42 }
    }

    fn next(self) {
        let foo = read(self.foo);
        send(join(), self.s, foo);
        write(self.foo, foo + u32:1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, MultipleStateElements) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]
proc Producer {
    s: chan<u32> out;
    config(input_s: chan<u32> out) {
        (input_s,)
    }
    init {
        (u32:42, u32:100)
    }
    next(x: u32, y: u32) {
        send(join(), s, x + y);
        (x + u32:1, y - u32:1)
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
    x: u32,
    y: u32,
}

impl Producer {
    fn new(input_s: chan<u32> out) -> Self {
        Producer { s: input_s, x: u32:42, y: u32:100 }
    }

    fn next(self) {
        let x = read(self.x);
        let y = read(self.y);
        send(join(), self.s, x + y);
        let next_state = (x + u32:1, y - u32:1);
        write(self.x, next_state.0);
        write(self.y, next_state.1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, MultipleStateElementsWithInitStatements) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]
proc Producer {
    s: chan<u32> out;
    config(input_s: chan<u32> out) {
        (input_s,)
    }
    init {
        let a = u32:42;
        let b = u32:100;
        (a, b)
    }
    next(x: u32, y: u32) {
        send(join(), s, x + y);
        (x + u32:1, y - u32:1)
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Producer {
    s: chan<u32> out,
    x: u32,
    y: u32,
}

impl Producer {
    fn new(input_s: chan<u32> out) -> Self {
        let a = u32:42;
        let b = u32:100;
        Producer { s: input_s, x: a, y: b }
    }

    fn next(self) {
        let x = read(self.x);
        let y = read(self.y);
        send(join(), self.s, x + y);
        let next_state = (x + u32:1, y - u32:1);
        write(self.x, next_state.0);
        write(self.y, next_state.1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, PreExistingExplicitStateAccess) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]
proc Counter {
    init { 0 }
    config() { }
    next(state: u32) {
        let current = read(state);
        write(state, current + 1);
    }
}
)",
      R"(#![feature(explicit_state_access)]

proc Counter {
    state: u32,
}

impl Counter {
    fn new() -> Self {
        Counter { state: 0 }
    }

    fn next(self) {
        let current = read(self.state);
        write(self.state, current + 1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, MultipleStateElementsWithFunctionCallInit) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]
fn get_initial_tuple() -> (u32, u32) {
    (u32:42, u32:100)
}

proc Producer {
    s: chan<u32> out;
    config(input_s: chan<u32> out) {
        (input_s,)
    }
    init {
        get_initial_tuple()
    }
    next(x: u32, y: u32) {
        send(join(), s, x + y);
        (x + u32:1, y - u32:1)
    }
}
)",
      R"(#![feature(explicit_state_access)]

fn get_initial_tuple() -> (u32, u32) { (u32:42, u32:100) }

proc Producer {
    s: chan<u32> out,
    x: u32,
    y: u32,
}

impl Producer {
    fn new(input_s: chan<u32> out) -> Self {
        let init_state = get_initial_tuple();
        Producer { s: input_s, x: init_state.0, y: init_state.1 }
    }

    fn next(self) {
        let x = read(self.x);
        let y = read(self.y);
        send(join(), self.s, x + y);
        let next_state = (x + u32:1, y - u32:1);
        write(self.x, next_state.0);
        write(self.y, next_state.1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, AlreadyImplStyleNotFormatted) {
  DoLegacyProcConversionFmt(
      R"(#![feature(explicit_state_access)]
proc AlreadyImplStyle { in_ch: chan<u32> in, out_ch: chan<u32> out, state: u32 }
impl AlreadyImplStyle {
  fn new(in_ch: chan<u32> in, out_ch: chan<u32> out) -> Self { AlreadyImplStyle { in_ch, out_ch, state: u32:42 } }
  fn next(self) {
    let state = read(self.state);
    let (tok, data) = recv(join(), self.in_ch);
    let tok = send(tok, self.out_ch, data + state);
    write(self.state, state + u32:1);
  }
}
)",
      R"(#![feature(explicit_state_access)]

proc AlreadyImplStyle {
    in_ch: chan<u32> in,
    out_ch: chan<u32> out,
    state: u32,
}

impl AlreadyImplStyle {
    fn new(in_ch: chan<u32> in, out_ch: chan<u32> out) -> Self {
        AlreadyImplStyle { in_ch, out_ch, state: u32:42 }
    }

    fn next(self) {
        let state = read(self.state);
        let (tok, data) = recv(join(), self.in_ch);
        let tok = send(tok, self.out_ch, data + state);
        write(self.state, state + u32:1);
    }
}
)");
}

TEST_F(LegacyProcConverterTest, StructWithNewNextMethods) {
  DoLegacyProcConversionFmt(
      R"(struct MyStruct {
    x: u32,
}

impl MyStruct {
    fn new(x: u32) -> Self {
        MyStruct { x }
    }

    fn next(self) -> u32 {
        self.x + u32:1
    }
}
)",
      R"(#![feature(explicit_state_access)]

struct MyStruct { x: u32 }

impl MyStruct {
    fn new(x: u32) -> Self { MyStruct { x } }

    fn next(self) -> u32 { self.x + u32:1 }
}
)");
}

}  // namespace
}  // namespace xls::dslx
