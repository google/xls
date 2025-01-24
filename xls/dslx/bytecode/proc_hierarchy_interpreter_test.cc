// Copyright 2021 The XLS Authors
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
#include "xls/dslx/bytecode/proc_hierarchy_interpreter.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {
namespace {

absl::StatusOr<TestResultData> ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) {
  return DslxInterpreterTestRunner().ParseAndTest(program, module_name,
                                                  filename, options);
}

absl::StatusOr<TypecheckedModule> ParseAndTypecheckOrPrintError(
    std::string_view program, ImportData* import_data) {
  // Parse/typecheck and print a helpful error.
  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(program, "test.x", "test", import_data);
  if (!tm.ok()) {
    UniformContentFilesystem vfs(program);
    TryPrintError(tm.status(), import_data->file_table(), vfs);
  }
  return tm;
}

using ::testing::HasSubstr;

class ProcHierarchyInterpreterTest : public ::testing::Test {
 public:
  void SetUp() override { import_data_.emplace(CreateImportDataForTest()); }

  absl::StatusOr<TestProc*> ParseAndGetTestProc(
      std::string_view program, std::string_view test_proc_name) {
    absl::StatusOr<TypecheckedModule> tm =
        ParseAndTypecheckOrPrintError(program, &import_data_.value());
    XLS_RETURN_IF_ERROR(tm.status());
    tm_.emplace(*tm);
    return tm_->module->GetTestProc(test_proc_name);
  }

  absl::Status Run(TestProc* test_proc,
                   const BytecodeInterpreterOptions& options) {
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, tm_->type_info->GetTopLevelProcTypeInfo(
                                            test_proc->proc()));
    XLS_ASSIGN_OR_RETURN(
        InterpValue terminator,
        ti->GetConstExpr(test_proc->proc()->config().params()[0]));
    std::vector<std::string> trace_output;
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<ProcHierarchyInterpreter> hierarchy_interpreter,
        ProcHierarchyInterpreter::Create(&import_data_.value(), ti,
                                         test_proc->proc(), options));
    // There should be a single top config argument: a reference to the
    // terminator channel. Determine the actual channel object.
    XLS_RET_CHECK_EQ(hierarchy_interpreter->InterfaceArgs().size(), 1);
    std::string terminal_channel_name =
        std::string{hierarchy_interpreter->GetInterfaceChannelName(0)};

    // Run until a single output appears in the terminal channel.
    XLS_RETURN_IF_ERROR(
        hierarchy_interpreter->TickUntilOutput({{terminal_channel_name, 1}})
            .status());

    return absl::OkStatus();
  }

 protected:
  std::optional<ImportData> import_data_;
  std::optional<TypecheckedModule> tm_;
};

// https://github.com/google/xls/issues/981
TEST_F(ProcHierarchyInterpreterTest, AssertEqFailProcIterations) {
  constexpr std::string_view kProgram = R"(
#[test_proc]
proc BTester {
    terminator: chan<bool> out;

    init { (u32:0) }

    config(terminator: chan<bool> out) {
        (terminator,)
    }

    next(state: u32) {
        assert_eq(state, u32:0);
        // ensure at least 2 `next()` iterations to create an interpreter frame.
        let tok = send_if(join(), terminator, state > u32:1, true);
        (state + u32:1)
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr std::string_view kModuleName = "test";
  ParseAndTestOptions options;
  ::testing::internal::CaptureStderr();
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, std::string{temp_file.path()},
                   options));
  std::string stdcerr(::testing::internal::GetCapturedStderr());
  EXPECT_EQ(result.result(), TestResult::kSomeFailed);
  EXPECT_THAT(stdcerr, HasSubstr("were not equal"));
}

TEST_F(ProcHierarchyInterpreterTest, DistinctNestedParametricProcs) {
  // Tests that B, which has one set of parameters, can instantiate A, which has
  // a different set of parameters.
  // TODO(rspringer): Once this goes in, open a bug: if init() is changed to
  // "{ N }", it fails, because N can't be evaluated to a value: it's computed
  // without applying the caller bindings.
  constexpr std::string_view kProgram = R"(
proc A<N: u32> {
    data_in: chan<u32> in;
    data_out: chan<u32> out;

    init {
        N
    }
    config(data_in: chan<u32> in, data_out: chan<u32> out) {
        (data_in, data_out)
    }
    next(state: u32) {
        let (tok, x) = recv(join(), data_in);
        let tok = send(tok, data_out, x + N + state);
        state + u32:1
    }
}

proc B<M: u32, N: u32> {
    data_in: chan<u32> in;
    data_out: chan<u32> out;

    init { () }
    config(data_in: chan<u32> in, data_out: chan<u32> out) {
        spawn A<N>(data_in, data_out);
        (data_in, data_out)
    }
    next(state: ()) {
        ()
    }
}

#[test_proc]
proc BTester {
    data_in: chan<u32> out;
    data_out: chan<u32> in;
    terminator: chan<bool> out;

    init { () }

    config(terminator: chan<bool> out) {
        let (data_in_p, data_in_c) = chan<u32>("data_in");
        let (data_out_p, data_out_c) = chan<u32>("data_out");
        spawn B<u32:5, u32:3>(data_in_c, data_out_p);
        (data_in_p, data_out_c, terminator)
    }

    next(state: ()) {
        let tok = send(join(), data_in, u32:3);
        let (tok, result) = recv(tok, data_out);
        assert_eq(result, u32:9);
        let tok = send(tok, terminator, true);
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr std::string_view kModuleName = "test";
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, std::string{temp_file.path()},
                   options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

// https://github.com/google/xls/issues/1502
TEST_F(ProcHierarchyInterpreterTest, TwoTestProcsWithSameDeepNetwork) {
  constexpr std::string_view kProgram = R"(
pub proc BarA {
    bar_in_r: chan<u1> in;
    bar_out_s: chan<u1> out;

    config(
        bar_in_r: chan<u1> in,
        bar_out_s: chan<u1> out,
    ) {
        (bar_in_r, bar_out_s)
    }

    init {  }

    next (state: ()) {
        let tok = join();
        let (tok, a) = recv(tok, bar_in_r);
        trace_fmt!("BarA recv {}", a);
        send(tok, bar_out_s, a);
        trace_fmt!("BarA sent {}", a);
    }
}

pub proc Bar {
    bar_in_r: chan<u1> in;
    bar_s: chan<u1> out;

    config(
        bar_in_r: chan<u1> in,
        bar_out_s: chan<u1> out,
    ) {
        let (bar_s, bar_r) = chan<u1, u32:1>("bar");

        spawn BarA(bar_r, bar_out_s);

        (bar_in_r, bar_s)
    }

    init {  }

    next (state: ()) {
        let tok = join();
        let (tok, a) = recv(tok, bar_in_r);
        trace_fmt!("Bar recv {}", a);
        send(tok, bar_s, a);
        trace_fmt!("Bar sent {}", a);
    }
}

proc Foo {
    config(
        foo_in_r: chan<u1> in,
        foo_out_s: chan<u1> out,
    ) {
        spawn Bar(foo_in_r, foo_out_s);
    }

    init {  }

    next (state: ()) { }
}

#[test_proc]
proc Foo_test_0 {
    terminator: chan<bool> out;

    foo_in_s: chan<u1> out;
    foo_out_r: chan<u1> in;

    config (terminator: chan<bool> out) {
        let (foo_in_s, foo_in_r) = chan<u1, u32:1>("foo_in");
        let (foo_out_s, foo_out_r) = chan<u1, u32:1>("foo_out");

        spawn Foo(foo_in_r, foo_out_s);

        (terminator, foo_in_s, foo_out_r)
    }

    init { }

    next (state: ()) {
        let tok = join();
        let tok = send(tok, foo_in_s, u1:0);
        trace_fmt!("Test 0 sent");
        let (tok, data) = recv(tok, foo_out_r);
        trace_fmt!("Test 0 recv {}", data);

        assert_eq(u1:0, data);

        send(tok, terminator, true);
    }
}

#[test_proc]
proc Foo_test_1 {
    terminator: chan<bool> out;

    foo_in_s: chan<u1> out;
    foo_out_r: chan<u1> in;

    config (terminator: chan<bool> out) {
        let (foo_in_s, foo_in_r) = chan<u1, u32:1>("foo_in");
        let (foo_out_s, foo_out_r) = chan<u1, u32:1>("foo_out");

        spawn Foo(foo_in_r, foo_out_s);

        (terminator, foo_in_s, foo_out_r)
    }

    init { }

    next (state: ()) {
        let tok = join();
        let tok = send(tok, foo_in_s, u1:1);
        trace_fmt!("Test 1 sent");
        let (tok, data) = recv(tok, foo_out_r);
        trace_fmt!("Test 1 recv {}", data);

        assert_eq(u1:1, data);

        send(tok, terminator, true);
    }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  constexpr std::string_view kModuleName = "test";
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, kModuleName, std::string{temp_file.path()},
                   options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, TraceChannels) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>("input");
    let (output_p, output_c) = chan<u32>("output");
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(state: ()) {
    let tok = send(join(), data_out, u32:42);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, u32:100);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(Run(
      test_proc, BytecodeInterpreterOptions().trace_channels(true).trace_hook(
                     [&](const Span&, std::string_view s) {
                       trace_output.push_back(std::string{s});
                     })));
  EXPECT_THAT(trace_output,
              testing::ElementsAre(
                  "Sent data on channel `tester_proc::data_out`:\n  u32:42",
                  "Received data on channel "
                  "`tester_proc->incrementer#0::in_ch`:\n  u32:42",
                  "Sent data on channel "
                  "`tester_proc->incrementer#0::out_ch`:\n  u32:43",
                  "Received data on channel `tester_proc::data_in`:\n  u32:43",
                  "Sent data on channel `tester_proc::data_out`:\n  u32:100",
                  "Received data on channel "
                  "`tester_proc->incrementer#0::in_ch`:\n  u32:100",
                  "Sent data on channel "
                  "`tester_proc->incrementer#0::out_ch`:\n  u32:101",
                  "Received data on channel `tester_proc::data_in`:\n  u32:101",
                  "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
}

TEST_F(ProcHierarchyInterpreterTest, TraceChannelsHexValues) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>("input");
    let (output_p, output_c) = chan<u32>("output");
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(state: ()) {
    let tok = send(join(), data_out, u32:42);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, u32:100);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(
      Run(test_proc, BytecodeInterpreterOptions()
                         .trace_channels(true)
                         .format_preference(FormatPreference::kHex)
                         .trace_hook([&](const Span&, std::string_view s) {
                           trace_output.push_back(std::string{s});
                         })));
  EXPECT_THAT(
      trace_output,
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out`:\n  u32:0x2a",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:0x2a",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:0x2b",
          "Received data on channel `tester_proc::data_in`:\n  u32:0x2b",
          "Sent data on channel `tester_proc::data_out`:\n  u32:0x64",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:0x64",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:0x65",
          "Received data on channel `tester_proc::data_in`:\n  u32:0x65",
          "Sent data on channel `tester_proc::terminator`:\n  u1:0x1"));
}

TEST_F(ProcHierarchyInterpreterTest, TraceChannelsWithNonblockingReceive) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i, valid) = recv_non_blocking(join(), in_ch, u32:0);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>("input");
    let (output_p, output_c) = chan<u32>("output");
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(state: ()) {
    let tok = send_if(join(), data_out, false, u32:42);
    let (tok, result) = recv(tok, data_in);
    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(Run(
      test_proc, BytecodeInterpreterOptions().trace_channels(true).trace_hook(
                     [&](const Span&, std::string_view s) {
                       trace_output.push_back(std::string{s});
                     })));
  EXPECT_THAT(
      trace_output,
      testing::ElementsAre(
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  u32:1",
          "Received data on channel `tester_proc::data_in`:\n  u32:1",
          "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
}

TEST_F(ProcHierarchyInterpreterTest, TraceStructChannels) {
  constexpr std::string_view kProgram = R"(
struct Foo {
  a: u32,
  b: u16
}

proc incrementer {
  in_ch: chan<Foo> in;
  out_ch: chan<Foo> out;

  init { () }

  config(in_ch: chan<Foo> in,
         out_ch: chan<Foo> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
    let tok = send(tok, out_ch, Foo { a:i.a + u32:1, b:i.b + u16:1 });
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<Foo> out;
  data_in: chan<Foo> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<Foo>("input");
    let (output_p, output_c) = chan<Foo>("output");
    spawn incrementer(input_c, output_p);
    (input_p, output_c, terminator)
  }

  next(state: ()) {

    let tok = send(join(), data_out, Foo { a:u32:42, b:u16:100 });
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, Foo{ a:u32:555, b:u16:123 });
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(Run(
      test_proc, BytecodeInterpreterOptions().trace_channels(true).trace_hook(
                     [&](const Span&, std::string_view s) {
                       trace_output.push_back(std::string{s});
                     })));
  EXPECT_EQ(trace_output[0],
            R"(Sent data on channel `tester_proc::data_out`:
  Foo {
    a: u32:42,
    b: u16:100
})");
  EXPECT_EQ(trace_output[1],
            R"(Received data on channel `tester_proc->incrementer#0::in_ch`:
  Foo {
    a: u32:42,
    b: u16:100
})");
}

TEST_F(ProcHierarchyInterpreterTest, TraceArrayOfChannels) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32>[1] out;
  data_in: chan<u32>[1] in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>[1]("input");
    let (output_p, output_c) = chan<u32>[1]("output");
    spawn incrementer(input_c[0], output_p[0]);
    (input_p, output_c, terminator)
  }

  next(state: ()) {

    let tok = send(join(), data_out[0], u32:42);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(tok, data_out[0], u32:100);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(Run(
      test_proc, BytecodeInterpreterOptions().trace_channels(true).trace_hook(
                     [&](const Span&, std::string_view s) {
                       trace_output.push_back(std::string{s});
                     })));
  EXPECT_THAT(
      trace_output,
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out[0]`:\n  u32:42",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:42",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:43",
          "Received data on channel `tester_proc::data_in[0]`:\n  u32:43",
          "Sent data on channel `tester_proc::data_out[0]`:\n  u32:100",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:100",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:101",
          "Received data on channel `tester_proc::data_in[0]`:\n  u32:101",
          "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
}

// Tests https://github.com/google/xls/issues/1552
TEST_F(ProcHierarchyInterpreterTest, TraceChannelsWithMultiProcInstances) {
  constexpr std::string_view kProgram = R"(
proc incrementer {
  in_ch: chan<u32> in;
  out_ch: chan<u32> out;

  init { () }

  config(in_ch: chan<u32> in,
         out_ch: chan<u32> out) {
    (in_ch, out_ch)
  }
  next(_: ()) {
    let (tok, i) = recv(join(), in_ch);
    let tok = send(tok, out_ch, i + u32:1);
  }
}

#[test_proc]
proc tester_proc {
  data_out: chan<u32> out;
  data_in: chan<u32> in;
  data_out_2: chan<u32> out;
  data_in_2: chan<u32> in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>("input");
    let (output_p, output_c) = chan<u32>("output");
    spawn incrementer(input_c, output_p);

    let (input_2_p, input_2_c) = chan<u32>("input2");
    let (output_2_p, output_2_c) = chan<u32>("output2");
    spawn incrementer(input_2_c, output_2_p);

    (input_p, output_c, input_2_p, output_2_c, terminator)
  }

  next(state: ()) {
    let tok = send(join(), data_out, u32:42);
    let (tok, result) = recv(tok, data_in);

    let tok = send(tok, data_out, u32:100);
    let (tok, result) = recv(tok, data_in);

    let tok = send(join(), data_out_2, u32:43);
    let (tok, result) = recv(tok, data_in_2);

    let tok = send(tok, data_out_2, u32:101);
    let (tok, result) = recv(tok, data_in_2);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  std::vector<std::string> trace_output;
  XLS_ASSERT_OK(Run(
      test_proc, BytecodeInterpreterOptions().trace_channels(true).trace_hook(
                     [&](const Span&, std::string_view s) {
                       trace_output.push_back(std::string{s});
                     })));
  EXPECT_THAT(
      trace_output,
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out`:\n  u32:42",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:42",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:43",
          "Received data on channel `tester_proc::data_in`:\n  u32:43",
          "Sent data on channel `tester_proc::data_out`:\n  u32:100",
          "Received data on channel `tester_proc->incrementer#0::in_ch`:\n  "
          "u32:100",
          "Sent data on channel `tester_proc->incrementer#0::out_ch`:\n  "
          "u32:101",
          "Received data on channel `tester_proc::data_in`:\n  u32:101",
          "Sent data on channel `tester_proc::data_out_2`:\n  u32:43",
          "Received data on channel `tester_proc->incrementer#1::in_ch`:\n  "
          "u32:43",
          "Sent data on channel `tester_proc->incrementer#1::out_ch`:\n  "
          "u32:44",
          "Received data on channel `tester_proc::data_in_2`:\n  u32:44",
          "Sent data on channel `tester_proc::data_out_2`:\n  u32:101",
          "Received data on channel `tester_proc->incrementer#1::in_ch`:\n  "
          "u32:101",
          "Sent data on channel `tester_proc->incrementer#1::out_ch`:\n  "
          "u32:102",
          "Received data on channel `tester_proc::data_in_2`:\n  u32:102",
          "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
}

}  // namespace
}  // namespace xls::dslx
