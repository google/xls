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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
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

using ::absl_testing::StatusIs;
using ::testing::AllOf;
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

  absl::StatusOr<ProcDef*> ParseAndGetTestProcDef(
      std::string_view program, std::string_view test_proc_name) {
    absl::StatusOr<TypecheckedModule> tm =
        ParseAndTypecheckOrPrintError(program, &import_data_.value());
    XLS_RETURN_IF_ERROR(tm.status());
    tm_.emplace(*tm);
    return tm_->module->GetTestProcDef(test_proc_name);
  }

  absl::StatusOr<std::unique_ptr<ProcHierarchyInterpreter>> Create(
      TestProc* test_proc, const BytecodeInterpreterOptions& options) {
    XLS_ASSIGN_OR_RETURN(TypeInfo * ti, tm_->type_info->GetTopLevelProcTypeInfo(
                                            test_proc->proc()));
    return ProcHierarchyInterpreter::Create(&import_data_.value(), ti,
                                            test_proc->proc(), options);
  }

  absl::StatusOr<std::unique_ptr<ProcHierarchyInterpreter>> Create(
      ProcDef* test_proc, const BytecodeInterpreterOptions& options) {
    return ProcHierarchyInterpreter::Create(&import_data_.value(),
                                            tm_->type_info, test_proc, options);
  }

  absl::Status Run(ProcHierarchyInterpreter& hierarchy_interpreter,
                   const BytecodeInterpreterOptions& /*options*/) {
    // There should be a single top config argument: a reference to the
    // terminator channel. Determine the actual channel object.
    XLS_RET_CHECK_EQ(hierarchy_interpreter.InterfaceArgs().size(), 1);
    std::string terminal_channel_name =
        std::string{hierarchy_interpreter.GetInterfaceChannelName(0)};

    // Run until a single output appears in the terminal channel.
    XLS_RETURN_IF_ERROR(
        hierarchy_interpreter.TickUntilOutput({{terminal_channel_name, 1}})
            .status());

    return absl::OkStatus();
  }

  absl::StatusOr<ProcInstance*> GetProcInstance(
      ProcHierarchyInterpreter& hierarchy_interpreter,
      std::string_view proc_id_str) {
    for (ProcInstance& instance : hierarchy_interpreter.proc_instances()) {
      const std::optional<ProcId>& pid = instance.interpreter().proc_id();
      if (pid.has_value() && pid->ToString() == proc_id_str) {
        return &instance;
      }
    }
    return absl::NotFoundError(
        absl::StrFormat("No ProcInstance found with ProcId `%s`", proc_id_str));
  }

  // Sender's two sends must appear atomic within a tick; mid-tick yield lets
  // the tester interleave between them.
  static constexpr std::string_view kSendAtomicityProgram = R"(
proc Sender {
    ready_s: chan<bool> out;
    data_s: chan<u32> out;
    config(ready_s: chan<bool> out, data_s: chan<u32> out) { (ready_s, data_s) }
    init { () }
    next(state: ()) {
        let tok = send(join(), ready_s, true);
        send(tok, data_s, u32:42);
    }
}

#[test_proc]
proc SendAtomicityTest {
    terminator: chan<bool> out;
    ready_r: chan<bool> in;
    data_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (ready_s, ready_r) = chan<bool, u32:1>("ready");
        let (data_s, data_r) = chan<u32, u32:1>("data");
        spawn Sender(ready_s, data_s);
        (terminator, ready_r, data_r)
    }

    init { u32:0 }

    next(tick: u32) {
        let (tok, _, ready) = recv_non_blocking(join(), ready_r, false);
        let (tok, _, data_valid) = recv_non_blocking(tok, data_r, u32:0);
        assert_eq(!ready || data_valid, true);
        send_if(tok, terminator, tick == u32:9, true);
        tick + u32:1
    }
})";

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
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(GetProcInstance(*interpreter, "tester_proc:0")
                  .value()
                  ->events()
                  .GetTraceMessageStrings(),
              testing::ElementsAre(
                  "Sent data on channel `tester_proc::data_out`:\n  u32:42",
                  "Received data on channel `tester_proc::data_in`:\n  u32:43",
                  "Sent data on channel `tester_proc::data_out`:\n  u32:100",
                  "Received data on channel `tester_proc::data_in`:\n  u32:101",
                  "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc->incrementer:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:42",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:43",
                           "Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:100",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:101"));
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
  auto options =
      BytecodeInterpreterOptions().trace_channels(true).format_preference(
          FormatPreference::kHex);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out`:\n  u32:0x2a",
          "Received data on channel `tester_proc::data_in`:\n  u32:0x2b",
          "Sent data on channel `tester_proc::data_out`:\n  u32:0x64",
          "Received data on channel `tester_proc::data_in`:\n  u32:0x65",
          "Sent data on channel `tester_proc::terminator`:\n  u1:0x1"));
  EXPECT_THAT(GetProcInstance(*interpreter, "tester_proc->incrementer:0")
                  .value()
                  ->events()
                  .GetTraceMessageStrings(),
              testing::ElementsAre(
                  "Received data on channel "
                  "`tester_proc->incrementer#0::in_ch`:\n  u32:0x2a",
                  "Sent data on channel "
                  "`tester_proc->incrementer#0::out_ch`:\n  u32:0x2b",
                  "Received data on channel "
                  "`tester_proc->incrementer#0::in_ch`:\n  u32:0x64",
                  "Sent data on channel "
                  "`tester_proc->incrementer#0::out_ch`:\n  u32:0x65"));
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
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc->incrementer:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:1"));
  EXPECT_THAT(GetProcInstance(*interpreter, "tester_proc:0")
                  .value()
                  ->events()
                  .GetTraceMessageStrings(),
              testing::ElementsAre(
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
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_EQ(GetProcInstance(*interpreter, "tester_proc:0")
                .value()
                ->events()
                .GetTraceMessageStrings()[0],
            R"(Sent data on channel `tester_proc::data_out`:
  Foo {
    a: u32:42,
    b: u16:100
})");
  EXPECT_EQ(GetProcInstance(*interpreter, "tester_proc->incrementer:0")
                .value()
                ->events()
                .GetTraceMessageStrings()[0],
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
  data_out: chan<u32>[2] out;
  data_in: chan<u32>[2] in;
  terminator: chan<bool> out;

  init { () }

  config(terminator: chan<bool> out) {
    let (input_p, input_c) = chan<u32>[2]("input");
    let (output_p, output_c) = chan<u32>[2]("output");
    spawn incrementer(input_c[0], output_p[0]);
    spawn incrementer(input_c[1], output_p[1]);
    (input_p, output_c, terminator)
  }

  next(state: ()) {
    let tok = send(join(), data_out[0], u32:42);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(join(), data_out[1], u32:42);
    let (tok, result) = recv(tok, data_in[1]);

    let tok = send(tok, data_out[0], u32:100);
    let (tok, result) = recv(tok, data_in[0]);

    let tok = send(tok, data_out[1], u32:100);
    let (tok, result) = recv(tok, data_in[1]);

    let tok = send(tok, terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "tester_proc"));
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out[0]`:\n  u32:42",
          "Received data on channel `tester_proc::data_in[0]`:\n  u32:43",
          "Sent data on channel `tester_proc::data_out[1]`:\n  u32:42",
          "Received data on channel `tester_proc::data_in[1]`:\n  u32:43",
          "Sent data on channel `tester_proc::data_out[0]`:\n  u32:100",
          "Received data on channel `tester_proc::data_in[0]`:\n  u32:101",
          "Sent data on channel `tester_proc::data_out[1]`:\n  u32:100",
          "Received data on channel `tester_proc::data_in[1]`:\n  u32:101",
          "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc->incrementer:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:42",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:43",
                           "Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:100",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:101"));
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
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre(
          "Sent data on channel `tester_proc::data_out`:\n  u32:42",
          "Received data on channel `tester_proc::data_in`:\n  u32:43",
          "Sent data on channel `tester_proc::data_out`:\n  u32:100",
          "Received data on channel `tester_proc::data_in`:\n  u32:101",
          "Sent data on channel `tester_proc::data_out_2`:\n  u32:43",
          "Received data on channel `tester_proc::data_in_2`:\n  u32:44",
          "Sent data on channel `tester_proc::data_out_2`:\n  u32:101",
          "Received data on channel `tester_proc::data_in_2`:\n  u32:102",
          "Sent data on channel `tester_proc::terminator`:\n  u1:1"));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc->incrementer:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:42",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:43",
                           "Received data on channel "
                           "`tester_proc->incrementer#0::in_ch`:\n  u32:100",
                           "Sent data on channel "
                           "`tester_proc->incrementer#0::out_ch`:\n  u32:101"));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "tester_proc->incrementer:1")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Received data on channel "
                           "`tester_proc->incrementer#1::in_ch`:\n  u32:43",
                           "Sent data on channel "
                           "`tester_proc->incrementer#1::out_ch`:\n  u32:44",
                           "Received data on channel "
                           "`tester_proc->incrementer#1::in_ch`:\n  u32:101",
                           "Sent data on channel "
                           "`tester_proc->incrementer#1::out_ch`:\n  u32:102"));
}

TEST_F(ProcHierarchyInterpreterTest, AssertsInProcsShowHierarchyInErrors) {
  constexpr std::string_view kProgram[4] = {R"(
#[test_proc]
proc TestAssert {
    terminator: chan<bool> out;

    init {}
    config(terminator: chan<bool> out) { (terminator,) }
    next(state: ()) {
        assert!(false, "assert_label");
        send(join(), terminator, true);
    }
})",
                                            R"(
#[test_proc]
proc TestFail {
    terminator: chan<bool> out;

    init {}
    config(terminator: chan<bool> out) { (terminator,) }
    next(state: ()) {
        fail!("fail_label", ());
        send(join(), terminator, true);
    }
})",
                                            R"(
#[test_proc]
proc TestAssertLt {
    terminator: chan<bool> out;

    init {}
    config(terminator: chan<bool> out) { (terminator,) }
    next(state: ()) {
        assert_lt(u32:1, u32:0);
        send(join(), terminator, true);
    }
})",
                                            R"(
#[test_proc]
proc TestAssertEq {
    terminator: chan<bool> out;

    init {}
    config(terminator: chan<bool> out) { (terminator,) }
    next(state: ()) {
        assert_eq(u32:100, u32:99);
        send(join(), terminator, true);
    }
})"};

  for (int i = 0; i < 4; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        auto temp_file, TempFile::CreateWithContent(kProgram[i], "_test.x"));
    constexpr std::string_view kModuleName = "test";
    ParseAndTestOptions options;
    ::testing::internal::CaptureStderr();
    XLS_ASSERT_OK_AND_ASSIGN(
        TestResultData result,
        ParseAndTest(kProgram[i], kModuleName, std::string{temp_file.path()},
                     options));
    std::string stdcerr(::testing::internal::GetCapturedStderr());
    EXPECT_EQ(result.result(), TestResult::kSomeFailed);
    EXPECT_THAT(stdcerr, HasSubstr("called from"));
  }
}

TEST_F(ProcHierarchyInterpreterTest,
       AssertsInFunctionsShowNoHierarchyInErrors) {
  constexpr std::string_view kProgram = R"(
#[test]
fn test_assert() {
    assert!(false, "assert_label");
}

#[test]
fn test_fail() {
    fail!("fail_label", ());
}

#[test]
fn test_assert_eq() {
    assert_eq(u32:100, u32:99);
}

#[test]
fn test_assert_lt() {
    assert_lt(u32:1, u32:0);
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
  EXPECT_THAT(stdcerr, Not(HasSubstr("called from")));
}

TEST_F(ProcHierarchyInterpreterTest, SimpleExplicitStateAccess) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

#[test_proc]
proc Counter {
    terminator: chan<bool> out;

    init { 0 }

    config(terminator: chan<bool> out) {
        (terminator,)
    }

    next(i: u32) {
        let old_value = read(i);
        send_if(token(), terminator, old_value > 1, true);
        write(i, old_value + 1);
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

TEST_F(ProcHierarchyInterpreterTest, MultiElementExplicitStateAccess) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

#[test_proc]
proc Counter {
    terminator: chan<bool> out;

    init { (0, 1) }

    config(terminator: chan<bool> out) {
        (terminator,)
    }

    next(i: u32, j: u32) {
        let old_i = read(i);
        let old_j = read(j);
        send_if(token(), terminator, old_j > 5 && old_j == old_i + 1, true);
        write(i, old_i + 1);
        write(j, old_j + 1);
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

TEST_F(ProcHierarchyInterpreterTest, SimpleProcDef) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

#[test]
proc Counter {
  terminator: chan<bool> out,
  i: u32,
}

impl Counter {
  fn new(terminator: chan<bool> out) -> Self {
    Counter { terminator: terminator, i: 0 }
  }

  fn next(self) {
    let old_value = read(self.i);
    send_if(token(), self.terminator, old_value > 1, true);
    write(self.i, old_value + 1);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, FailingProcDef) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

#[test]
proc Counter {
  terminator: chan<bool> out,
  i: u32,
}

impl Counter {
  fn new(terminator: chan<bool> out) -> Self {
    Counter { terminator: terminator, i: 0 }
  }

  fn next(self) {
    let old_value = read(self.i);
    assert_eq(old_value, 1);
    send_if(token(), self.terminator, old_value > 1, true);
    write(self.i, old_value + 1);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kSomeFailed);
}

TEST_F(ProcHierarchyInterpreterTest, MultiLevelProcDef) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

proc Counter {
  ch_out: chan<u32> out,
  i: u32,
}

impl Counter {
  fn new(ch_out: chan<u32> out) -> Self {
    Counter { ch_out: ch_out, i: 0 }
  }

  fn next(self) {
    let old_value = read(self.i);
    send(token(), self.ch_out, old_value);
    write(self.i, old_value + 1);
  }
}

#[test]
proc CounterTest {
  terminator: chan<bool> out,
  ch_in: chan<u32> in,
}

impl CounterTest {
  fn new(terminator: chan<bool> out) -> Self {
    let (counter_out, counter_in) = chan<u32>("counter");
    Counter::new(counter_out).spawn();
    CounterTest { terminator: terminator, ch_in: counter_in }
  }

  fn next(self) {
    let (_, value) = recv(token(), self.ch_in);
    send_if(token(), self.terminator, value > 1, true);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, ParametricProcDef) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]
#![feature(generics)]

proc Counter<T: type> {
  ch_out: chan<T> out,
  i: T,
}

impl Counter {
  fn new(ch_out: chan<T> out) -> Self {
    Counter<T> { ch_out: ch_out, i: 0 }
  }

  fn next(self) {
    let old_value = read(self.i);
    send(token(), self.ch_out, old_value);
    write(self.i, old_value + 1);
  }
}

#[test]
proc CounterTest {
  terminator: chan<bool> out,
  ch_in0: chan<u32> in,
  ch_in1: chan<s16> in,
}

impl CounterTest {
  fn new(terminator: chan<bool> out) -> Self {
    let (counter_out0, counter_in0) = chan<u32>("counter0");
    let (counter_out1, counter_in1) = chan<s16>("counter1");
    Counter<u32>::new(counter_out0).spawn();
    Counter<s16>::new(counter_out1).spawn();
    CounterTest {
      terminator: terminator,
      ch_in0: counter_in0,
      ch_in1: counter_in1
    }
  }

  fn next(self) {
    let (_, value0) = recv(token(), self.ch_in0);
    let (_, value1) = recv(token(), self.ch_in1);
    send_if(token(), self.terminator, value0 > 2 && value1 > 2, true);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, MultiLevelProcDefWithForwardedChannel) {
  constexpr std::string_view kProgram = R"(
#![feature(explicit_state_access)]

proc Counter {
  ch_out: chan<u32> out,
  i: u32,
}

impl Counter {
  fn new(ch_out: chan<u32> out) -> Self {
    Counter { ch_out: ch_out, i: 0 }
  }

  fn next(self) {
    let old_value = read(self.i);
    send(token(), self.ch_out, old_value);
    write(self.i, old_value + 1);
  }
}

proc CounterDelegate {}

impl CounterDelegate {
  fn new(ch_out: chan<u32> out) -> Self {
    Counter::new(ch_out).spawn();
    CounterDelegate {}
  }
}

#[test]
proc CounterTest {
  terminator: chan<bool> out,
  ch_in: chan<u32> in,
}

impl CounterTest {
  fn new(terminator: chan<bool> out) -> Self {
    let (counter_out, counter_in) = chan<u32>("counter");
    CounterDelegate::new(counter_out).spawn();
    CounterTest { terminator: terminator, ch_in: counter_in }
  }

  fn next(self) {
    let (_, value) = recv(token(), self.ch_in);
    send_if(token(), self.terminator, value > 1, true);
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, ProcDefSpawnWithChannelArray) {
  constexpr std::string_view kProgram = R"(
proc SomeProc {
  ins: chan<u32>[2] in,
}

impl SomeProc {
  fn new(ins: chan<u32>[2] in) -> Self {
    SomeProc { ins }
  }

  fn next(self) {
    const for (i, _) in u32:0..2 {
      let (_, v) = recv(token(), self.ins[i]);
      trace_fmt!("recv: {}", v);
    }(());
  }
}

#[test]
proc TopProc {
  terminator: chan<bool> out,
  outs: chan<u32>[2] out,
}

impl TopProc {
  fn new(terminator: chan<bool> out) -> Self {
    let (outs, ins) = chan<u32>[2]("ins_outs");
    SomeProc::new(ins).spawn();
    TopProc { terminator, outs }
  }

  fn next(self) {
    let t = const for (i, t) in u32:0..2 {
      send(t, self.outs[i], i)
    }(token());
    send(t, self.terminator, true);
  }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, TraceProcDefChannelArray) {
  constexpr std::string_view kProgram = R"(
proc Incrementer {
  in_ch: chan<u32> in,
  out_ch: chan<u32> out,
}

impl Incrementer {
  fn new(in_ch: chan<u32> in,
         out_ch: chan<u32> out) -> Self {
    Incrementer { in_ch, out_ch }
  }

  fn next(self) {
    let (tok, i) = recv(join(), self.in_ch);
    let tok = send(tok, self.out_ch, i + u32:1);
  }
}

#[test]
proc TestProc {
  data_out: chan<u32>[2] out,
  data_in: chan<u32>[2] in,
  terminator: chan<bool> out,
}

impl TestProc {
  fn new(terminator: chan<bool> out) -> Self {
    let (input_p, input_c) = chan<u32>[2]("input");
    let (output_p, output_c) = chan<u32>[2]("output");
    Incrementer::new(input_c[0], output_p[0]).spawn();
    Incrementer::new(input_c[1], output_p[1]).spawn();
    TestProc { data_out: input_p, data_in: output_c, terminator: terminator }
  }

  fn next(self) {
    let tok = send(join(), self.data_out[0], u32:42);
    let (tok, result) = recv(tok, self.data_in[0]);

    let tok = send(join(), self.data_out[1], u32:42);
    let (tok, result) = recv(tok, self.data_in[1]);

    let tok = send(tok, self.data_out[0], u32:100);
    let (tok, result) = recv(tok, self.data_in[0]);

    let tok = send(tok, self.data_out[1], u32:100);
    let (tok, result) = recv(tok, self.data_in[1]);

    let tok = send(tok, self.terminator, true);
 }
})";

  XLS_ASSERT_OK_AND_ASSIGN(ProcDef * test_proc,
                           ParseAndGetTestProcDef(kProgram, "TestProc"));
  auto options = BytecodeInterpreterOptions().trace_channels(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
  EXPECT_THAT(GetProcInstance(*interpreter, "TestProc:0")
                  .value()
                  ->events()
                  .GetTraceMessageStrings(),
              testing::ElementsAre(
                  "Sent data on channel `TestProc::data_out[0]`:\n  u32:42",
                  "Received data on channel `TestProc::data_in[0]`:\n  u32:43",
                  "Sent data on channel `TestProc::data_out[1]`:\n  u32:42",
                  "Received data on channel `TestProc::data_in[1]`:\n  u32:43",
                  "Sent data on channel `TestProc::data_out[0]`:\n  u32:100",
                  "Received data on channel `TestProc::data_in[0]`:\n  u32:101",
                  "Sent data on channel `TestProc::data_out[1]`:\n  u32:100",
                  "Received data on channel `TestProc::data_in[1]`:\n  u32:101",
                  "Sent data on channel `TestProc::terminator`:\n  u1:1"));
  EXPECT_THAT(
      GetProcInstance(*interpreter, "TestProc->Incrementer:0")
          .value()
          ->events()
          .GetTraceMessageStrings(),
      testing::ElementsAre("Received data on channel "
                           "`TestProc->Incrementer#0::in_ch`:\n  u32:42",
                           "Sent data on channel "
                           "`TestProc->Incrementer#0::out_ch`:\n  u32:43",
                           "Received data on channel "
                           "`TestProc->Incrementer#0::in_ch`:\n  u32:100",
                           "Sent data on channel "
                           "`TestProc->Incrementer#0::out_ch`:\n  u32:101"));
}

TEST_F(ProcHierarchyInterpreterTest, ProcDefDealingOutBothEndsOfChannel) {
  std::string_view kProgram = R"(
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

proc C {
    c_in: chan<u32> in,
    terminator: chan<bool> out,
    i: u32,
}

impl C {
    fn new(c_in: chan<u32> in, terminator: chan<bool> out) -> Self {
        C { c_in: c_in, terminator: terminator, i: 0 }
    }
    fn next(self) {
        let last_i = read(self.i);
        let (tok1, e) = recv(join(), self.c_in);
        write(self.i, e + last_i);
        send_if(token(), self.terminator, last_i > 1, true);
    }
}

#[test]
proc TestProc {}

impl TestProc {
    fn new(terminator: chan<bool> out) -> Self {
        let (c_out, c_in) = chan<u32>("my_chan");
        P::new(c_out).spawn();
        let c = C::new(c_in, terminator);
        c.spawn();
        TestProc {}
    }

    fn next(self) {}
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto temp_file,
                           TempFile::CreateWithContent(kProgram, "_test.x"));
  ParseAndTestOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      TestResultData result,
      ParseAndTest(kProgram, "test", std::string{temp_file.path()}, options));
  EXPECT_EQ(result.result(), TestResult::kAllPassed);
}

TEST_F(ProcHierarchyInterpreterTest, BoundedFifoBlocksAndResumes) {
  // Depth-1 channel forces the second send (via send_if) to block until
  // drained; checks stack save/restore keeps values in order.
  constexpr std::string_view kProgram = R"(
proc Passthrough {
    data_r: chan<u32> in;
    result_s: chan<u32> out;

    config(data_r: chan<u32> in, result_s: chan<u32> out) { (data_r, result_s) }

    init { () }

    next(state: ()) {
        let (tok, data) = recv(join(), data_r);
        send(tok, result_s, data);
    }
}

#[test_proc]
proc BoundedFifoTest {
    terminator: chan<bool> out;
    data_s: chan<u32> out;
    result_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (data_s, data_r) = chan<u32, u32:1>("data");
        let (result_s, result_r) = chan<u32, u32:1>("result");
        spawn Passthrough(data_r, result_s);
        (terminator, data_s, result_r)
    }

    init { }

    next(_: ()) {
        let tok = join();
        let tok = send(tok, data_s, u32:1);
        let tok = send_if(tok, data_s, true, u32:2);

        let (tok, result) = recv(tok, result_r);
        assert_eq(result, u32:1);
        let (tok, result) = recv(tok, result_r);
        assert_eq(result, u32:2);

        send(tok, terminator, true);
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "BoundedFifoTest"));
  auto options =
      BytecodeInterpreterOptions().max_ticks(20).simulate_bounded_fifos(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
}

TEST_F(ProcHierarchyInterpreterTest, ImplStyleBoundedFifoBlocksAndResumes) {
  // Same as above, but impl-style procs resolve member-channel depth via a
  // separate path (GetFifoDepth), so it needs its own coverage.
  constexpr std::string_view kProgram = R"(
proc Passthrough {
  data_in: chan<u32> in,
  result_out: chan<u32> out,
}

impl Passthrough {
  fn new(data_in: chan<u32> in, result_out: chan<u32> out) -> Self {
    Passthrough { data_in, result_out }
  }

  fn next(self) {
    let (tok, data) = recv(join(), self.data_in);
    send(tok, self.result_out, data);
  }
}

#[test]
proc BoundedFifoProcDefTest {
  terminator: chan<bool> out,
  data_out: chan<u32> out,
  result_in: chan<u32> in,
}

impl BoundedFifoProcDefTest {
  fn new(terminator: chan<bool> out) -> Self {
    let (data_out, data_in) = chan<u32, u32:1>("data");
    let (result_out, result_in) = chan<u32, u32:1>("result");
    Passthrough::new(data_in, result_out).spawn();
    BoundedFifoProcDefTest { terminator, data_out, result_in }
  }

  fn next(self) {
    let tok = send(join(), self.data_out, u32:1);
    let tok = send(tok, self.data_out, u32:2);

    let (tok, result) = recv(tok, self.result_in);
    assert_eq(result, u32:1);
    let (tok, result) = recv(tok, self.result_in);
    assert_eq(result, u32:2);

    send(tok, self.terminator, true);
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      ProcDef * test_proc,
      ParseAndGetTestProcDef(kProgram, "BoundedFifoProcDefTest"));
  auto options =
      BytecodeInterpreterOptions().max_ticks(20).simulate_bounded_fifos(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
}

TEST_F(ProcHierarchyInterpreterTest, FifoDepthAppliesPerChannelArrayElement) {
  // Depth on a channel array must apply per-element, not to the array as a
  // whole: element 0 blocks/resumes, element 1 never blocks.
  constexpr std::string_view kProgram = R"(
proc Passthrough {
    in_ch: chan<u32> in;
    out_ch: chan<u32> out;

    init { () }

    config(in_ch: chan<u32> in, out_ch: chan<u32> out) {
        (in_ch, out_ch)
    }
    next(_: ()) {
        let (tok, v) = recv(join(), in_ch);
        send(tok, out_ch, v);
    }
}

#[test_proc]
proc FifoDepthArrayTest {
    terminator: chan<bool> out;
    data_out: chan<u32>[2] out;
    data_in: chan<u32>[2] in;

    init { () }

    config(terminator: chan<bool> out) {
        let (input_p, input_c) = chan<u32, u32:1>[2]("input");
        let (output_p, output_c) = chan<u32, u32:1>[2]("output");
        spawn Passthrough(input_c[0], output_p[0]);
        spawn Passthrough(input_c[1], output_p[1]);
        (terminator, input_p, output_c)
    }

    next(_: ()) {
        let tok = join();
        let tok = send(tok, data_out[0], u32:1);
        let tok = send(tok, data_out[0], u32:2);
        let tok = send(tok, data_out[1], u32:100);

        let (tok, r0) = recv(tok, data_in[0]);
        assert_eq(r0, u32:1);
        let (tok, r0) = recv(tok, data_in[0]);
        assert_eq(r0, u32:2);
        let (tok, r1) = recv(tok, data_in[1]);
        assert_eq(r1, u32:100);

        send(tok, terminator, true);
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(TestProc * test_proc,
                           ParseAndGetTestProc(kProgram, "FifoDepthArrayTest"));
  auto options =
      BytecodeInterpreterOptions().max_ticks(20).simulate_bounded_fifos(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));
  XLS_ASSERT_OK(Run(*interpreter, options));
}

TEST_F(ProcHierarchyInterpreterTest, ProcScheduleSeedIsDeterministic) {
// PriorityMux's output depends on proc execution order, which makes this a
// good scenario for checking a fixed schedule seed reproduces exactly.
  constexpr std::string_view kProgram = R"(
pub proc Writer<BASE_COUNT: u32> {
    out_s: chan<u32> out;
    config(out_s: chan<u32> out) { (out_s,) }
    init { u32:0 }
    next(count: u32) {
        send(join(), out_s, BASE_COUNT + count);
        count + u32:1
    }
}

pub proc PriorityMux {
    a_r: chan<u32> in;
    b_r: chan<u32> in;
    out_s: chan<u32> out;
    config(a_r: chan<u32> in, b_r: chan<u32> in, out_s: chan<u32> out) {
        (a_r, b_r, out_s)
    }
    init { () }
    next(state: ()) {
        let (tok, val_a, got_a) = recv_non_blocking(join(), a_r, u32:0);
        let tok = if got_a {
            send(tok, out_s, val_a)
        } else {
            let (tok, val_b) = recv(tok, b_r);
            send(tok, out_s, val_b)
        };
    }
}

#[test_proc]
proc PriorityMuxTest {
    terminator: chan<bool> out;
    out_r: chan<u32> in;

    config(terminator: chan<bool> out) {
        let (a_s, a_r) = chan<u32>("writer_a");
        let (b_s, b_r) = chan<u32>("writer_b");
        let (out_s, out_r) = chan<u32>("mux_out");
        spawn Writer<u32:0>(a_s);
        spawn Writer<u32:100>(b_s);
        spawn PriorityMux(a_r, b_r, out_s);
        (terminator, out_r)
    }

    init { u32:0 }

    next(count: u32) {
        let (tok, val) = recv_if(join(), out_r, count < u32:3, u32:0);
        assert_eq(val < u32:10, true);
        let done = count == u32:2;
        send_if(tok, terminator, done, true);
        if done { u32:0 } else { count + u32:1 }
    }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      TestProc * test_proc,
      ParseAndGetTestProc(kProgram, "PriorityMuxTest"));
  auto options =
      BytecodeInterpreterOptions().max_ticks(20).proc_schedule_seed(0);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcHierarchyInterpreter> first,
                           Create(test_proc, options));
  absl::Status first_status = Run(*first, options);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcHierarchyInterpreter> second,
                           Create(test_proc, options));
  absl::Status second_status = Run(*second, options);

  EXPECT_EQ(first_status, second_status);
}

TEST_F(ProcHierarchyInterpreterTest, MidTickYieldInterleavesChannelOps) {
  XLS_ASSERT_OK_AND_ASSIGN(
      TestProc * test_proc,
      ParseAndGetTestProc(kSendAtomicityProgram, "SendAtomicityTest"));

  auto default_options = BytecodeInterpreterOptions().max_ticks(20);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> default_interpreter,
      Create(test_proc, default_options));
  XLS_ASSERT_OK(Run(*default_interpreter, default_options));

  auto yield_options =
      BytecodeInterpreterOptions().max_ticks(20).mid_tick_yield(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> yield_interpreter,
      Create(test_proc, yield_options));
  EXPECT_THAT(Run(*yield_interpreter, yield_options),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("were not equal")));
}

// Yield + schedule seed together hit the random (not FIFO) reinsertion path;
// must stay deterministic for a fixed seed.
TEST_F(ProcHierarchyInterpreterTest,
       MidTickYieldWithScheduleSeedIsDeterministic) {
  XLS_ASSERT_OK_AND_ASSIGN(
      TestProc * test_proc,
      ParseAndGetTestProc(kSendAtomicityProgram, "SendAtomicityTest"));
  auto options = BytecodeInterpreterOptions()
                     .max_ticks(20)
                     .mid_tick_yield(true)
                     .proc_schedule_seed(0);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcHierarchyInterpreter> first,
                           Create(test_proc, options));
  absl::Status first_status = Run(*first, options);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ProcHierarchyInterpreter> second,
                           Create(test_proc, options));
  absl::Status second_status = Run(*second, options);

  EXPECT_EQ(first_status, second_status);
}

TEST_F(ProcHierarchyInterpreterTest, DeadlockDiagnosticsIncludeBlockedSend) {
  // Deadlock message must mention blocked-on-send procs too, not just
  // blocked-on-receive.
  constexpr std::string_view kProgram = R"(
proc StuckSender {
  out_ch: chan<u32> out,
}
impl StuckSender {
  fn new(out_ch: chan<u32> out) -> Self { StuckSender { out_ch } }
  fn next(self) {
    let tok = send(join(), self.out_ch, u32:1);
    send(tok, self.out_ch, u32:2);
  }
}

#[test]
proc DeadlockOnSendTest {
  terminator: chan<bool> out,
  never_in: chan<u32> in,
}
impl DeadlockOnSendTest {
  fn new(terminator: chan<bool> out) -> Self {
    let (data_out, _) = chan<u32, u32:1>("data");
    StuckSender::new(data_out).spawn();
    let (_, never_in) = chan<u32>("never");
    DeadlockOnSendTest { terminator, never_in }
  }
  fn next(self) {
    let (tok, _) = recv(join(), self.never_in);
    send(tok, self.terminator, true);
  }
})";
  XLS_ASSERT_OK_AND_ASSIGN(
      ProcDef * test_proc,
      ParseAndGetTestProcDef(kProgram, "DeadlockOnSendTest"));
  auto options =
      BytecodeInterpreterOptions().max_ticks(20).simulate_bounded_fifos(true);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ProcHierarchyInterpreter> interpreter,
      Create(test_proc, options));

  std::string terminal_channel_name{interpreter->GetInterfaceChannelName(0)};
  absl::Status status =
      interpreter->TickUntilOutput({{terminal_channel_name, 1}}).status();
  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       AllOf(HasSubstr("is blocked on receive on channel"),
                             HasSubstr("is blocked on send on channel"))));
}

}  // namespace
}  // namespace xls::dslx
