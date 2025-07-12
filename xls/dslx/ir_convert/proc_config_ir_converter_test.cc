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
#include "xls/dslx/ir_convert/proc_config_ir_converter.h"

#include <memory>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/ir_convert/test_utils.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

namespace m = ::xls::op_matchers;

void ExpectIr(std::string_view got) {
  return ::xls::dslx::ExpectIr(got, TestName(),
                               "proc_config_ir_converter_test");
}

PackageConversionData MakeConversionData(std::string_view n) {
  return {.package = std::make_unique<Package>(n)};
}

absl::Status ParseAndAcceptWithConverter(std::string_view module_text,
                                         PackageConversionData& conv,
                                         const ProcId& proc_id,
                                         ProcConversionData& proc_data) {
  auto import_data = CreateImportDataForTest();

  ParametricEnv bindings;

  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ParseAndTypecheck(module_text, "test_module.x",
                                         "test_module", &import_data));

  XLS_ASSIGN_OR_RETURN(
      Function * f, tm.module->GetMemberOrError<Function>("test_proc.config"));

  ChannelScope channel_scope(&conv, &import_data, ConvertOptions{});
  channel_scope.EnterFunctionContext(tm.type_info, bindings);
  ProcConfigIrConverter converter(f, tm.type_info, &import_data, &proc_data,
                                  &channel_scope, bindings, proc_id);
  return f->Accept(&converter);
}

TEST(ProcConfigIrConverterTest, BasicConversion) {
  constexpr std::string_view kModule = R"(
proc test_proc {
  c: chan<u32> in;
  x: u32;
  init { u32: 0 }
  config(c: chan<u32> in, ham_sandwich: u32) {
    (c, ham_sandwich)
  }
  next(y: u32) {
    let y = y + x;
    y
  }
}

proc main {
  c: chan<u32> out;
  init { () }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn test_proc(c, u32:7);
    (p,)
  }
  next(state: ()) {
    ()
  }
}
)";

  PackageConversionData conv = MakeConversionData("the_package");
  StreamingChannel channel("the_channel", /*id=*/0, ChannelOps::kSendReceive,
                           conv.package->GetBitsType(32), {}, ChannelConfig(),
                           FlowControl::kNone,
                           ChannelStrictness::kProvenMutuallyExclusive);
  ProcConversionData proc_data;
  ProcId proc_id;
  proc_data.id_to_config_args[proc_id].push_back(&channel);
  proc_data.id_to_config_args[proc_id].push_back(Value(UBits(8, 32)));
  XLS_EXPECT_OK(ParseAndAcceptWithConverter(kModule, conv, proc_id, proc_data));
}

TEST(ProcConfigIrConverterTest, CatchesMissingArgMap) {
  constexpr std::string_view kModule = R"(
proc test_proc {
  c: chan<u32> in;
  init { () }
  config(c: chan<u32> in) {
    (c,)
  }
  next(state: ()) {
    ()
  }
}

proc main {
  c: chan<u32> out;
  init { () }
  config() {
    let (p, c) = chan<u32>("my_chan");
    spawn test_proc(c);
    (p,)
  }
  next(state: ()) {
    ()
  }
}
)";

  ProcId proc_id;
  ProcConversionData proc_data;
  PackageConversionData conv = MakeConversionData("the_package");
  EXPECT_THAT(ParseAndAcceptWithConverter(kModule, conv, proc_id, proc_data),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not found in arg mapping")));
}

// TODO: https://github.com/google/xls/issues/2078 - Re-enable this test after
// proc-scoped channels can be natively generated.
TEST(ProcConfigIrConverterTest,
     DISABLED_ConvertsParametricExpressionForInternalChannelFifoDepth) {
  constexpr std::string_view kModule = R"(
proc passthrough {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    (c_in, c_out)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), c_in);
    let tok = send(tok, c_out, data);
    ()
  }
}

proc test_proc<X: u32, Y: u32> {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    let (p, c) = chan<u32, {X + Y}>("my_chan");
    spawn passthrough(c_in, p);
    spawn passthrough(c, c_out);
    (c_in, c_out)
  }
  next(state: ()) {
    ()
  }
}

proc main {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init { () }
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    spawn test_proc<u32:3, u32:4>(c_in, c_out);
    (c_in, c_out)
  }
  next(state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(
          tm.module, &import_data,
          ConvertOptions{.lower_to_proc_scoped_channels = true}));

  EXPECT_THAT(conv.package->channels(), Contains(m::Channel("my_chan")));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel,
                           conv.package->GetChannel("my_chan"));
  ASSERT_EQ(channel->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(down_cast<StreamingChannel*>(channel)->GetFifoDepth(), 7);
}

// TODO: https://github.com/google/xls/issues/2078 - Re-enable this test after
// proc-scoped channels can be natively generated.
TEST(ProcConfigIrConverterTest,
     DISABLED_ConvertMultipleInternalChannelsWithSameNameInSameProc) {
  constexpr std::string_view kModule = R"(
proc main {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  internal_in0: chan<u32> in;
  internal_out0: chan<u32> out;
  internal_in1: chan<u32> in;
  internal_out1: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    let (p0, c0) = chan<u32>("my_chan");
    let (p1, c1) = chan<u32>("my_chan");
    (c_in, c_out, c0, p0, c1, p1)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), c_in);
    let tok = send(tok, internal_out0, data);
    let (tok, data) = recv(tok, internal_in0);
    let tok = send(tok, internal_out1, data);
    let (tok, data) = recv(tok, internal_in1);
    let tok = send(tok, c_out, data);
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(
          tm.module, &import_data,
          ConvertOptions{.lower_to_proc_scoped_channels = true}));

  EXPECT_THAT(conv.package->channels(),
              AllOf(Contains(m::Channel("my_chan")),
                    Contains(m::Channel("my_chan__1"))));
}

// TODO: https://github.com/google/xls/issues/2078 - Re-enable this test after
// proc-scoped channels can be natively generated.
TEST(ProcConfigIrConverterTest, DISABLED_ChannelArrayDestructureWithWildcard) {
  constexpr std::string_view kModule = R"(
  proc SomeProc {
      some_chan_array: chan<u32>[4] in;

      config() {
          let (_, new_chan_array_r) = chan<u32>[4]("the_chan_array");
          (new_chan_array_r,)
      }

      init {  }

      next(state: ()) {  }
  }
)";
  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(
          tm.module, &import_data,
          ConvertOptions{.verify_ir = false,
                         .lower_to_proc_scoped_channels = true}));
  EXPECT_THAT(conv.package->channels(),
              UnorderedElementsAre(m::Channel("the_chan_array__0"),
                                   m::Channel("the_chan_array__1"),
                                   m::Channel("the_chan_array__2"),
                                   m::Channel("the_chan_array__3")));
}

// TODO: https://github.com/google/xls/issues/2078 - Re-enable this test after
// proc-scoped channels can be natively generated.
TEST(ProcConfigIrConverterTest,
     DISABLED_ChannelArrayDestructureWithRestOfTuple) {
  constexpr std::string_view kModule = R"(
  proc SomeProc {
      some_chan_array: chan<u32>[4] in;

      config() {
          let (.., new_chan_array_r) = chan<u32>[4]("the_chan_array");
          (new_chan_array_r,)
      }

      init {  }

      next(state: ()) {  }
  }
)";
  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(
          tm.module, &import_data,
          ConvertOptions{.verify_ir = false,
                         .lower_to_proc_scoped_channels = true}));
  EXPECT_THAT(conv.package->channels(),
              UnorderedElementsAre(m::Channel("the_chan_array__0"),
                                   m::Channel("the_chan_array__1"),
                                   m::Channel("the_chan_array__2"),
                                   m::Channel("the_chan_array__3")));
}

// TODO: https://github.com/google/xls/issues/2078 - Re-enable this test after
// proc-scoped channels can be natively generated.
TEST(ProcConfigIrConverterTest, DISABLED_DealOutChannelArrayElementsToSpawnee) {
  constexpr std::string_view kModule = R"(
  proc B {
    input: chan<u32> in;
    output: chan<u32> out;

    init { () }

    config(input: chan<u32> in, output: chan<u32> out) {
      (input, output)
    }

    next(state: ()) {
      let (tok, data) = recv(join(), input);
      let tok = send(tok, output, data);
    }
  }

  proc A {
    inputs: chan<u32>[2][2] in;
    outputs: chan<u32>[2][2] out;

    init { () }

    config() {
      let (output_to_b, input_from_b) = chan<u32>[2][2]("toward_b");
      let (output_to_a, input_from_a) = chan<u32>[2][2]("toward_a");
      unroll_for!(i, _) : (u32, ()) in u32:0..u32:2 {
        unroll_for!(j, _) : (u32, ()) in u32:0..u32:2 {
          spawn B(input_from_a[i][j], output_to_a[i][j]);
        }(())
      }(());
      (input_from_b, output_to_b)
    }

    next(state: ()) {
      unroll_for!(i, (tok, data)): (u32, (token, u32)) in u32:0..u32:2 {
        unroll_for!(j, (tok, data)): (u32, (token, u32)) in u32:0..u32:2 {
          let tok = send(tok, outputs[i][j], data);
          let (tok, data) = recv(tok, inputs[i][j]);
          (tok, data)
        }((tok, data))
      }((join(), u32:0));
    }
  }
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(
          tm.module, &import_data,
          ConvertOptions{.lower_to_proc_scoped_channels = true}));
  EXPECT_THAT(conv.package->channels(),
              UnorderedElementsAre(
                  m::Channel("toward_a__0_0"), m::Channel("toward_a__0_1"),
                  m::Channel("toward_a__1_0"), m::Channel("toward_a__1_1"),
                  m::Channel("toward_b__0_0"), m::Channel("toward_b__0_1"),
                  m::Channel("toward_b__1_0"), m::Channel("toward_b__1_1")));
}

TEST(ProcConfigIrConverterTest, MultipleNonLeafSpawnsOfSameProc) {
  constexpr std::string_view kModule = R"(
proc C {
    init { () }
    config() { () }
    next(state: ()) { () }
}

proc B {
    init { () }
    config() {
      spawn C();
      ()
    }
    next(state: ()) { () }
}

proc A {
    init { () }
    config() {
        spawn B();
        spawn B();
        ()
    }
    next(state: ()) { () }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(tm.module, &import_data, ConvertOptions{}));

  EXPECT_THAT(conv.package->procs(),
              UnorderedElementsAre(m::Proc("__test_module__A_0_next"),
                                   m::Proc("__test_module__A__B_0__C_0_next"),
                                   m::Proc("__test_module__A__B_0_next"),
                                   m::Proc("__test_module__A__B_1__C_0_next"),
                                   m::Proc("__test_module__A__B_1_next")));
}

TEST(ProcConfigIrConverterTest,
     MultipleInternalChannelsWithSameNameInDifferentProcs) {
  constexpr std::string_view kModule = R"(
proc passthrough {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  internal_in: chan<u32> in;
  internal_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    let (p, c) = chan<u32>("my_chan");
    (c_in, c_out, c, p)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), c_in);
    let tok = send(tok, internal_out, data);
    let (tok, data) = recv(tok, internal_in);
    let tok = send(tok, c_out, data);
    ()
  }
}

proc main {
  c_out: chan<u32> out;
  internal_in: chan<u32> in;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    let (p, c) = chan<u32>("my_chan");
    spawn passthrough(c_in, p);
    (c_out, c)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), internal_in);
    let tok = send(tok, c_out, data);
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      PackageConversionData conv,
      ConvertModuleToPackage(tm.module, &import_data, ConvertOptions{}));

  EXPECT_THAT(conv.package->channels(),
              AllOf(Contains(m::Channel("test_module__my_chan")),
                    Contains(m::Channel("test_module__my_chan__1"))));
}

TEST(ProcConfigIrConverterTest, ProcScopedChannels) {
  constexpr std::string_view kModule = R"(
proc passthrough {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    (c_in, c_out)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), c_in);
    let tok = send(tok, c_out, data);
    ()
  }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  PackageConversionData conv{.package =
                                 std::make_unique<Package>(tm.module->name())};
  XLS_ASSERT_OK(ConvertOneFunctionIntoPackage(
      tm.module, "passthrough", &import_data,
      /* parametric_env */ nullptr,
      ConvertOptions{.verify_ir = true, .proc_scoped_channels = true}, &conv));
  EXPECT_THAT(conv.package->channels(), IsEmpty());

  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc, conv.package->GetTopAsProc());
  EXPECT_TRUE(proc->is_new_style_proc());
  EXPECT_THAT(proc->interface(), SizeIs(2));

  ExpectIr(conv.DumpIr());
}

TEST(ProcConfigIrConverterTest, GlobalScopedChannels) {
  constexpr std::string_view kModule = R"(
proc passthrough {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    (c_in, c_out)
  }
  next(state: ()) {
    let (tok, data) = recv(join(), c_in);
    let tok = send(tok, c_out, data);
    ()
  }
}
)";
  auto import_data = CreateImportDataForTest();
  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  PackageConversionData conv{.package =
                                 std::make_unique<Package>(tm.module->name())};
  XLS_ASSERT_OK(ConvertOneFunctionIntoPackage(
      tm.module, "passthrough", &import_data,
      /* parametric_env */ nullptr, ConvertOptions{}, &conv));
  EXPECT_THAT(conv.package->channels(),
              AllOf(Contains(m::Channel("test_module__c_in")),
                    Contains(m::Channel("test_module__c_out"))));

  XLS_ASSERT_OK_AND_ASSIGN(xls::Proc * proc, conv.package->GetTopAsProc());
  EXPECT_FALSE(proc->is_new_style_proc());
}

}  // namespace
}  // namespace xls::dslx
