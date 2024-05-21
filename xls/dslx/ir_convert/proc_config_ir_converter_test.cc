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
#include <optional>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/extract_conversion_order.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

using ::testing::Contains;
using ::testing::HasSubstr;
using ::xls::status_testing::StatusIs;

namespace m = ::xls::op_matchers;

TEST(ProcConfigIrConverterTest, BasicConversion) {
  constexpr std::string_view kModule = R"(
proc test_proc {
  c: chan<u32> in;
  x: u32;
  init { u32: 0 }
  config(c: chan<u32> in, ham_sandwich: u32) {
    (c, ham_sandwich)
  }
  next(tok: token, y: u32) {
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
  next(tok: token, state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  ParametricEnv bindings;
  ProcId proc_id{/*proc_stack=*/{}, /*instance=*/0};

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, tm.module->GetMemberOrError<Function>("test_proc.config"));

  Package package("the_package");
  ChannelMetadataProto metadata;
  StreamingChannel channel("the_channel", /*id=*/0, ChannelOps::kSendReceive,
                           package.GetBitsType(32), {},
                           /*fifo_config=*/std::nullopt, FlowControl::kNone,
                           ChannelStrictness::kProvenMutuallyExclusive,
                           metadata);

  ProcConversionData proc_data;
  proc_data.id_to_config_args[proc_id].push_back(&channel);
  proc_data.id_to_config_args[proc_id].push_back(Value(UBits(8, 32)));

  ProcConfigIrConverter converter(&package, f, tm.type_info, &import_data,
                                  &proc_data, bindings, proc_id);
  XLS_EXPECT_OK(f->Accept(&converter));
}

TEST(ProcConfigIrConverterTest, CatchesMissingArgMap) {
  constexpr std::string_view kModule = R"(
proc test_proc {
  c: chan<u32> in;
  init { () }
  config(c: chan<u32> in) {
    (c,)
  }
  next(tok: token, state: ()) {
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
  next(tok: token, state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  ParametricEnv bindings;
  ProcId proc_id{/*proc_stack=*/{}, /*instance=*/0};

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, tm.module->GetMemberOrError<Function>("test_proc.config"));

  Package package("the_package");
  ProcConversionData proc_data;
  ProcConfigIrConverter converter(&package, f, tm.type_info, &import_data,
                                  &proc_data, bindings, proc_id);
  EXPECT_THAT(f->Accept(&converter),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not found in arg mapping")));
}

TEST(ProcConfigIrConverterTest,
     ConvertsParametricExpressionForInternalChannelFifoDepth) {
  constexpr std::string_view kModule = R"(
proc passthrough {
  c_in: chan<u32> in;
  c_out: chan<u32> out;
  init {}
  config(c_in: chan<u32> in, c_out: chan<u32> out) {
    (c_in, c_out)
  }
  next(tok: token, state: ()) {
    let (tok, data) = recv(tok, c_in);
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
  next(tok: token, state: ()) {
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
  next(tok: token, state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  ParametricEnv bindings;

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      ConvertModuleToPackage(tm.module, &import_data, ConvertOptions{}));

  EXPECT_THAT(package->channels(),
              Contains(m::Channel("test_module__my_chan")));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel,
                           package->GetChannel("test_module__my_chan"));
  ASSERT_EQ(channel->kind(), ChannelKind::kStreaming);
  EXPECT_EQ(down_cast<StreamingChannel*>(channel)->GetFifoDepth(), 7);
}

TEST(ProcConfigIrConverterTest,
     ConvertMultipleInternalChannelsWithSameNameInSameProc) {
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
  next(tok: token, state: ()) {
    let (tok, data) = recv(tok, c_in);
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

  ParametricEnv bindings;

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      ConvertModuleToPackage(tm.module, &import_data, ConvertOptions{}));

  EXPECT_THAT(package->channels(),
              AllOf(Contains(m::Channel("test_module__my_chan")),
                    Contains(m::Channel("test_module__my_chan__1"))));
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
  next(tok: token, state: ()) {
    let (tok, data) = recv(tok, c_in);
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
  next(tok: token, state: ()) {
    let (tok, data) = recv(tok, internal_in);
    let tok = send(tok, c_out, data);
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  ParametricEnv bindings;

  XLS_ASSERT_OK_AND_ASSIGN(
      TypecheckedModule tm,
      ParseAndTypecheck(kModule, "test_module.x", "test_module", &import_data));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Package> package,
      ConvertModuleToPackage(tm.module, &import_data, ConvertOptions{}));

  EXPECT_THAT(package->channels(),
              AllOf(Contains(m::Channel("test_module__my_chan")),
                    Contains(m::Channel("test_module__my_chan__1"))));
}

}  // namespace
}  // namespace xls::dslx
