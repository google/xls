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
#include "xls/dslx/proc_config_ir_converter.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using testing::HasSubstr;

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
    let (p, c) = chan<u32>;
    spawn test_proc(c, u32:7);
    (p,)
  }
  next(tok: token, state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  SymbolicBindings bindings;
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
                           /*fifo_depth=*/absl::nullopt, FlowControl::kNone,
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
    let (p, c) = chan<u32>;
    spawn test_proc(c);
    (p,)
  }
  next(tok: token, state: ()) {
    ()
  }
}
)";

  auto import_data = CreateImportDataForTest();

  SymbolicBindings bindings;
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

}  // namespace
}  // namespace xls::dslx
