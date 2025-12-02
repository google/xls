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
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/channel_scope.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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

  ChannelScope channel_scope(
      &conv, &import_data,
      ConvertOptions{.lower_to_proc_scoped_channels = false});
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

}  // namespace
}  // namespace xls::dslx
