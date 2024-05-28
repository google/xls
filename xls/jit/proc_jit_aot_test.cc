// Copyright 2024 The XLS Authors
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

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/events.h"
#include "xls/ir/proc.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_proc_runtime.h"
#include "xls/jit/jit_runtime.h"
#include "xls/public/ir_parser.h"

extern "C" {
// Top proc entrypoint
int64_t proc_0(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
int64_t proc_1(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
}

namespace xls {
namespace {
using testing::Optional;

static_assert(std::is_same_v<JitFunctionType, decltype(&proc_0)>,
              "Jit function ABI updated. This test needs to be tweaked.");
static_assert(std::is_same_v<JitFunctionType, decltype(&proc_1)>,
              "Jit function ABI updated. This test needs to be tweaked.");

static constexpr std::string_view kTestAotEntrypointsProto =
    "xls/jit/specialized_caps_aot.pb";
static constexpr std::string_view kGoldIr = "xls/jit/some_caps_no_idents.ir";

absl::StatusOr<AotPackageEntrypointsProto> GetEntrypointsProto() {
  AotPackageEntrypointsProto proto;
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(kTestAotEntrypointsProto));
  XLS_ASSIGN_OR_RETURN(std::string bin, GetFileContents(path));
  XLS_RET_CHECK(proto.ParseFromString(bin));
  return proto;
}
bool AreSymbolsAsExpected() {
  auto v = GetEntrypointsProto();
  if (!v.ok()) {
    return false;
  }
  return absl::c_any_of(v->entrypoint(),
                        [](const AotEntrypointProto& p) {
                          return p.has_function_symbol() &&
                                 p.function_symbol() == "proc_0";
                        }) &&
         absl::c_any_of(v->entrypoint(), [](const AotEntrypointProto& p) {
           return p.has_function_symbol() && p.function_symbol() == "proc_1";
         });
}

// Not really a test just to make sure that if all other tests are disabled due
// to linking failure we have *something* that fails.
TEST(SymbolNames, AreAsExpected) {
  ASSERT_TRUE(AreSymbolsAsExpected())
      << "Symbols are not what we expected. This test needs to be updated to "
         "match new jit-compiler symbol naming scheme. Symbols are: "
      << GetEntrypointsProto();
}

class ProcJitAotTest : public testing::Test {
  void SetUp() override {
    if (!AreSymbolsAsExpected()) {
      GTEST_SKIP() << "Linking probably failed. AOTEntrypoints lists "
                      "unexpected symbol names";
    }
  }
};

Value StrValue(const char sv[8]) {
  auto add_failure_and_return_zero = [&](auto reason) {
    ADD_FAILURE() << "Unable to make value with " << sv
                  << " because: " << reason;
    TypeManager tm;
    return ZeroOfType(tm.GetArrayType(8, tm.GetBitsType(8)));
  };
  std::array<uint64_t, 8> ret;
  absl::c_copy_n(std::string_view(sv, 8), ret.size(), ret.begin());
  auto value = ValueBuilder::UBitsArray(ret, 8).Build();
  if (value.ok()) {
    return *value;
  }
  return add_failure_and_return_zero(value);
}

TEST_F(ProcJitAotTest, Tick) {
  XLS_ASSERT_OK_AND_ASSIGN(AotPackageEntrypointsProto proto,
                           GetEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p0, p->GetProc("proc_0"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p1, p->GetProc("proc_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto aot_runtime,
      CreateAotSerialProcRuntime(
          p.get(), proto,
          {ProcAotEntrypoints{.proc = p0, .unpacked = proc_0},
           ProcAotEntrypoints{.proc = p1, .unpacked = proc_1}}));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueueManager * chan_man,
                           aot_runtime->GetJitChannelQueueManager());
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * chan_input,
                           chan_man->GetQueueByName("chan_0"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * chan_output,
                           chan_man->GetQueueByName("chan_1"));
  XLS_EXPECT_OK(chan_input->Write(StrValue("abcdefgh")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("ijklmnop")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("qrstuvwx")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("yz012345")));
  XLS_EXPECT_OK(aot_runtime->Tick());
  XLS_EXPECT_OK(aot_runtime->Tick());
  XLS_EXPECT_OK(aot_runtime->Tick());
  XLS_EXPECT_OK(aot_runtime->Tick());
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("ABCDEFGH")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("ijklmnop")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("QrStUvWx")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("YZ012345")));
}

TEST_F(ProcJitAotTest, TickUntilBlocked) {
  XLS_ASSERT_OK_AND_ASSIGN(AotPackageEntrypointsProto proto,
                           GetEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p0, p->GetProc("proc_0"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p1, p->GetProc("proc_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto aot_runtime,
      CreateAotSerialProcRuntime(
          p.get(), proto,
          {ProcAotEntrypoints{.proc = p0, .unpacked = proc_0},
           ProcAotEntrypoints{.proc = p1, .unpacked = proc_1}}));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueueManager * chan_man,
                           aot_runtime->GetJitChannelQueueManager());
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * chan_input,
                           chan_man->GetQueueByName("chan_0"));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelQueue * chan_output,
                           chan_man->GetQueueByName("chan_1"));
  XLS_EXPECT_OK(chan_input->Write(StrValue("abcdefgh")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("ijklmnop")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("qrstuvwx")));
  XLS_EXPECT_OK(chan_input->Write(StrValue("yz012345")));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t count, aot_runtime->TickUntilBlocked());
  // 1 for each of the inputs and then another tick succeeds by making progress
  // until it tries to recv from the empty channel so 5 count.
  EXPECT_EQ(count, 5);
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("ABCDEFGH")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("ijklmnop")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("QrStUvWx")));
  EXPECT_THAT(chan_output->Read(), Optional(StrValue("YZ012345")));
}
}  // namespace
}  // namespace xls
