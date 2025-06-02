// Copyright 2025 The XLS Authors
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
#include "xls/dev_tools/extract_interface.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/ir/value_builder.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_ir_interface.pb.h"
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
int64_t __multi_proc__proc_ten__proc_quad_0_next(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
int64_t __multi_proc__proc_ten__proc_double_0_next(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
int64_t __multi_proc__proc_ten_0_next(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
}

namespace xls {
namespace {
using ::testing::Optional;

static_assert(std::is_same_v<JitFunctionType, decltype(&proc_0)>,
              "Jit function ABI updated. This test needs to be tweaked.");
static_assert(std::is_same_v<JitFunctionType, decltype(&proc_1)>,
              "Jit function ABI updated. This test needs to be tweaked.");

static constexpr std::string_view kTestCapsAotEntrypointsProto =
    "xls/jit/specialized_caps_aot.pb";
static constexpr std::string_view kCapsGoldIr =
    "xls/jit/some_caps_no_idents.ir";
static constexpr std::string_view kTestMultiAotEntrypointsProto =
    "xls/jit/multi_proc_aot_psc.pb";
static constexpr std::string_view kMultiGoldIr = "xls/jit/multi_proc_psc.ir";

absl::StatusOr<AotPackageEntrypointsProto> GetMultiEntrypointsProto() {
  AotPackageEntrypointsProto proto;
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(kTestMultiAotEntrypointsProto));
  XLS_ASSIGN_OR_RETURN(std::string bin, GetFileContents(path));
  XLS_RET_CHECK(proto.ParseFromString(bin));
  return proto;
}

bool AreMultiSymbolsAsExpected() {
  auto v = GetMultiEntrypointsProto();
  if (!v.ok()) {
    return false;
  }
  return absl::c_any_of(v->entrypoint(),
                        [](const AotEntrypointProto& p) {
                          return p.has_function_symbol() &&
                                 p.function_symbol() ==
                                     "__multi_proc__proc_ten_0_next";
                        }) &&
         absl::c_any_of(
             v->entrypoint(),
             [](const AotEntrypointProto& p) {
               return p.has_function_symbol() &&
                      p.function_symbol() ==
                          "__multi_proc__proc_ten__proc_double_0_next";
             }) &&
         absl::c_any_of(v->entrypoint(), [](const AotEntrypointProto& p) {
           return p.has_function_symbol() &&
                  p.function_symbol() ==
                      "__multi_proc__proc_ten__proc_quad_0_next";
         });
}

absl::StatusOr<AotPackageEntrypointsProto> GetCapsEntrypointsProto() {
  AotPackageEntrypointsProto proto;
  XLS_ASSIGN_OR_RETURN(std::filesystem::path path,
                       GetXlsRunfilePath(kTestCapsAotEntrypointsProto));
  XLS_ASSIGN_OR_RETURN(std::string bin, GetFileContents(path));
  XLS_RET_CHECK(proto.ParseFromString(bin));
  return proto;
}

bool AreCapsSymbolsAsExpected() {
  auto v = GetCapsEntrypointsProto();
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
  EXPECT_TRUE(AreCapsSymbolsAsExpected())
      << "Symbols are not what we expected. This test needs to be updated to "
         "match new jit-compiler symbol naming scheme. Symbols are: "
      << GetCapsEntrypointsProto();
  EXPECT_TRUE(AreMultiSymbolsAsExpected())
      << "Symbols are not what we expected. This test needs to be updated to "
         "match new jit-compiler symbol naming scheme. Symbols are: "
      << GetMultiEntrypointsProto();
}

class ProcJitAotPscTest : public testing::Test {
  void SetUp() override {
    if (!AreCapsSymbolsAsExpected() || !AreMultiSymbolsAsExpected()) {
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

TEST_F(ProcJitAotPscTest, Tick) {
  XLS_ASSERT_OK_AND_ASSIGN(AotPackageEntrypointsProto proto,
                           GetCapsEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kCapsGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kCapsGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p0, p->GetProc("proc_0"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p1, p->GetProc("proc_1"));
  PackageInterfaceProto::Proc p0_interface = ExtractProcInterface(p0);
  PackageInterfaceProto::Proc p1_interface = ExtractProcInterface(p1);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto aot_runtime,
      CreateAotSerialProcRuntime(
          p.get(), proto,
          {ProcAotEntrypoints{.proc_interface_proto = p0_interface,
                              .unpacked = proc_0},
           ProcAotEntrypoints{.proc_interface_proto = p1_interface,
                              .unpacked = proc_1}}));
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

TEST_F(ProcJitAotPscTest, TickUntilBlocked) {
  XLS_ASSERT_OK_AND_ASSIGN(AotPackageEntrypointsProto proto,
                           GetCapsEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kCapsGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kCapsGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p0, p->GetProc("proc_0"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * p1, p->GetProc("proc_1"));
  PackageInterfaceProto::Proc p0_interface = ExtractProcInterface(p0);
  PackageInterfaceProto::Proc p1_interface = ExtractProcInterface(p1);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto aot_runtime,
      CreateAotSerialProcRuntime(
          p.get(), proto,
          {ProcAotEntrypoints{.proc_interface_proto = p0_interface,
                              .unpacked = proc_0},
           ProcAotEntrypoints{.proc_interface_proto = p1_interface,
                              .unpacked = proc_1}}));
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

TEST_F(ProcJitAotPscTest, MultipleProcsCanHitSameFunction) {
  XLS_ASSERT_OK_AND_ASSIGN(AotPackageEntrypointsProto proto,
                           GetMultiEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kMultiGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kMultiGoldIr));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * pquad, p->GetProc("__multi_proc__proc_ten__proc_quad_0_next"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * pdouble, p->GetProc("__multi_proc__proc_ten__proc_double_0_next"));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * ptop,
                           p->GetProc("__multi_proc__proc_ten_0_next"));
  PackageInterfaceProto::Proc pquad_interface = ExtractProcInterface(pquad);
  PackageInterfaceProto::Proc pdouble_interface = ExtractProcInterface(pdouble);
  PackageInterfaceProto::Proc ptop_interface = ExtractProcInterface(ptop);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto aot_runtime,
      CreateAotSerialProcRuntime(
          ptop, proto,
          {
              ProcAotEntrypoints{
                  .proc_interface_proto = pdouble_interface,
                  .unpacked = __multi_proc__proc_ten__proc_double_0_next},
              ProcAotEntrypoints{
                  .proc_interface_proto = pquad_interface,
                  .unpacked = __multi_proc__proc_ten__proc_quad_0_next},
              ProcAotEntrypoints{.proc_interface_proto = ptop_interface,
                                 .unpacked = __multi_proc__proc_ten_0_next},
          }));
  XLS_ASSERT_OK_AND_ASSIGN(JitChannelQueueManager * chan_man,
                           aot_runtime->GetJitChannelQueueManager());
  ProcInstance* top_instance = aot_runtime->elaboration().top();
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * input_channel,
      top_instance->GetChannelInstance("multi_proc__bytes_src"));
  ChannelQueue& chan_input = chan_man->GetQueue(input_channel);
  XLS_ASSERT_OK_AND_ASSIGN(
      ChannelInstance * output_channel,
      top_instance->GetChannelInstance("multi_proc__bytes_result"));
  ChannelQueue& chan_output = chan_man->GetQueue(output_channel);

  XLS_EXPECT_OK(chan_input.Write(Value(UBits(4, 32))));
  XLS_EXPECT_OK(chan_input.Write(Value(UBits(8, 32))));
  XLS_EXPECT_OK(chan_input.Write(Value(UBits(16, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(int64_t count, aot_runtime->TickUntilBlocked());
  // 1 for each of the inputs and then another tick succeeds by making progress
  // until it tries to recv from the empty channel so 4 count.
  EXPECT_EQ(count, 4);
  EXPECT_THAT(chan_output.Read(), Optional(Value(UBits(40, 32))));
  EXPECT_THAT(chan_output.Read(), Optional(Value(UBits(80, 32))));
  EXPECT_THAT(chan_output.Read(), Optional(Value(UBits(160, 32))));
}

}  // namespace
}  // namespace xls
