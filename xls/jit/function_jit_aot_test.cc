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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <type_traits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/aot_entrypoint.pb.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_runtime.h"
#include "xls/public/ir_parser.h"

extern "C" {
// The actual symbols the AOT generates.
int64_t __multi_func_with_trace__multi_function_one_packed(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
int64_t __multi_func_with_trace__multi_function_one(  // NOLINT
    const uint8_t* const* inputs, uint8_t* const* outputs, void* temp_buffer,
    xls::InterpreterEvents* events, xls::InstanceContext* instance_context,
    xls::JitRuntime* jit_runtime, int64_t continuation_point);
}

namespace xls {
namespace {
using status_testing::StatusIs;
using testing::ContainsRegex;
using testing::ElementsAre;
using testing::UnorderedElementsAre;

static_assert(
    std::is_same_v<JitFunctionType,
                   decltype(&__multi_func_with_trace__multi_function_one)>,
    "Jit function ABI updated. This test needs to be tweaked.");
static_assert(
    std::is_same_v<
        JitFunctionType,
        decltype(&__multi_func_with_trace__multi_function_one_packed)>,
    "Jit function ABI updated. This test needs to be tweaked.");

static constexpr std::string_view kExpectedSymbolNameUnpacked =
    "__multi_func_with_trace__multi_function_one";
static constexpr std::string_view kExpectedSymbolNamePacked =
    "__multi_func_with_trace__multi_function_one_packed";
static constexpr std::string_view kGoldIr =
    "xls/jit/multi_function_with_trace.ir";
static constexpr std::string_view kGoldTopName =
    "__multi_func_with_trace__multi_function_one";
static constexpr std::string_view kTestAotEntrypointsProto =
    "xls/jit/multi_function_aot.pb";

absl::StatusOr<AotEntrypointProto> GetEntrypointsProto() {
  AotEntrypointProto proto;
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
  return v->has_function_symbol() &&
         v->function_symbol() == kExpectedSymbolNameUnpacked &&
         v->has_packed_function_symbol() &&
         v->packed_function_symbol() == kExpectedSymbolNamePacked;
}

// Not really a test just to make sure that if all other tests are disabled due
// to linking failure we have *something* that fails.
TEST(SymbolNames, AreAsExpected) {
  ASSERT_TRUE(AreSymbolsAsExpected())
      << "Symbols are not what we expected. This test needs to be updated to "
         "match new jit-compiler symbol naming scheme";
}

class FunctionJitAotTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!AreSymbolsAsExpected()) {
      GTEST_SKIP() << "Linking probably failed. AOTEntrypoints lists "
                      "unexpected symbol names";
    }
  }
};

TEST_F(FunctionJitAotTest, CallAot) {
  XLS_ASSERT_OK_AND_ASSIGN(AotEntrypointProto proto, GetEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction(kGoldTopName));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto test_aot, FunctionJit::CreateFromAot(
                         f, proto, __multi_func_with_trace__multi_function_one,
                         __multi_func_with_trace__multi_function_one_packed));
  // Value
  {
    XLS_ASSERT_OK_AND_ASSIGN(auto res, test_aot->Run({Value(UBits(3, 8))}));
    EXPECT_EQ(res.value, Value(UBits(15, 8)));
    EXPECT_THAT(res.events.trace_msgs,
                UnorderedElementsAre(TraceMessage("mf_2(6) -> 12", 0),
                                     TraceMessage("mf_1(3) -> 15", 0)));
  }

  // Packed
  {
    uint8_t inp = 4;
    uint8_t out = 0;
    PackedBitsView<8> inp_pv(&inp, 0);
    PackedBitsView<8> out_pv(&out, 0);
    XLS_ASSERT_OK(test_aot->RunWithPackedViews(inp_pv, out_pv));
    EXPECT_EQ(out, 20);
  }

  // unpacked
  {
    uint8_t inp = 5;
    uint8_t out = 0;
    BitsView<8> inp_pv(&inp);
    MutableBitsView<8> out_pv(&out);
    XLS_ASSERT_OK(test_aot->RunWithUnpackedViews(inp_pv, out_pv));
    EXPECT_EQ(out, 25);
  }
}

TEST_F(FunctionJitAotTest, InterceptCallAot) {
  XLS_ASSERT_OK_AND_ASSIGN(AotEntrypointProto proto, GetEntrypointsProto());
  XLS_ASSERT_OK_AND_ASSIGN(auto gold_file, GetXlsRunfilePath(kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(std::string pkg_text, GetFileContents(gold_file));
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(pkg_text, kGoldIr));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction(kGoldTopName));

  static thread_local int64_t called_unpacked_cnt = 0;
  static thread_local int64_t called_packed_cnt = 0;

  called_unpacked_cnt = 0;
  called_packed_cnt = 0;

  XLS_ASSERT_OK_AND_ASSIGN(
      auto test_aot,
      FunctionJit::CreateFromAot(
          f, proto,
          [](const uint8_t* const* inputs, uint8_t* const* outputs,
             void* temp_buffer, xls::InterpreterEvents* events,
             xls::InstanceContext* instance_context,
             xls::JitRuntime* jit_runtime,
             int64_t continuation_point) -> int64_t {
            ++called_unpacked_cnt;
            instance_context->vtable.record_trace(
                instance_context, new std::string("Stuck in the middle"), 42,
                events);
            return __multi_func_with_trace__multi_function_one(
                inputs, outputs, temp_buffer, events, instance_context,
                jit_runtime, continuation_point);
          },
          [](const uint8_t* const* inputs, uint8_t* const* outputs,
             void* temp_buffer, xls::InterpreterEvents* events,
             xls::InstanceContext* instance_context,
             xls::JitRuntime* jit_runtime,
             int64_t continuation_point) -> int64_t {
            ++called_packed_cnt;
            instance_context->vtable.record_assertion(instance_context,
                                                      "with you", events);
            return __multi_func_with_trace__multi_function_one_packed(
                inputs, outputs, temp_buffer, events, instance_context,
                jit_runtime, continuation_point);
          }));

  // Value
  {
    XLS_ASSERT_OK_AND_ASSIGN(auto res, test_aot->Run({Value(UBits(3, 8))}));
    EXPECT_EQ(res.value, Value(UBits(15, 8)));
    EXPECT_THAT(res.events.trace_msgs,
                UnorderedElementsAre(TraceMessage("Stuck in the middle", 42),
                                     TraceMessage("mf_2(6) -> 12", 0),
                                     TraceMessage("mf_1(3) -> 15", 0)));
    EXPECT_EQ(called_unpacked_cnt, 1);
    EXPECT_EQ(called_packed_cnt, 0);
  }

  // Packed
  {
    uint8_t inp = 4;
    uint8_t out = 0;
    PackedBitsView<8> inp_pv(&inp, 0);
    PackedBitsView<8> out_pv(&out, 0);
    EXPECT_THAT(
        test_aot->RunWithPackedViews(inp_pv, out_pv),
        StatusIs(absl::StatusCode::kAborted, ContainsRegex("with you")));
    // NB The computation does finish so we still expect the result here.
    EXPECT_EQ(out, 20);
    EXPECT_EQ(called_unpacked_cnt, 1);
    EXPECT_EQ(called_packed_cnt, 1);
  }

  // unpacked
  {
    uint8_t inp = 5;
    uint8_t out = 0;
    BitsView<8> inp_pv(&inp);
    MutableBitsView<8> out_pv(&out);
    XLS_ASSERT_OK(test_aot->RunWithUnpackedViews(inp_pv, out_pv));
    EXPECT_EQ(out, 25);
    EXPECT_EQ(called_unpacked_cnt, 2);
    EXPECT_EQ(called_packed_cnt, 1);
  }
}

}  // namespace
}  // namespace xls
