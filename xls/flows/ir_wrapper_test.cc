// Copyright 2022 The XLS Authors
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

#include "xls/flows/ir_wrapper.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/golden_files.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_view.h"
#include "xls/jit/function_jit.h"

namespace xls {
namespace {

void ExpectIr(std::string_view got, std::string_view test_name) {
  ExpectEqualToGoldenFile(
      absl::StrFormat("xls/flows/testdata/ir_wrapper_test_%s.ir", test_name),
      got);
}

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(IrWrapperTest, DslxToIrOk) {
  constexpr std::string_view kParamsDslx = R"(pub struct ParamsProto {
    latency: sN[64],
}
pub const params = ParamsProto { latency: sN[64]:7 };)";

  constexpr std::string_view kTopDslx = R"(import param;
pub fn GetLatency() -> s64 {
    param::params.latency
})";

  dslx::FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> params_module,
      dslx::ParseModule(kParamsDslx, "params_module.x", "param", file_table));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> top_module,
      dslx::ParseModule(kTopDslx, "top_module.x", "top", file_table));

  XLS_ASSERT_OK_AND_ASSIGN(
      IrWrapper ir_wrapper,
      IrWrapper::Create("test_package", std::move(top_module), "top_module.x",
                        std::move(params_module), "params_module.x"));

  // Test that modules can be retrieved.
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Module * params,
                           ir_wrapper.GetDslxModule("param"));
  EXPECT_EQ(params->ToString(), kParamsDslx);

  XLS_ASSERT_OK_AND_ASSIGN(dslx::Module * top, ir_wrapper.GetDslxModule("top"));
  EXPECT_EQ(top->ToString(), kTopDslx);

  EXPECT_THAT(
      ir_wrapper.GetDslxModule("not_a_module"),
      status_testing::StatusIs(absl::StatusCode::kNotFound,
                               testing::HasSubstr("Could not find module")));

  // Test that the ir can be compiled and retrieved.
  XLS_ASSERT_OK_AND_ASSIGN(Function * get_latency,
                           ir_wrapper.GetIrFunction("GetLatency"));

  XLS_VLOG_LINES(3, get_latency->DumpIr());
  ExpectIr(get_latency->DumpIr(), TestName() + "_get_latency");

  XLS_ASSERT_OK_AND_ASSIGN(Package * package, ir_wrapper.GetIrPackage());
  ExpectIr(package->DumpIr(), TestName() + "_package");

  // Test that that the jit for the function can be retrieved and run.
  XLS_ASSERT_OK_AND_ASSIGN(
      FunctionJit * jit, ir_wrapper.GetAndMaybeCreateFunctionJit("GetLatency"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> ret_val,
                           jit->Run(absl::Span<const Value>{}));
  EXPECT_EQ(ret_val.value, Value(UBits(7, 64)));
}

TEST(IrWrapperTest, DslxProcsToIrOk) {
  constexpr std::string_view kTopDslx = R"(proc foo {
    in_0: chan<u32> in;
    in_1: chan<u32> in;
    output: chan<u32> out;
    config(in_0: chan<u32> in, in_1: chan<u32> in, output: chan<u32> out) {
        (in_0, in_1, output)
    }
    init {
        ()
    }
    next(state: ()) {
        let tok: token = join();
        let (tok, a) = recv(tok, in_0);
        let (tok, b) = recv(tok, in_1);
        let tok = send(tok, output, a + b);
        ()
    }
})";

  dslx::FileTable file_table;
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> top_module,
      dslx::ParseModule(kTopDslx, "top_module.x", "top", file_table));

  XLS_ASSERT_OK_AND_ASSIGN(
      IrWrapper ir_wrapper,
      IrWrapper::Create("test_package", std::move(top_module), "top_module.x"));

  // Test that modules can be retrieved.
  XLS_ASSERT_OK_AND_ASSIGN(dslx::Module * top, ir_wrapper.GetDslxModule("top"));
  EXPECT_EQ(top->ToString(), kTopDslx);

  // Test that the ir proc can be compiled and retrieved.
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top_proc, ir_wrapper.GetIrProc("foo_0_next"));
  XLS_VLOG_LINES(3, top_proc->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(Package * package, ir_wrapper.GetIrPackage());
  ExpectIr(package->DumpIr(), TestName());

  // Test that that the jit for the proc can be retrieved and run.
  XLS_ASSERT_OK_AND_ASSIGN(SerialProcRuntime * proc_runtime,
                           ir_wrapper.GetAndMaybeCreateProcRuntime());
  XLS_ASSERT_OK_AND_ASSIGN(
      JitChannelQueueWrapper in_0,
      ir_wrapper.CreateJitChannelQueueWrapper("test_package__in_0"));
  XLS_ASSERT_OK_AND_ASSIGN(
      JitChannelQueueWrapper in_1,
      ir_wrapper.CreateJitChannelQueueWrapper("test_package__in_1"));
  XLS_ASSERT_OK_AND_ASSIGN(
      JitChannelQueueWrapper out,
      ir_wrapper.CreateJitChannelQueueWrapper("test_package__output"));

  // Send data.
  EXPECT_TRUE(in_0.Empty());
  EXPECT_TRUE(in_1.Empty());
  EXPECT_TRUE(out.Empty());

  XLS_ASSERT_OK(in_0.WriteWithUint64(10));
  XLS_ASSERT_OK(in_1.WriteWithUint64(20));

  EXPECT_FALSE(in_0.Empty());
  EXPECT_FALSE(in_1.Empty());
  EXPECT_TRUE(out.Empty());

  // Run one tick
  XLS_ASSERT_OK(proc_runtime->Tick());

  EXPECT_TRUE(in_0.Empty());
  EXPECT_TRUE(in_1.Empty());
  EXPECT_FALSE(out.Empty());

  // Receive data.
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t out_value, out.ReadWithUint64());

  EXPECT_TRUE(in_0.Empty());
  EXPECT_TRUE(in_1.Empty());
  EXPECT_TRUE(out.Empty());

  EXPECT_EQ(out_value, 30);

  // Send data.
  absl::Span<uint8_t> in_0_buffer = in_0.buffer();
  absl::Span<uint8_t> in_1_buffer = in_1.buffer();
  using MutableInt32View = MutableBitsView<32>;
  auto in0_data = MutableInt32View(in_0_buffer.data());
  in0_data.SetValue(20);
  auto in1_data = MutableInt32View(in_1_buffer.data());
  in1_data.SetValue(20);
  XLS_ASSERT_OK(in_0.Write(in_0_buffer));
  XLS_ASSERT_OK(in_1.Write(in_1_buffer));

  EXPECT_FALSE(in_0.Empty());
  EXPECT_FALSE(in_1.Empty());

  // Run one tick
  XLS_ASSERT_OK(proc_runtime->Tick());

  EXPECT_TRUE(in_0.Empty());
  EXPECT_TRUE(in_1.Empty());
  EXPECT_FALSE(out.Empty());

  // Receive data.
  absl::Span<uint8_t> out_buffer = out.buffer();
  XLS_ASSERT_OK(out.Read(out_buffer));

  EXPECT_TRUE(out.Empty());

  auto out_data = MutableInt32View(out_buffer.data());
  out_value = out_data.GetValue();

  EXPECT_EQ(out_value, 40);
}

}  // namespace
}  // namespace xls
