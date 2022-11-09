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
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/value_view.h"

namespace xls {
namespace {

TEST(IrWrapperTest, DslxToIrOk) {
  constexpr std::string_view kParamsDslx = R"(pub struct ParamsProto {
  latency: sN[64],
}
pub const params = ParamsProto { latency: sN[64]:7 };)";

  constexpr std::string_view kTopDslx = R"(import param
pub fn GetLatency() -> s64 {
  param::params.latency
})";

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> params_module,
      dslx::ParseModule(kParamsDslx, "params_module.x", "param"));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> top_module,
                           dslx::ParseModule(kTopDslx, "top_module.x", "top"));

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
  EXPECT_EQ(get_latency->DumpIr(),
            R"(fn __top__GetLatency() -> bits[64] {
  ret params_latency: bits[64] = literal(value=7, id=3, pos=[(0,2,15)])
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Package * package, ir_wrapper.GetIrPackage());
  EXPECT_EQ(package->DumpIr(),
            R"(package test_package

file_number 0 "fake_file.x"

fn __top__GetLatency() -> bits[64] {
  ret params_latency: bits[64] = literal(value=7, id=3, pos=[(0,2,15)])
}
)");

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
  next(tok: token, state: ()) {
    let (tok, a) = recv(tok, in_0);
    let (tok, b) = recv(tok, in_1);
    let tok = send(tok, output, (a) + (b));
    ()
  }
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> top_module,
                           dslx::ParseModule(kTopDslx, "top_module.x", "top"));

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
  EXPECT_EQ(package->DumpIr(),
            R"(package test_package

file_number 0 "fake_file.x"

chan test_package__in_0(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan test_package__in_1(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan test_package__output(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

proc __top__foo_0_next(__token: token, init={}) {
  receive.4: (token, bits[32]) = receive(__token, channel_id=0, id=4)
  tok: token = tuple_index(receive.4, index=0, id=6, pos=[(0,11,9)])
  receive.8: (token, bits[32]) = receive(tok, channel_id=1, id=8)
  a: bits[32] = tuple_index(receive.4, index=1, id=7, pos=[(0,11,14)])
  b: bits[32] = tuple_index(receive.8, index=1, id=11, pos=[(0,12,14)])
  tok__1: token = tuple_index(receive.8, index=0, id=10, pos=[(0,12,9)])
  add.12: bits[32] = add(a, b, id=12, pos=[(0,13,36)])
  tok__2: token = send(tok__1, add.12, channel_id=2, id=13)
  after_all.15: token = after_all(__token, tok, tok__1, tok__2, id=15)
  next (after_all.15)
}
)");

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
