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

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/parse_and_typecheck.h"

namespace xls {
namespace {

TEST(IrWrapperTest, DslxToIrOk) {
  constexpr absl::string_view kParamsDslx = R"(pub struct ParamsProto {
  latency: sN[64],
}
pub const params = ParamsProto { latency: sN[64]:7 };)";

  constexpr absl::string_view kTopDslx = R"(import param
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
  ret params_latency: bits[64] = literal(value=7, id=5, pos=0,2,15)
}
)");

  XLS_ASSERT_OK_AND_ASSIGN(Package * package, ir_wrapper.GetIrPackage());
  EXPECT_EQ(package->DumpIr(),
            R"(package test_package

file_number 0 "fake_file.x"

fn __top__GetLatency() -> bits[64] {
  ret params_latency: bits[64] = literal(value=7, id=5, pos=0,2,15)
}
)");

  // Test that that the jit for the function can be retrieved and run.
  XLS_ASSERT_OK_AND_ASSIGN(
      FunctionJit * jit, ir_wrapper.GetAndMaybeCreateFunctionJit("GetLatency"));
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> ret_val,
                           jit->Run(absl::Span<const Value>{}));
  EXPECT_EQ(ret_val.value, Value(UBits(7, 64)));
}

}  // namespace
}  // namespace xls
