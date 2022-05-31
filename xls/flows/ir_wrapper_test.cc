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
  constexpr absl::string_view kParamsDslx = R"(
pub struct ParamsProto {
  latency: sN[64],
}

pub const params = ParamsProto { latency: sN[64]:1 } ;
)";

  constexpr absl::string_view kTopDslx = R"(
import param

pub fn GetLatency() -> s64 {
  param::params.latency
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<dslx::Module> params_module,
      dslx::ParseModule(kParamsDslx, "params_module", "param"));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<dslx::Module> top_module,
                           dslx::ParseModule(kTopDslx, "top_module", "top"));

  XLS_ASSERT_OK_AND_ASSIGN(
      IrWrapper ir_wrapper,
      IrWrapper::Create("test_package", std::move(top_module), "top_module.x",
                        std::move(params_module), "params_module.x"));

  XLS_ASSERT_OK_AND_ASSIGN(Function * get_latency,
                           ir_wrapper.GetIrFunction("GetLatency"));

  XLS_VLOG_LINES(3, get_latency->DumpIr());
  EXPECT_EQ(get_latency->DumpIr(),
            R"(fn __top__GetLatency() -> bits[64] {
  ret params_latency: bits[64] = literal(value=1, id=5, pos=0,4,15)
}
)");
}

}  // namespace
}  // namespace xls
