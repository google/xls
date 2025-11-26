#include "xls/codegen_v_1_5/convert_to_block.h"
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

#include <memory>

#include "gtest/gtest.h"
#include "xls/codegen/codegen_options.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls::codegen {
namespace {

using verilog::CodegenOptions;

class ConvertToBlockTest : public IrTestBase {
 protected:
  CodegenOptions codegen_options() {
    return CodegenOptions().module_name(TestName());
  }

  SchedulingOptions scheduling_options() {
    return SchedulingOptions{}.pipeline_stages(2);
  }

  TestDelayEstimator delay_estimator_;
};

TEST_F(ConvertToBlockTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK(p->SetTop(top));
  TestDelayEstimator delay_estimator;

  XLS_ASSERT_OK(ConvertToBlock(p.get(), codegen_options(), scheduling_options(),
                               &delay_estimator_));

  // TODO: https://github.com/google/xls/issues/3356 - assert stuff.
}

}  // namespace
}  // namespace xls::codegen
