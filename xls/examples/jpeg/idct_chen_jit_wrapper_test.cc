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

#include "xls/examples/jpeg/idct_chen_jit_wrapper.h"

#include <array>
#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/value_view.h"

namespace xls {
namespace {

TEST(IdctChenJitWrapperTest, SmokeTest) {
  std::array<int32_t, 64> empty = {0};
  XLS_ASSERT_OK_AND_ASSIGN(auto idct, xls::jpeg::IdctChen::Create());
  std::array<int32_t, 64> result;
  XLS_ASSERT_OK(idct->Run(PackedArrayView<PackedBitsView<32>, 64>(&empty),
                          PackedArrayView<PackedBitsView<32>, 64>(&result)));
  EXPECT_EQ(empty, result);
}

}  // namespace
}  // namespace xls
