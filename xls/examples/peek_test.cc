// Copyright 2026 The XLS Authors
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

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/examples/peek_Peek_true_jit_wrapper.h"
#include "xls/examples/peek_PeekIf_true_jit_wrapper.h"
#include "xls/ir/value.h"

namespace xls {
namespace examples {
namespace {

TEST(PeekJitTest, TestOverrides) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Peek> peek,
      Peek::Create());

  XLS_ASSERT_OK(peek->SendToReqR(Value(UBits(0x5, 32))));
  XLS_ASSERT_OK(peek->Tick());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<unsigned int> result, peek->ReceiveFromRespS());
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), 0x5);
}

TEST(PeekIfJitTest, TestOverrides) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PeekIf> peek,
      PeekIf::Create());

  // First packet which should be passed through
  // as peek guard isn't enabled.
  XLS_ASSERT_OK(peek->SendToReqR(Value(UBits(0x3, 32))));
  XLS_ASSERT_OK(peek->Tick());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<unsigned int> result, peek->ReceiveFromRespS());
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), 0x3);

  XLS_ASSERT_OK(peek->SendToEnableR(Value(UBits(1, 1))));

  // Second packet which should be ignored after enabling peek guard.
  XLS_ASSERT_OK(peek->SendToReqR(Value(UBits(0x2, 32))));
  XLS_ASSERT_OK(peek->Tick());
  XLS_ASSERT_OK_AND_ASSIGN(result, peek->ReceiveFromRespS());
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result.value(), 0x0);
}

}  // namespace
}  // namespace examples
}  // namespace xls
