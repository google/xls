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

// Link/run test for the AOT C++ header: compiles a small function to an AOT
// object, includes the generated header and calls the packed entrypoint
// through the stable generated API. The code
// here depends only on the standard library and the AOT object itself.

#include <array>
#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"
#include "xls/jit/aot_cpp_header_second_test_generated.h"
#include "xls/jit/aot_cpp_header_test_generated.h"

namespace {

namespace first = aot_cpp::generated::Step;
namespace second = aot_cpp::generated::SecondStep;

TEST(AotCppHeaderLinkTest, InvokePacked) {
  // x = 0b101 -> bit 2 (=1) is the high bit of the result tuple.
  uint8_t x = 0x05;
  // arr[0] = 0x03 -> low 7 bits (0b0000011) is the low element.
  uint8_t arr[2] = {0x03, 0xAB};
  uint8_t result = 0;

  const uint8_t* const inputs[] = {&x, arr};
  uint8_t* const outputs[] = {&result};

  static constexpr std::size_t kTempSize =
      first::kTemporaryBufferSize == 0 ? 1 : first::kTemporaryBufferSize;
  static constexpr std::size_t kTempAlign =
      first::kTemporaryBufferAlignment == 0
          ? 1
          : first::kTemporaryBufferAlignment;
  alignas(kTempAlign) std::array<uint8_t, kTempSize> temp_buffer = {};

  int64_t continuation = first::kPackedFunction(
      inputs, outputs, temp_buffer.data(), /*events=*/nullptr,
      /*instance_context=*/nullptr, /*runtime=*/nullptr,
      /*continuation_point=*/0);
  EXPECT_EQ(continuation, 0);
  EXPECT_EQ(result, 0x83);
}

TEST(AotCppHeaderLinkTest, InvokePackedMatchesInvokePackedSymbol) {
  // Both spellings of the stable API resolve to the same function pointer.
  EXPECT_EQ(first::kPackedFunction, &first::StepInvokePacked);
}

TEST(AotCppHeaderLinkTest, MultipleHeadersCallDistinctPackedSymbols) {
  EXPECT_NE(first::kPackedFunction, second::kPackedFunction);

  uint8_t input = 0x5a;
  uint8_t result = 0;
  const uint8_t* const inputs[] = {&input};
  uint8_t* const outputs[] = {&result};
  static constexpr std::size_t kTempSize =
      second::kTemporaryBufferSize == 0 ? 1 : second::kTemporaryBufferSize;
  static constexpr std::size_t kTempAlign =
      second::kTemporaryBufferAlignment == 0
          ? 1
          : second::kTemporaryBufferAlignment;
  alignas(kTempAlign) std::array<uint8_t, kTempSize> temp_buffer = {};

  EXPECT_EQ(second::kPackedFunction(
                inputs, outputs, temp_buffer.data(), /*events=*/nullptr,
                /*instance_context=*/nullptr, /*runtime=*/nullptr,
                /*continuation_point=*/0),
            0);
  EXPECT_EQ(result, 0x5a);
}

}  // namespace
