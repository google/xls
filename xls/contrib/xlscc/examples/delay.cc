// Copyright 2023 The XLS Authors
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

#include "/xls_builtin.h"  // NOLINT
#include "xls_int.h"       // NOLINT

template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

#define DEFAULT_OUTPUT_VALUE 3
#define MEMORY_SIZE 1024

using uint32_t = unsigned int;
using uint64_t = unsigned long long;  // NOLINT(runtime/int)

// TODO(rigge): support arbitrary types for memories
using memory_t = uint64_t;
using addr_t = ac_int<10, /*signed=*/false>;

#pragma hls_top
void delay(__xls_channel<uint32_t>& in,
           __xls_memory<memory_t, MEMORY_SIZE>& memory,
           __xls_channel<uint32_t>& out) {
  static uint32_t prev_write = 0;
  static uint32_t prev_read = 0;
  static bool reading = true;
  static bool memory_initialized = false;
  static addr_t addr = 0;

  const auto input_value = in.read();
  uint32_t output_value = 0;

  if (reading) {
    prev_write = input_value;
    if (memory_initialized) {
      const memory_t read_value = memory[addr];
      prev_read = read_value & 0xFFFFFFFFUL;
      output_value = (read_value >> 32) & 0xFFFFFFFFUL;
    } else {
      prev_read = DEFAULT_OUTPUT_VALUE;
      output_value = DEFAULT_OUTPUT_VALUE;
    }
  } else {
    memory[addr] = (static_cast<uint64_t>(prev_write) << 32) | input_value;
    output_value = prev_read;
  }

  const addr_t next_addr = addr + (reading ? 0 : 1);
  memory_initialized = memory_initialized || (!reading && (next_addr == 0));
  addr = next_addr;

  out.write(output_value);
  reading = !reading;
}
