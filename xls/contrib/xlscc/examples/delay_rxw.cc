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

template <typename T>
struct pair_t {
  T first;
  T second;

  explicit pair_t() : first(), second() {}
  explicit pair_t(T first, T second) : first(first), second(second) {}
};

using memory_t = pair_t<uint32_t>;
using addr_t = ac_int<10, /*signed=*/false>;

template <typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

template <typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;

// A delay block implemented using RAM with read-xor-write (abbreviated RXW,
// i.e. it either reads or writes in a proc tick, but never both) . The delay
// block takes an input transaction and delays it DELAY transactions later. The
// output is DEFAULT_OUTPUT_VALUE for the first DELAY transactions.
struct Delay {
  InputChannel<uint32_t> in;
  __xls_memory<memory_t, MEMORY_SIZE> memory;
  OutputChannel<uint32_t> out;

  uint32_t prev_write = 0;
  uint32_t prev_read = 0;
  bool reading = true;
  bool memory_initialized = false;
  addr_t addr = 0;

#pragma hls_top
  void delay() {
    const auto input_value = in.read();
    uint32_t output_value = 0;

    if (!memory_initialized) {
      prev_read = DEFAULT_OUTPUT_VALUE;
      output_value = DEFAULT_OUTPUT_VALUE;
    }

    if (reading) {
      prev_write = input_value;
      if (memory_initialized) {
        const memory_t read_value = memory[addr];
        output_value = read_value.first;
        prev_read = read_value.second;
      }
    } else {
      memory[addr] = memory_t(prev_write, input_value);
      output_value = prev_read;
    }

    const addr_t next_addr = addr + (reading ? 0 : 1);
    memory_initialized = memory_initialized || (!reading && (next_addr == 0));
    addr = next_addr;

    out.write(output_value);
    reading = !reading;
  }
};
