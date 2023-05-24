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

#include <cstdint>

#include "/xls_builtin.h"  // NOLINT
#include "xls_int.h"       // NOLINT

template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

#define DEFAULT_OUTPUT_VALUE 3
#define DELAY 2048
#define MEMORY_SIZE static_cast<uint32_t>(DELAY)

using std::uint32_t;

using memory_t = uint32_t;
using addr_t = ac_int<11, /*signed=*/false>;

template <typename T>
struct my_optional {
  bool has_value;
  T value;
};

template <typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

template <typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;

// A delay block implemented using RAM with read-and-write (abbreviated RAW,
// i.e. it reads and writes in the same proc tick) . The delay block takes an
// input transaction and delays it DELAY transactions later. The output is
// DEFAULT_OUTPUT_VALUE for the first DELAY transactions.
struct Delay {
  InputChannel<uint32_t> in;
  __xls_memory<memory_t, MEMORY_SIZE> memory;
  OutputChannel<uint32_t> out;

  addr_t write_addr = 0;
  addr_t read_addr = 0;
  bool full = false;
  uint32_t initialization_count = 0;

  my_optional<uint32_t> fifo(bool out_ready) {
    bool next_full = false;
    addr_t next_write_addr = write_addr;
    addr_t next_read_addr = read_addr;
    if (!full) {
      uint32_t input_value;
      if (in.nb_read(input_value)) {
        memory[write_addr] = input_value;
        next_write_addr = write_addr + 1;
        if (static_cast<uint32_t>(next_write_addr) > MEMORY_SIZE) {
          next_write_addr = 0;
        }
        if (next_write_addr == read_addr) {
          next_full = true;
        }
      }
    }

    bool empty = (write_addr == read_addr) && !full;
    my_optional<uint32_t> output{.has_value = false};

    if (!empty && out_ready) {
      output.value = memory[read_addr];
      output.has_value = true;
      next_read_addr = read_addr + 1;
      if (static_cast<uint32_t>(next_read_addr) > MEMORY_SIZE) {
        next_read_addr = 0;
      }
      next_full = false;
    }

    write_addr = next_write_addr;
    read_addr = next_read_addr;
    full = next_full;
    return output;
  }

#pragma hls_top
  void delay() {
    bool initializating = initialization_count < MEMORY_SIZE;
    my_optional<uint32_t> output = fifo(/*out_ready=*/!initializating);
    if (initializating) {
      output.has_value = true;
      output.value = DEFAULT_OUTPUT_VALUE;
      initialization_count++;
    }
    if (output.has_value) {
      out.write(output.value);
    }
  }
};
