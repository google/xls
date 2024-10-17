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

#ifndef READ_ITERATIONS
#error "Must define READ_ITERATIONS in BUILD"
#endif /* READ_ITERATIONS */

class MemIteratedRead {
  __xls_memory<ac_int<8>, 22> mem;
  __xls_channel<ac_int<8>, __xls_channel_dir_Out> res;
  __xls_channel<ac_int<1>, __xls_channel_dir_In> rsel;

  template <int N>
  ac_int<8> read_n() {
    if constexpr (N > 1) {
      return mem[N - 1] + read_n<N - 1>();
    }
    if constexpr (N == 1) {
      return mem[0];
    }
    return 0;
  }

  void Run() {
    ac_int<8> a = 0;

    if (rsel.read()) {
      a = read_n<READ_ITERATIONS>();
      res.write(a);
    } else {
      mem[11] = 10;
    }
  }
};
