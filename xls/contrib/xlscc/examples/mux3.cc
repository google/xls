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

// NOLINTNEXTLINE
#include "xls_int.h"

#define ac_channel __xls_channel
template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

struct CsrRegisters {
  ac_int<1, false> enable_0;
  ac_int<1, false> enable_1;
};

struct ChannelType {
  ac_int<8, false> data;
};

struct Mux3Impl {
  void Run(CsrRegisters& csrs, ac_channel<struct ChannelType>& mux_in0,
           ac_channel<struct ChannelType>& mux_in1,
           ac_channel<struct ChannelType>& mux_in2,
           ac_channel<struct ChannelType>& mux_out) {
    ChannelType out;
    if (csrs.enable_0) {
      out = mux_in0.read();
    } else if (csrs.enable_1) {
      out = mux_in1.read();
    } else {
      out = mux_in2.read();
    }
    mux_out.write(out);
  }
};

#pragma hls_design top
void Mux3_f(CsrRegisters& csrs, ac_channel<struct ChannelType>& mux_in0,
            ac_channel<struct ChannelType>& mux_in1,
            ac_channel<struct ChannelType>& mux_in2,
            ac_channel<struct ChannelType>& mux_out) {
  static Mux3Impl i;
  i.Run(csrs, mux_in0, mux_in1, mux_in2, mux_out);
}
