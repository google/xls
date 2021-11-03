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

struct ChannelType {
  ac_int<8, false> data;
};

struct Mux1To2Impl {
  void Run(ac_channel<ac_int<1, false>>& sel,
           ac_channel<struct ChannelType>& mux_in,
           ac_channel<struct ChannelType>& mux_out0,
           ac_channel<struct ChannelType>& mux_out1) {
    ChannelType in_val = mux_in.read();
    ac_int<1, false> sel_val = sel.read();

    if (sel_val == 0) {
      mux_out0.write(in_val);
    } else {
      mux_out1.write(in_val);
    }
  }
};

#pragma hls_design top
void Mux1To2_f(ac_channel<ac_int<1, false>>& sel,
               ac_channel<struct ChannelType>& mux_in,
               ac_channel<struct ChannelType>& mux_out0,
               ac_channel<struct ChannelType>& mux_out1) {
  static Mux1To2Impl i;
  i.Run(sel, mux_in, mux_out0, mux_out1);
}
