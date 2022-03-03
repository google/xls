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

struct PacketSelectorImpl {
  void Run(ac_channel<struct ChannelType>& in0,
           ac_channel<struct ChannelType>& in1,
           ac_channel<struct ChannelType>& out) {
    ChannelType in0_c_val = in0.read();
    ChannelType in1_c_val = in1.read();
    ac_int<8, false> in0_val = in0_c_val.data >> 1;
    ac_int<8, false> in1_val = in1_c_val.data << 1;
    ChannelType result;
    if (in0_val == in1_val) {
      result = in0_c_val;
    } else {
      result = in1_c_val;
    }
    out.write(result);
  }
};

#pragma hls_design top
void PacketSelector_f(ac_channel<struct ChannelType>& in0,
                      ac_channel<struct ChannelType>& in1,
                      ac_channel<struct ChannelType>& out) {
  static PacketSelectorImpl i;
  i.Run(in0, in1, out);
}
