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

// This example uses a switch to ensure that the optimizer turns LookupFunc
// into a lookup rather than random logic.

#define ac_channel __xls_channel
template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

using U2 = ac_int<2, false>;
using U4 = ac_int<4, false>;

static U4 LookupFunc(const U2& i0, const U2& i1) {
  switch (i0 * 4 + i1) {
    default:
    case 0:
      return 12;
    case 0 + 1:
      return 15;
    case 0 + 2:
      return 7;
    case 0 + 3:
      return 4;
    case 4:
      return 8;
    case 4 + 1:
      return 4;
    case 4 + 2:
      return 6;
    case 4 + 3:
      return 9;
    case 8:
      return 2;
    case 8 + 1:
      return 5;
    case 8 + 2:
      return 8;
    case 8 + 3:
      return 13;
    case 12:
      return 8;
    case 12 + 1:
      return 9;
    case 12 + 2:
      return 3;
    case 12 + 3:
      return 2;
  }

  return 0;
}

struct SwitchImpl {
  void Run(ac_channel<U2>& in0, ac_channel<U2>& in1, ac_channel<U4>& out) {
    U2 in0_data = in0.read();
    U2 in1_data = in1.read();
    U4 out_data = LookupFunc(in0_data, in1_data);

    out.write(out_data);
  }
};

#pragma hls_design top
void Switch_f(ac_channel<U2>& in0, ac_channel<U2>& in1, ac_channel<U4>& out) {
  static SwitchImpl i;
  i.Run(in0, in1, out);
}
