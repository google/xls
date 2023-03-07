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

typedef ac_int<16, false> MyInt;

class MyBlock {
  __xls_channel<MyInt, __xls_channel_dir_In> val_in;
  __xls_channel<MyInt, __xls_channel_dir_Out> val_out;
  ac_int<16, false> acc;

  void Run() {
    auto val = val_in.read();
    acc += val;
    val_out.write(val);
  }
};
