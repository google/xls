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

#ifndef __AC_FIXED_H__
#define __AC_FIXED_H__

#include "ac_int.h"

// Non-functional stub for ac_fixed
template <int W, int I, bool S = true, ac_q_mode Q = AC_TRN,
          ac_o_mode O = AC_WRAP>
class ac_fixed {
 public:
  ac_fixed() { (void)__xlscc_unimplemented(); }
  template <typename T>
  ac_fixed(const T& o) {
    (void)__xlscc_unimplemented();
  }
  int to_int() {
    (void)__xlscc_unimplemented();
    return 0;
  }
};

#endif  //__AC_FIXED_H__
