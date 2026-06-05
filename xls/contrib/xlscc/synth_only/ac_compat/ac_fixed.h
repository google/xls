// Copyright 2022 The XLS Authors
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

#ifndef __AC_FIXED_COMPAT_H__
#define __AC_FIXED_COMPAT_H__

#include <ac_int.h>

#include "xls_fixed.h"

template <int Width, int IntegerWidth, bool Signed,
          ac_datatypes::ac_q_mode Quantization = AC_TRN,
          ac_datatypes::ac_o_mode Overflow = AC_WRAP>
using ac_fixed = XlsFixed<Width, IntegerWidth, Signed, Quantization, Overflow>;

#endif  // __AC_FIXED_COMPAT_H__
