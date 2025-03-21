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

#ifndef __AC_INT_COMPAT_H__
#define __AC_INT_COMPAT_H__

#include "xls_int.h"

template <int Width, bool Signed = true>
using ac_int = XlsInt<Width, Signed>;

using ac_datatypes::ac_o_mode;
using ac_datatypes::ac_q_mode;

using ac_datatypes::AC_RND;
using ac_datatypes::AC_RND_CONV;
using ac_datatypes::AC_RND_CONV_ODD;
using ac_datatypes::AC_RND_INF;
using ac_datatypes::AC_RND_MIN_INF;
using ac_datatypes::AC_RND_ZERO;
using ac_datatypes::AC_SAT;
using ac_datatypes::AC_SAT_SYM;
using ac_datatypes::AC_SAT_ZERO;
using ac_datatypes::AC_TRN;
using ac_datatypes::AC_TRN_ZERO;
using ac_datatypes::AC_WRAP;

using ac_datatypes::ac_special_val;
using ac_datatypes::AC_VAL_0;
using ac_datatypes::AC_VAL_DC;
using ac_datatypes::AC_VAL_MAX;
using ac_datatypes::AC_VAL_MIN;
using ac_datatypes::AC_VAL_QUANTUM;

namespace ac {
using ac_datatypes::ac::log2_ceil;
using ac_datatypes::ac::log2_floor;
using ac_datatypes::ac::nbits;
}  // namespace ac

#endif  //__AC_INT_COMPAT_H__
